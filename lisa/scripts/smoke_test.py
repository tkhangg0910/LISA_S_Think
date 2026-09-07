"""Verify the environment can actually build and run LISA on this GPU.

Stages, each reported independently so a failure tells you where it broke:
  1. torch sees the GPU and has kernels for its compute capability
  2. the transformers / peft / deepspeed stack imports
  3. LISA's own modules import (llava + segment_anything + LISAForCausalLM)
  4. a real matmul + a SAM ViT-H forward pass on GPU
  5. (with --version) full checkpoint load and one end-to-end query

    python scripts/smoke_test.py
    python scripts/smoke_test.py --version checkpoints/LISA-7B-v1-explanatory \
                                 --image imgs/example1.jpg
"""

import argparse
import os
import sys
import traceback

# run from anywhere: put the repo root (parent of scripts/) on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OK, FAIL = "  OK  ", " FAIL "
_failures = []


def stage(name):
    def deco(fn):
        def wrapped(*a, **kw):
            print(f"\n--- {name} " + "-" * max(0, 60 - len(name)))
            try:
                out = fn(*a, **kw)
                print(f"[{OK}] {name}")
                return out
            except Exception:
                traceback.print_exc()
                print(f"[{FAIL}] {name}")
                _failures.append(name)
                return None
        return wrapped
    return deco


@stage("1. torch / GPU / arch")
def check_torch():
    import torch
    print(f"torch          {torch.__version__}")
    print(f"cuda build     {torch.version.cuda}")
    print(f"cuda available {torch.cuda.is_available()}")
    assert torch.cuda.is_available(), "no CUDA device visible"
    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    arch_list = torch.cuda.get_arch_list()
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"device         {name}  sm_{cap[0]}{cap[1]}  {total:.1f} GiB")
    print(f"arch_list      {arch_list}")
    # binary compatibility holds within a major compute-capability version.
    # "sm_86" -> (8, 6), "sm_100" -> (10, 0): the digits are one minor digit
    # preceded by the major, so major = int(digits) // 10.
    majors = {int(a.split("_")[1]) // 10 for a in arch_list if a.startswith("sm_")}
    assert cap[0] in majors, (
        f"this torch build has no kernels for sm_{cap[0]}x "
        f"(device is sm_{cap[0]}{cap[1]}, build targets {sorted(arch_list)})"
    )
    return torch


@stage("2. stack imports")
def check_stack():
    import accelerate
    import peft
    import transformers
    print(f"transformers   {transformers.__version__}")
    print(f"peft           {peft.__version__}")
    print(f"accelerate     {accelerate.__version__}")
    import deepspeed
    print(f"deepspeed      {deepspeed.__version__}")
    try:
        import bitsandbytes  # only needed for --load_in_8bit / --load_in_4bit
        print(f"bitsandbytes   {bitsandbytes.__version__}")
    except Exception as e:
        print(f"bitsandbytes   not installed ({type(e).__name__}) "
              f"-- fine unless you use 8/4-bit")


@stage("3. LISA modules import")
def check_lisa_imports():
    from model.LISA import LISAForCausalLM  # noqa: F401
    from model.llava import conversation as conversation_lib
    from model.segment_anything import build_sam_vit_h  # noqa: F401
    from utils.dataset import collate_fn  # noqa: F401
    print(f"conv templates {len(conversation_lib.conv_templates)}")


@stage("4. GPU compute: matmul + SAM ViT-H forward")
def check_gpu_compute():
    import torch
    from model.segment_anything import build_sam_vit_h

    a = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
    b = (a @ a).float()
    assert torch.isfinite(b).all(), "matmul produced non-finite values"
    print(f"bf16 matmul    ok (mean {b.mean().item():.4f})")

    sam = build_sam_vit_h(None).cuda().bfloat16().eval()
    with torch.no_grad():
        emb = sam.image_encoder(torch.randn(1, 3, 1024, 1024, device="cuda",
                                            dtype=torch.bfloat16))
    assert emb.shape == (1, 256, 64, 64), emb.shape
    assert torch.isfinite(emb.float()).all(), "SAM encoder produced non-finite values"
    print(f"SAM ViT-H      ok, image embedding {tuple(emb.shape)}")
    print(f"peak mem       {torch.cuda.max_memory_allocated()/1024**3:.2f} GiB")
    del sam, emb
    torch.cuda.empty_cache()


@stage("5. end-to-end LISA query")
def check_end_to_end(version, image, prompt, precision):
    sys.argv = ["infer.py"]
    from scripts.infer import build_model, parse_args, run_one
    from model.segment_anything.utils.transforms import ResizeLongestSide
    import torch

    args = parse_args([
        "--version", version, "--precision", precision,
        "--out_dir", os.path.join("vis_output", "smoke_test"),
    ])
    os.makedirs(args.out_dir, exist_ok=True)
    tokenizer, model, clip_proc = build_model(args)
    transform = ResizeLongestSide(args.image_size)
    r = run_one(args, tokenizer, model, clip_proc, transform, image, prompt, "smoke")
    assert r is not None, "inference returned nothing"
    assert r["answer"], "model produced an empty answer"
    print(f"peak mem       {torch.cuda.max_memory_allocated()/1024**3:.2f} GiB")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--version", default=None,
                   help="LISA checkpoint dir; stage 5 is skipped without it")
    p.add_argument("--image", default="imgs/example1.jpg")
    p.add_argument("--prompt", default=(
        "Where can the driver see the car speed in this image? "
        "Please output segmentation mask."))
    p.add_argument("--precision", default="bf16")
    a = p.parse_args()

    print("=" * 68)
    print(f"LISA smoke test   env={os.environ.get('LISA_ENV_NAME', '?')}   "
          f"host={os.uname().nodename}")
    print("=" * 68)

    check_torch()
    check_stack()
    check_lisa_imports()
    check_gpu_compute()
    if a.version:
        check_end_to_end(a.version, a.image, a.prompt, a.precision)
    else:
        print("\n--- 5. end-to-end LISA query --- skipped (no --version) ---")

    print("\n" + "=" * 68)
    if _failures:
        print("FAILED stages: " + ", ".join(_failures))
        sys.exit(1)
    print("all stages passed")


if __name__ == "__main__":
    main()
