"""Non-interactive LISA reasoning-segmentation inference.

chat.py is an interactive REPL, which is useless inside a SLURM batch job.
This is the same pipeline driven from the command line or a JSONL manifest.

Single query:
    python scripts/infer.py \
        --version checkpoints/LISA-7B-v1-explanatory \
        --image imgs/example1.jpg \
        --prompt "Where can the driver see the car speed in this image? Please output segmentation mask."

Batch (one JSON object per line, keys: image, prompt, optional id):
    python scripts/infer.py --version ... --jsonl queries.jsonl --out_dir vis_output/run1
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, BitsAndBytesConfig, CLIPImageProcessor

from model.LISA import LISAForCausalLM
from model.llava import conversation as conversation_lib
from model.llava.mm_utils import tokenizer_image_token
from model.segment_anything.utils.transforms import ResizeLongestSide
from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX)


_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_LOCAL_CLIP = os.path.join(_ROOT, "checkpoints", "clip-vit-large-patch14")

# transformers treats any unknown from_pretrained kwarg that matches a config
# field as a CONFIG OVERRIDE. So passing vision_tower="openai/clip-vit-large-patch14"
# silently overwrites the local path baked into the checkpoint config and the
# model then tries to reach the Hub -- which fails under HF_HUB_OFFLINE=1.
# Default to the local copy whenever it exists.
DEFAULT_VISION_TOWER = (
    _LOCAL_CLIP if os.path.isdir(_LOCAL_CLIP) else "openai/clip-vit-large-patch14"
)


def parse_args(argv):
    p = argparse.ArgumentParser(description="LISA batch inference")
    p.add_argument("--version", required=True,
                   help="path to a LISA HF checkpoint, e.g. checkpoints/LISA-7B-v1-explanatory")
    p.add_argument("--image", default=None, help="single image path")
    p.add_argument("--prompt", default=None, help="single text query")
    p.add_argument("--jsonl", default=None,
                   help="manifest with one {'image':..., 'prompt':...} per line")
    p.add_argument("--out_dir", default="./vis_output")
    p.add_argument("--precision", default="bf16", choices=["fp32", "bf16", "fp16"])
    p.add_argument("--load_in_8bit", action="store_true", default=False)
    p.add_argument("--load_in_4bit", action="store_true", default=False)
    p.add_argument("--image_size", default=1024, type=int)
    p.add_argument("--model_max_length", default=512, type=int)
    p.add_argument("--max_new_tokens", default=512, type=int)
    p.add_argument("--vision-tower", default=DEFAULT_VISION_TOWER, type=str)
    p.add_argument("--use_mm_start_end", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--conv_type", default="llava_v1",
                   choices=["llava_v1", "llava_llama_2"])
    p.add_argument("--local-rank", default=0, type=int)
    return p.parse_args(argv)


def preprocess_sam(
    x,
    pixel_mean=torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1),
    pixel_std=torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1),
    img_size=1024,
) -> torch.Tensor:
    """Normalize pixel values and pad to a square input (same as chat.py)."""
    x = (x - pixel_mean) / pixel_std
    h, w = x.shape[-2:]
    return F.pad(x, (0, img_size - w, 0, img_size - h))


def build_model(args):
    tokenizer = AutoTokenizer.from_pretrained(
        args.version,
        cache_dir=None,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token
    seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]

    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half

    kwargs = {"torch_dtype": torch_dtype}
    if args.load_in_4bit:
        kwargs.update({
            "torch_dtype": torch.half,
            "quantization_config": BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                llm_int8_skip_modules=["visual_model"],
            ),
        })
    elif args.load_in_8bit:
        kwargs.update({
            "torch_dtype": torch.half,
            "quantization_config": BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_skip_modules=["visual_model"],
            ),
        })

    model = LISAForCausalLM.from_pretrained(
        args.version,
        low_cpu_mem_usage=True,
        vision_tower=args.vision_tower,
        seg_token_idx=seg_token_idx,
        **kwargs,
    )
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch_dtype)

    if args.load_in_4bit or args.load_in_8bit:
        pass  # bitsandbytes already placed the weights on GPU
    elif args.precision == "bf16":
        model = model.bfloat16().cuda()
    elif args.precision == "fp16":
        model = model.half().cuda()
    else:
        model = model.float().cuda()

    model.get_model().get_vision_tower().to(device="cuda")
    model.eval()

    clip_image_processor = CLIPImageProcessor.from_pretrained(model.config.vision_tower)
    return tokenizer, model, clip_image_processor


def cast(t, precision):
    if precision == "bf16":
        return t.bfloat16()
    if precision == "fp16":
        return t.half()
    return t.float()


@torch.no_grad()
def run_one(args, tokenizer, model, clip_image_processor, transform, image_path, prompt, tag):
    image_np = cv2.imread(image_path)
    if image_np is None:
        print(f"[skip] cannot read {image_path}", flush=True)
        return None
    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    original_size_list = [image_np.shape[:2]]

    conv = conversation_lib.conv_templates[args.conv_type].copy()
    conv.messages = []
    text = DEFAULT_IMAGE_TOKEN + "\n" + prompt
    if args.use_mm_start_end:
        text = text.replace(
            DEFAULT_IMAGE_TOKEN,
            DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN,
        )
    conv.append_message(conv.roles[0], text)
    conv.append_message(conv.roles[1], "")
    full_prompt = conv.get_prompt()

    image_clip = clip_image_processor.preprocess(image_np, return_tensors="pt")[
        "pixel_values"][0].unsqueeze(0).cuda()
    image_clip = cast(image_clip, args.precision)

    image = transform.apply_image(image_np)
    resize_list = [image.shape[:2]]
    image = preprocess_sam(
        torch.from_numpy(image).permute(2, 0, 1).contiguous()
    ).unsqueeze(0).cuda()
    image = cast(image, args.precision)

    input_ids = tokenizer_image_token(full_prompt, tokenizer, return_tensors="pt")
    input_ids = input_ids.unsqueeze(0).cuda()

    output_ids, pred_masks = model.evaluate(
        image_clip, image, input_ids, resize_list, original_size_list,
        max_new_tokens=args.max_new_tokens, tokenizer=tokenizer,
    )
    output_ids = output_ids[0][output_ids[0] != IMAGE_TOKEN_INDEX]
    text_output = tokenizer.decode(output_ids, skip_special_tokens=False)
    # strip the echoed prompt: llava_v1 answers after "ASSISTANT:", llama_2 after "[/INST]"
    for marker in ("ASSISTANT:", "[/INST]"):
        if marker in text_output:
            text_output = text_output.split(marker)[-1]
            break
    text_output = text_output.replace("\n", "").replace("  ", " ").strip()

    print(f"\n=== {tag} ===")
    print(f"image : {image_path}")
    print(f"prompt: {prompt}")
    print(f"answer: {text_output}", flush=True)

    # pred_masks holds one entry per image, each of shape [n_seg, H, W]. One answer
    # can carry several [SEG] tokens ("... is [SEG]. The other is [SEG]."), so write
    # them all -- chat.py only ever saves row 0 and silently drops the rest.
    saved = []
    for i, mask_set in enumerate(pred_masks):
        if mask_set.shape[0] == 0:
            continue
        for j in range(mask_set.shape[0]):
            pred_mask = mask_set[j].detach().float().cpu().numpy() > 0
            suffix = f"{i}" if mask_set.shape[0] == 1 else f"{i}_{j}"

            mask_path = os.path.join(args.out_dir, f"{tag}_mask_{suffix}.png")
            cv2.imwrite(mask_path, pred_mask.astype(np.uint8) * 255)

            overlay = image_np.copy()
            overlay[pred_mask] = (
                image_np * 0.5
                + pred_mask[:, :, None].astype(np.uint8) * np.array([255, 0, 0]) * 0.5
            )[pred_mask]
            overlay_path = os.path.join(args.out_dir, f"{tag}_overlay_{suffix}.jpg")
            cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            saved += [mask_path, overlay_path]
            print(f"saved : {mask_path} | {overlay_path} "
                  f"({100 * pred_mask.mean():.2f}% of image)", flush=True)

    if not saved:
        print("saved : (no [SEG] token produced -- text-only answer)", flush=True)

    return {"id": tag, "image": image_path, "prompt": prompt,
            "answer": text_output, "outputs": saved}


def main(argv):
    args = parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)

    if args.jsonl:
        with open(args.jsonl) as f:
            queries = [json.loads(l) for l in f if l.strip()]
    elif args.image and args.prompt:
        queries = [{"image": args.image, "prompt": args.prompt}]
    else:
        sys.exit("need either --image and --prompt, or --jsonl")

    tokenizer, model, clip_image_processor = build_model(args)
    transform = ResizeLongestSide(args.image_size)

    results = []
    for n, q in enumerate(queries):
        tag = str(q.get("id") or
                  os.path.splitext(os.path.basename(q["image"]))[0] + f"_{n:04d}")
        r = run_one(args, tokenizer, model, clip_image_processor, transform,
                    q["image"], q["prompt"], tag)
        if r:
            results.append(r)
        torch.cuda.empty_cache()

    out = os.path.join(args.out_dir, "results.jsonl")
    with open(out, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n{len(results)} queries done -> {out}")


if __name__ == "__main__":
    main(sys.argv[1:])
