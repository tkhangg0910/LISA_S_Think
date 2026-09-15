"""Per-class IoU / DICE for the brain-MRI LISA model.

train_ds.py's validate() pools all masks into one gIoU/cIoU number, which hides
the class that matters least in pixels and most clinically: MTA is non-empty in
only ~6% of rows, so a pooled score is essentially GCA's score. This reports
MTA / GCA / Koedam separately plus an overall.

Measurement is teacher-forced (the answer with its three [SEG] is fed in, the
model only produces the masks) -- same path as validate(). That is deliberate:
it keeps the k-th [SEG] bound to the k-th class, so per-class numbers are
comparable across checkpoints. It does not measure whether the model would emit
[SEG] on its own.

Reported per class, and for OVERALL (all classes pooled):
  gIoU  mean per-sample IoU
  cIoU  sum(intersection) / sum(union)  -- area-weighted
  DICE  mean per-sample Dice
Each over two populations:
  present  samples whose ground-truth mask is non-empty  <- the headline number
  all      every sample; an empty GT predicted empty counts as 1.0

    python scripts/eval_brain_seg.py --version runs/<exp>/hf_model
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.llava import conversation as conversation_lib  # noqa: E402
from scripts.infer import build_model, parse_args as infer_parse_args  # noqa: E402
from utils.brain_seg_dataset import CLASSES, BrainSegValDataset  # noqa: E402
from utils.dataset import collate_fn  # noqa: E402
from utils.utils import dict_to_cuda  # noqa: E402


def parse_args(argv):
    p = argparse.ArgumentParser(description="brain-MRI per-class IoU/DICE")
    p.add_argument("--reasoning_mode", default="full", choices=["full", "localization_only", "morph_only", "malig_only"])
    p.add_argument("--model_max_length", type=int, default=512)
    p.add_argument("--version", required=True, help="LISA checkpoint dir")
    p.add_argument("--parquet",
                   default="data/brain_data_split_v1/data/test-00000-of-00001.parquet")
    p.add_argument("--precision", default="bf16", choices=["fp32", "bf16", "fp16"])
    p.add_argument("--out", default=None, help="write metrics json here")
    p.add_argument("--limit", default=0, type=int, help="debug: stop after N samples")
    p.add_argument("--thresholds", default="", type=str,
                   help="comma-separated logit thresholds to sweep, e.g. '-1,-0.5,0,0.5'. "
                        "The mask decoder emits logits and the stock binarisation is >0 "
                        "(sigmoid 0.5). Measured on brainM the predictions are "
                        "systematically SMALLER than ground truth (median pred/gt area "
                        "0.87x GCA, 0.81x Koedam), so a lower threshold may fit better. "
                        "Costs one extra comparison per threshold, no extra forward pass.")
    p.add_argument("--answer_order", default="reason_first",
                   choices=["reason_first", "seg_first", "a4_seg"],
                   help="must match how the checkpoint was trained")
    p.add_argument("--clip_view", default="center_crop",
                   choices=["center_crop", "pad_square", "tight_pad"],
                   help="must match how the checkpoint was trained")
    p.add_argument("--use_q1", action="store_true", default=False,
                   help="must match how the checkpoint was trained")
    p.add_argument("--neutral_reasoning", action="store_true", default=False,
                   help="strip the findings/severity/presence text from the forced "
                        "answer, keeping only the three [SEG]. Measures how much of "
                        "the score is driven by the image rather than by the "
                        "ground-truth reasoning that teacher forcing supplies.")
    return p.parse_args(argv)


class Acc:
    """Accumulates the three statistics for one population."""

    def __init__(self):
        self.inter = 0.0
        self.union = 0.0
        self.ious = []
        self.dices = []

    def add(self, inter, union, psum, gsum):
        self.inter += inter
        self.union += union
        self.ious.append(1.0 if union == 0 else inter / union)
        self.dices.append(1.0 if psum + gsum == 0 else 2 * inter / (psum + gsum))

    def report(self):
        if not self.ious:
            return {"n": 0, "gIoU": float("nan"), "cIoU": float("nan"),
                    "DICE": float("nan")}
        return {
            "n": len(self.ious),
            "gIoU": float(np.mean(self.ious)),
            "cIoU": float(self.inter / (self.union + 1e-10)),
            "DICE": float(np.mean(self.dices)),
        }


def main(argv):
    args = parse_args(argv)

    margs = infer_parse_args(["--version", args.version, "--precision", args.precision, "--model_max_length", str(args.model_max_length)])

    # collate_fn splits rounds on the *default* conversation's separators, while the
    # dataset renders prompts with the same global. train_ds.py sets this; infer.py
    # does not, so without it the dataset writes vicuna_v0 turns that collate_fn then
    # fails to split -> AssertionError: (1, 'A').
    conversation_lib.default_conversation = conversation_lib.conv_templates[
        margs.conv_type
    ]

    tokenizer, model, _ = build_model(margs)
    model.eval()

    ds = BrainSegValDataset(".", tokenizer, model.config.vision_tower, args.parquet,
                            neutral_reasoning=args.neutral_reasoning,
                            answer_order=args.answer_order,
                            clip_view=args.clip_view, use_q1=args.use_q1,
                            reasoning_mode=args.reasoning_mode)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=lambda b: collate_fn(b, tokenizer=tokenizer,
                                        conv_type=margs.conv_type,
                                        use_mm_start_end=margs.use_mm_start_end),
    )

    THR = [float(x) for x in args.thresholds.split(",")] if args.thresholds else [0.0]
    pops = ["present", "all"]
    acc = {t: {c: {p: Acc() for p in pops} for c in CLASSES + ["OVERALL"]} for t in THR}
    n_empty_gt = {c: 0 for c in CLASSES}
    n_empty_correct = {c: 0 for c in CLASSES}
    per_sample = []  # one record per (sample, class) with non-empty GT

    for i, batch in enumerate(tqdm.tqdm(loader)):
        if args.limit and i >= args.limit:
            break
        batch = dict_to_cuda(batch)
        cast = {"bf16": torch.bfloat16, "fp16": torch.half, "fp32": torch.float}[
            args.precision]
        batch["images"] = batch["images"].to(cast)
        batch["images_clip"] = batch["images_clip"].to(cast)

        with torch.no_grad():
            out = model(**batch)

        gt = out["gt_masks"][0].int()            # [3, H, W]
        logits = out["pred_masks"][0]            # [3, H, W], raw logits
        assert gt.shape[0] == len(CLASSES) == logits.shape[0], (gt.shape, logits.shape)

        for thr in THR:
            pred = (logits > thr).int()
            for k, cls in enumerate(CLASSES):
                g, p = gt[k], pred[k]
                inter = torch.logical_and(g == 1, p == 1).sum().item()
                gsum, psum = (g == 1).sum().item(), (p == 1).sum().item()
                union = gsum + psum - inter

                acc[thr][cls]["all"].add(inter, union, psum, gsum)
                acc[thr]["OVERALL"]["all"].add(inter, union, psum, gsum)
                if gsum > 0:
                    acc[thr][cls]["present"].add(inter, union, psum, gsum)
                    acc[thr]["OVERALL"]["present"].add(inter, union, psum, gsum)
                    if thr == THR[0]:
                        per_sample.append({
                            "id": batch["image_paths"][0], "cls": cls,
                            "IoU": inter / union if union else 1.0,
                            "DICE": 2 * inter / (psum + gsum) if psum + gsum else 1.0,
                            "gt_px": gsum, "pred_px": psum,
                        })
                elif thr == THR[0]:
                    n_empty_gt[cls] += 1
                    n_empty_correct[cls] += int(psum == 0)

    if len(THR) > 1:
        print(f"\n{'thr':>6} " + " ".join(f"{c:>18}" for c in CLASSES + ["OVERALL"]))
        print(f"{'':>6} " + " ".join(f"{'IoU     DICE':>18}" for _ in CLASSES + ["OVERALL"]))
        print("-" * 82)
        for t in THR:
            row = []
            for c in CLASSES + ["OVERALL"]:
                r = acc[t][c]["present"].report()
                row.append(f"{r['gIoU']:>8.4f} {r['DICE']:>9.4f}")
            print(f"{t:>6.2f} " + " ".join(row))
        best = max(THR, key=lambda t: acc[t]["OVERALL"]["present"].report()["gIoU"])
        print(f"\nbest threshold by OVERALL IoU: {best}")
    acc = acc[THR[0]]
    result = {c: {p: acc[c][p].report() for p in pops} for c in CLASSES + ["OVERALL"]}
    result["empty_gt"] = {
        c: {"n": n_empty_gt[c],
            "predicted_empty": n_empty_correct[c],
            "rate": (n_empty_correct[c] / n_empty_gt[c]) if n_empty_gt[c] else None}
        for c in CLASSES
    }
    result["_meta"] = {"version": args.version, "parquet": args.parquet,
                       "teacher_forced": True,
                       "neutral_reasoning": args.neutral_reasoning,
                       "answer_order": args.answer_order,
                       "clip_view": args.clip_view,
                       "use_q1": args.use_q1,
                       "reasoning_mode": args.reasoning_mode}

    # per-sample IoU so the distribution can be inspected, not just its mean
    for cls in CLASSES + ["OVERALL"]:
        v = np.array([r["IoU"] for r in per_sample
                      if cls == "OVERALL" or r["cls"] == cls])
        if len(v):
            result[cls]["present"]["IoU_percentiles"] = {
                k: float(np.percentile(v, q))
                for k, q in [("p10", 10), ("p25", 25), ("median", 50),
                             ("p75", 75), ("p90", 90)]
            }
            result[cls]["present"]["IoU_frac_below_0.25"] = float((v < 0.25).mean())
            result[cls]["present"]["IoU_frac_above_0.75"] = float((v > 0.75).mean())
    result["per_sample"] = per_sample

    print(f"\ncheckpoint : {args.version}")
    print(f"data       : {args.parquet}\n")
    hdr = f"{'class':<9} | {'n':>4} {'gIoU':>7} {'cIoU':>7} {'DICE':>7}  |  " \
          f"{'n':>4} {'gIoU':>7} {'cIoU':>7} {'DICE':>7}"
    print(f"{'':<9} | {'--- GT non-empty ---':^30}  |  {'--- all samples ---':^30}")
    print(hdr)
    print("-" * len(hdr))
    for c in CLASSES + ["OVERALL"]:
        a, b = result[c]["present"], result[c]["all"]
        print(f"{c:<9} | {a['n']:>4} {a['gIoU']:>7.4f} {a['cIoU']:>7.4f} {a['DICE']:>7.4f}"
              f"  |  {b['n']:>4} {b['gIoU']:>7.4f} {b['cIoU']:>7.4f} {b['DICE']:>7.4f}")
    print("\nempty-GT samples predicted empty (specificity):")
    for c in CLASSES:
        e = result["empty_gt"][c]
        r = "n/a" if e["rate"] is None else f"{e['rate']:.3f}"
        print(f"  {c:<8} {e['predicted_empty']:>4}/{e['n']:<4}  {r}")

    out = args.out or os.path.join("runs", "eval_brain_seg.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main(sys.argv[1:])
