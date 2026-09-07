"""Re-load the LISA-specific weights that train_ds.py throws away.

THE PROBLEM
-----------
train_ds.py does, in this order:

    model = LISAForCausalLM.from_pretrained(args.version, ...)   # loads everything
    ...
    if not args.eval_only:
        model.get_model().initialize_lisa_modules(config)        # <-- destructive

`initialize_lisa_modules` does not *load* anything -- it CONSTRUCTS fresh modules:

    self.visual_model   = build_sam_vit_h(self.vision_pretrained)   # vanilla SAM
    self.text_hidden_fcs = nn.ModuleList([nn.Sequential(*text_fc)]) # random init

That is exactly right when you start from a plain LLaVA checkpoint, which has no
SAM and no projection layer. But when you fine-tune **from a LISA checkpoint**
(e.g. checkpoints/LISA-7B-v1), it silently discards the trained mask decoder and
the trained [SEG] projection layer -- so the segmentation head restarts from
scratch and the first epochs undo the released model.

Nothing warns you. Loss just starts high.

THE FIX
-------
Call `restore(model, ckpt_dir)` right after `initialize_lisa_modules`. It copies
`visual_model.*` and `text_hidden_fcs.*` back out of the checkpoint shards. If the
checkpoint has no such keys (a plain LLaVA start), it is a no-op and reports so.
"""

import json
import os

import torch

PREFIXES = ("model.visual_model.", "model.text_hidden_fcs.")


def _shard_paths(ckpt_dir):
    index = os.path.join(ckpt_dir, "pytorch_model.bin.index.json")
    if os.path.exists(index):
        with open(index) as f:
            shards = sorted(set(json.load(f)["weight_map"].values()))
        return [os.path.join(ckpt_dir, s) for s in shards]
    single = os.path.join(ckpt_dir, "pytorch_model.bin")
    return [single] if os.path.exists(single) else []


def restore(model, ckpt_dir, verbose=True):
    """Copy visual_model.* and text_hidden_fcs.* from ckpt_dir into `model`.

    Returns the number of tensors restored.
    """
    paths = _shard_paths(ckpt_dir)
    if not paths:
        if verbose:
            print(f"[restore_lisa_modules] no pytorch_model*.bin under {ckpt_dir}")
        return 0

    target = dict(model.named_parameters())
    target.update(dict(model.named_buffers()))

    restored, skipped = 0, []
    for p in paths:
        shard = torch.load(p, map_location="cpu")
        for key, value in shard.items():
            if not key.startswith(PREFIXES):
                continue
            dst = target.get(key)
            if dst is None:
                skipped.append(key)
                continue
            if dst.shape != value.shape:
                skipped.append(f"{key} (shape {tuple(dst.shape)} vs {tuple(value.shape)})")
                continue
            with torch.no_grad():
                dst.copy_(value.to(dtype=dst.dtype, device=dst.device))
            restored += 1
        del shard

    if verbose:
        if restored:
            print(f"[restore_lisa_modules] restored {restored} tensors "
                  f"(visual_model + text_hidden_fcs) from {ckpt_dir}")
        else:
            print(f"[restore_lisa_modules] {ckpt_dir} carries no LISA module weights "
                  f"-- fresh SAM + random projection layer (correct for a LLaVA start)")
        if skipped:
            print(f"[restore_lisa_modules] skipped {len(skipped)}: {skipped[:5]}")
    return restored
