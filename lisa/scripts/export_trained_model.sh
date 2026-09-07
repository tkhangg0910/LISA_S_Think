#!/bin/bash
# Turn a finished train_ds.py run into a loadable HF checkpoint.
#
#   train_ds.py leaves runs/<exp>/ckpt_model/  (a DeepSpeed ZeRO checkpoint).
#   Two steps are needed before anything can load it:
#     1. zero_to_fp32.py     shards -> one pytorch_model.bin
#     2. merge_lora_...py    fold LoRA into the base and write HF format
#
#   bash scripts/export_trained_model.sh <exp_name> [base_ckpt]
#
# Result: runs/<exp_name>/hf_model, which scripts/eval_brain_seg.py can load.
set -euo pipefail

LISA_ROOT="${LISA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source "$LISA_ROOT/env.sh"

EXP="${1:?usage: export_trained_model.sh <exp_name> [base_ckpt]}"
BASE="${2:-$LISA_ROOT/checkpoints/LISA-7B-v1}"
RUN="$LISA_ROOT/runs/$EXP"

[ -d "$RUN/ckpt_model" ] || { echo "no $RUN/ckpt_model -- did training save?" >&2; exit 1; }

# DeepSpeed 0.16's generated zero_to_fp32.py imports
# huggingface_hub.split_torch_state_dict_into_shards (added in hf_hub 0.23), but the
# stack is pinned to hf_hub 0.17.3 for transformers 4.31. The in-memory API does the
# same consolidation without touching huggingface_hub.
echo "==> consolidating ZeRO shards -> $RUN/pytorch_model.bin"
python - <<PYEOF
import torch
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint

sd = get_fp32_state_dict_from_zero_checkpoint("$RUN/ckpt_model")
print(f"    {len(sd)} tensors")
torch.save(sd, "$RUN/pytorch_model.bin")
PYEOF

echo "==> merging LoRA -> $RUN/hf_model"
cd "$LISA_ROOT"
python merge_lora_weights_and_save_hf_model.py \
    --version="$BASE" \
    --vision-tower="$LISA_ROOT/checkpoints/clip-vit-large-patch14" \
    --vision_pretrained="$LISA_ROOT/checkpoints/sam_vit_h_4b8939.pth" \
    --weight="$RUN/pytorch_model.bin" \
    --save_path="$RUN/hf_model"

# merge_lora_weights... drops the vision tower from the state dict, and the saved
# config keeps whatever --vision-tower was passed, so make sure it points local.
python - <<PY
import json, os
p = os.path.join("$RUN", "hf_model", "config.json")
cfg = json.load(open(p))
for k in ("vision_tower", "mm_vision_tower"):
    cfg[k] = "$LISA_ROOT/checkpoints/clip-vit-large-patch14"
json.dump(cfg, open(p, "w"), indent=2)
print("config vision_tower ->", cfg["vision_tower"])
PY

echo
echo "done: $RUN/hf_model"
echo "eval: python scripts/eval_brain_seg.py --version $RUN/hf_model"
