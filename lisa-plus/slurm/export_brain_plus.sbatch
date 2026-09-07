#!/bin/bash
# ZeRO shards -> fp32 bin -> merge LoRA -> runs/<exp>/hf_model, for LISA++.
#
# Must run on THIS branch: only lisa_plus's merge script + model/llava1p5 rebuild
# the mlp2x_gelu projector LISA++ was trained with. The main branch would
# silently randomise it -- no error, wrong weights.
#
#   sbatch slurm/export_brain_plus.sbatch brainPlus7B-q1
#SBATCH --job-name=export-plus
#SBATCH --account=rc-rse
#SBATCH --qos=rc-rse
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=250gb
#SBATCH --time=03:00:00
#SBATCH --output=logs/export_plus_%j.log
set -euo pipefail
LISA_ROOT="${LISA_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"   # submit from the repo root
export LISA_ENV_NAME=env-b200; export HF_HUB_OFFLINE=1
source "$LISA_ROOT/env.sh"

EXP="${1:?usage: sbatch slurm/export_brain_plus.sbatch <exp_name>}"
RUN="$LISA_ROOT/runs/$EXP"
CLIP="$LISA_ROOT/checkpoints/clip-vit-large-patch14-336"

echo "==> consolidating ZeRO shards -> $RUN/pytorch_model.bin"
python - "$RUN" <<'PYEOF'
import sys, torch
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
run = sys.argv[1]
sd = get_fp32_state_dict_from_zero_checkpoint(f"{run}/ckpt_model")
print(f"    {len(sd)} tensors")
torch.save(sd, f"{run}/pytorch_model.bin")
PYEOF

echo "==> merging LoRA -> $RUN/hf_model"
python merge_lora_weights_and_save_hf_model.py \
    --no-use_mm_start_end \
    --version="$LISA_ROOT/checkpoints/LISA_Plus_7b" \
    --vision-tower="$CLIP" \
    --vision_pretrained="$LISA_ROOT/checkpoints/sam_vit_h_4b8939.pth" \
    --weight="$RUN/pytorch_model.bin" \
    --save_path="$RUN/hf_model"

# the saved config keeps whatever --vision-tower was passed; pin it to the local copy
python - "$RUN/hf_model/config.json" "$CLIP" <<'PYEOF'
import json, sys
p, clip = sys.argv[1], sys.argv[2]
c = json.load(open(p))
for k in ("vision_tower", "mm_vision_tower"):
    c[k] = clip
json.dump(c, open(p, "w"), indent=2)
print("vision_tower ->", c["vision_tower"])
PYEOF

echo; echo "done: $RUN/hf_model"
