#!/usr/bin/env bash
set -euo pipefail
source /workspace/venvs/lisa-a100/bin/activate
cd "$(dirname "${BASH_SOURCE[0]}")/.."
EXP="${1:?Pass the localization training experiment name}"
[[ "$EXP" == brain13B-localization_only-* && "$EXP" != */* ]] || exit 2
RUN="$PWD/runs/$EXP"
export CUDA_VISIBLE_DEVICES="${GPU_ID:-0}"
export TOKENIZERS_PARALLELISM=false
python - "$RUN" <<'PYEXPORT'
import sys
import torch
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
run = sys.argv[1]
state = get_fp32_state_dict_from_zero_checkpoint(f"{run}/ckpt_model")
torch.save(state, f"{run}/pytorch_model.bin")
PYEXPORT
python merge_lora_weights_and_save_hf_model.py \
    --version "$PWD/checkpoints/LISA-13B-llama2-v1" \
    --vision-tower "$PWD/checkpoints/clip-vit-large-patch14" \
    --vision_pretrained "$PWD/checkpoints/sam_vit_h_4b8939.pth" \
    --model_max_length 2048 --lora_r 8 \
    --weight "$RUN/pytorch_model.bin" --save_path "$RUN/hf_model"
python scripts/eval_brain_seg.py \
    --version "$RUN/hf_model" \
    --parquet data/brain_data_split_v1/data/test-00000-of-00001.parquet \
    --reasoning_mode localization_only --answer_order reason_first \
    --clip_view center_crop --precision bf16 --model_max_length 2048 \
    --out "$RUN/eval_test.json" 2>&1 | tee "$RUN/eval_test.log"
rm -- "$RUN/pytorch_model.bin"
