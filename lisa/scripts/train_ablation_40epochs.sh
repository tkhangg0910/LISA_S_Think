#!/usr/bin/env bash
# Run from an already activated LISA-compatible Python environment on RunPod.
set -euo pipefail
MODE="${1:-morph_only}"
MAX_LENGTH=1024
[[ "$MODE" != localization_only ]] || MAX_LENGTH=2048
case "$MODE" in localization_only|morph_only|malig_only) ;; *) echo 'Use localization_only, morph_only or malig_only' >&2; exit 2;; esac
LISA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$LISA_ROOT"
BASE="${LISA_BASE:-$LISA_ROOT/checkpoints/LISA-13B-llama2-v1}"
CLIP="${CLIP_PATH:-$LISA_ROOT/checkpoints/clip-vit-large-patch14}"
SAM="${SAM_PATH:-$LISA_ROOT/checkpoints/sam_vit_h_4b8939.pth}"
DATA="${BRAIN_DATA_DIR:-$LISA_ROOT/data/brain_data_split_v1/data}"
EXP="${EXP_NAME:-brain13B-${MODE}-40epochs-$(date -u +%Y%m%dT%H%M%S)}"
for path in "$BASE/config.json" "$CLIP/config.json" "$SAM" "$DATA/train-00000-of-00001.parquet" "$DATA/test-00000-of-00001.parquet"; do
    [[ -f "$path" ]] || { echo "Missing: $path" >&2; exit 1; }
done
[[ ! -e "$LISA_ROOT/runs/$EXP" ]] || { echo "Run already exists: $EXP. Set a new EXP_NAME to avoid auto-resume." >&2; exit 1; }
mkdir -p "$LISA_ROOT/logs" "$LISA_ROOT/.cache/torch_extensions"
export TORCH_EXTENSIONS_DIR="$LISA_ROOT/.cache/torch_extensions"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
python scripts/check_ablation_data.py --version "$BASE" --data "$DATA" --mode "$MODE" --model_max_length "$MAX_LENGTH"
unset CUDA_VISIBLE_DEVICES
# GPU_ID selects one physical GPU, including on a future two-GPU Pod.
deepspeed --include "localhost:${GPU_ID:-0}" --master_port "${MASTER_PORT:-29501}" train_ds.py \
    --version "$BASE" --restore_lisa_modules \
    --vision_pretrained "$SAM" --vision-tower "$CLIP" \
    --dataset_dir "$LISA_ROOT" --log_base_dir "$LISA_ROOT/runs" --exp_name "$EXP" \
    --dataset brain_seg --sample_rates 1 --val_dataset brain_seg \
    --brain_seg_data "$DATA/train-00000-of-00001.parquet" \
    --brain_seg_val_data "$DATA/test-00000-of-00001.parquet" \
    --reasoning_mode "$MODE" --answer_order reason_first --clip_view center_crop \
    --train_vision_encoder_blocks 8 --precision bf16 --model_max_length "$MAX_LENGTH" \
    --batch_size 1 --grad_accumulation_steps 20 --steps_per_epoch 50 --epochs 40 \
    --lr 3e-4 --lora_r 8 --ce_loss_weight 1.0 --bce_loss_weight 2.0 --dice_loss_weight 2.0 \
    --workers 4 --print_freq 1 2>&1 | tee "$LISA_ROOT/logs/$EXP.log"
echo "Timing: $LISA_ROOT/runs/$EXP/epoch_timing.jsonl"
