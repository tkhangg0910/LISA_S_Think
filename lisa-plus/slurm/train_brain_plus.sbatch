#!/bin/bash
# Fine-tune LISA++ 7B on brain_data_split_v1 with the brainM recipe.
# Runs from the lisa_plus worktree, which is the only tree whose LlavaMetaModel builds
# the mlp2x_gelu projector LISA++ was trained with; the main tree would silently
# randomise it. LISA++ also ships a 336px CLIP tower (576 patches, not 256) and a
# tokenizer with no <im_start>/<im_end>, hence --no-use_mm_start_end.
#
# model_max_length must be 1536, not the 1024 used for the LISA runs: lisa_plus's
# collate_fn truncates to model_max_length - 575 (it budgets for 576 CLIP patches, not
# 256). With Q1 boxes the target reaches 483 tokens, so 1024 leaves a 449-token budget
# and clips the trailing [SEG] on 0.2% of rows -- two samples out of a thousand, enough
# to kill the run at a random step with
#   AssertionError: gt_mask.shape [3,H,W], pred_mask.shape [0,H,W]
# and few enough that a 4-step smoke test never sees it.
#SBATCH --job-name=brainPlus
#SBATCH --account=rc-rse
#SBATCH --qos=rc-rse
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=250gb
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/brain_plus_%j.log
set -euo pipefail
LISA_ROOT="${LISA_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"   # submit from the repo root
export LISA_ENV_NAME=env-b200
source $LISA_ROOT/env.sh
export HF_HUB_OFFLINE=1
module load cuda/12.8.1 2>/dev/null || true
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

deepspeed --num_gpus=1 --master_port=$((24000 + RANDOM % 1000)) train_ds.py \
    --version="$LISA_ROOT/checkpoints/LISA_Plus_7b" \
    --restore_lisa_modules \
    --no-use_mm_start_end \
    --vision_pretrained="$LISA_ROOT/checkpoints/sam_vit_h_4b8939.pth" \
    --vision-tower="$LISA_ROOT/checkpoints/clip-vit-large-patch14-336" \
    --dataset_dir="$LISA_ROOT" --log_base_dir="$LISA_ROOT/runs" \
    --dataset="brain_seg" --sample_rates="1" \
    --brain_seg_data="$LISA_ROOT/data/brain_data_split_v1/data/train-00000-of-00001.parquet" \
    --val_dataset="brain_seg" \
    --brain_seg_val_data="$LISA_ROOT/data/brain_data_split_v1/data/test-00000-of-00001.parquet" \
    --use_q1 --answer_order reason_first --clip_view center_crop \
    --train_vision_encoder_blocks 8 \
    --precision=bf16 --model_max_length=1536 \
    --batch_size=2 --grad_accumulation_steps=10 --steps_per_epoch=50 --epochs=40 \
    --lr=3e-4 --lora_r=8 \
    --ce_loss_weight=1.0 --bce_loss_weight=2.0 --dice_loss_weight=2.0 \
    --workers=8 "$@"
