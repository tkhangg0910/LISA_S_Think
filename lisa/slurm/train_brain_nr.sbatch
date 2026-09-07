#!/bin/bash
# Non-reasoning ABLATION -- reproduces the "LISA (7B/13B) ours-NR" column.
#
# Identical to slurm/train_brain_b200.sbatch except the target contains NO
# reasoning: the answer is "Status (<label>)." followed by the three [SEG], with
# no A1 boxes, no A2 findings, no presence sentence and no A3 severity. That is
# the like-for-like measurement of what the reasoning chain contributes.
#
# Note it trains for 80 epochs, twice the reasoning runs' 40 -- the ablation is
# not handicapped by budget. It still scores lower. See README.md.
#
#   sbatch slurm/train_brain_nr.sbatch --exp_name brainNR7B-a4seg
#   LISA_BASE=$PWD/checkpoints/LISA-13B-llama2-v1 \
#     sbatch slurm/train_brain_nr.sbatch --exp_name brainNR13B-a4seg
#SBATCH --job-name=lisa-brain-nr
#SBATCH --account=rc-rse
#SBATCH --qos=rc-rse
#SBATCH --partition=hpg-b200
#SBATCH --nodes=1
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200gb
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/brain_train_nr_%j.log
set -euo pipefail
LISA_ROOT="${LISA_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"   # submit from the repo root
export LISA_ENV_NAME=env-b200
source "$LISA_ROOT/env.sh"
export HF_HUB_OFFLINE=1
module load cuda/12.8.1 2>/dev/null || true

deepspeed --num_gpus=1 --master_port=$((24000 + RANDOM % 1000)) train_ds.py \
    --version="${LISA_BASE:-$LISA_ROOT/checkpoints/LISA-7B-v1}" \
    --restore_lisa_modules \
    --vision_pretrained="$LISA_ROOT/checkpoints/sam_vit_h_4b8939.pth" \
    --vision-tower="$LISA_ROOT/checkpoints/clip-vit-large-patch14" \
    --dataset_dir="$LISA_ROOT" \
    --log_base_dir="$LISA_ROOT/runs" \
    --dataset="brain_seg" \
    --sample_rates="1" \
    --brain_seg_data="$LISA_ROOT/data/brain_data_split_v1/data/train-00000-of-00001.parquet" \
    --val_dataset="brain_seg" \
    --brain_seg_val_data="$LISA_ROOT/data/brain_data_split_v1/data/test-00000-of-00001.parquet" \
    --answer_order=a4_seg \
    --clip_view=center_crop \
    --train_vision_encoder_blocks=8 \
    --precision=bf16 \
    --model_max_length=1024 \
    --batch_size=2 \
    --grad_accumulation_steps=10 \
    --steps_per_epoch=50 \
    --epochs=80 \
    --lr=3e-4 \
    --lora_r=8 \
    --ce_loss_weight=1.0 \
    --bce_loss_weight=2.0 \
    --dice_loss_weight=2.0 \
    --workers=8 \
    "$@"
