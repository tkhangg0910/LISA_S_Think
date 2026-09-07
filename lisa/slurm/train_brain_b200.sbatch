#!/bin/bash
# Fine-tune LISA on brain_data_split_v1 for segmentation quality.
#
# The defaults below ARE the published configuration -- they reproduce the
# "LISA (7B/13B) (-R)" column of results/matrix.csv. Nothing needs to be
# appended on the command line except the experiment name (and the base
# checkpoint for 13B). See REPRODUCE.md.
#
#   sbatch slurm/train_brain_b200.sbatch --exp_name brainM-q1          # 7B
#   LISA_BASE=$PWD/checkpoints/LISA-13B-llama2-v1 \
#     sbatch slurm/train_brain_b200.sbatch --exp_name brain13B-q1      # 13B
#   sbatch slurm/train_brain_b200.sbatch --steps_per_epoch 5 --epochs 1 --exp_name smoke
#
# Why these values:
#   --dice_loss_weight 2.0   lesions occupy 0.5-5.6% of the image; at that
#                            imbalance BCE alone collapses to predicting empty
#   --train_vision_encoder_blocks 8
#                            SAM ViT-H was pretrained on natural images; on MRI
#                            the frozen encoder is the bottleneck, not the decoder
#   --use_q1                 prepend A1's localisation boxes to the reasoning chain
#   --answer_order reason_first
#                            reasoning text precedes the three [SEG]; attention is
#                            causal, so the mask head can read it (see README.md)
#   --restore_lisa_modules   REQUIRED. Without it train_ds.py rebuilds SAM and
#                            re-randomises text_hidden_fcs after loading, silently
#                            resetting the segmentation head to scratch.
#   batch 2 x grad_accum 10 x 50 steps = 1000 samples = exactly one pass over train
#
# Submit from the repository root (SLURM_SUBMIT_DIR is used as LISA_ROOT).
#SBATCH --job-name=lisa-brain-train
#SBATCH --account=rc-rse
#SBATCH --qos=rc-rse
#SBATCH --partition=hpg-b200
#SBATCH --nodes=1
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200gb
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/brain_train_%j.log
set -euo pipefail
LISA_ROOT="${LISA_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"   # submit from the repo root
export LISA_ENV_NAME=env-b200
source "$LISA_ROOT/env.sh"
export HF_HUB_OFFLINE=1
module load cuda/12.8.1 2>/dev/null || true
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

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
    --use_q1 \
    --answer_order=reason_first \
    --clip_view=center_crop \
    --train_vision_encoder_blocks=8 \
    --precision=bf16 \
    --model_max_length=1024 \
    --batch_size=2 \
    --grad_accumulation_steps=10 \
    --steps_per_epoch=50 \
    --epochs=40 \
    --lr=3e-4 \
    --lora_r=8 \
    --ce_loss_weight=1.0 \
    --bce_loss_weight=2.0 \
    --dice_loss_weight=2.0 \
    --workers=8 \
    "$@"
