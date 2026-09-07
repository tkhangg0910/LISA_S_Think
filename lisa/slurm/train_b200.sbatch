#!/bin/bash
# Fine-tune LISA-7B on ReasonSeg (LoRA + DeepSpeed ZeRO-2) on B200 GPUs.
#
# --restore_lisa_modules is ON because we fine-tune from a LISA checkpoint:
# train_ds.py otherwise rebuilds SAM from the vanilla .pth and re-randomises
# text_hidden_fcs *after* loading, silently resetting the segmentation head.
# Drop the flag if you ever start from a plain LLaVA checkpoint instead.
#
# Defaults to reason_seg only, because that is the one dataset present locally.
# For the paper recipe you also need sem_seg / refer_seg / vqa data and:
#   --dataset="sem_seg||refer_seg||vqa||reason_seg" --sample_rates="9,3,3,1"
#
#   sbatch slurm/train_b200.sbatch
#   sbatch slurm/train_b200.sbatch --epochs 20 --lr 1e-4
#SBATCH --job-name=lisa-train
#SBATCH --account=rc-rse
#SBATCH --qos=rc-rse
#SBATCH --partition=hpg-b200
#SBATCH --nodes=1
#SBATCH --gres=gpu:b200:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=256gb
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/train_%j.log
set -euo pipefail
LISA_ROOT="${LISA_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"   # submit from the repo root
export LISA_ENV_NAME=env-b200
source $LISA_ROOT/env.sh
export HF_HUB_OFFLINE=1
module load cuda/12.8.1 2>/dev/null || true

NGPU=$(nvidia-smi -L | wc -l)
echo "training on $NGPU GPU(s)"
nvidia-smi

deepspeed --num_gpus=$NGPU --master_port=$((24000 + RANDOM % 1000)) train_ds.py \
    --version="${LISA_BASE:-$LISA_ROOT/checkpoints/LISA-7B-v1}" \
    --vision_pretrained="$LISA_ROOT/checkpoints/sam_vit_h_4b8939.pth" \
    --vision-tower="$LISA_ROOT/checkpoints/clip-vit-large-patch14" \
    --dataset_dir="$LISA_ROOT/dataset" \
    --log_base_dir="$LISA_ROOT/runs" \
    --exp_name="${LISA_EXP:-lisa-7b-reasonseg}" \
    --dataset="reason_seg" \
    --sample_rates="1" \
    --reason_seg_data="ReasonSeg|train" \
    --val_dataset="ReasonSeg|val" \
    --precision=bf16 \
    --batch_size=2 \
    --grad_accumulation_steps=10 \
    --steps_per_epoch=500 \
    --epochs=10 \
    --lr=3e-4 \
    --lora_r=8 \
    --explanatory=0.1 \
    --restore_lisa_modules "$@"
