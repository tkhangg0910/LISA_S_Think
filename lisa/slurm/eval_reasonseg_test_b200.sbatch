#!/bin/bash
# gIoU / cIoU on the ReasonSeg val split, via train_ds.py --eval_only.
# NOTE: train_ds.py builds the *train* HybridDataset before it checks --eval_only,
# so --dataset must be restricted to what actually exists on disk, otherwise
# init_ade20k()/init_cocostuff() blow up on missing directories during an eval run.
#
# NOTE: this path is teacher-forced -- "[SEG]." is already in the prompt, so it
# measures mask quality, not whether the LLM decides to emit [SEG] on its own.
#SBATCH --job-name=lisa-eval-rs-test
#SBATCH --account=rc-rse
#SBATCH --qos=rc-rse
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128gb
#SBATCH --time=04:00:00
#SBATCH --output=logs/eval_reasonseg_test_%j.log
set -euo pipefail
LISA_ROOT="${LISA_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"   # submit from the repo root
export LISA_ENV_NAME=env-b200
source $LISA_ROOT/env.sh
export HF_HUB_OFFLINE=1
module load cuda/12.8.1 2>/dev/null || true

deepspeed --num_gpus=1 --master_port=$((24000 + RANDOM % 1000)) train_ds.py \
    --version="${LISA_CKPT:-$LISA_ROOT/checkpoints/LISA-7B-v1-explanatory}" \
    --vision_pretrained="$LISA_ROOT/checkpoints/sam_vit_h_4b8939.pth" \
    --vision-tower="$LISA_ROOT/checkpoints/clip-vit-large-patch14" \
    --dataset_dir="$LISA_ROOT/dataset" \
    --val_dataset="ReasonSeg|test" \
    --dataset="reason_seg" \
    --sample_rates="1" \
    --reason_seg_data="ReasonSeg|train" \
    --exp_name="eval-lisa-7b-test" \
    --log_base_dir="$LISA_ROOT/runs" \
    --precision=bf16 \
    --eval_only
