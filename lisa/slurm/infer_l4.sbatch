#!/bin/bash
# Reasoning-segmentation inference on one L4 (24 GB).
#   sbatch slurm/infer_l4.sbatch --image imgs/example1.jpg --prompt "Can you segment ..."
#   sbatch slurm/infer_l4.sbatch --jsonl queries.jsonl --out_dir vis_output/run1
# 7B in bf16 is ~17 GB of weights; add --load_in_8bit if you hit OOM.
#SBATCH --job-name=lisa-infer-l4
#SBATCH --account=rc-rse
#SBATCH --qos=rc-rse
#SBATCH --partition=hpg-turin
#SBATCH --gres=gpu:l4:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --time=04:00:00
#SBATCH --output=logs/infer_l4_%j.log
set -euo pipefail
LISA_ROOT="${LISA_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"   # submit from the repo root
export LISA_ENV_NAME=env
source $LISA_ROOT/env.sh
export HF_HUB_OFFLINE=1
python scripts/infer.py \
    --version "${LISA_CKPT:-$LISA_ROOT/checkpoints/LISA-7B-v1-explanatory}" \
    --precision bf16 "$@"
