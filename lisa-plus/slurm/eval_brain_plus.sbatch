#!/bin/bash
#SBATCH --job-name=ev-plus
#SBATCH --account=rc-rse
#SBATCH --qos=rc-rse
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200gb
#SBATCH --time=02:00:00
#SBATCH --output=logs/eval_plus_%j.log
set -euo pipefail
LISA_ROOT="${LISA_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"   # submit from the repo root
export LISA_ENV_NAME=env-b200; export HF_HUB_OFFLINE=1
source $LISA_ROOT/env.sh
python scripts/eval_brain_seg.py --parquet "$LISA_ROOT/data/brain_data_split_v1/data/test-00000-of-00001.parquet" "$@"
