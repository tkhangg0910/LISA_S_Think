#!/bin/bash
#SBATCH --job-name=lisa-smoke-b200
#SBATCH --account=rc-rse
#SBATCH --qos=rc-rse
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128gb
#SBATCH --time=01:00:00
#SBATCH --output=logs/smoke_b200_%j.log
set -euo pipefail
LISA_ROOT="${LISA_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"   # submit from the repo root
export LISA_ENV_NAME=env-b200
source $LISA_ROOT/env.sh
export HF_HUB_OFFLINE=1
nvidia-smi
python scripts/smoke_test.py "$@"
