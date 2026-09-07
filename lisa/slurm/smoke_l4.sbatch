#!/bin/bash
#SBATCH --job-name=lisa-smoke-l4
#SBATCH --account=rc-rse
#SBATCH --qos=rc-rse
#SBATCH --partition=hpg-turin
#SBATCH --gres=gpu:l4:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --time=01:00:00
#SBATCH --output=logs/smoke_l4_%j.log
set -euo pipefail
LISA_ROOT="${LISA_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"   # submit from the repo root
export LISA_ENV_NAME=env
source $LISA_ROOT/env.sh
export HF_HUB_OFFLINE=1
nvidia-smi
python scripts/smoke_test.py "$@"
