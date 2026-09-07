#!/bin/bash
# Per-class IoU/DICE on the held-out brain split.
#   sbatch slurm/eval_brain_b200.sbatch --version runs/<exp>/hf_model --out runs/<exp>/metrics.json
#SBATCH --job-name=brain-eval
#SBATCH --account=rc-rse
#SBATCH --qos=rc-rse
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128gb
#SBATCH --time=02:00:00
#SBATCH --output=logs/brain_eval_%j.log
set -euo pipefail
LISA_ROOT="${LISA_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"   # submit from the repo root
export LISA_ENV_NAME=env-b200
source $LISA_ROOT/env.sh
export HF_HUB_OFFLINE=1
python scripts/eval_brain_seg.py "$@"
