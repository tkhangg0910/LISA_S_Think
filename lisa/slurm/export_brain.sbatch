#!/bin/bash
#SBATCH --job-name=brain-export
#SBATCH --account=rc-rse
#SBATCH --qos=rc-rse
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200gb
#SBATCH --time=03:00:00
#SBATCH --output=logs/brain_export_%j.log
set -euo pipefail
LISA_ROOT="${LISA_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"   # submit from the repo root
export LISA_ENV_NAME=env-b200
export HF_HUB_OFFLINE=1
for EXP in "$@"; do
  echo "########## $EXP"
  bash $LISA_ROOT/scripts/export_trained_model.sh "$EXP"
done
