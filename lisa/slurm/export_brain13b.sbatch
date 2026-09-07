#!/bin/bash
#SBATCH --job-name=export-13b
#SBATCH --account=rc-rse
#SBATCH --qos=rc-rse
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=250gb
#SBATCH --time=03:00:00
#SBATCH --output=logs/brain_export_13b_%j.log
set -euo pipefail
LISA_ROOT="${LISA_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"   # submit from the repo root
export LISA_ENV_NAME=env-b200
export HF_HUB_OFFLINE=1
bash $LISA_ROOT/scripts/export_trained_model.sh brain13B-q1 $LISA_ROOT/checkpoints/LISA-13B-llama2-v1
