#!/bin/bash
#SBATCH --job-name=lisa-setup-b200
#SBATCH --account=rc-rse
#SBATCH --qos=rc-rse
#SBATCH --partition=hpg-default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32gb
#SBATCH --time=03:00:00
#SBATCH --output=logs/setup_env_b200_%j.log
set -euo pipefail
LISA_ROOT="${LISA_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"   # submit from the repo root
bash $LISA_ROOT/setup/01b_create_env_b200.sh
