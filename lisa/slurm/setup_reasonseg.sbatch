#!/bin/bash
#SBATCH --job-name=lisa-reasonseg
#SBATCH --account=rc-rse
#SBATCH --qos=rc-rse
#SBATCH --partition=hpg-default
#SBATCH --cpus-per-task=4
#SBATCH --mem=16gb
#SBATCH --time=02:00:00
#SBATCH --output=logs/setup_reasonseg_%j.log
set -uo pipefail
bash $LISA_ROOT/setup/03_download_reasonseg.sh
