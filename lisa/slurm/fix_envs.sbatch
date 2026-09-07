#!/bin/bash
#SBATCH --job-name=lisa-fix-envs
#SBATCH --account=rc-rse
#SBATCH --qos=rc-rse
#SBATCH --partition=hpg-default
#SBATCH --cpus-per-task=8
#SBATCH --mem=32gb
#SBATCH --time=02:00:00
#SBATCH --output=logs/fix_envs_%j.log
set -euo pipefail
LISA_ROOT="${LISA_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"   # submit from the repo root
R=$LISA_ROOT

echo "########## L4 env"
LISA_ENV_NAME=env source $R/env.sh
python -m pip install "setuptools<81" scikit-image==0.22.0
python -c "import pkg_resources, skimage, torch.utils.cpp_extension; print('L4 ok', skimage.__version__)"
conda deactivate

echo "########## B200 env"
LISA_ENV_NAME=env-b200 source $R/env.sh
python -m pip install scikit-image==0.22.0
python -c "import skimage; print('B200 ok', skimage.__version__)"
