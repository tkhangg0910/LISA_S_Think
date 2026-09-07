#!/bin/bash
# Cross-check: drive upstream chat.py through stdin, to prove scripts/infer.py
# is faithful to it (or find where it isn't).
#SBATCH --job-name=lisa-chatpy-xc
#SBATCH --account=rc-rse
#SBATCH --qos=rc-rse
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128gb
#SBATCH --time=01:00:00
#SBATCH --output=logs/chatpy_xc_%j.log
set -uo pipefail
export LISA_ENV_NAME=env-b200
source $LISA_ROOT/env.sh
export HF_HUB_OFFLINE=1
printf 'Can you segment the steering wheel in this image?\nimgs/example1.jpg\nWhere can the driver see the car speed in this image? Please output segmentation mask.\nimgs/example1.jpg\n'  | python chat.py      --version="$LISA_ROOT/checkpoints/LISA-7B-v1-explanatory"      --vision-tower="$LISA_ROOT/checkpoints/clip-vit-large-patch14"      --precision=bf16      --vis_save_path="$LISA_ROOT/vis_output/chatpy_xc"
