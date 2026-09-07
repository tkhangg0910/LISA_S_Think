#!/bin/bash
# Create the B200 (sm_100 / Blackwell) conda env at $LISA_ROOT/env-b200.
# Safe to re-run.
set -euo pipefail

LISA_ROOT="${LISA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export LISA_ENV_NAME="env-b200"
source "$LISA_ROOT/env.sh" || true

LISA_ENV="$LISA_ROOT/env-b200"

echo "==> env prefix: $LISA_ENV"

if [ ! -d "$LISA_ENV" ]; then
    conda create -y -p "$LISA_ENV" python=3.10
fi

# shellcheck disable=SC1091
. "$(dirname "$(dirname "$CONDA_EXE")")/etc/profile.d/conda.sh"
conda activate "$LISA_ENV"

echo "==> python: $(which python) -- $(python -V)"

python -m pip install --upgrade pip setuptools wheel

echo "==> installing torch 2.7.1 + cu128 (sm_100 / B200 compatible)"
python -m pip install \
    torch==2.7.1 torchvision==0.22.1 \
    --index-url https://download.pytorch.org/whl/cu128

echo "==> installing LISA requirements (B200 pinset)"
python -m pip install -r "$LISA_ROOT/setup/requirements-b200.txt"

echo
echo "==> installed versions"
python - <<'PY'
import importlib
for m in ["torch","torchvision","transformers","tokenizers","peft","accelerate",
          "deepspeed","numpy","cv2","pycocotools"]:
    try:
        mod = importlib.import_module(m)
        print(f"  {m:<15} {getattr(mod,'__version__','?')}")
    except Exception as e:
        print(f"  {m:<15} FAILED: {type(e).__name__}: {e}")
import torch
print(f"  torch.version.cuda   {torch.version.cuda}")
print(f"  arch_list            {torch.cuda.get_arch_list()}")
print(f"  sm_100 supported     {'sm_100' in torch.cuda.get_arch_list()}")
PY

echo
echo "DONE. Activate with:  LISA_ENV_NAME=env-b200 source $LISA_ROOT/env.sh"
