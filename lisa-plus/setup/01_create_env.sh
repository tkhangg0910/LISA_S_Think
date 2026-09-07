#!/bin/bash
# Create the LISA conda env at $LISA_ROOT/env. Safe to re-run.
set -euo pipefail

LISA_ROOT="${LISA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source "$LISA_ROOT/env.sh" || true    # env.sh warns if env is missing; that's expected here

LISA_ENV="$LISA_ROOT/env"

echo "==> conda: $(command -v conda)"
echo "==> env prefix: $LISA_ENV"
echo "==> pkgs cache: $CONDA_PKGS_DIRS"

if [ ! -d "$LISA_ENV" ]; then
    conda create -y -p "$LISA_ENV" python=3.10
fi

# shellcheck disable=SC1091
. "$(dirname "$(dirname "$CONDA_EXE")")/etc/profile.d/conda.sh"
conda activate "$LISA_ENV"

echo "==> python: $(which python) -- $(python -V)"

python -m pip install --upgrade pip wheel
# torch 2.1.2 still imports pkg_resources (removed in setuptools 81+)
python -m pip install "setuptools<81"

echo "==> installing torch 2.1.2 + cu121 (sm_89 / L4 compatible)"
python -m pip install \
    torch==2.1.2 torchvision==0.16.2 \
    --index-url https://download.pytorch.org/whl/cu121

echo "==> installing LISA requirements"
python -m pip install -r "$LISA_ROOT/setup/requirements-hpg.txt"

echo
echo "==> installed versions"
python - <<'PY'
import importlib
for m in ["torch","torchvision","transformers","tokenizers","peft","accelerate",
          "deepspeed","bitsandbytes","numpy","cv2","pycocotools"]:
    try:
        mod = importlib.import_module(m)
        print(f"  {m:<15} {getattr(mod,'__version__','?')}")
    except Exception as e:
        print(f"  {m:<15} FAILED: {type(e).__name__}: {e}")
import torch
print(f"  torch.version.cuda   {torch.version.cuda}")
print(f"  arch_list            {torch.cuda.get_arch_list()}")
PY

echo
echo "DONE. Activate with:  source $LISA_ROOT/env.sh"
