#!/usr/bin/env bash
set -euo pipefail

cd /workspace/LISA_S_Think/lisa

# Thư viện hệ thống và công cụ build cho DeepSpeed.
apt-get update
apt-get install -y \
    python3-venv python3-dev build-essential ninja-build \
    libgl1 libglib2.0-0 libaio-dev

# Environment riêng trên volume.
python3 -m venv /workspace/venvs/lisa-a100
source /workspace/venvs/lisa-a100/bin/activate

export PIP_DEFAULT_TIMEOUT=120
export PIP_RETRIES=10
export MAX_JOBS=4
export TORCH_EXTENSIONS_DIR=/workspace/LISA_S_Think/lisa/.cache/torch_extensions
mkdir -p "$TORCH_EXTENSIONS_DIR"

python -m pip install --upgrade pip
python -m pip install \
    "setuptools==69.5.1" "wheel==0.43.0" \
    "numpy==1.26.4" ninja packaging

python -m pip install \
    torch==2.1.2 torchvision==0.16.2 \
    --index-url https://download.pytorch.org/whl/cu121

DS_BUILD_OPS=0 python -m pip install --no-build-isolation \
    -r setup/requirements-hpg.txt \
    "matplotlib==3.8.4"

python -m pip check

# Kiểm tra cả import thực tế của repo và CUDA optimizer.
python - <<'PY'
import importlib
import sys

modules = [
    "torch", "torchvision", "transformers", "tokenizers",
    "huggingface_hub", "peft", "accelerate", "deepspeed",
    "bitsandbytes", "numpy", "cv2", "PIL",
    "pyarrow.parquet", "pycocotools.mask", "skimage",
    "matplotlib", "scipy", "einops", "sentencepiece",
    "tensorboard", "sklearn",
    "train_ds", "scripts.eval_brain_seg",
]

failed = []
for name in modules:
    try:
        mod = importlib.import_module(name)
        print(f"OK  {name}: {getattr(mod, '__version__', '')}")
    except Exception as exc:
        failed.append(name)
        print(f"FAIL {name}: {type(exc).__name__}: {exc}")

if failed:
    sys.exit("Dependency check failed: " + ", ".join(failed))

import torch
assert torch.cuda.is_available(), "PyTorch không thấy GPU"
print("GPU:", torch.cuda.get_device_name(0))
print("Torch CUDA:", torch.version.cuda)

from torch.utils.cpp_extension import CUDA_HOME
print("CUDA_HOME:", CUDA_HOME)
assert CUDA_HOME, "Thiếu CUDA toolkit/nvcc để build DeepSpeed FusedAdam"

from deepspeed.ops.adam import FusedAdam
param = torch.nn.Parameter(torch.ones(16, device="cuda"))
optimizer = FusedAdam([param], lr=1e-3)
param.square().sum().backward()
optimizer.step()
torch.cuda.synchronize()

print("\nDEPENDENCIES + CUDA + FUSED ADAM: OK")
PY
