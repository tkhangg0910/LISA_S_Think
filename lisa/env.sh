# =============================================================================
#  LISA project environment  --  source this file before doing anything.
#
#      source /path/to/this/repo/env.sh
#
#  Every cache / package / weight / temp file is redirected INSIDE $LISA_ROOT.
#  Nothing is written to $HOME (except conda's ~/.conda/environments.txt
#  registry line, which conda writes unconditionally and is a few bytes).
# =============================================================================

export LISA_ROOT="${LISA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"

# Two envs, because the two GPU generations on this cluster need different
# CUDA builds and nothing links them:
#   env       torch 2.1.2+cu121  -> L4        (sm_89, 24 GB)  inference
#   env-b200  torch 2.7.1+cu128  -> B200      (sm_100, 180GB) training + inference
# Override with:  LISA_ENV_NAME=env-b200 source env.sh
if [ -z "${LISA_ENV_NAME:-}" ]; then
    if command -v nvidia-smi >/dev/null 2>&1 && \
       nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | grep -qi "b200\|blackwell"; then
        LISA_ENV_NAME="env-b200"
    else
        LISA_ENV_NAME="env"
    fi
fi
export LISA_ENV_NAME
export LISA_ENV="$LISA_ROOT/$LISA_ENV_NAME"

# --- keep every cache inside the project -------------------------------------
export XDG_CACHE_HOME="$LISA_ROOT/.cache"
export PIP_CACHE_DIR="$LISA_ROOT/.cache/pip"
export CONDA_PKGS_DIRS="$LISA_ROOT/.cache/conda_pkgs"
export CONDA_ENVS_DIRS="$LISA_ROOT/.cache/conda_envs"
export HF_HOME="$LISA_ROOT/.cache/huggingface"
export HUGGINGFACE_HUB_CACHE="$LISA_ROOT/.cache/huggingface/hub"
export TRANSFORMERS_CACHE="$LISA_ROOT/.cache/huggingface/transformers"
export TORCH_HOME="$LISA_ROOT/.cache/torch"
export TORCH_EXTENSIONS_DIR="$LISA_ROOT/.cache/torch_extensions"
export TRITON_CACHE_DIR="$LISA_ROOT/.cache/triton"
export TRITON_HOME="$LISA_ROOT/.cache"          # otherwise triton writes ~/.triton
export TRITON_DUMP_DIR="$LISA_ROOT/.cache/triton"
export MPLCONFIGDIR="$LISA_ROOT/.cache/matplotlib"
export TMPDIR="$LISA_ROOT/.cache/tmp"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"

mkdir -p "$PIP_CACHE_DIR" "$CONDA_PKGS_DIRS" "$CONDA_ENVS_DIRS" \
         "$HUGGINGFACE_HUB_CACHE" "$TORCH_HOME" "$TORCH_EXTENSIONS_DIR" \
         "$TRITON_CACHE_DIR" "$MPLCONFIGDIR" "$TMPDIR" \
         "$LISA_ROOT/checkpoints" "$LISA_ROOT/logs" "$LISA_ROOT/runs" \
         "$LISA_ROOT/dataset" "$LISA_ROOT/vis_output"

# --- runtime knobs -----------------------------------------------------------
export TOKENIZERS_PARALLELISM=false
export PYTHONNOUSERSITE=1          # ignore ~/.local/lib/python*/site-packages
export OMP_NUM_THREADS=8

# --- activate the conda env --------------------------------------------------
if [ -z "$_CONDA_ROOT" ]; then
    module load conda/25.7.0 >/dev/null 2>&1 || true
fi
__conda_base="$(dirname "$(dirname "${CONDA_EXE:-/apps/conda/25.7.0/bin/conda}")")"
if [ -f "$__conda_base/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1091
    . "$__conda_base/etc/profile.d/conda.sh"
fi
unset __conda_base

if [ -d "$LISA_ENV" ]; then
    conda activate "$LISA_ENV"
else
    echo "[env.sh] conda env not found at $LISA_ENV -- run setup/01_create_env.sh first" >&2
fi

cd "$LISA_ROOT" || return 1
