#!/usr/bin/env bash
# Python environment for CAT's DenseTNT + counterfactual responsibility on a
# Linux GPU server (tested target: Ubuntu 24.04, H200).
#
# Why these versions:
#   Python 3.9   advgen/utils_cython.cpython-39-x86_64-linux-gnu.so is built for
#                it (rebuilding needs a C compiler and Cython), and TF 2.12 runs on it
#   torch 2.4.1  CAT pins torch 1.12.0+cu116, which has no kernels for Hopper
#   + cu121      (sm_90, H100/H200); 2.4.1 still ships Python 3.9 wheels and loads
#                densetnt.bin unchanged (scripts/responsibility/verify_densetnt.py
#                checks the result against CAT's own pipeline)
#   tensorflow-cpu 2.12   only used to build DenseTNT's input tensors; the CPU
#                build keeps TensorFlow off the GPU
#   numpy < 1.24 required by TF 2.12
# MetaDrive and the RL stack are not needed for responsibility.
#
# usage (from the repository root):
#   bash scripts/responsibility/setup_env.sh               # venv at ~/venvs/cat39 via uv
#   ENV_DIR=/scratch/venvs/cat39 bash scripts/responsibility/setup_env.sh
#   BACKEND=conda ENV_NAME=cat bash scripts/responsibility/setup_env.sh
#   CUDA_TAG=cu124 bash scripts/responsibility/setup_env.sh  # other torch CUDA build
set -euo pipefail

BACKEND=${BACKEND:-uv}
ENV_DIR=${ENV_DIR:-$HOME/venvs/cat39}
ENV_NAME=${ENV_NAME:-cat}
CUDA_TAG=${CUDA_TAG:-cu121}

if command -v nvidia-smi >/dev/null; then
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
else
  echo "WARNING: no nvidia-smi; installing anyway (CPU only)" >&2
fi

case "$BACKEND" in
  uv)
    if ! command -v uv >/dev/null; then
      curl -LsSf https://astral.sh/uv/install.sh | sh
      export PATH="$HOME/.local/bin:$PATH"
    fi
    uv python install 3.9
    [ -d "$ENV_DIR" ] || uv venv --python 3.9 --seed "$ENV_DIR"
    # shellcheck disable=SC1091
    source "$ENV_DIR/bin/activate"
    PIP="uv pip"
    ;;
  conda)
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda env list | grep -q "^$ENV_NAME " || conda create -y -n "$ENV_NAME" python=3.9
    conda activate "$ENV_NAME"
    PIP="python -m pip"
    ;;
  *) echo "BACKEND must be uv or conda" >&2; exit 1 ;;
esac

$PIP install "torch==2.4.1" "torchvision==0.19.1" --index-url "https://download.pytorch.org/whl/$CUDA_TAG"
$PIP install "tensorflow-cpu==2.12.0" "numpy<1.24" pyyaml matplotlib tqdm pytest scipy

python - <<'PY'
import pickle, sys
sys.modules.setdefault("pickle5", pickle)
import torch
import advgen.utils_cython  # noqa: F401  (the prebuilt extension loads)
from advgen.modeling.vectornet import VectorNet  # noqa: F401
print(f"python {sys.version.split()[0]}, torch {torch.__version__}, CUDA build {torch.version.cuda}")
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f"cuda:0 {p.name} sm_{p.major}{p.minor}, {torch.cuda.device_count()} GPU(s)")
    torch.ones(8, device="cuda").sum().item()
else:
    print("no CUDA device visible")
PY

echo
if [ "$BACKEND" = uv ]; then echo "activate: source $ENV_DIR/bin/activate"; else echo "activate: conda activate $ENV_NAME"; fi
echo "then check:"
echo "  python -m pytest tests/responsibility -q"
echo "  python -m scripts.responsibility.verify_densetnt --n 3 --device cuda"
