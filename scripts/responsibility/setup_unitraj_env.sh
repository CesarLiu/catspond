#!/usr/bin/env bash
# Python environments for UniTraj's MTR as the responsibility model's second
# motion model (responsibility/UNITRAJ_PLAN.md; runbook step 11), on a Linux GPU
# server (target: Ubuntu 24.04, H200). Two environments, kept apart because
# their pins conflict:
#
#   unitraj39  training MTR and using it for responsibility: torch 2.4.1+cu121
#              (Hopper kernels, Python 3.9 wheels, as for CAT), PyTorch Lightning,
#              UniTraj with MTR's CUDA ops compiled for sm_90, and ScenarioNet +
#              MetaDrive to read ScenarioNet datasets during training
#   womd39     converting WOMD tfrecords to ScenarioNet: TensorFlow and the
#              waymo-open-dataset package ScenarioNet's converter needs
#
# Not verified on the target server yet: the exact waymo-open-dataset build
# (check ScenarioNet's installation docs if conversion fails to import it) and
# the compilation of MTR's CUDA ops (needs nvcc from a CUDA 12.x toolkit).
#
# usage (from the cat repository root; UniTraj checked out next to it or at $UNITRAJ):
#   bash scripts/responsibility/setup_unitraj_env.sh            # both environments
#   PART=train bash scripts/responsibility/setup_unitraj_env.sh # only unitraj39
#   PART=convert bash scripts/responsibility/setup_unitraj_env.sh
# variables: UNITRAJ (UniTraj checkout), TRAIN_ENV (~/venvs/unitraj39),
#   CONVERT_ENV (~/venvs/womd39), SRC (~/src: where ScenarioNet is cloned),
#   CUDA_TAG (cu121), TORCH_CUDA_ARCH_LIST (9.0)
set -euo pipefail

REPO=$(cd "$(dirname "$0")/../.." && pwd)
PART=${PART:-all}
UNITRAJ=${UNITRAJ:-$(cd "$REPO/.." && pwd)/UniTraj}
TRAIN_ENV=${TRAIN_ENV:-$HOME/venvs/unitraj39}
CONVERT_ENV=${CONVERT_ENV:-$HOME/venvs/womd39}
SRC=${SRC:-$HOME/src}
CUDA_TAG=${CUDA_TAG:-cu121}
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-9.0}

[ -d "$UNITRAJ/unitraj" ] || { echo "UniTraj not found at $UNITRAJ (set UNITRAJ=...)" >&2; exit 1; }
if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
uv python install 3.9
mkdir -p "$SRC"
[ -d "$SRC/scenarionet" ] || git clone https://github.com/metadriverse/scenarionet.git "$SRC/scenarionet"

if [ "$PART" = all ] || [ "$PART" = train ]; then
  echo "=== $TRAIN_ENV"
  if ! command -v nvcc >/dev/null; then
    echo "nvcc not found: MTR's CUDA ops need a CUDA 12.x toolkit (e.g. module load cuda/12.1, or" >&2
    echo "  conda install -c nvidia cuda-nvcc=12.1 cuda-cudart-dev=12.1), then re-run" >&2
    exit 1
  fi
  nvcc --version | tail -1
  [ -d "$TRAIN_ENV" ] || uv venv --python 3.9 --seed "$TRAIN_ENV"
  # shellcheck disable=SC1091
  source "$TRAIN_ENV/bin/activate"
  uv pip install "torch==2.4.1" --index-url "https://download.pytorch.org/whl/$CUDA_TAG"
  uv pip install "numpy<2" "pytorch-lightning==2.4.0" "hydra-core==1.3.2" wandb scikit-learn einops easydict \
    h5py scipy matplotlib pyyaml tqdm pytest pillow metadrive-simulator
  uv pip install -e "$SRC/scenarionet"
  (cd "$UNITRAJ" && python setup.py develop)  # compiles MTR's knn and attention ops
  python - <<'PY'
import torch
import unitraj.models.mtr.ops.knn.knn_cuda  # noqa: F401
import unitraj.models.mtr.ops.attention.attention_cuda  # noqa: F401
import scenarionet.common_utils  # noqa: F401
print(f"torch {torch.__version__} (CUDA {torch.version.cuda}); MTR ops and scenarionet import")
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f"cuda:0 {p.name} sm_{p.major}{p.minor}, {torch.cuda.device_count()} GPU(s)")
PY
  deactivate
fi

if [ "$PART" = all ] || [ "$PART" = convert ]; then
  echo "=== $CONVERT_ENV"
  [ -d "$CONVERT_ENV" ] || uv venv --python 3.9 --seed "$CONVERT_ENV"
  # shellcheck disable=SC1091
  source "$CONVERT_ENV/bin/activate"
  uv pip install "tensorflow==2.12.0" "waymo-open-dataset-tf-2-12-0==1.6.4" metadrive-simulator tqdm
  uv pip install -e "$SRC/scenarionet"
  python -c "import waymo_open_dataset, scenarionet; print('conversion environment ready')"
  deactivate
fi

echo
echo "training / responsibility: source $TRAIN_ENV/bin/activate"
echo "  python -m scripts.responsibility.verify_unitraj --checkpoint random --n 2   # environment test on the GPU"
echo "conversion:                source $CONVERT_ENV/bin/activate"
