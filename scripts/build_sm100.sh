#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python was not found at $PYTHON_BIN" >&2
  exit 1
fi

git submodule update --init --recursive

"$PYTHON_BIN" -m venv --clear .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip setuptools wheel ninja numpy

CUDA_HOME="$(python3 scripts/detect_cuda_home.py)"
export CUDA_HOME
export PATH="$CUDA_HOME/bin:$PATH"

CUDA_VERSION="$(
  "$CUDA_HOME/bin/nvcc" --version |
    sed -n 's/.*release \([0-9]*\.[0-9]*\).*/\1/p'
)"
case "$CUDA_VERSION" in
  12.9) TORCH_CUDA_INDEX=cu129 ;;
  13.0) TORCH_CUDA_INDEX=cu130 ;;
  *)
    echo "No documented PyTorch wheel pairing for CUDA $CUDA_VERSION" >&2
    exit 1
    ;;
esac

python3 -m pip install torch==2.12.1 \
  --index-url "https://download.pytorch.org/whl/$TORCH_CUDA_INDEX"

python3 - <<'PY'
import os
import re
import subprocess

import torch

nvcc_output = subprocess.check_output(
    [os.path.join(os.environ["CUDA_HOME"], "bin", "nvcc"), "--version"],
    text=True,
)
toolkit_cuda = re.search(r"release\s+(\d+\.\d+)", nvcc_output).group(1)

print("PyTorch:", torch.__version__)
print("PyTorch CUDA:", torch.version.cuda)
print("CUDA Toolkit:", toolkit_cuda)
print("GPU:", torch.cuda.get_device_name(0))

assert torch.cuda.is_available(), "PyTorch cannot access a CUDA GPU"
assert torch.version.cuda == toolkit_cuda, (torch.version.cuda, toolkit_cuda)
PY

export FLASH_MLA_DISABLE_SM90="${FLASH_MLA_DISABLE_SM90:-1}"
export NVCC_THREADS="${NVCC_THREADS:-8}"
export MAX_JOBS="${MAX_JOBS:-8}"

echo "Building with CUDA_HOME=$CUDA_HOME, NVCC_THREADS=$NVCC_THREADS, MAX_JOBS=$MAX_JOBS"
python3 setup.py build_ext --inplace -v

echo "Build complete. Run with .venv/bin/python3, or activate it with:"
echo "  source .venv/bin/activate"
