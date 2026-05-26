#!/usr/bin/env bash
# Bootstrap a fresh Kaggle kernel into the run state.
#
# Usage (from a notebook cell):
#   !bash scripts/kaggle_bootstrap.sh
#
# Assumes the repo is already cloned to the CWD.

set -euo pipefail

echo "==> python version"
python --version

echo "==> nvidia-smi (if present)"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
else
  echo "  no NVIDIA GPU visible"
fi

echo "==> installing pinned requirements"
pip install -q -r requirements.txt

# FlashAttention often needs a separate install on Kaggle because of CUDA wheels.
echo "==> installing flash-attn (best-effort)"
pip install -q flash-attn --no-build-isolation || echo "  flash-attn install failed; will fall back to SDPA"

echo "==> installing DeepSpeed (best-effort)"
pip install -q deepspeed || echo "  deepspeed install failed"

echo "==> done."
