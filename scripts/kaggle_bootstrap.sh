#!/usr/bin/env bash
# Bootstrap a fresh Kaggle / Colab kernel into the run state.
#
# Usage (from a notebook cell):
#   !bash scripts/kaggle_bootstrap.sh
#
# Assumes the repo is already cloned to CWD.
#
# Wheel cache (flash-attn only — it's the slow one):
#   wheels/${WHEEL_PROFILE}/  (default: wheels/kaggle/2xt4)
#
# Lookup order is delegated to scripts/install_flash_attn.py:
#   1. Local cache → 2. Community prebuilt (mjun0812/flash-attention-prebuild-wheels)
#   → 3. Source build (slow fallback)

set -euo pipefail

PROFILE="${WHEEL_PROFILE:-kaggle/2xt4}"
WHEEL_DIR="wheels/${PROFILE}"
ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-7.5}"

mkdir -p "$WHEEL_DIR"

echo "==> python version"
python --version

echo "==> nvidia-smi (if present)"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv
else
  echo "  no NVIDIA GPU visible"
fi

echo "==> profile: $PROFILE  (TORCH_CUDA_ARCH_LIST=$ARCH_LIST)"

echo "==> installing pinned requirements"
pip install -q -r requirements.txt

echo "==> installing delta-Mem (upstream)"
pip install -q "git+https://github.com/declare-lab/delta-Mem" \
  || echo "  delta-Mem install failed; cells 2 and 7 will fail with a clear RuntimeError"

echo "==> flash-attn (cache → community → source)"
python scripts/install_flash_attn.py \
  --cache-dir "$WHEEL_DIR" \
  --arch-list "$ARCH_LIST" \
  || echo "  install_flash_attn.py reported a non-zero exit; harness will fall back to SDPA"

# --- DeepSpeed: usually has prebuilt wheels on PyPI ------------------------
echo "==> DeepSpeed"
DS_CACHED=$(ls "$WHEEL_DIR"/deepspeed-*.whl 2>/dev/null | head -1 || true)
if [[ -n "$DS_CACHED" ]]; then
  echo "  cached wheel found: $(basename "$DS_CACHED")"
  pip install -q --no-deps "$DS_CACHED" || echo "  cached install failed"
else
  pip install -q deepspeed || echo "  deepspeed install failed"
fi

echo "==> done."
