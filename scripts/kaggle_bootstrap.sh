#!/usr/bin/env bash
# Bootstrap a fresh Kaggle / Colab kernel into the run state.
#
# Usage (from a notebook cell):
#   !bash scripts/kaggle_bootstrap.sh
#
# Assumes the repo is already cloned to CWD.
#
# Wheel cache: slow-to-build packages (flash-attn) are written to
# wheels/${PROFILE}/ where PROFILE defaults to "kaggle/2xt4". To benefit
# on subsequent sessions, commit the resulting .whl files back to the repo
# (the bootstrap prints the exact command at the end).

set -euo pipefail

PROFILE="${WHEEL_PROFILE:-kaggle/2xt4}"
WHEEL_DIR="wheels/${PROFILE}"
# T4 is sm_75; constrain the build so the wheel stays under GitHub's 100 MB
# per-file limit. Override with TORCH_CUDA_ARCH_LIST if you're caching for
# a different accelerator (e.g. "8.0" for A100, "8.9" for L4, "7.5;8.0" for
# a multi-target cache).
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

# --- flash-attn: cache-aware install ---------------------------------------
echo "==> flash-attn"
FA_CACHED=$(ls "$WHEEL_DIR"/flash_attn-*.whl 2>/dev/null | head -1 || true)
if [[ -n "$FA_CACHED" ]]; then
  echo "  cached wheel found: $(basename "$FA_CACHED")"
  if pip install -q --no-deps "$FA_CACHED"; then
    echo "  installed from cache."
  else
    echo "  cached install failed — rebuilding."
    FA_CACHED=""
  fi
fi
if [[ -z "$FA_CACHED" ]]; then
  echo "  building from source for sm_${ARCH_LIST/./} (slow; first time only)..."
  if TORCH_CUDA_ARCH_LIST="$ARCH_LIST" \
       pip wheel flash-attn --no-build-isolation -w "$WHEEL_DIR" >/tmp/fa-build.log 2>&1; then
    FA_BUILT=$(ls -t "$WHEEL_DIR"/flash_attn-*.whl 2>/dev/null | head -1)
    if [[ -n "$FA_BUILT" ]]; then
      pip install -q --no-deps "$FA_BUILT" || echo "  install of fresh wheel failed"
      FA_SIZE_MB=$(du -m "$FA_BUILT" | cut -f1)
      echo "  built + installed: $(basename "$FA_BUILT") (${FA_SIZE_MB} MB)"
      if [[ $FA_SIZE_MB -lt 95 ]]; then
        echo "  → commit to skip the rebuild next time:"
        echo "    git add $FA_BUILT && git commit -m 'cache: $(basename "$FA_BUILT")' && git push"
      else
        echo "  ⚠️ wheel is ${FA_SIZE_MB} MB — too big for direct git commit (>100 MB)."
        echo "    Options: (a) narrow TORCH_CUDA_ARCH_LIST and rebuild, (b) git-lfs, (c) GitHub Release."
      fi
    fi
  else
    echo "  flash-attn build failed (see /tmp/fa-build.log); will fall back to SDPA"
  fi
fi

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
