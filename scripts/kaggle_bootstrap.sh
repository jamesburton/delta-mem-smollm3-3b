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

# Cache setup (data caches + wheels). See docs/cache-profiles.md.
# This is opportunistic: if the Kaggle Dataset isn't attached and the wheel
# URLs in the manifest aren't published yet, it falls through cleanly and
# the rest of the bootstrap downloads the fresh way.
if [[ -f "${WHEEL_DIR}/manifest.json" ]]; then
  echo "==> running cache_setup.py for profile $PROFILE"
  python scripts/cache_setup.py --profile "$PROFILE" || \
    echo "  cache_setup reported non-zero; continuing with fresh downloads"
else
  echo "  no manifest at ${WHEEL_DIR}/manifest.json; skipping cache_setup"
fi

# If the cache symlinked a pip wheelhouse, use it to speed up pip.
PIP_FIND_LINKS=""
if [[ -d "/kaggle/working/pip_wheelhouse" ]]; then
  PIP_FIND_LINKS="--find-links=/kaggle/working/pip_wheelhouse"
  echo "==> using pip wheelhouse at /kaggle/working/pip_wheelhouse"
fi

echo "==> installing pinned requirements"
pip install -q $PIP_FIND_LINKS -r requirements.txt

echo "==> setting up delta-Mem (upstream — clone + PYTHONPATH, not pip)"
DM_DIR=".deps/delta-Mem"
mkdir -p .deps
if [[ ! -d "$DM_DIR/deltamem" ]]; then
  git clone --depth=1 https://github.com/declare-lab/delta-Mem "$DM_DIR" \
    || echo "  clone failed; cells 2 and 7 will fail at deltamem import"
fi
if [[ -f "$DM_DIR/requirements.txt" ]]; then
  echo "  installing upstream's pinned requirements (conflicts are non-fatal)"
  pip install -q -r "$DM_DIR/requirements.txt" 2>&1 | tail -2 || true
fi
# Sanity check
if [[ -d "$DM_DIR/deltamem" ]]; then
  python -c "import sys; sys.path.insert(0, '$DM_DIR'); from deltamem.core import attach_delta_mem; print('  deltamem.core imports OK from', '$DM_DIR')" \
    || echo "  deltamem.core failed to import even with PYTHONPATH set"
fi

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
