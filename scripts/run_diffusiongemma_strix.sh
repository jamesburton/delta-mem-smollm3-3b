#!/usr/bin/env bash
# ============================================================================
# run_diffusiongemma_strix.sh
#
# Launcher for DiffusionGemma-26B-A4B-it on the Strix Halo workstation
# (T2 in the tier model: Ryzen AI Max 395 + Radeon 8060S iGPU, gfx1151).
#
# This script assumes:
#   - You are on the Strix Halo box (Linux, ROCm 7.x with gfx1151 support).
#     If you're on Windows on Strix, use WSL2 Ubuntu 24.04+ for the ROCm path,
#     OR fall back to the prebuilt Lychee-Technology binaries (see plan doc).
#   - You have already cloned llama.cpp + checked out PR #24423 (or merged main
#     if the PR has landed). See docs/strix_diffusiongemma_plan.md → Build.
#   - GGUF weights are already downloaded under $MODEL_DIR.
#
# DO NOT run this on the T1 RTX 3060 workstation — it is gfx1151 / Strix-only.
# Coordinate with the user before launching on Strix (iGPU may be in use).
# ============================================================================

set -euo pipefail

# ----------------------------------------------------------------------------
# Configuration — edit these paths to match the Strix box layout
# ----------------------------------------------------------------------------
LLAMA_DIR="${LLAMA_DIR:-$HOME/llama.cpp-diffusiongemma}"
RUNNER="${RUNNER:-$LLAMA_DIR/build/bin/llama-diffusion-cli}"

# Download these from HuggingFace ahead of time (see plan doc → step 2):
#   huggingface-cli download corsairnui/diffusiongemma-26b-a4b-it-strix-halo-fp16 \
#     weights/diffusiongemma-26B-A4B-it-Q5_K_M-self.gguf \
#     --local-dir "$HOME/models/diffusiongemma-strix"
MODEL_DIR="${MODEL_DIR:-$HOME/models/diffusiongemma-strix/weights}"

# Default to Q5_K_M (19.15 GB) — passes 5/5 quality gates vs FP16.
# Override with: VARIANT=BF16 ./run_diffusiongemma_strix.sh
#   - BF16   = 50.54 GB, headline fp16 number (~134 tok/s with AOTriton)
#   - Q5_K_M = 19.15 GB, recommended primary, ~124 tok/s
#   - Q4_K_M = 16.81 GB, smaller but only 3/5 quality gates pass
VARIANT="${VARIANT:-Q5_K_M}"
case "$VARIANT" in
  BF16)    MODEL_FILE="$MODEL_DIR/diffusiongemma-26B-A4B-it-BF16.gguf" ;;
  Q5_K_M)  MODEL_FILE="$MODEL_DIR/diffusiongemma-26B-A4B-it-Q5_K_M-self.gguf" ;;
  Q4_K_M)  MODEL_FILE="$MODEL_DIR/diffusiongemma-26B-A4B-it-Q4_K_M-self.gguf" ;;
  *) echo "Unknown VARIANT=$VARIANT (use BF16, Q5_K_M, or Q4_K_M)"; exit 2 ;;
esac

# Generation budget — DiffusionGemma denoises a fixed-length canvas, so -n is
# also the canvas length. 2048 matches the corsairnui benchmark setup.
N_TOKENS="${N_TOKENS:-2048}"

# Prompt — short, deterministic, easy to eyeball for incoherence (which is
# the diffusion-sampler-missing failure mode; see plan doc).
PROMPT="${PROMPT:-Explain text diffusion in three concise bullets.}"

# Pilot mode = short canvas to validate the build picks up the iGPU before
# committing 30+ seconds to the full benchmark.
PILOT="${PILOT:-1}"

OUT_DIR="${OUT_DIR:-$HOME/diffusiongemma-strix-results}"
mkdir -p "$OUT_DIR"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
LOG="$OUT_DIR/run_${VARIANT}_${TS}.log"

# ----------------------------------------------------------------------------
# Sanity checks
# ----------------------------------------------------------------------------
echo "=== DiffusionGemma Strix Halo launcher ===" | tee "$LOG"
echo "Timestamp:  $TS"                              | tee -a "$LOG"
echo "Runner:     $RUNNER"                          | tee -a "$LOG"
echo "Variant:    $VARIANT"                         | tee -a "$LOG"
echo "Model:      $MODEL_FILE"                      | tee -a "$LOG"
echo "N tokens:   $N_TOKENS"                        | tee -a "$LOG"
echo "Log:        $LOG"                             | tee -a "$LOG"

if [[ ! -x "$RUNNER" ]]; then
  echo "ERROR: llama-diffusion-cli not found at $RUNNER" | tee -a "$LOG"
  echo "Build it per docs/strix_diffusiongemma_plan.md (Build section)." | tee -a "$LOG"
  exit 1
fi

if [[ ! -f "$MODEL_FILE" ]]; then
  echo "ERROR: GGUF not found at $MODEL_FILE" | tee -a "$LOG"
  echo "Download it per docs/strix_diffusiongemma_plan.md (Download section)." | tee -a "$LOG"
  exit 1
fi

# Show ROCm visibility — fail loud if no iGPU is visible.
if command -v rocminfo >/dev/null 2>&1; then
  echo "--- rocminfo (gfx1151 expected) ---" | tee -a "$LOG"
  rocminfo | grep -E 'Name:|gfx' | head -n 20 | tee -a "$LOG" || true
fi

# ----------------------------------------------------------------------------
# Pilot run — 64-token canvas, ~2 seconds. Validates iGPU + sampler path.
# ----------------------------------------------------------------------------
if [[ "$PILOT" == "1" ]]; then
  echo "" | tee -a "$LOG"
  echo "=== PILOT: short 64-token canvas to validate iGPU + sampler ===" | tee -a "$LOG"
  "$RUNNER" \
    -m "$MODEL_FILE" \
    -p "Say hello in one short sentence." \
    -n 64 \
    -ngl 99 \
    --diffusion-eb auto \
    --diffusion-kv-cache auto \
    --perf \
    2>&1 | tee -a "$LOG"
  echo "" | tee -a "$LOG"
  echo "=== Pilot finished. Inspect output above for:" | tee -a "$LOG"
  echo "    1. 'gfx1151' shown by the HIP backend (else it ran on CPU)"   | tee -a "$LOG"
  echo "    2. coherent text (else diffusion sampler is missing — see plan)" | tee -a "$LOG"
  echo "    3. a reported tok/s number"                                   | tee -a "$LOG"
  echo "If any of those failed, STOP and investigate before the main run." | tee -a "$LOG"
  echo "" | tee -a "$LOG"
fi

# ----------------------------------------------------------------------------
# Main benchmark — 2048-token canvas, matches corsairnui's reference setup.
# Tunables match the corsairnui README's recommended command.
# ----------------------------------------------------------------------------
echo "=== MAIN: $VARIANT, $N_TOKENS-token canvas ===" | tee -a "$LOG"
START="$(date +%s)"
"$RUNNER" \
  -m "$MODEL_FILE" \
  -p "$PROMPT" \
  -n "$N_TOKENS" \
  -ngl 99 \
  -cnv \
  --diffusion-eb auto \
  --diffusion-kv-cache auto \
  --perf \
  2>&1 | tee -a "$LOG"
END="$(date +%s)"
echo "" | tee -a "$LOG"
echo "Wall time: $((END - START))s" | tee -a "$LOG"
echo "Done. Full log: $LOG"          | tee -a "$LOG"
