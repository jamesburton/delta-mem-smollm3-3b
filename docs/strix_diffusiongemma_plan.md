# Strix Halo — DiffusionGemma-26B-A4B-it run plan

**Target machine:** T2 Strix Halo workstation (Ryzen AI Max 395, Radeon 8060S iGPU `gfx1151`, 128 GB unified, ROCm 7.x).
**Target model:** `google/diffusiongemma-26B-A4B-it` (26 B total / A4B activated, MoE, Apache-2.0).
**Goal:** measured tok/s baseline using the `corsairnui/diffusiongemma-26b-a4b-it-strix-halo-fp16` GGUFs on a HIP / `gfx1151` llama.cpp build with PR #24423 (DiffusionGemma kernels + sampler).
**Authored from:** T1 desk research, **not** Strix. Nothing here was verified on the box — treat first-run output as ground truth.

---

## TL;DR — the one-liner

After the build + download steps below succeed, the morning run is:

```bash
bash scripts/run_diffusiongemma_strix.sh
```

That runs a 64-token pilot (validates iGPU + sampler) then the 2048-token Q5_K_M benchmark. Override with `VARIANT=BF16` or `VARIANT=Q4_K_M` for the other quants. Default log lands in `~/diffusiongemma-strix-results/`.

---

## CRITICAL FINDING — diffusion sampling support is in an *unmerged PR*

DiffusionGemma needs **PR #24423** on `ggml-org/llama.cpp` and a dedicated `llama-diffusion-cli` binary. **`llama-cli` / `llama-server` from mainline llama.cpp cannot generate from this model** — the architecture (`diffusion-gemma`) and the sampler (`entropy_bounded_denoising`, linear-decay temperature 0.8 → 0.4, adaptive stopping) are not in the released code path.

If you build mainline llama.cpp and run the GGUF through `llama-cli`, one of two things happens:
1. it refuses to load (architecture unknown), or
2. it loads via a generic Gemma path and produces **incoherent output** at AR-style sampling — because the weights were trained for masked-canvas denoising, not next-token prediction.

**Status as of June 2026:** PR #24423 is **open / draft**, not merged into main. You must check out the PR branch and build the dedicated `llama-diffusion-cli` target.

### If you measure incoherent text, this is the cause

Symptoms:
- repeated tokens / loops
- broken syntax inside the canvas
- output that doesn't match the prompt at all

→ You're running through the wrong sampler. Stop and re-verify you built from PR #24423 and you're invoking `llama-diffusion-cli`, not `llama-cli`.

---

## Coordination checklist — read before launching

- [ ] **Confirm with the user that the Strix iGPU is free.** Coordination is user-mediated until we ship a file lock. Tell them "I want to run DiffusionGemma on Strix — is the iGPU free?" and wait for confirmation.
- [ ] **Disk:** ~21 GB free for Q5_K_M GGUF, ~17 GB for Q4_K_M, ~51 GB for BF16. Plus ~5 GB for the llama.cpp source tree + build artefacts.
- [ ] **Estimated wall time:**
  - llama.cpp build (HIP, first time): **20–40 min** (single `cmake --build` over ~24 cores)
  - GGUF download (Q5_K_M, 19 GB): **5–20 min** depending on link speed
  - Pilot run (64-token canvas): **~2–5 s wall**
  - Main benchmark (2048-token canvas): **~17 s wall at the reported ~14 tok/s** (PR-thread number from TinyComputers) up to **~17 s at ~124 tok/s** (corsairnui README claim — see "Benchmark expectations" below for why these disagree)
- [ ] **Fallbacks identified:** Vulkan llama.cpp, prebuilt Lychee-Technology binaries — see "Fallbacks" section.
- [ ] **Post-run:** tell the user "done with Strix" and append measured numbers to `~/.claude/OTHER_MACHINES.md` (Strix section, Notes / measurements) and `~/.claude/LLMs.md` (DiffusionGemma entry).

---

## Step 1 — Build llama.cpp with DiffusionGemma + HIP/gfx1151

### Prereqs on Strix

- Linux 6.16+ (or recent WSL2 Ubuntu 24.04 with ROCm passthrough — untested, see Fallbacks)
- **ROCm 7.2.0** from AMD's APT repo (the known-good version called out in [llama.cpp Discussion #20856][disc20856]). ROCm 7.x is the first family with first-class `gfx1151` support; do not try 6.x.
- `cmake`, `git`, `build-essential`, `ninja-build` (or use `make`)

Verify ROCm sees the iGPU before doing anything else:

```bash
rocminfo | grep -E 'gfx|Name:' | head
# Expect: "Name: gfx1151"
```

### Clone + checkout PR #24423

```bash
cd ~
git clone https://github.com/ggml-org/llama.cpp.git llama.cpp-diffusiongemma
cd llama.cpp-diffusiongemma
git fetch --no-tags origin pull/24423/head:diffusiongemma
git checkout diffusiongemma
# Latest reviewed commit in the PR thread: 9b4dae8 (context sizing by RAM fix).
# If 'gh' is installed: `gh pr checkout 24423` also works.
```

### Configure + build (HIP, gfx1151, rocWMMA flash-attn, no-VMM)

These are the **known-good** flags from [Discussion #20856][disc20856] for gfx1151. `GGML_HIP_NO_VMM=ON` is the load-bearing one — without it you get misleading hangs on model load that look like driver issues.

```bash
HIPCXX="$(hipconfig -l)/clang" \
HIP_PATH="$(hipconfig -R)" \
cmake -S . -B build \
  -DGGML_HIP=ON \
  -DGPU_TARGETS=gfx1151 \
  -DGGML_HIP_ROCWMMA_FATTN=ON \
  -DGGML_HIP_NO_VMM=ON \
  -DGGML_HIP_MMQ_MFMA=ON \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build --config Release -j"$(nproc)" --target llama-diffusion-cli
# Also useful: --target llama-cli llama-bench  (for AR sanity checks on the same build)
```

Binary lands at `build/bin/llama-diffusion-cli`.

### Runtime env vars

None of the HIP build flags above translate to runtime env vars. You should **not** need `HSA_OVERRIDE_GFX_VERSION` because `gfx1151` is a real target in ROCm 7.2+ (the override hack is for older / unsupported targets). If load OOMs on the iGPU's address space, try:

```bash
export DG_UBATCH=8192   # PR-thread workaround for ubatch-vs-graph-reserve OOM on Strix
```

(From the PR comments — one Strix user hit a graph-reservation OOM that this env var avoids.)

### Prebuilt binaries (if you want to skip the build)

- [Lychee-Technology/llama-cpp-for-strix-halo][lychee] ships nightly Strix-tuned binaries. **However** these track mainline llama.cpp and won't include PR #24423 unless/until it merges. Useful for AR sanity-check workloads on this box; not useful for DiffusionGemma until the PR lands.
- [gbuznote-beep/llama-diffusion-cli-prebuilt][prebuilt] claims prebuilt `llama-diffusion-cli` artefacts from the PR. **Unverified provenance** — only use if you accept the trust trade-off, and verify checksums against a self-built binary if you do.

---

## Step 2 — Download GGUF weights

Use [`corsairnui/diffusiongemma-26b-a4b-it-strix-halo-fp16`][corsairnui]. It exists explicitly to ship gfx1151-tuned + therock-benchmarked GGUFs for this model.

Three files live under `weights/`:

| Variant | File | Size | Notes |
|---|---|---|---|
| **BF16** | `diffusiongemma-26B-A4B-it-BF16.gguf` | **50.54 GB** | Headline fp16 number; reference for quality gates |
| **Q5_K_M** | `diffusiongemma-26B-A4B-it-Q5_K_M-self.gguf` | **19.15 GB** | **Recommended primary**; 5/5 quality gates vs BF16 |
| **Q4_K_M** | `diffusiongemma-26B-A4B-it-Q4_K_M-self.gguf` | **16.81 GB** | Smaller; only 3/5 quality gates pass — use only for tight-memory comparison |

Download just Q5_K_M (recommended morning run):

```bash
huggingface-cli download corsairnui/diffusiongemma-26b-a4b-it-strix-halo-fp16 \
  weights/diffusiongemma-26B-A4B-it-Q5_K_M-self.gguf \
  manifests/WEIGHTS_SHA256SUMS.txt \
  --local-dir "$HOME/models/diffusiongemma-strix"
```

Add BF16 + Q4_K_M to that command if you want the full sweep (and have the disk).

Verify after download:

```bash
cd "$HOME/models/diffusiongemma-strix"
sha256sum -c manifests/WEIGHTS_SHA256SUMS.txt 2>&1 | grep -v ': OK$' || true
# Empty grep output = all good.
```

---

## Step 3 — Run the benchmark

The script `scripts/run_diffusiongemma_strix.sh` does pilot + main back-to-back. To kick the whole thing off:

```bash
cd <your llm-model-tests clone on Strix>
bash scripts/run_diffusiongemma_strix.sh
# overrides:
#   VARIANT=BF16 bash scripts/run_diffusiongemma_strix.sh      # 50 GB run
#   VARIANT=Q4_K_M bash scripts/run_diffusiongemma_strix.sh    # 17 GB run
#   N_TOKENS=512 PILOT=0 bash scripts/run_diffusiongemma_strix.sh  # quick check
```

The script:
1. Validates `llama-diffusion-cli` and the GGUF exist
2. Runs `rocminfo` so the log captures whether `gfx1151` is visible
3. Runs a 64-token pilot — **STOP and investigate** if this doesn't show `gfx1151` in the HIP backend register or if output is incoherent
4. Runs the 2048-token main benchmark with `--perf` for tok/s reporting

Equivalent manual command (matches corsairnui's recommended invocation):

```bash
~/llama.cpp-diffusiongemma/build/bin/llama-diffusion-cli \
  -m ~/models/diffusiongemma-strix/weights/diffusiongemma-26B-A4B-it-Q5_K_M-self.gguf \
  -p "Explain text diffusion in three concise bullets." \
  -n 2048 \
  -ngl 99 \
  -cnv \
  --diffusion-eb auto \
  --diffusion-kv-cache auto \
  --perf
```

### What the flags mean

- `-ngl 99` — offload all layers to the iGPU (Strix unified mem makes this cheap)
- `-cnv` — conversation mode; matches the variant the corsairnui README benchmarked
- `--diffusion-eb auto` — entropy-bound sampler; the PR's main contribution. `auto` lets it pick defaults (`entropy_bound=0.1`, `confidence=0.005`, temperature 0.8 → 0.4)
- `--diffusion-kv-cache auto` — KV-cache toggle; on single-GPU Strix the PR enables this by default
- `--perf` — print perf summary (the tok/s number we care about)
- `-n 2048` — canvas length; this is also the generated length cap

---

## Step 4 — Benchmark expectations (and a discrepancy worth noting)

Two reference numbers, and they disagree by ~9×:

| Source | Variant | Reported tok/s | Notes |
|---|---|---|---|
| [corsairnui README][corsairnui] | Q5_K_M, conv mode | **124.49** | 2048-token canvas, gfx1151, therock build |
| corsairnui README | BF16 + AOTriton | **134.65** | Transformers path with `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`, **not** llama.cpp |
| corsairnui README | BF16 baseline | 115.42 | Transformers, 1536-token target |
| [TinyComputers post][tinycomp] | Q8_0 (Unsloth) | **~14** | 256 tokens in 17.4 s; `-cnv -n 2048 --diffusion-visual`, llama.cpp PR #24423 |

The corsairnui ≥100 tok/s numbers in the README's model-index are explicitly for the **Transformers path with AOTriton**, *not* for the GGUF llama.cpp path. The llama.cpp Q-series numbers in the same README ("124.49 tok/s estimate", "122.40 tok/s estimate") are flagged as **estimates** from a scaling formula, not direct measurements — easy to misread. The TinyComputers ~14 tok/s number is a **direct measurement** on Strix with the PR build.

**Working hypothesis for the morning:** expect on the order of **10–30 tok/s** from the llama.cpp / PR #24423 path on Q5_K_M. Anything above that is a pleasant surprise; anything in the 100s suggests we're hitting the Transformers+AOTriton path or measuring something other than steady-state generation. Anything below 5 tok/s suggests the iGPU isn't being used (CPU fallback) or you're paying a load-time penalty in the wall-clock measurement — re-run with the pilot's load amortised.

The headline measurement to report back: **steady-state tok/s on Q5_K_M, 2048-token canvas, gfx1151 HIP build of PR #24423**.

---

## Fallbacks (in order)

1. **`-fa off`** — if `GGML_HIP_ROCWMMA_FATTN=ON` causes load failures or correctness issues on the diffusion path (the PR comments note FA was set to "disabled" when running on Strix in the TinyComputers post). Pass `-fa off` to `llama-diffusion-cli`.
2. **`DG_UBATCH=8192`** env var — for the PR-reported ubatch-vs-graph-reserve OOM on Strix.
3. **Vulkan build** — drop ROCm entirely, build llama.cpp with `-DGGML_VULKAN=ON`. Slower per the Strix Halo wiki but more portable; still need PR #24423 for the sampler. Vulkan works on the PR per its compatibility notes.
4. **Mainline llama.cpp + AR sampling** — *not recommended*. You'd get a tok/s number but it would be measuring the wrong thing (AR generation from diffusion weights = incoherent output). Only do this to prove the build works and the GGUF loads.
5. **Transformers path (BF16) with AOTriton** — the path the corsairnui README's headline numbers came from. Different harness, different question. If you want to reproduce the ~135 tok/s claim, this is the route: clone the Transformers DiffusionGemma path on Strix, `export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`, run via `transformers` generate(). Out of scope for this morning's first run, but document as next-up.
6. **Prebuilt Lychee-Technology binary** — won't have PR #24423 until merge, so only useful for sanity-checking the AR pipeline on Strix (e.g. a Qwen3-4B run) to confirm the iGPU is wired correctly before sinking time into the diffusion build.

---

## What to report back / update afterwards

After the run:

1. Append measured numbers to `C:\Users\james\.claude\OTHER_MACHINES.md` under the Strix section's **Notes / measurements** (currently empty) — date, variant, tok/s, llama.cpp commit, ROCm version.
2. Append the same to `C:\Users\james\.claude\LLMs.md` under the DiffusionGemma entry. Update the **Status** line from "not yet downloaded" to "VIABLE T2 — N tok/s on Q5_K_M".
3. If PR #24423 status changed (merged, closed, force-pushed), note that in `LLMs.md` → "Server / API parity notes" → llama.cpp line — currently says "PR #24423 adds DiffusionGemma kernels but block-diffusion sampling support in the server still pending".
4. If the build failed, write a `docs/strix_setup.md` capturing the exact error + fix — the build path is fragile enough that a 20-min build with an unhelpful error halfway through is worth a documented postmortem.

---

## Open questions I couldn't resolve from the desk

- **Exact ROCm version installed on the Strix box right now.** Discussion #20856 calls out ROCm 7.2.0 from AMD's APT repo as the known-good combo. If Strix is on something older, the build will likely fail at `cmake` time with an unknown target.
- **Whether Strix is on Linux or WSL2.** This plan assumes native Linux. If it's WSL2 the ROCm passthrough story is murkier — fall back to Vulkan or use Windows-native prebuilt Lychee binaries (but those don't have the PR yet).
- **Whether `-fa on` (rocWMMA flash attention) works through the diffusion path.** The TinyComputers post says FA was auto-disabled. The script defaults to letting `--perf`-able defaults stand; if generation fails or hangs, force `-fa off`.
- **Whether the corsairnui "Q5_K_M_self" suffix matters** vs vanilla Q5_K_M. The naming suggests a self-attention-specific quant variant; the corsairnui README treats them as drop-in replacements. Flag and move on; revisit if quality gates show drift.

---

## References

- [llama.cpp Discussion #20856 — Known-Good Strix Halo ROCm + llama.cpp Stack][disc20856]
- [llama.cpp PR #24423 — DiffusionGemma support][pr24423]
- [corsairnui/diffusiongemma-26b-a4b-it-strix-halo-fp16 (HF)][corsairnui]
- [Strix Halo wiki — llama.cpp with ROCm][wiki]
- [Lychee-Technology/llama-cpp-for-strix-halo — nightly Strix-tuned binaries][lychee]
- [TinyComputers — Running DiffusionGemma on Strix Halo and Tesla P40s][tinycomp]
- [WayneTechLab/llama-diffusion-gemma — community fork of PR #24423][waynetech]
- [gbuznote-beep/llama-diffusion-cli-prebuilt — unofficial prebuilt binaries (trust at own risk)][prebuilt]
- [Unsloth — DiffusionGemma run guide + alternate GGUFs][unsloth]

[disc20856]: https://github.com/ggml-org/llama.cpp/discussions/20856
[pr24423]: https://github.com/ggml-org/llama.cpp/pull/24423
[corsairnui]: https://huggingface.co/corsairnui/diffusiongemma-26b-a4b-it-strix-halo-fp16
[wiki]: https://strixhalo.wiki/AI/llamacpp-with-ROCm
[lychee]: https://github.com/Lychee-Technology/llama-cpp-for-strix-halo/releases
[tinycomp]: https://tinycomputers.io/posts/running-diffusiongemma-on-strix-halo-and-tesla-p40s.html
[waynetech]: https://github.com/WayneTechLab/llama-diffusion-gemma/tree/main
[prebuilt]: https://github.com/gbuznote-beep/llama-diffusion-cli-prebuilt
[unsloth]: https://unsloth.ai/docs/models/diffusiongemma
