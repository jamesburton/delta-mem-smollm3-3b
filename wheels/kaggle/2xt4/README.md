# Kaggle T4×2 cache profile

This profile lets a Kaggle notebook **skip the 5–8 minutes of model downloads**
and the 30-second flash-attn community lookup on every fresh kernel by mounting
a Kaggle Dataset that holds the HF cache, and pulling the flash-attn wheel from
a GitHub Release.

## Why flash-attn is NOT cached for this profile

T4 is sm_75 (Turing). FlashAttention-2 needs sm_80+ (Ampere or newer) — on
T4 the kernel raises `RuntimeError("FlashAttention only supports Ampere GPUs
or newer")` on first forward. `install_flash_attn.py` now has a hardware gate
that skips the install entirely on Turing. The harness's
`_attn_impl_for_hardware` gate (`harness/backbone.py`) detects the same thing
and falls back to PyTorch's SDPA mem-efficient kernel at model load.

So this profile has **no `wheels` entries**. The cache pattern is data-only.

## How re-use works

Two layers, each cached the right way for its size:

| Artefact | Size | Where cached |
|---|---|---|
| **HF model cache** (Qwen3-4B base + δ-Mem adapter + Qwen3.5-MTP GGUF) | ~12 GB | Kaggle Dataset attached at `/kaggle/input/delta-mem-smollm3-3b-cache/` |
| **pip wheelhouse** (transformers, accelerate, etc.) | ~500 MB | Same Kaggle Dataset (optional subdir) |
| **delta-Mem upstream clone** | <1 MB | Re-cloned each session (negligible) |

When `scripts/cache_setup.py --profile kaggle/2xt4` runs at notebook bootstrap:

1. For each wheel in `manifest.json`, it tries primary URL (GitHub Release),
   then `fallback_urls` (the always-working community wheel). It verifies SHA256
   and `pip install --no-deps`.
2. For each data cache, it looks for the Kaggle Dataset attached at
   `/kaggle/input/<slug>/`. If present, it symlinks the destination
   (`~/.cache/huggingface/hub` or `/kaggle/working/pip_wheelhouse`) to the
   attachment. If missing, it prints how to attach and continues — downloads
   fall back to fresh just like today.

So **the cache is opportunistic** — the notebook works on day one with no
dataset, and gets faster once the dataset exists.

## One-time setup (~10 minutes)

These are the steps to publish the cache. Do them once per project; future
notebook reruns then attach the dataset and skip download.

### Step 1: First-ever notebook run with `KAGGLE_CACHE_BUILD=1`

```python
# At the top of the notebook cell that runs bootstrap:
import os
os.environ['KAGGLE_CACHE_BUILD'] = '1'
```

Run the notebook normally. It will:
- Download everything fresh (5–8 min for the 7.5 GB Qwen3-4B model).
- After the run, **copy** the HF cache to `/kaggle/working/cache_export/`
  (resolving symlinks so the copy is self-contained).

### Step 2: Save Version → publish as Kaggle Dataset

1. On Kaggle: **Save Version → Save & Run All (Commit)**. Make sure "Save
   Output" is enabled so `/kaggle/working/` persists.
2. Once the version is saved, click the **Output** tab on the version page.
3. Click **New Dataset → From Notebook Output**.
4. Title: **delta-mem-smollm3-3b-cache** (must match the `kaggle_dataset`
   field in `manifest.json`). Owner: **jamesburton** (or yours; update
   manifest if different).
5. Inside the dataset, the path layout should be:
   ```
   delta-mem-smollm3-3b-cache/
   ├── huggingface_hub/         # ~/.cache/huggingface/hub contents
   │   ├── models--Qwen--Qwen3-4B-Instruct-2507/
   │   ├── models--declare-lab--delta-mem_qwen3_4b-instruct/
   │   └── ...
   └── pip_wheelhouse/          # (optional) pip wheels cached here
   ```

## Per-run workflow (after one-time setup)

1. Open the notebook on Kaggle.
2. Click **+ Add Data** → search **delta-mem-smollm3-3b-cache** → attach.
3. Run all. Bootstrap detects the attachment, symlinks the HF cache, pulls the
   flash-attn wheel from GitHub Release, installs in seconds.
4. Net saving: ~5–8 min per run.

## Updating the cache

When you add a new model or adapter:

1. Run with `KAGGLE_CACHE_BUILD=1` to refresh `/kaggle/working/cache_export/`.
2. **Update Dataset → New Version** from the notebook output.
3. Old runs continue using the previous version unless explicitly bumped.

## Adapting for a new profile (Colab L4, RunPod A100, …)

Copy the directory:

```bash
cp -r wheels/kaggle/2xt4 wheels/colab/l4
```

Edit:
- `profile`, `python`, `torch`, `cuda`, `platform` in `manifest.json`
- `kaggle_dataset` slugs (Colab equivalent: a GCS bucket or shared Drive folder
  — see `docs/cache-profiles.md` for the convention)
- Wheel URLs / sha256s for the new accelerator

Bootstrap your new notebook with:

```bash
python scripts/cache_setup.py --profile colab/l4
```

No code changes needed in `cache_setup.py`. The pattern is generic.
