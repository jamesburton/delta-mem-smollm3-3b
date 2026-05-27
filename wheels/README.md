# Wheel cache

Pre-built wheels for slow-to-compile packages (today: just `flash-attn`),
grouped by accelerator profile so each Kaggle/Colab session installs in
seconds instead of rebuilding from source.

## Layout

```
wheels/
└── <profile>/
    └── <wheelname>.whl
```

A "profile" is a short slug describing the **target accelerator + Python ABI**.
The path the bootstrap reads is controlled by the env var `WHEEL_PROFILE`
(default `kaggle/2xt4`).

Current profiles:

| Profile         | GPU                | CUDA cap | Python | Notes                       |
|---------------- |--------------------|----------|--------|----------------------------|
| `kaggle/2xt4`   | Tesla T4 × 2       | sm_75    | 3.12   | Free Kaggle accelerator    |

When you add a new profile (e.g. `kaggle/l4`, `colab/a100`):

1. Set `WHEEL_PROFILE=<slug>` and `TORCH_CUDA_ARCH_LIST=<cap>` before running
   `scripts/kaggle_bootstrap.sh`.
2. After the build, commit the produced `.whl` under `wheels/<slug>/`.
3. Add a row to the table above.

## Why constrain `TORCH_CUDA_ARCH_LIST`?

Default flash-attn wheels embed kernels for every supported compute capability
(sm_70 – sm_120) and weigh **~150-200 MB** — over GitHub's 100 MB per-file
limit. Narrowing the build to one cap (e.g. `7.5` for T4) drops the wheel to
~50-80 MB and it commits cleanly without Git LFS.

If you need a multi-target wheel (e.g. you bounce between T4 and L4 in the
same notebook), set `TORCH_CUDA_ARCH_LIST="7.5;8.9"` — but expect the wheel
to grow proportionally and possibly cross the 100 MB threshold; at that point
you'll need to use a GitHub Release asset instead of a tracked file.

## Where wheels come from

The bootstrap delegates to `scripts/install_flash_attn.py`, which tries three
sources in order:

1. **This local cache** (`wheels/<profile>/`). If a matching wheel already lives
   here, install in seconds, no network call.
2. **Community prebuilt at [mjun0812/flash-attention-prebuild-wheels](https://github.com/mjun0812/flash-attention-prebuild-wheels)**.
   The script queries the GitHub Releases API, finds a wheel matching the exact
   `(cuda, torch, python, platform)` tuple of the current kernel, downloads it
   into the local cache, then installs. Wheels are ~150-180 MB so this is a
   one-time ~30 s network hit. After this step you can `git add` the downloaded
   wheel and push it — future sessions skip the community lookup entirely.
3. **Source build** as a last resort. Constrained to `TORCH_CUDA_ARCH_LIST=7.5`
   for T4-only kernels to keep the resulting wheel under GitHub's 100 MB limit.

The community wheels are *much* bigger than a T4-only source build (~170 MB vs
~70 MB) because they target multiple compute capabilities. If repo size matters
to you, prefer the source-build path and commit the slimmer wheel back.

## What's *not* cached here

- Model weights (`.safetensors`, `.gguf`, etc.) — those go in HF Hub.
- Regular PyPI wheels — those are already binary on PyPI and install fast.
- Build artifacts other than the final `.whl`.
- DeepSpeed — PyPI ships prebuilt wheels for most environments, so caching it
  saves nothing.
