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

## What's *not* cached here

- Model weights (`.safetensors`, `.gguf`, etc.) — those go in HF Hub.
- Regular PyPI wheels — those are already binary on PyPI and install fast.
- Build artifacts other than the final `.whl`.
