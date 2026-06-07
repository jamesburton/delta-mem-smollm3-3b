# FlashAttention-2 on Windows (RTX 3060)

How to get FA2 working in the local Windows dev env, why this specific torch
version, and how to validate the install.

## TL;DR

```powershell
cd E:\Development\llm-model-tests\delta-mem-smollm3-3b
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts\install_local_windows.py
```

That last line:

1. Downloads `torch-2.9.0+cu128` (2.7 GB) into `wheels/local/3060/`.
2. Downloads `flash_attn-2.8.3` Windows wheel matching that torch ABI (121 MB).
3. SHA256-verifies both.
4. Installs them with `pip install --no-deps`.
5. Smoke-tests FA2 with a bf16 forward pass.

Expected end state:

```
torch 2.9.0+cu128   flash_attn 2.8.3
GPU 0 compute capability: sm_86
FA2 forward OK, output shape (2, 4, 8, 64)
```

After that, the test harness `harness/backbone.py:_attn_impl_for_hardware` will
elect `attn_implementation="flash_attention_2"` automatically — no further code
change required.

## Why torch 2.9 (not 2.10 or 2.11)

FlashAttention-2's CUDA module links against torch's `_C` Python extension. The
ABI breaks across minor torch versions, particularly on Windows where there's
no manylinux-style ABI stability story.

As of June 2026, the only community sources publishing Windows FA2 wheels are
[`bdashore3/flash-attention`](https://github.com/bdashore3/flash-attention) and
[`kingbri1/flash-attention`](https://github.com/kingbri1/flash-attention). Both
top out at **torch 2.9.0 + cu128** in their latest release (v2.8.3, August 2025).

We previously ran torch 2.11.0+cu128 locally. Installing the torch-2.9 FA2 wheel
on that env produced:

```
ImportError: DLL load failed while importing flash_attn_2_cuda:
  The specified procedure could not be found.
```

Downgrading torch to 2.9.0+cu128 makes the wheel import cleanly.

If a newer torch wheel appears, update `wheels/local/3060/manifest.json` with
the new entry and re-run `install_local_windows.py`.

## Why not build from source

Building FA2 from source on Windows needs:

- MSVC C++ build tools (`cl.exe` in PATH).
- A matching CUDA toolkit (`nvcc` already present locally at v12.8).
- 30+ GB of free RAM during link (the linker holds a lot of state).
- 30–90 minutes per build.

For a one-off iteration env, the prebuilt wheels are dramatically faster. If
this changes (e.g. need a torch version no one prebuilds for), see
[`Dao-AILab/flash-attention`](https://github.com/Dao-AILab/flash-attention)
README for the Windows source-build recipe.

## What the wheels weigh

| Wheel                      | Size    | Cached at                            | Git status |
|----------------------------|---------|--------------------------------------|------------|
| `torch-2.9.0+cu128 cp311`  | 2.7 GB  | `wheels/local/3060/torch-2.9.0+...`  | gitignored |
| `flash_attn-2.8.3 cp311`   | 121 MB  | `wheels/local/3060/flash_attn-...`   | gitignored |

Both are over GitHub's 100 MB per-file limit, so we don't commit them. The
`manifest.json` next to them records SHA256 + URL so the installer can fetch
them deterministically.

## The harness FA2 gate

`harness/backbone.py:_attn_impl_for_hardware()` does the gating:

1. If the request isn't `"flash_attention_2"`, pass it through.
2. Walk `torch.cuda.get_device_capability(i)` — if any visible GPU is sm < 80
   (Turing / Volta / earlier), return `"sdpa"`.
3. Try `import flash_attn`. If `ImportError`, return `"sdpa"`.
4. Otherwise return `"flash_attention_2"`.

Step 3 is what makes a missing FA2 install graceful — the harness falls back to
PyTorch's SDPA mem-efficient path instead of crashing at first forward. So
running without FA2 still produces valid (slower) results.

The SDP backend preferences are set by `configure_sdp_backends()` in the same
file: `flash` > `mem_efficient` > `cudnn` enabled, `math` **disabled** by
default to prevent the math kernel from materialising an N×N attention matrix
and OOM-ing at 4K+ context. Override via `ENABLE_MATH_SDP=1`.

## Performance expectations

Pre-FA2 (SDPA mem-efficient) on the 3060 at 1.5K NIH context:

| Cell             | NIH  | Peak VRAM | decode tok/s | Wall (s) |
|------------------|------|-----------|--------------|----------|
| 1 vanilla        | 1.00 | 7.9 GiB   | 1.0          | 134      |
| 2 +δ-Mem         | 1.00 | 8.4 GiB   | 0.8          | 229      |
| 6 +spec-decode   | 1.00 | 9.3 GiB   | 2.4          | 413      |

FA2 is generally expected to add 3-5× decode throughput on Ampere+ at these
sequence lengths. See `results/LOCAL_FA2/` for the post-FA2 numbers after the
sweep completes.

## Troubleshooting

### `DLL load failed while importing flash_attn_2_cuda`

The FA2 wheel was built against a different torch ABI than the installed torch.
Run `python -c "import torch; print(torch.__version__)"`; it MUST be exactly
`2.9.0+cu128` for the cached wheel. If you upgraded torch, re-pin to 2.9.0:

```powershell
pip uninstall -y torch flash-attn
python scripts\install_local_windows.py
```

### `RuntimeError: FlashAttention only supports Ampere GPUs or newer`

Your GPU is sm_75 (T4) or earlier. FA2 needs sm_80+. The harness should catch
this in `_attn_impl_for_hardware` and fall back to SDPA — if you're seeing this
at runtime, it means the gate is being bypassed (custom code path?). File at
`harness/backbone.py:147`.

### `OOM at 4K+ context`

Likely the math SDP kernel got picked because mem-efficient couldn't handle the
shape. Check the load log for `SDP backends:` — `math` should be `False`. If
it's `True`, unset `ENABLE_MATH_SDP`.

### Slow downloads from pytorch.org

The `download.pytorch.org` CDN is rate-limited per source IP for very large
wheels. The 2.7 GB torch wheel took ~35 minutes at 1.3 MB/s during initial
setup on this profile. That's expected, not a bug. Once cached in
`wheels/local/3060/`, subsequent installs are file-copy speed.

## File map

```
wheels/local/3060/
├── manifest.json                 # SHA256 + URL pointers (committed)
├── README.md                     # quick reference (committed)
├── torch-2.9.0+cu128-...whl      # 2.7 GB, gitignored
└── flash_attn-2.8.3+...whl       # 121 MB, gitignored

scripts/install_local_windows.py  # idempotent fetch + verify + install
docs/windows-fa2-setup.md         # this file
.gitignore                        # wheels/local/**/*.whl excluded
```
