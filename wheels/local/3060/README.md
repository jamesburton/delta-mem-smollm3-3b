# Local Windows wheels — RTX 3060 12 GB profile

Pre-built wheels for the **local Windows** development env that the test matrix
uses for fast iteration before paid cloud runs. These match the toolchain pinned
by `requirements.txt` plus `torch==2.9.0+cu128`.

**Why these aren't committed:** `torch-2.9.0+cu128` is 2.7 GB and `flash_attn`
is 121 MB — both blow past GitHub's 100 MB per-file limit. The `.whl` files
in this directory are gitignored; this README + the `manifest.json` are the
authoritative pointers, and `scripts/install_local_windows.py` fetches them.

## Profile

| Field                  | Value                                  |
|------------------------|----------------------------------------|
| GPU                    | RTX 3060 12 GB (Ampere, sm_86)         |
| OS                     | Windows 11 Pro                         |
| Python                 | 3.11 (cp311)                           |
| Torch                  | 2.9.0+cu128                            |
| CUDA toolkit (runtime) | 12.8 (bundled in torch wheel)          |
| flash-attn             | 2.8.3 (cu128 + torch2.9 ABI)           |

## Files in this directory

```
torch-2.9.0+cu128-cp311-cp311-win_amd64.whl                            (~2.7 GB)
flash_attn-2.8.3+cu128torch2.9.0cxx11abiFALSE-cp311-cp311-win_amd64.whl  (~121 MB)
manifest.json                                                          (SHA256, URLs)
README.md                                                              (this file)
```

## How to populate

```powershell
# From repo root, with the venv already created:
.\.venv\Scripts\Activate.ps1
python scripts\install_local_windows.py
```

The script:

1. Reads `wheels/local/3060/manifest.json`.
2. Downloads any wheel missing from this directory.
3. Verifies SHA256 against the manifest.
4. Installs with `pip install --no-deps`.
5. Verifies `import flash_attn` and runs a tiny CUDA forward.

## Why these specific versions

- **torch 2.9.0+cu128** is the highest torch version with a community-prebuilt
  FA2 Windows wheel (`bdashore3/flash-attention` / `kingbri1/flash-attention`,
  release v2.8.3). Newer torch (2.10, 2.11) ABI-breaks the FA2 `_C` symbols on
  Windows, so importing FA2 fails with `DLL load failed`.
- **flash_attn 2.8.3** is the matching wheel. It bundles kernels for sm_80
  through sm_120 so it works on Ampere (3060) and Ada/Hopper alike.
- **cu128** keeps us aligned with Kaggle's torch+cu128 stack.

See `docs/windows-fa2-setup.md` for the full write-up of why this combination,
what went wrong before we landed on it, and how to upgrade when newer wheels
become available.
