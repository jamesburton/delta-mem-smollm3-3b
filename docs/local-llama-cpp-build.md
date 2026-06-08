# Building llama-cpp-python with CUDA on this Windows machine

Why this needs a custom build:

- This box has an **Intel Xeon X5670** (Westmere, 2010). It lacks AVX
  entirely, never mind AVX2. Every prebuilt `llama-cpp-python` wheel
  from PyPI or the abetlen release page assumes AVX or AVX2 (the
  CPU-only PyPI wheel actually imports OK but the GPU-enabled
  `cu130` / `cu128` ones crash on first native call with
  `WinError 0xC000001D` = `ILLEGAL_INSTRUCTION`).
- The RTX 3060 is fully functional via CUDA, so we just need to
  build with `GGML_AVX=OFF GGML_AVX2=OFF GGML_FMA=OFF` and let CUDA
  do the work.

## Prerequisites

Already on this machine:

- Visual Studio 2026 Build Tools (cl.exe 14.50, MSBuild 18).
- CUDA Toolkit 12.8 at `E:\CUDA_v12.8.1` (nvcc 12.8.93).
- Python 3.11 venv at `.venv\`.
- `ninja` package (Python wrapper) installed; bundles `ninja.exe`
  under `.venv\Scripts\`.

If any of these are missing, install them first.

## Build script

`scripts/build_llama_cpp_local.bat` does it (calls vcvars64.bat,
sets CMAKE_ARGS, runs `pip install --force-reinstall llama-cpp-python`).
The PowerShell helper in `scripts/build_llama_cpp_local.ps1` is
equivalent and produces a more usable log.

Critical CMAKE_ARGS:

```
-G Ninja
-DCMAKE_C_COMPILER=cl
-DCMAKE_CXX_COMPILER=cl
-DCMAKE_CUDA_COMPILER=E:/CUDA_v12.8.1/bin/nvcc.exe
-DCMAKE_CUDA_FLAGS_INIT=-allow-unsupported-compiler
-DGGML_CUDA=on
-DGGML_NATIVE=OFF
-DGGML_AVX=OFF
-DGGML_AVX2=OFF
-DGGML_FMA=OFF
-DCMAKE_CUDA_ARCHITECTURES=86
```

- `-G Ninja` skips MSBuild's CUDA integration check (which needs
  CUDA's VS extension files installed under Program Files\VS\... —
  we don't have admin to drop them in).
- `-allow-unsupported-compiler` passed to nvcc so it accepts MSBuild
  Tools 18's cl 14.50, which CUDA 12.8's nvcc considers "future"
  and would otherwise reject.
- `GGML_NATIVE=OFF` stops CMake auto-probing the host CPU for
  instruction sets; we set them by hand.
- `GGML_AVX=OFF GGML_AVX2=OFF GGML_FMA=OFF` produce a SSE4.2-only
  CPU backend, which is what the X5670 supports.
- `CMAKE_CUDA_ARCHITECTURES=86` targets Ampere only. Smaller binary,
  faster build, only compute capability the local GPU supports anyway.

Build takes about 15–20 minutes on this machine (12 cores, no AVX).

## Verifying

```powershell
.\.venv\Scripts\python.exe -c "import llama_cpp; print(llama_cpp.llama_supports_gpu_offload())"
```

Should print `True`. If it instead prints `False`, the build fell back
to CPU-only — re-check the `GGML_CUDA=on` flag survived environment
substitution and that `nvcc` was on PATH when CMake ran.

## Smoke test

```powershell
.\.venv\Scripts\python.exe -c "
from llama_cpp import Llama
from huggingface_hub import hf_hub_download
p = hf_hub_download(repo_id='unsloth/Qwen3.5-4B-MTP-GGUF',
                    filename='Qwen3.5-4B-Q4_K_M.gguf')
llm = Llama(model_path=p, n_ctx=4096, n_gpu_layers=-1, verbose=False)
print(llm('The capital of France is', max_tokens=8, temperature=0)['choices'][0]['text'])
"
```

Expect `Paris.` (or similar) and ~3-4 GiB of additional GPU memory used.

## When the prebuilt wheel finally works

If a future `abetlen/llama-cpp-python` release ships SSE4.2-only or
no-AVX prebuilt CUDA wheels, drop them into `wheels/local/3060/` and
update `manifest.json` so `scripts/install_local_windows.py` can use
them. The current build is reproducible but slow; a wheel would
collapse the setup to a single `pip install`.
