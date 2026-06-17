@echo off
REM ar_gguf_qwen3_4b.bat — run llama.cpp GGUF Qwen3-4B against the bnb-4bit floor.
REM
REM This MUST use the project venv (E:\Development\llm-model-tests\delta-mem-smollm3-3b\.venv\),
REM NOT the system Python311 (C:\Python311), because llama-cpp-python 0.3.28
REM was built from source ONLY into the venv (the system Python311 has no
REM llama_cpp module — verified 2026-06-17).
REM
REM The custom CUDA + no-AVX llama.cpp build is described in
REM `docs/local-llama-cpp-build.md`. Prebuilt CUDA wheels from PyPI / abetlen
REM CRASH on first call here because Xeon X5670 lacks AVX/AVX2 — WinError
REM 0xC000001D ILLEGAL_INSTRUCTION. Do not "fix" by reinstalling from PyPI.

set PYTHONIOENCODING=utf-8
set SCRIPT_DIR=%~dp0

REM The venv lives in the repo root, two parents above this worktree.
REM Hardcode the absolute path since the project's worktree layout doesn't
REM vendor the venv inside the worktree.
set VENV_PY=E:\Development\llm-model-tests\delta-mem-smollm3-3b\.venv\Scripts\python.exe

if not exist "%VENV_PY%" (
    echo [fatal] project venv not found at %VENV_PY%
    echo Rebuild llama-cpp-python with: scripts\build_llama_cpp_local.bat
    exit /b 1
)

REM Pass all args through (--quant, --ctx, --new-tokens, --n-batch).
"%VENV_PY%" "%SCRIPT_DIR%ar_gguf.py" %*
