@echo off
REM ar_marlin.bat — run GPTQ-Marlin Qwen3-4B against the bnb-4bit floor.
REM
REM This must use the SYSTEM Python311 (C:\Python311), NOT the project venv,
REM because gptqmodel/optimum/bnb live in the system site only. The project
REM venv has torch 2.9+cu128 and no gptqmodel — different stack entirely.
REM
REM Setting PYTHONIOENCODING + PYTHONUTF8 here because gptqmodel's
REM import-time ASCII-art logo crashes Windows cp1252 consoles
REM (see CUDA_NOTES.md "Common pitfalls" 2026-06-16). The Python script
REM also sets these defensively, but launcher-level is the belt.

set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
set SCRIPT_DIR=%~dp0

REM Pass all args through (--attn, --backend, --ctx, --new-tokens).
C:\Python311\python.exe -X utf8 "%SCRIPT_DIR%ar_marlin.py" %*
