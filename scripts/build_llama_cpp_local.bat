@echo on
REM Build llama-cpp-python from source with CUDA + AVX disabled.
REM Logs everything to scripts\build_llama_cpp_local.log

set LOG=%~dp0build_llama_cpp_local.log
echo === BUILD START %DATE% %TIME% > "%LOG%"

call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >> "%LOG%" 2>&1
if errorlevel 1 (
    echo vcvars64 FAILED >> "%LOG%"
    exit /b 1
)

set CUDA_PATH=E:\CUDA_v12.8.1
set PATH=%CUDA_PATH%\bin;%PATH%
set CMAKE_ARGS=-DGGML_CUDA=on -DGGML_NATIVE=OFF -DGGML_AVX=OFF -DGGML_AVX2=OFF -DGGML_FMA=OFF -DCMAKE_CUDA_ARCHITECTURES=86
set FORCE_CMAKE=1

echo === CL.EXE >> "%LOG%"
where cl >> "%LOG%" 2>&1
echo === NVCC >> "%LOG%"
where nvcc >> "%LOG%" 2>&1
echo === CMAKE_ARGS=%CMAKE_ARGS% >> "%LOG%"
echo === >> "%LOG%"

cd /d %~dp0..
.venv\Scripts\python.exe -m pip install --no-cache-dir --force-reinstall llama-cpp-python==0.3.28 >> "%LOG%" 2>&1
set RC=%ERRORLEVEL%
echo === BUILD END %DATE% %TIME% rc=%RC% >> "%LOG%"
exit /b %RC%
