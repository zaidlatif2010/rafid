@echo off
rem ============================================================================
rem  Rafid - Windows build
rem  Open "x64 Native Tools Command Prompt for VS" from the Start menu, go to this
rem  folder, and run:  build_windows.bat
rem  Needs: Visual Studio 2022 (or newer) with "Desktop development with C++",
rem         CUDA Toolkit 12.8 or newer (RTX 50 series), Git, internet for the first build.
rem ============================================================================
setlocal
cd /d "%~dp0"

where cl >nul 2>nul
if errorlevel 1 (
  echo [!] The C++ compiler was not found.
  echo     Run this file from "x64 Native Tools Command Prompt for VS".
  exit /b 1
)
where cmake >nul 2>nul
if errorlevel 1 (
  echo [!] CMake was not found. It comes with Visual Studio's C++ workload.
  exit /b 1
)
where git >nul 2>nul
if errorlevel 1 (
  echo [!] Git was not found. Install it from https://git-scm.com and reopen the prompt.
  exit /b 1
)

set CUDA_FLAG=
where nvcc >nul 2>nul
if errorlevel 1 (
  echo [!] nvcc was not found: building a CPU-only version.
  echo     For the NVIDIA GPU install CUDA Toolkit 12.8+ and run this again.
) else (
  set CUDA_FLAG=-DGGML_CUDA=ON
  echo [+] CUDA found: building with GPU support.
)

set GEN=
where ninja >nul 2>nul
if not errorlevel 1 set GEN=-G Ninja

cmake -S . -B build %GEN% -DCMAKE_BUILD_TYPE=Release %CUDA_FLAG%
if errorlevel 1 exit /b 1
cmake --build build --config Release -j
if errorlevel 1 exit /b 1

echo.
if exist build\rafid.exe (
  echo [+] Done: build\rafid.exe
) else (
  echo [+] Done: build\Release\rafid.exe
)
echo     Start it with run_rafid.bat
endlocal
