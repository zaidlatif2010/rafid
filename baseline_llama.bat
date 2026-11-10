@echo off
rem Speed of plain llama.cpp on the same model and the same computer, to compare with Rafid.
rem It is the same llama.cpp build Rafid uses (b12169), its own benchmark tool (llama-bench), set up the usual
rem way for a model bigger than memory: the layers on the GPU, the experts read from the disk through mmap.
rem Close Rafid first: both need the memory.
rem   baseline_llama.bat                      the model in LM Studio's folder
rem   baseline_llama.bat "D:\path\first.gguf" another place (the first file of the split)
setlocal
cd /d "%~dp0"
set "MODEL=%~1"
if "%MODEL%"=="" set "MODEL=%USERPROFILE%\.lmstudio\models\unsloth\Qwen3.5-397B-A17B-GGUF\Qwen3.5-397B-A17B-UD-IQ2_XXS-00001-of-00004.gguf"
if not exist "%MODEL%" (
  echo [!] Model not found: %MODEL%
  echo     Run it as:  baseline_llama.bat "full\path\to\the\first.gguf"
  pause
  exit /b 1
)
set "OUT=profiles\tests\results"
if not exist "%OUT%" mkdir "%OUT%"
set "LOG=%OUT%\baseline-llamacpp.log"
echo Model: %MODEL% > "%LOG%"

echo.
echo [1/3] Writing: 64 tokens, twice (after a warm-up). This can take 10-20 minutes.
set NGL=99
:gen
echo   trying with %NGL% layers on the GPU...
echo --- gen ngl=%NGL% >> "%LOG%"
bin\llama-bench.exe -m "%MODEL%" -ngl %NGL% -ncmoe 99 -p 0 -n 64 -r 2 -o json -oe md > "%OUT%\baseline-llamacpp-gen.json" 2>> "%LOG%"
if errorlevel 1 (
  if "%NGL%"=="99" ( set NGL=40& goto gen )
  if "%NGL%"=="40" ( set NGL=0& goto gen )
  echo [!] llama.cpp could not run the model. The details are in %LOG%
  pause
  exit /b 1
)
type "%LOG%" | findstr /c:"| qwen" 

echo.
echo [2/3] Reading a question: 128 tokens, twice.
echo --- pp ngl=%NGL% >> "%LOG%"
bin\llama-bench.exe -m "%MODEL%" -ngl %NGL% -ncmoe 99 -p 128 -n 0 -r 2 -o json -oe md > "%OUT%\baseline-llamacpp-pp.json" 2>> "%LOG%"
type "%LOG%" | findstr /c:"| qwen"

echo.
echo [3/3] Longer runs: writing 512 tokens and reading a 512-token question, once each. About 10-20 minutes.
echo --- long ngl=%NGL% >> "%LOG%"
bin\llama-bench.exe -m "%MODEL%" -ngl %NGL% -ncmoe 99 -p 512 -n 512 -r 1 -o json -oe md > "%OUT%\baseline-llamacpp-long.json" 2>> "%LOG%"
type "%LOG%" | findstr /c:"| qwen"
echo.
echo Done. The results are in %OUT% (baseline-llamacpp-*.json and .log).
pause
