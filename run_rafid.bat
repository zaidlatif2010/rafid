@echo off
rem Starts Rafid and opens the web interface. Extra options are passed through,
rem for example:  run_rafid.bat --port 8090
setlocal
cd /d "%~dp0"

rem Apply a pending update (a new build is delivered as bin\rafid-new.exe while the old one may be running)
if exist "bin\rafid-new.exe" (
  move /y "bin\rafid-new.exe" "bin\rafid.exe" >nul 2>&1 && echo Rafid was updated.
)

set "EXE=bin\rafid.exe"
if not exist "%EXE%" set "EXE=build\rafid.exe"
if not exist "%EXE%" set "EXE=build\Release\rafid.exe"
if not exist "%EXE%" (
  echo [!] rafid.exe was not found in the bin folder.
  pause
  exit /b 1
)

rem The llama.cpp DLLs need the Microsoft Visual C++ runtime.
if not exist "%SystemRoot%\System32\msvcp140.dll" goto novc
if not exist "%SystemRoot%\System32\vcruntime140_1.dll" goto novc

echo Starting Rafid. Keep this window open while you use it; close it to stop Rafid.
"%EXE%" --models-dir "%~dp0models" --profiles-dir "%~dp0profiles" %*
if errorlevel 1 pause
exit /b

:novc
echo [!] The Microsoft Visual C++ runtime is not installed (msvcp140.dll is missing).
echo     Download and install it from Microsoft, then run this file again:
echo     https://aka.ms/vs/17/release/vc_redist.x64.exe
pause
exit /b 1
