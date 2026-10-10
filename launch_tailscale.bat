@echo off
setlocal EnableExtensions DisableDelayedExpansion
title ComfyUI Wrapper - Tailscale

rem Change this port if 8000 is already in use.
set "WRAPPER_PORT=8000"
set "TAILSCALE_EXE="
set "TAILSCALE_IP="
set "WRAPPER_PYTHON="

pushd "%~dp0"
if errorlevel 1 (
    echo ERROR: Could not open the wrapper directory.
    pause
    exit /b 1
)

if exist "%ProgramFiles%\Tailscale\tailscale.exe" set "TAILSCALE_EXE=%ProgramFiles%\Tailscale\tailscale.exe"
if not defined TAILSCALE_EXE (
    for /f "delims=" %%T in ('where.exe tailscale.exe 2^>nul') do if not defined TAILSCALE_EXE set "TAILSCALE_EXE=%%T"
)
if not defined TAILSCALE_EXE (
    echo ERROR: Tailscale was not found. Install it and connect to your tailnet.
    goto failed
)

for /f "delims=" %%I in ('""%TAILSCALE_EXE%" ip -4"') do set "TAILSCALE_IP=%%I"
if not defined TAILSCALE_IP (
    echo ERROR: Could not read your Tailscale IPv4 address.
    echo Open Tailscale, connect to your tailnet, and run this launcher again.
    goto failed
)

if exist "%~dp0.venv\Scripts\python.exe" set "WRAPPER_PYTHON=%~dp0.venv\Scripts\python.exe"
if not defined WRAPPER_PYTHON if exist "%~dp0venv\Scripts\python.exe" set "WRAPPER_PYTHON=%~dp0venv\Scripts\python.exe"
if not defined WRAPPER_PYTHON (
    for /f "delims=" %%P in ('where.exe python.exe 2^>nul') do if not defined WRAPPER_PYTHON set "WRAPPER_PYTHON=%%P"
)
if not defined WRAPPER_PYTHON (
    echo ERROR: Python was not found. Install Python or activate your wrapper environment.
    goto failed
)

echo.
echo Starting the ComfyUI wrapper on port %WRAPPER_PORT%.
echo Once startup completes, open either address:
echo   This PC:        http://127.0.0.1:%WRAPPER_PORT%
echo   Your tailnet:   http://%TAILSCALE_IP%:%WRAPPER_PORT%
echo.
echo ComfyUI must already be running. Keep this window open.
echo Press Ctrl+C to stop the wrapper.
echo.

"%WRAPPER_PYTHON%" "%~dp0webui.py" --listen --port "%WRAPPER_PORT%" --no-browser
set "WRAPPER_EXIT=%ERRORLEVEL%"
echo.
if not "%WRAPPER_EXIT%"=="0" echo ERROR: The wrapper exited with code %WRAPPER_EXIT%. See the message above.
pause
popd
exit /b %WRAPPER_EXIT%

:failed
echo.
pause
popd
exit /b 1
