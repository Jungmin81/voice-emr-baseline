@echo off
chcp 65001 > nul
setlocal

REM Voice EMR Baseline - CLI runner

if "%~1"=="" (
    echo Usage: run_cli.bat ^<audio_file^> [options]
    echo.
    echo Examples:
    echo   run_cli.bat sample.wav
    echo   run_cli.bat sample.wav --skip-llm
    echo   run_cli.bat sample.wav --whisper-model small
    echo.
    pause
    exit /b 1
)

if exist "venv\Scripts\activate.bat" (
    call "venv\Scripts\activate.bat"
)

python baseline.py %*
pause
