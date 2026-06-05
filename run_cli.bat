@echo off
chcp 65001 > nul
echo ======================================================
echo  Voice EMR Baseline - CLI 실행
echo ======================================================
echo.

if "%~1"=="" (
    echo 사용법: run_cli.bat ^<음성파일경로^> [옵션]
    echo.
    echo 예시:
    echo   run_cli.bat sample.wav
    echo   run_cli.bat sample.wav --skip-llm
    echo   run_cli.bat sample.wav --whisper-model small
    echo.
    pause
    exit /b 1
)

if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
)

python baseline.py %*
pause
