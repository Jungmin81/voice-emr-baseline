@echo off
chcp 65001 > nul
setlocal

echo ======================================================
echo   Voice EMR Baseline - Web UI
echo ======================================================
echo.

if exist "venv\Scripts\activate.bat" (
    call "venv\Scripts\activate.bat"
    echo [OK] Virtual env activated
) else (
    echo [!] No venv found. Using system Python.
    echo     Create venv: python -m venv venv ^&^& venv\Scripts\activate ^&^& pip install -r requirements.txt
)

echo.
echo Starting Gradio server...
echo Open browser at: http://localhost:7860
echo.

python app.py
pause
