@echo off
chcp 65001 > nul
setlocal enabledelayedexpansion

echo ======================================================
echo   Voice EMR Benchmark
echo ======================================================
echo.

REM 가상환경 활성화
if exist "venv\Scripts\activate.bat" (
    call "venv\Scripts\activate.bat"
)

REM 샘플 파일 확인
if not exist "samples\sample_short.wav" (
    echo [!] samples\sample_short.wav not found
    echo     First run: python generate_samples.py
    pause
    exit /b 1
)

if not exist "samples\sample_long.wav" (
    echo [!] samples\sample_long.wav not found
    echo     First run: python generate_samples.py
    pause
    exit /b 1
)

echo Select benchmark mode:
echo.
echo   [1] Quick  - STT only, fast (5-10 min)
echo   [2] Full   - STT + SOAP summary (20-40 min)
echo   [3] GPU only
echo   [4] CPU only
echo.

set /p choice="Select (1-4, default=1): "

if "%choice%"=="" set choice=1

if "%choice%"=="1" (
    python benchmark.py --samples "samples\sample_short.wav" "samples\sample_long.wav" --preset quick
) else if "%choice%"=="2" (
    python benchmark.py --samples "samples\sample_short.wav" "samples\sample_long.wav" --preset full
) else if "%choice%"=="3" (
    python benchmark.py --samples "samples\sample_short.wav" "samples\sample_long.wav" --preset full --devices cuda
) else if "%choice%"=="4" (
    python benchmark.py --samples "samples\sample_short.wav" "samples\sample_long.wav" --preset full --devices cpu
) else (
    echo Invalid choice, running quick mode...
    python benchmark.py --samples "samples\sample_short.wav" "samples\sample_long.wav" --preset quick
)

echo.
echo ======================================================
echo   Done - check bench_results\report.md
echo ======================================================
pause
