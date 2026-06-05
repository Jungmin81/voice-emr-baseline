@echo off
chcp 65001 > nul
echo ======================================================
echo  Voice EMR Baseline - 웹 UI 실행
echo ======================================================
echo.

REM 가상환경이 있으면 활성화
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
    echo [OK] 가상환경 활성화됨
) else (
    echo [경고] 가상환경 없음. 시스템 Python 사용.
    echo        venv 만들기: python -m venv venv ^&^& venv\Scripts\activate ^&^& pip install -r requirements.txt
)

echo.
echo Gradio 서버 시작 중...
echo 브라우저에서 http://localhost:7860 접속하세요
echo.
python app.py
pause
