#!/usr/bin/env bash
# Voice EMR Baseline — Ubuntu 자동 셋업 스크립트
#
# 사용: bash setup_ubuntu.sh
#       또는 chmod +x setup_ubuntu.sh && ./setup_ubuntu.sh

set -e  # 에러 발생 시 즉시 중단

echo "======================================================"
echo "  Voice EMR Baseline — Ubuntu Setup"
echo "======================================================"
echo ""

# 1. 시스템 패키지 확인
echo "[1/5] 시스템 패키지 확인..."
for cmd in python3 pip ffmpeg git; do
    if ! command -v $cmd &> /dev/null; then
        echo "  ❌ $cmd 없음 — 설치 필요:"
        echo "     sudo apt install -y python3 python3-pip python3-venv ffmpeg git"
        exit 1
    fi
done
echo "  ✅ 시스템 패키지 OK"

# 2. GPU 확인
echo ""
echo "[2/5] GPU 확인..."
if command -v nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    echo "  ✅ GPU 감지: $GPU_NAME"
    HAS_GPU=1
else
    echo "  ⚠️  NVIDIA GPU 없음 — CPU 모드로 동작"
    HAS_GPU=0
fi

# 3. 가상환경
echo ""
echo "[3/5] 가상환경 생성..."
if [ -d "venv" ]; then
    echo "  이미 venv 존재 — 재사용"
else
    python3 -m venv venv
    echo "  ✅ venv 생성"
fi
source venv/bin/activate
pip install --upgrade pip --quiet

# 4. PyTorch
echo ""
echo "[4/5] PyTorch 설치..."
if [ $HAS_GPU -eq 1 ]; then
    echo "  GPU용 PyTorch (CUDA 12.1) 설치 중... (약 2~5분)"
    pip install torch --index-url https://download.pytorch.org/whl/cu121 --quiet
else
    echo "  CPU용 PyTorch 설치 중..."
    pip install torch --index-url https://download.pytorch.org/whl/cpu --quiet
fi
echo "  ✅ PyTorch 완료"

# 5. 의존성
echo ""
echo "[5/5] 나머지 의존성 설치..."
pip install -r requirements.txt --quiet
echo "  ✅ 모든 패키지 설치 완료"

# 검증
echo ""
echo "======================================================"
echo "  설치 검증"
echo "======================================================"
python check_gpu.py 2>/dev/null || python -c "
import torch
print('PyTorch:', torch.__version__)
print('CUDA 가능:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
"

echo ""
echo "🎉 셋업 완료!"
echo ""
echo "다음 단계:"
echo "  1. 샘플 생성:        python generate_samples.py"
echo "  2. CLI 실행:         python baseline.py samples/sample_short.wav"
echo "  3. 웹 UI 실행:        python app.py"
echo "  4. 벤치마크 실행:     bash run_benchmark.sh"
echo ""
echo "(venv 활성화 명령:  source venv/bin/activate)"
