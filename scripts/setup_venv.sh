#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────
# 이 PC(2×RTX A6000 48GB, driver 595 / CUDA 13.2) 전용 학습 venv 설치
# HANDOFF.md §8 "(1) 환경 구축" 을 이 머신에 맞게 follow-up 한 버전.
#   - torchvision / bitsandbytes 는 설치하지 않는다 (원 서버에서 import 체인 깨뜨림).
#   - torch 는 CUDA 12.4 휠 (driver 595 와 호환).
# 사용법:  bash scripts/setup_venv.sh
# ──────────────────────────────────────────────────────────────────────────
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$HERE/venv_train"

echo ">> venv 생성: $VENV"
python3.10 -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"

python -m pip install -U pip wheel

echo ">> torch (cu124) 설치"
pip install torch --index-url https://download.pytorch.org/whl/cu124

echo ">> 학습 스택 설치 (torchvision / bitsandbytes 제외)"
pip install \
  "transformers>=4.45,<5" "peft>=0.13" "accelerate>=0.34" "datasets>=2.20" \
  tensorboard "gradio>=4.44" librosa soundfile jiwer pandas psutil faster-whisper

echo ""
echo ">> 검증"
python - <<'PY'
import torch, transformers, peft, accelerate, datasets, librosa
print("torch        ", torch.__version__, "| cuda:", torch.cuda.is_available(),
      "| ngpu:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print("   GPU", i, torch.cuda.get_device_name(i))
print("transformers ", transformers.__version__)
print("peft         ", peft.__version__)
print("accelerate   ", accelerate.__version__)
print("datasets     ", datasets.__version__)
print("librosa      ", librosa.__version__)
print("OK")
PY
echo ""
echo ">> 완료. 활성화:  source $VENV/bin/activate"
