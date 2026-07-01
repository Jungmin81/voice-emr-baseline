#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────
# 평가 — 파인튜닝 체크포인트 vs 베이스 medium 비교 (이 PC 전용 래퍼)
#
# 사용:
#   bash scripts/run_eval.sh outputs/phase_b_small/checkpoint-1000   # 파인튜닝
#   bash scripts/run_eval.sh openai/whisper-medium                   # 베이스 비교
#
# ⚠️ HANDOFF §8(4): --max-samples 300 은 test 앞 300개(전부 환자)만 봄(표본 편향).
#    의사/간호사 포함 정식 수치는 --max-samples 를 빼고 전체 test 로 평가할 것.
# ──────────────────────────────────────────────────────────────────────────
set -euo pipefail

MODEL="${1:?모델 경로를 넘기세요 (예: outputs/phase_b_small/checkpoint-1000 또는 openai/whisper-medium)}"
shift || true

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$HERE/venv_train/bin/activate"
cd "$HERE/voice_emr_baseline/training"

python evaluate.py \
  --model-dir "$MODEL" \
  --base-model openai/whisper-medium \
  --manifest-dir manifests/100p \
  --split test \
  "$@"
