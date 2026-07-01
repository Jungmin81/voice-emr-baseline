#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────
# Phase B 학습 — 이 PC(2×RTX A6000 48GB) 전용 실행 스크립트
#
# 원 서버(4×A100 80GB)는  batch 16 × 4 GPU × accum 1 = effective 64.
# 이 PC 는        batch  8 × 2 GPU × accum 4 = effective 64  (동일 → step수·LR 스케줄 유지).
#   - A6000 48GB 는 A100 80GB 의 ~60% → per-device batch 를 16→8 로 절반.
#   - GPU 2장이라 부족한 effective batch 는 grad-accum=4 로 보전.
#   - gradient_checkpointing 은 그대로 OFF (medium LoRA, batch8 이면 48GB 에 들어감).
#     혹시 OOM 이면: --batch-size 4 --grad-accum 8  로 더 줄이거나, config.py 에서
#     gradient_checkpointing=True (단 DDP "mark ready once" 이슈 주의, HANDOFF §4 참고).
#
# 선행조건:
#   1) bash scripts/setup_venv.sh  로 venv_train 준비
#   2) AI Hub 데이터 확보 후 manifest 재생성 (데이터 경로가 원 서버와 다르므로 필수):
#        export AIHUB_DATA_DIR="<데이터>/aihub/Training"
#        cd voice_emr_baseline/training
#        python prepare_dataset.py --data-dir "$AIHUB_DATA_DIR" \
#               --output manifests/100p --categories 환자 의사 간호사 --max-speakers 100
# ──────────────────────────────────────────────────────────────────────────
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$HERE/venv_train/bin/activate"
cd "$HERE/voice_emr_baseline/training"

# 데이터 경로 (필요 시 여기서 export 하거나 셸에서 미리 설정)
: "${AIHUB_DATA_DIR:?AIHUB_DATA_DIR 가 설정되지 않았습니다. export AIHUB_DATA_DIR=<...>/aihub/Training}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
NPROC="$(awk -F',' '{print NF}' <<< "$CUDA_VISIBLE_DEVICES")"

# exp1 = 균형(stratified, 화자당 200발화 상한) manifest, 기존 checkpoint 와 분리된 출력 폴더
MANIFEST_DIR="${MANIFEST_DIR:-manifests/exp1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/exp1}"
PHASE="${PHASE:-b}"            # b=중규모, c=전체

echo ">> GPUs: $CUDA_VISIBLE_DEVICES (nproc=$NPROC)"
echo ">> phase: $PHASE   manifest: $MANIFEST_DIR   output: $OUTPUT_ROOT   data: $AIHUB_DATA_DIR"

# --resume 를 인자로 넘기면 checkpoint 에서 재개됨 (데이터 경로가 원본과 동일할 때만 안전)
torchrun --nproc_per_node="$NPROC" train_whisper.py \
  --phase "$PHASE" \
  --manifest-dir "$MANIFEST_DIR" \
  --output-root "$OUTPUT_ROOT" \
  --base-model openai/whisper-medium \
  --batch-size 8 \
  --grad-accum 4 \
  --num-workers 8 \
  "$@"

# 모니터링(별도 터미널):
#   source venv_train/bin/activate
#   tensorboard --logdir voice_emr_baseline/training/outputs/phase_b_small/tensorboard \
#               --host 0.0.0.0 --port 6006
