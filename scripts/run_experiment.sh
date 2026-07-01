#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────
# 전이학습 실험 1회 완주: 학습 → (완료 후) 최신 체크포인트 after-eval → before/after 비교
#   · before(base) 평가는 별도로 먼저 돌려 outputs/exp1/eval_BEFORE_base_medium.json 에 저장돼 있어야 함
#   · 학습이 성공해야 after-eval 진행
# 사용:  bash scripts/run_experiment.sh
# ──────────────────────────────────────────────────────────────────────────
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$HERE/venv_train/bin/activate"
export AIHUB_DATA_DIR="${AIHUB_DATA_DIR:-/mnt/nas_raw/rawdata/etc/voice/aihub/Training}"
TDIR="$HERE/voice_emr_baseline/training"
OUT="$TDIR/outputs/exp1/phase_b_small"
BEFORE="$TDIR/outputs/exp1/eval_BEFORE_base_medium.json"
AFTER="$TDIR/outputs/exp1/eval_AFTER_ft.json"

echo "######## [1/3] 학습 시작 ########"
bash "$HERE/scripts/run_train_a6000.sh" || { echo "학습 실패 — 중단"; exit 1; }

echo "######## [2/3] after-eval (최신 체크포인트) ########"
CKPT="$(ls -d "$OUT"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)"
[ -z "$CKPT" ] && { echo "체크포인트 없음: $OUT"; exit 1; }
echo ">> 평가 대상: $CKPT"
cd "$TDIR"
python evaluate.py --model-dir "$CKPT" --base-model openai/whisper-medium \
  --manifest-dir manifests/exp1 --split test --output "$AFTER" || { echo "after-eval 실패"; exit 1; }

echo "######## [3/3] before/after 비교 ########"
python "$HERE/scripts/compare_eval.py" "$BEFORE" "$AFTER"
echo ""
echo "완료. before=$BEFORE  after=$AFTER"
