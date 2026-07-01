#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────
# 중규모(exp1) 학습이 끝나기를 대기 → ① 중간 결과 보관 → ② 전체 데이터 학습
# 을 자동으로 이어서 진행. 백그라운드로 띄워두면 됨.
#
#   · 동일 yardstick: 전/후 평가는 모두 exp1 의 test(2400, 균형)로.
#     (base before = outputs/exp1/.../eval_BEFORE_base_medium.json 재사용)
#   · leakage 방지: 전체 학습 train 에서 exp1 의 val/test 화자를 제외.
#   · 전체 런은 phase c(lr 5e-5) + EarlyStopping(patience=3) + eval/save 1000 step.
#
# 사용:  nohup bash scripts/run_full_after_mid.sh > full_pipeline.log 2>&1 &
# ──────────────────────────────────────────────────────────────────────────
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$HERE/venv_train/bin/activate"
export AIHUB_DATA_DIR="${AIHUB_DATA_DIR:-/mnt/nas_raw/rawdata/etc/voice/aihub/Training}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

TDIR="$HERE/voice_emr_baseline/training"
EXP1="$TDIR/outputs/exp1"
MARKER="$EXP1/eval_AFTER_ft.json"          # exp1 완료 신호 (after-eval 산출물)
BEFORE="$EXP1/eval_BEFORE_base_medium.json"
RESULTS="$HERE/results"
SD="/tmp/claude-1003/-home-jungmin-cheon/d7395b5d-7ffe-41da-abb0-c9754751c34f/scratchpad"

log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

# ── 0) exp1 완료 대기 ──────────────────────────────────────────────
log "대기: 중규모(exp1) 학습+after-eval 완료까지 (marker=$MARKER)"
while [ ! -f "$MARKER" ]; do
  sleep 60
  if [ ! -f "$MARKER" ] && ! pgrep -f "run_experiment.sh|train_whisper.py|evaluate.py" >/dev/null; then
    sleep 60   # eval 단계 전환 틈 한 번 더 확인
    [ -f "$MARKER" ] && break
    if ! pgrep -f "run_experiment.sh|train_whisper.py|evaluate.py" >/dev/null; then
      log "❌ exp1 프로세스가 사라졌고 결과 마커도 없음 → 중단 (experiment.log 확인 필요)"
      exit 1
    fi
  fi
done
log "✓ exp1 완료 감지"

# ── 1) 중간 결과 보관 ──────────────────────────────────────────────
DST="$RESULTS/exp1_midscale"
mkdir -p "$DST"
cp -f "$BEFORE" "$DST/" 2>/dev/null || true
cp -f "$MARKER" "$DST/" 2>/dev/null || true
cp -f "$TDIR/manifests/exp1/stats.json" "$DST/manifest_stats.json" 2>/dev/null || true
for f in cer_log.jsonl progress.json sample_predictions.jsonl status.json; do
  cp -f "$EXP1/phase_b_small/$f" "$DST/" 2>/dev/null || true
done
python "$HERE/scripts/compare_eval.py" "$BEFORE" "$MARKER" > "$DST/compare.txt" 2>&1 || true
log "✓ 중간 결과 보관: $DST"
cat "$DST/compare.txt" || true

# ── 2) 전체 manifest 생성 (전 화자/전 발화, 층화) ──────────────────
cd "$TDIR"
log "전체 manifest 생성 중 (NFS 라벨 ~113만건 읽기 → 시간 소요)..."
python prepare_dataset.py --data-dir "$AIHUB_DATA_DIR" --output manifests/full \
  --categories 환자 의사 간호사 --stratify || { log "❌ 전체 manifest 실패"; exit 1; }

# ── 3) exp1 val/test 화자 제외 → full_train 구성 (동일 yardstick, leakage 방지) ──
python - <<'PY'
import json, os, shutil
exp1='manifests/exp1'; full='manifests/full'; out='manifests/full_train'
held=set()
for sp in ('val','test'):
    for line in open(f'{exp1}/{sp}.jsonl', encoding='utf-8'):
        held.add(json.loads(line)['speaker'])
os.makedirs(out, exist_ok=True)
nin=nout=0
with open(f'{out}/train.jsonl','w',encoding='utf-8') as w:
    for sp in ('train','val','test'):
        p=f'{full}/{sp}.jsonl'
        if not os.path.exists(p): continue
        for line in open(p, encoding='utf-8'):
            nin+=1; r=json.loads(line)
            if r.get('speaker') in held: continue
            w.write(line); nout+=1
shutil.copy(f'{exp1}/val.jsonl', f'{out}/val.jsonl')   # 동일 val 로 early stop
print(f'full_train: {nout}/{nin} 샘플 (exp1 held {len(held)}화자 제외)')
PY
[ -s manifests/full_train/train.jsonl ] || { log "❌ full_train 비어있음"; exit 1; }

# ── 4) 전체 학습용 대시보드 재기동 (이전 exp1 대시보드 정리) ────────
pkill -f "tensorboard --logdir" 2>/dev/null || true
pkill -f "monitor.py" 2>/dev/null || true
sleep 2
mkdir -p outputs/full/phase_c_full/tensorboard
nohup tensorboard --logdir outputs/full/phase_c_full/tensorboard --host 0.0.0.0 --port 6006 \
  > "$SD/tb_full.log" 2>&1 &
nohup python monitor.py --output-dir outputs/full/phase_c_full > "$SD/monitor_full.log" 2>&1 &
log "✓ 대시보드 재기동 (TensorBoard :6006 / monitor :7861 → 전체 런 대상)"

# ── 5) 전체 학습 (phase c, early stop, 출력 분리) ──────────────────
log "전체 학습 시작 (phase c, EarlyStopping patience=3, eval/save 1000step)"
MANIFEST_DIR=manifests/full_train OUTPUT_ROOT=outputs/full PHASE=c \
  bash "$HERE/scripts/run_train_a6000.sh" \
    --early-stop-patience 3 --eval-steps 1000 --save-steps 1000 \
  || { log "❌ 전체 학습 실패"; exit 1; }

# ── 6) 전체 모델 after-eval (동일 exp1 test) + 비교 + 보관 ─────────
# load_best 가 켜져 있어 final/ = best 모델. 없으면 최신 checkpoint 로 폴백.
CKPT="outputs/full/phase_c_full/final"
[ -d "$CKPT" ] || CKPT="$(ls -d outputs/full/phase_c_full/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)"
[ -z "$CKPT" ] && { log "❌ 전체 체크포인트 없음"; exit 1; }
log "after-eval (전체 모델): $CKPT"
AFTER_FULL="outputs/full/eval_AFTER_full.json"
python evaluate.py --model-dir "$CKPT" --base-model openai/whisper-medium \
  --manifest-dir manifests/exp1 --split test --output "$AFTER_FULL" || { log "❌ after-eval 실패"; exit 1; }

DSTF="$RESULTS/full"
mkdir -p "$DSTF"
cp -f "$BEFORE" "$DSTF/" 2>/dev/null || true
cp -f "$AFTER_FULL" "$DSTF/" 2>/dev/null || true
cp -f manifests/full_train/../full/stats.json "$DSTF/manifest_full_stats.json" 2>/dev/null || true
for f in cer_log.jsonl progress.json sample_predictions.jsonl; do
  cp -f "outputs/full/phase_c_full/$f" "$DSTF/" 2>/dev/null || true
done
python "$HERE/scripts/compare_eval.py" "$BEFORE" "$AFTER_FULL" | tee "$DSTF/compare.txt"
log "✓ 전체 파이프라인 완료. 결과: $DSTF"
