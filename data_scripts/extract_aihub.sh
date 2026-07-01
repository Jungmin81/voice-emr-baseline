#!/usr/bin/env bash
# AI Hub 음성 데이터 압축 해제 — 전이학습용 구조로 정리
#
# 레이아웃:
#   aihub/Training/[T원천]...        (각 zip -> 동일 이름 폴더)
#   aihub/Training/[T]라벨링데이터/medsub/...
#   aihub/Validation/[V원천]...
#   aihub/Validation/[V]라벨링데이터/medsub/...
#
# - 재개 가능: 각 폴더에 .done 마커가 있으면 건너뜀
# - 병렬: 동시 JOBS개 unzip
set -u

SRC="/disk1/jungmin.cheon/voice_datas/datas"
DST="/disk1/jungmin.cheon/voice_datas/aihub"
LOG="/disk1/jungmin.cheon/voice_datas/extract.log"
JOBS=8

mkdir -p "$DST/Training" "$DST/Validation"
: > "$LOG"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

extract_one() {
  local zip="$1"
  local base dest sub
  base="$(basename "$zip" .zip)"
  # T -> Training, V -> Validation
  if [[ "$base" == \[T* ]]; then sub="Training"; else sub="Validation"; fi
  dest="$DST/$sub/$base"

  if [[ -f "$dest/.done" ]]; then
    log "SKIP (이미 완료): $base"
    return 0
  fi
  mkdir -p "$dest"
  log "START: $base -> $sub/"
  if unzip -q -o -O UTF-8 "$zip" -d "$dest" >>"$LOG" 2>&1; then
    touch "$dest/.done"
    log "DONE : $base ($(du -sh "$dest" 2>/dev/null | cut -f1))"
  else
    log "FAIL : $base (unzip 오류 — 로그 확인)"
  fi
}
export -f extract_one log
export SRC DST LOG

log "=== 압축 해제 시작: $(ls "$SRC"/*.zip | wc -l)개 zip, 병렬 $JOBS ==="
log "총 zip 용량: $(du -ch "$SRC"/*.zip 2>/dev/null | tail -1 | cut -f1)"

# 라벨 zip 먼저, 그다음 원천 — bash job-pool 로 안정적 병렬 처리
zips=()
for z in "$SRC"/*라벨링데이터.zip "$SRC"/*원천*.zip; do
  [[ -e "$z" ]] && zips+=("$z")
done

for z in "${zips[@]}"; do
  # 동시 실행 수가 JOBS 미만이 될 때까지 대기
  while (( $(jobs -rp | wc -l) >= JOBS )); do wait -n; done
  extract_one "$z" &
done
wait

log "=== 전체 완료 ==="
log "결과 구조:"
ls -d "$DST"/Training/*/ "$DST"/Validation/*/ 2>/dev/null | tee -a "$LOG"
