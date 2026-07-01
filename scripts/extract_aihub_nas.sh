#!/usr/bin/env bash
# AI Hub 음성 zip 추출 — 이 PC / NAS 경로 전용 (data_scripts/extract_aihub.sh 의 NAS판)
#   SRC: NAS 에 옮겨둔 zip 들
#   DST: NAS 에 추출 (414GB → 로컬 228GB 엔 안 들어감, NAS 3.2TB 에 추출)
# - 재개 가능(.done 마커), 병렬 JOBS개
# 사용:  bash scripts/extract_aihub_nas.sh
set -u

SRC="/mnt/nas_raw/rawdata/etc/voice"
DST="/mnt/nas_raw/rawdata/etc/voice/aihub"
LOG="/mnt/nas_raw/rawdata/etc/voice/extract.log"
JOBS=8

mkdir -p "$DST/Training" "$DST/Validation"
: > "$LOG"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

extract_one() {
  local zip="$1"
  local base dest sub
  base="$(basename "$zip" .zip)"
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

log "=== 압축 해제 시작: $(ls "$SRC"/*.zip 2>/dev/null | wc -l)개 zip, 병렬 $JOBS ==="
log "총 zip 용량: $(du -ch "$SRC"/*.zip 2>/dev/null | tail -1 | cut -f1)"

zips=()
for z in "$SRC"/*라벨링데이터.zip "$SRC"/*원천*.zip; do
  [[ -e "$z" ]] && zips+=("$z")
done

for z in "${zips[@]}"; do
  while (( $(jobs -rp | wc -l) >= JOBS )); do wait -n; done
  extract_one "$z" &
done
wait

log "=== 전체 완료 ==="
ls -d "$DST"/Training/*/ "$DST"/Validation/*/ 2>/dev/null | tee -a "$LOG"
