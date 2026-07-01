#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────
# 도착(복사 완료)한 zip 부터 먼저 추출 — 복사가 끝나는 족족 이어서 추출.
#
#  · 복사 중(미완성) zip 은 건너뜀: zip 목차(central directory)는 파일 끝에 있어
#    복사가 끝나야 `unzip -l` 이 성공 → 이걸 "완료 판정" 게이트로 사용.
#  · 추가로 size 가 N초간 변하지 않아야 처리(복사 중 파일 보호, 이중 안전장치).
#  · .done 마커로 재실행/이어받기 안전. flock 으로 중복 실행 방지.
#
# 사용:
#   bash scripts/extract_ready_nas.sh            # 1회: 지금 도착한 것만 추출
#   bash scripts/extract_ready_nas.sh --watch    # 감시: 새 zip 오면 계속 추출 (Ctrl-C 종료)
#   JOBS=2 bash scripts/extract_ready_nas.sh --watch 30   # 병렬 2, 30초 주기
# ──────────────────────────────────────────────────────────────────────────
set -u

SRC="/mnt/nas_raw/rawdata/etc/voice"
DST="/mnt/nas_raw/rawdata/etc/voice/aihub"
LOG="/mnt/nas_raw/rawdata/etc/voice/extract.log"
JOBS="${JOBS:-4}"            # 복사와 NAS I/O 경합하므로 기본 4 (env 로 조정)
STABLE_SEC=20               # 이 시간동안 size 안 변해야 "복사 완료"로 간주

WATCH=0
INTERVAL=60
for a in "$@"; do
  case "$a" in
    --watch) WATCH=1 ;;
    [0-9]*)  INTERVAL="$a" ;;
  esac
done

mkdir -p "$DST/Training" "$DST/Validation"
touch "$LOG"

log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

# zip 이 완전히 복사됐는지: (1) size 안정 (2) 목차 읽힘
is_ready() {
  local zip="$1"
  [[ -f "$zip" ]] || return 1
  local s1 s2
  s1=$(stat -c %s "$zip" 2>/dev/null) || return 1
  sleep "$STABLE_SEC"
  s2=$(stat -c %s "$zip" 2>/dev/null) || return 1
  [[ "$s1" == "$s2" && "$s1" -gt 0 ]] || return 1     # 아직 커지는 중 → 복사 중
  unzip -l "$zip" >/dev/null 2>&1                     # 목차 정상 → 복사 완료
}

extract_one() {
  local zip="$1"
  local base dest sub
  base="$(basename "$zip" .zip)"
  if [[ "$base" == \[T* ]]; then sub="Training"; else sub="Validation"; fi
  dest="$DST/$sub/$base"

  [[ -f "$dest/.done" ]] && return 0                  # 이미 완료
  mkdir -p "$dest"
  log "START: $base -> $sub/  ($(du -h "$zip" 2>/dev/null | cut -f1))"
  if unzip -q -o -O UTF-8 "$zip" -d "$dest" >>"$LOG" 2>&1; then
    touch "$dest/.done"
    log "DONE : $base ($(du -sh "$dest" 2>/dev/null | cut -f1))"
  else
    log "FAIL : $base (unzip 오류 — $LOG 확인)"
    rm -f "$dest/.done"
  fi
}
export -f extract_one log
export DST LOG

# 한 번 훑어서 '준비된 + 아직 안한' zip 들을 병렬 추출
do_pass() {
  local pending=0 ready=0 done_cnt=0 copying=0
  local zips=()
  # 라벨 먼저(작고 manifest 가 필요로 함), 그다음 원천
  for z in "$SRC"/*라벨링데이터.zip "$SRC"/*원천*.zip; do
    [[ -e "$z" ]] || continue
    zips+=("$z")
  done

  for z in "${zips[@]}"; do
    local base dest sub
    base="$(basename "$z" .zip)"
    if [[ "$base" == \[T* ]]; then sub="Training"; else sub="Validation"; fi
    dest="$DST/$sub/$base"
    if [[ -f "$dest/.done" ]]; then done_cnt=$((done_cnt+1)); continue; fi

    if is_ready "$z"; then
      ready=$((ready+1))
      while (( $(jobs -rp | wc -l) >= JOBS )); do wait -n; done
      extract_one "$z" &
    else
      copying=$((copying+1))
      log "WAIT : $base (복사 중/미완성 — 건너뜀)"
    fi
  done
  wait
  log "── 현황: 완료 $done_cnt / 이번에 처리 $ready / 복사중 $copying ──"
  return "$copying"   # 0이면 남은 복사 없음
}

# 중복 실행 방지
exec 9>"$SRC/.extract.lock"
if ! flock -n 9; then
  echo "이미 다른 추출 프로세스가 실행 중입니다 ($SRC/.extract.lock)"; exit 1
fi

log "=== extract_ready 시작 (JOBS=$JOBS, watch=$WATCH, interval=${INTERVAL}s) ==="
if (( WATCH )); then
  while true; do
    do_pass; copying=$?
    if (( copying == 0 )); then
      log "남은 복사 없음 → 한 번 더 확인 후 종료 대기 (${INTERVAL}s)"
    fi
    sleep "$INTERVAL"
  done
else
  do_pass
  log "=== 1회 패스 완료 (새로 도착한 zip 있으면 다시 실행하세요) ==="
fi
