# 데이터 노트 — AI Hub "비대면 진료를 위한 의료진 및 환자 음성"

⚠️ **데이터 자체는 용량(zip 272GB / 추출 414GB) 때문에 이 압축본에 포함되지 않음.**
새 PC로 별도 복사하거나 동일 스토리지를 마운트해야 함.

## 원 서버 위치
- **zip 원본**: `/disk1/jungmin.cheon/voice_datas/datas/` (18개, 272GB)
  ```
  [T원천]환자_1.zip ~ _6.zip          (환자: PA,PB,PC,PD,PE,PF)
  [T원천]의료진_간호사_1.zip ~ _4.zip  (간호사: HA)
  [T원천]의료진_의사_1.zip ~ _3.zip    (의사: HB)
  [T]라벨링데이터.zip                  (medsub/{환자,간호사,의사}/<화자>/<utt>.json)
  [V원천]환자_1 / 간호사_1 / 의사_1 + [V]라벨링데이터  (Validation 셋)
  ```
- **추출 결과**: `/disk1/jungmin.cheon/voice_datas/aihub/{Training,Validation}/`
  - 각 zip → 동일 이름 폴더로 추출(예: `Training/[T원천]환자_1/1/PA_0016/*.wav`)
  - 라벨: `Training/[T]라벨링데이터/medsub/<카테고리>/<화자>/<utt>.json` (`전사정보.LabelText`)

## 구조/통계
- WAV ↔ 라벨 **1:1**, Training 총 **1,130,811 발화** (환자 471k / 간호사 329k / 의사 330k)
- 화자: 환자 1,341 / 간호사 718 / 의사 366명
- 오디오 48kHz/16bit mono → 학습 시 librosa 로 16kHz 리샘플
- 발화 평균 ~20자(짧은 단문). Whisper 30초/448토큰 한계 내.

## 화자 prefix → 카테고리 (코드가 이 매핑 사용)
| prefix | 카테고리 |
|---|---|
| PA, PB, PC, PD, PE, PF | 환자 |
| HA | 간호사 |
| HB | 의사 |
(원본 코드는 PA/HA/DR 만 알아서 의사·대부분 환자를 누락하던 버그가 있었고, 수정함.)

## 새 PC에서 추출
`data_scripts/extract_aihub.sh` 의 `SRC`(zip 위치), `DST`(추출 대상) 경로를 새 PC 기준으로
수정 후 실행. 병렬 8개, 재개 가능(`.done` 마커). 128코어+NVMe에서 전체 ~10분 걸렸음.

## manifest 와 경로 의존성 (중요)
- `voice_emr_baseline/training/manifests/100p/{train,val,test}.jsonl` 의 각 줄 `audio` 필드가
  **`/disk1/jungmin.cheon/voice_datas/aihub/Training/...` 절대경로**로 박혀 있음.
- 새 PC에서 데이터 경로가 다르면:
  - (a) 같은 절대경로로 마운트하거나,
  - (b) `prepare_dataset.py` 로 manifest 재생성(권장), 또는
  - (c) jsonl 의 경로 prefix 를 sed 로 일괄 치환.
- Phase B 재생성 명령:
  ```bash
  export AIHUB_DATA_DIR="<새경로>/aihub/Training"
  python prepare_dataset.py --data-dir "$AIHUB_DATA_DIR" --output manifests/100p \
      --categories 환자 의사 간호사 --max-speakers 100
  ```
