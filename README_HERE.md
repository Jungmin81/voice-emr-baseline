# voice_emr — 이 PC에서 재구성한 작업본

`voice_emr_handoff_20260617.tar.gz`(원 서버 4×A100 인계본)를 **이 PC(2×RTX A6000)에서
바로 이어 쓸 수 있게** 재구성한 폴더입니다. 원본 압축본은 `../voice_emr_handoff/` 에 그대로 보존돼 있습니다.

## 먼저 읽을 것
1. **`docs/SETUP_A6000.md`** ← 이 머신 기준 설치·데이터·학습·평가 (가장 먼저)
2. `docs/HANDOFF.md` — 원 인계 문서(맥락·코드변경·크래시 이슈 전체)
3. `docs/DATA_NOTES.md` — AI Hub 데이터 위치/구조/재추출
4. `docs/ENVIRONMENT_NOTES.md` — 환경 함정(torchvision/bitsandbytes 금지 등)

## 폴더 구조
```
voice_emr/
├── README_HERE.md            ← 이 문서
├── docs/
│   ├── SETUP_A6000.md        ← ★ 이 PC 전용 follow-up 가이드 (신규)
│   ├── HANDOFF.md            ← 원 인계 문서
│   ├── DATA_NOTES.md
│   ├── ENVIRONMENT_NOTES.md
│   └── requirements_frozen.txt
├── scripts/                  ← 이 PC 전용 실행 스크립트 (신규)
│   ├── setup_venv.sh         ← 학습용 venv 설치(torch cu124, torchvision/bnb 제외)
│   ├── run_train_a6000.sh    ← Phase B 학습 (2GPU, batch8×accum4=effective64)
│   └── run_eval.sh           ← 평가 래퍼
├── data_scripts/
│   └── extract_aihub.sh      ← AI Hub zip 추출 (SRC/DST 경로 수정 후 사용)
├── .claude/settings.json     ← Claude 권한 설정 (프롬프트 감소)
├── venv_train/               ← scripts/setup_venv.sh 가 생성 (git 제외 대상)
└── voice_emr_baseline/       ← 프로젝트 본체
    ├── training/             ← 수정된 학습 코드 + manifests + outputs(checkpoint-500/1000)
    ├── STT_BASELINE_REPORT.md, accuracy_results/, results/, bench_results*/ ...
    └── baseline.py, app.py, benchmark.py, ...
```

## 3단계 요약 (상세는 SETUP_A6000.md)
```bash
cd /home/jungmin.cheon/jm_repo/voice_v2/voice_emr
# 1) 환경
bash scripts/setup_venv.sh && source venv_train/bin/activate
# 2) 데이터 확보 후 manifest 재생성 (이 PC엔 데이터 없음 → 절대경로 manifest 무효)
export AIHUB_DATA_DIR="<데이터>/aihub/Training"
cd voice_emr_baseline/training && python prepare_dataset.py \
    --data-dir "$AIHUB_DATA_DIR" --output manifests/100p \
    --categories 환자 의사 간호사 --max-speakers 100 && cd ../..
# 3) 학습 / 평가
bash scripts/run_train_a6000.sh
bash scripts/run_eval.sh outputs/phase_b_small/checkpoint-1000 --max-samples 300
```

## 원본 인계본과 달라진 점
- `scripts/`, `docs/SETUP_A6000.md`, `README_HERE.md` = **이 PC용으로 신규 작성**
- `training/train_whisper.py` 에 **`--grad-accum` CLI 추가** (A6000 effective-batch 보전용; 원본은 config 고정 1)
- 그 외 코드/체크포인트/리포트는 인계본과 동일

## 주의
- **데이터·체크포인트·결과물은 git 에 올리지 말 것**(환자 데이터, 용량/라이선스). `voice_emr_baseline/.gitignore` 에 이미 `*.safetensors/*.pt/results/*.wav` 등 제외 설정 있음.
- 기존 GitHub repo(`../../voice-emr-baseline`)는 건드리지 않음. 나중에 push 하려면 `training/` 코드와 문서만 선별 반영.
