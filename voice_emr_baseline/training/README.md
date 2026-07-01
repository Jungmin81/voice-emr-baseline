# Voice EMR 전이학습 모듈

AI Hub 의료 음성 데이터로 **Whisper large-v3 LoRA 전이학습** + 실시간 모니터링.

## 📁 파일 구성

```
training/
├── config.py                   # 설정 (경로, 모델, Phase별 하이퍼파라미터)
├── prepare_dataset.py          # AI Hub 데이터 → manifest 변환
├── train_whisper.py            # Whisper LoRA 학습 메인
├── monitor.py                  # Gradio 실시간 모니터링 대시보드
├── evaluate.py                 # 학습 후 평가 (CER, 카테고리별)
├── utils.py                    # CER 계산, 텍스트 정규화 등
├── requirements_training.txt   # 추가 의존성
└── README.md                   # 이 문서
```

## 🚀 빠른 시작

### Step 0. 의존성 설치

```bash
cd ~/jm_repo/voice-emr-baseline
source venv/bin/activate
pip install -r training/requirements_training.txt
```

### Step 1. 데이터 경로 설정

`config.py`의 `DEFAULT_DATA_DIR`을 수정하거나, 환경변수로:

```bash
export AIHUB_DATA_DIR="/path/to/Training"
# 예: /data/aihub/Training
```

### Step 2. Manifest 생성 (한 번만)

#### Phase A — 빠른 검증 (1명, ~5시간 음성, 2~3시간 학습)
```bash
cd training/
python prepare_dataset.py \
    --data-dir "$AIHUB_DATA_DIR" \
    --output manifests/quick \
    --categories 환자 \
    --max-speakers 1
```

#### Phase B — 100명 검증 (12~24시간 학습)
```bash
python prepare_dataset.py \
    --data-dir "$AIHUB_DATA_DIR" \
    --output manifests/100p \
    --categories 환자 의사 간호사 \
    --max-speakers 100
```

#### Phase C — 전체 데이터 (2~3일 학습)
```bash
python prepare_dataset.py \
    --data-dir "$AIHUB_DATA_DIR" \
    --output manifests/full \
    --categories 환자 의사 간호사
```

→ 출력: `manifests/<name>/train.jsonl`, `val.jsonl`, `test.jsonl`, `stats.json`

### Step 3. 학습 실행

#### 단일 GPU
```bash
python train_whisper.py --phase a --manifest-dir manifests/quick
```

#### A100 × 4 멀티 GPU
```bash
torchrun --nproc_per_node=4 train_whisper.py \
    --phase b --manifest-dir manifests/100p
```

→ 출력: `outputs/phase_<x>_<name>/`

### Step 4. 실시간 모니터링 (별도 터미널)

```bash
python monitor.py --output-dir outputs/phase_a_quick_verify
```

브라우저에서 `http://localhost:7861` 접속.

표시 내용:
- 현재 Epoch / Step / 진행률 / ETA
- Train Loss 그래프 (실시간 업데이트)
- Eval Loss / CER 그래프
- 샘플 예측 (GT vs PRED 비교)
- GPU 사용량 (4장 동시)

15초마다 자동 새로고침.

### Step 5. 평가

학습 끝나면 test split으로 평가:

```bash
python evaluate.py \
    --model-dir outputs/phase_a_quick_verify/final \
    --manifest-dir manifests/quick \
    --split test
```

출력:
- 전체 CER
- 카테고리별 (환자/의사/간호사) CER
- 성별·연령별 CER
- 샘플 비교 20개
- `eval_test.json` 저장

## 📊 Phase 비교

| Phase | 화자 수 | 데이터 양 | 학습 시간 (A100×4) | 용도 |
|---|---|---|---|---|
| **A** | 1명 (환자) | ~5h 음성 | 2~3시간 | 코드 동작 검증 |
| **B** | 100명 (전 카테고리) | ~50h 음성 | 12~24시간 | 초기 성능 측정 |
| **C** | 전체 | 1,450h 음성 | 2~3일 | 최종 모델 |

→ A 먼저 돌려서 파이프라인 확인 후 B, C로 확장 추천.

## 🔧 추가 옵션

### TensorBoard도 같이 보기
```bash
tensorboard --logdir outputs/phase_a_quick_verify/tensorboard --port 6006
```

### 중단된 학습 재개
```bash
python train_whisper.py --phase a --manifest-dir manifests/quick --resume
```

### 다른 베이스 모델
```bash
# medium으로 더 빠르게
python train_whisper.py --phase a --manifest-dir manifests/quick \
    --base-model openai/whisper-medium
```

## 📈 학습된 모델 활용

학습 끝나면 기존 베이스라인 코드에 바로 적용 가능:

```python
# baseline.py 안에서 (예시)
from peft import PeftModel
from transformers import WhisperForConditionalGeneration

base = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3")
model = PeftModel.from_pretrained(base, "training/outputs/phase_b_small/final")
```

또는 벤치마크에 바로 적용:
```bash
cd ..
python benchmark.py \
    --samples samples/aihub/*.wav \
    --whisper-checkpoint training/outputs/phase_b_small/final
```

## 🐛 트러블슈팅

### CUDA OOM
- `config.py`의 `batch_size` 줄이기 (16 → 8)
- `gradient_checkpointing=True` 유지
- 또는 더 작은 베이스 모델 (`whisper-medium`)

### 학습 너무 느림
- `dataloader_num_workers` 늘리기 (4 → 8)
- 데이터 풀려있는지 확인 (zip 압축 해제 안 됐으면 매번 디스크 I/O 폭증)
- `bf16=True` 확인 (A100은 bf16 가속)

### CER 안 떨어짐
- learning rate 줄이기 (1e-4 → 5e-5)
- 더 많은 데이터 필요 (Phase A → B)
- warmup steps 늘리기

### Gradio 대시보드 데이터 안 보임
- `--output-dir` 경로 정확한지 확인
- `outputs/<phase>/status.json` 파일 존재 확인
- 학습 첫 logging_steps(=50) 지나야 데이터 보이기 시작
