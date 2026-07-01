# Voice EMR 전이학습 — 작업 인수인계 (다른 PC / 새 Claude 세션용)

> 이 문서 하나로 이전 대화 없이 작업을 그대로 이어받을 수 있도록 정리했습니다.
> 작성 시점: 2026-06-17. 원 작업 서버: `nvidia-a100` (A100 80GB ×4).

---

## 0. 한 줄 요약 (현재 상태)

**Whisper-medium 한국어 의료음성 STT 를 AI Hub 데이터로 LoRA 전이학습 중.**
Phase B(카테고리당 100화자, 154k 샘플) 학습을 4×A100 DDP 로 돌리다가
**step 1000/7236(epoch 0.41)에서 서버가 갑자기 꺼져 중단됨.**
체크포인트는 디스크에 보존됨 → `--resume` 으로 재개 가능. **재개 전 크래시 원인 확인 필요.**

중간 성과(이미 베이스라인 대비 큰 개선):

| 모델 | test 300샘플(환자) CER |
|---|---:|
| 베이스 whisper-medium | 5.61% |
| **파인튜닝 step1000** | **2.28%** (−59% 상대) |

---

## 1. 프로젝트 개요

- **목표**: 일반 Whisper 의 한국어 의료대화 STT 성능을 베이스라인으로 측정하고, AI Hub
  "비대면 진료를 위한 의료진 및 환자 음성" 데이터로 **LoRA 전이학습**해 CER 을 낮춘다.
- **베이스 모델**: `openai/whisper-medium` (사용자 지정). large-v3 대비 medium 이 한국어
  실데이터에서 더 정확하고 안정적이었음(아래 베이스라인 리포트 참고).
- **원본 repo**: github.com/Jungmin81/voice-emr-baseline (단, 아래 수정사항은 **미푸시 상태**).
- **프로젝트 경로(원 서버)**: `/disk1/jungmin.cheon/jm_dev/voice_emr_baseline`

---

## 2. 지금까지 한 일 (시간순)

1. **베이스라인 성능 측정** — 전이학습 전 Whisper(tiny/small/medium/large-v3)의 STT
   소요시간·정확도(CER)를 기존 벤치 결과 + 신규 측정으로 산출.
   - 결과 문서: `voice_emr_baseline/STT_BASELINE_REPORT.md`,
     `bench_results_hub/`, `accuracy_results/`, `sample_bench_results/`
   - 핵심: AI Hub 실데이터에서 medium CER 3.35%(최저), 길이별/모델별 표 포함.
2. **AI Hub 데이터 압축 해제** — 272GB zip → 414GB, 전이학습용 구조로 정리(아래 §6).
3. **전이학습 코드 수정** — `training/` 모듈을 실제 데이터/환경에 맞게 대거 수정(아래 §4).
4. **환경 복구** — 깨진 패키지(torchvision/bitsandbytes) 비활성화, 학습 라이브러리 설치(아래 §5).
5. **스모크 테스트 통과** — medium LoRA end-to-end 검증(작은 manifest, 10~20 step).
6. **Phase B 학습 시작** — 4×A100 DDP, step 1000 도달 후 **서버 크래시로 중단**.
7. **평가** — checkpoint-1000 vs 베이스 medium 을 동일 test셋에서 비교(위 표).
8. **모니터링** — 실시간 대시보드(Gradio) + TensorBoard 구성.

---

## 3. 가장 중요한 미해결 이슈 — 서버 크래시

- **증상**: 4×A100 100% 풀로드 중 15:59 경 서버가 **비정상 재부팅**(정상 종료 로그 없음).
- **추정 원인**: 다중 GPU 풀로드 시 **전원(PSU/차단기) 과부하** 또는 **열 비상정지** 가 가장 유력.
- **확인 방법(원 서버, sudo 필요)**:
  ```bash
  sudo ipmitool sel elist | tail -40            # 하드웨어 이벤트(전원/열) — 가장 확실
  sudo grep -aiE "thermal|temperature|machine check|hardware error|MCE|panic|NVRM|Xid" /var/log/kern.log | tail -60
  sudo journalctl -k -b -1 -n 100 --no-pager    # 직전 부팅 커널 로그
  ```
  - 로그가 에러 없이 뚝 끊김 + IPMI `Power off`/`PSU failure` → 전원 문제 확정.
- **재개 시 권고**: 원인 확인 전까지 4-GPU 풀로드 재현 위험. 안전책으로
  **GPU 수 축소**(`CUDA_VISIBLE_DEVICES=0,1`, `--nproc_per_node=2`) 또는 배치 축소 고려.
- ※ **새 PC가 다른 하드웨어면** 이 크래시는 원 서버 전원 문제일 수 있으니 그냥 재현 안 될 수도 있음.

---

## 4. 코드 변경 요약 (`training/` — 새 PC에 그대로 포함됨)

원본 `training/` 코드에는 실제 데이터와 안 맞는 버그가 많았음. 수정 내용:

- **`utils.py`** — 화자 prefix→카테고리 매핑 수정. **실제 데이터: 환자=PA~PF, 간호사=HA, 의사=HB**
  (원본은 PA/HA/DR 만 알아서 의사 전체 + 환자 5/6 누락하던 버그). `SPEAKER_PREFIX_TO_CATEGORY` 추가.
- **`prepare_dataset.py`** — 폴더 탐색을 prefix 기반(고정 깊이 글롭)으로 재작성, 화자 단위
  train/val/test 분할에 **최소 1화자 보장** 로직 추가(소규모에서 val=0 방지).
- **`config.py`** — 베이스 모델 기본값 `whisper-medium`, **gradient_checkpointing=False**
  (medium+80GB 에선 불필요하고 DDP 와 충돌), checkpoint 보관 10개, logging_steps=25.
- **`train_whisper.py`** (대폭 수정):
  - **on-the-fly 데이터 로딩**: `datasets.map`(1.1M mel → 디스크 ~1TB, 데드락) 폐기 →
    torch `Dataset.__getitem__` 에서 librosa 로 즉시 로드(메모리 안전, 확장 가능).
  - **dtype 통일**: 모델 bf16 로드 + collator 가 input_features 를 bf16 캐스팅(학습/평가 generate
    dtype 충돌 제거).
  - **평가**: `predict_with_generate=False`(트레이너 eval=loss만), **CER 은 콜백에서 직접 generate
    하여 계산**(`cer_log.jsonl` + TensorBoard `eval/cer_pct`), best 기준 `eval_loss`.
  - **DDP 호환**: gradient checkpointing off + `ddp_find_unused_parameters=False`
    ("mark a variable ready only once" 오류 해결).
  - **대시보드 콜백**: `progress.json`(step/epoch/ETA/속도), `sample_predictions.jsonl`,
    `cer_log.jsonl` 기록(전부 rank0 한정). TensorBoard 에 CER scalar 도 기록.
  - **버전 호환**: `eval_strategy`/`evaluation_strategy`, `processing_class`/`tokenizer` 자동 분기.
  - **CLI 추가**: `--max-steps --eval-steps --save-steps --batch-size --num-workers`.
- **`monitor.py`** — `progress.json`/`cer_log.jsonl` 읽어 ETA·남은 step·CER 곡선 표시,
  이벤트 `queue=False`(원격 WS 막힘 회피). ※ tailscale 환경에선 그래도 안 떠서 TensorBoard 권장(§7).
- **`evaluate.py`** — test셋 평가(전체/카테고리/성별/연령별 CER). `--base-model whisper-medium` 필수.

---

## 5. 작업 환경 (자세한 건 `env/ENVIRONMENT_NOTES.md`)

- 원 서버 Python: conda env `jungmin.cheon_39` (Python 3.9) — `/usr/anaconda3/envs/jungmin.cheon_39/bin/python3`
- **핵심 동작 버전**: torch(런타임 2.8.0+cu128, CUDA OK) / transformers 4.57.6 / peft 0.17.1 /
  accelerate 1.10.1 / datasets 4.5.0 / gradio 4.44.1 / tensorboard 2.20.0 / librosa 0.11 /
  soundfile / jiwer / faster-whisper 1.2.1. 전체: `env/requirements_frozen.txt`.
- **반드시 알아야 할 gotcha (새 PC에서 환경 만들 때)**:
  - **torchvision / bitsandbytes 설치하지 말 것** — 원 서버에서 이 둘이 깨져 transformers/peft
    import 를 막았음(우리는 비활성화함). Whisper LoRA bf16 학습엔 둘 다 불필요.
  - datasets 4.x 의 `Audio` 피처는 torchcodec 요구 → 우리는 안 씀(librosa 직접 로드).
  - 새 PC에선 **이 깨진 env 복제 대신, §8의 깨끗한 설치**를 권장.

---

## 6. 데이터 (자세한 건 `DATA_NOTES.md`) — **이 압축본에 미포함 (414GB)**

- **원본 zip**: `/disk1/jungmin.cheon/voice_datas/datas/` (272GB, 18개)
  — `[T원천]환자_1~6`, `[T원천]의료진_간호사_1~4`, `[T원천]의료진_의사_1~3`, `[T]라벨링데이터`, `[V]*`
- **추출 구조**: `/disk1/jungmin.cheon/voice_datas/aihub/{Training,Validation}/` (414GB)
  - 추출 스크립트: `/disk1/jungmin.cheon/voice_datas/extract_aihub.sh` (이 압축본 `data_scripts/`에 사본)
- **WAV↔라벨 1:1, 총 113만 발화** (환자 471k/간호사 329k/의사 330k), 48kHz→16k 리샘플.
- ⚠️ **manifests(`training/manifests/100p`)의 audio 경로가 `/disk1/.../aihub/Training/...` 절대경로**임.
  새 PC에서 데이터 경로가 다르면 **manifest 재생성**(§8) 또는 경로 sed 치환 필요.

---

## 7. 모니터링

- **TensorBoard (권장, 잘 됨)**: `tensorboard --logdir <output>/tensorboard --host 0.0.0.0 --port 6006`
  → SCALARS 탭에 train loss, eval/loss, **eval/cer_pct**(우리가 추가).
- **Gradio 대시보드(`monitor.py`)**: 진행률/ETA/CER/샘플예측/GPU. 서버 로컬은 정상이나
  **tailscale 경유 시 "로딩중"에서 멈춤**(WS/이벤트 전달 차단). 우회: SSH 포트포워딩
  `ssh -L 7861:localhost:7861 ...` 후 `http://localhost:7861`.

---

## 8. 새 PC에서 재개하는 법 (순서대로)

### (1) 환경 구축 (깨끗하게 권장)
```bash
python3.10 -m venv venv_train && source venv_train/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu121   # GPU CUDA 버전
pip install "transformers>=4.45,<5" "peft>=0.13" "accelerate>=0.34" "datasets>=2.20" \
            tensorboard "gradio>=4.44" librosa soundfile jiwer pandas psutil faster-whisper
# torchvision / bitsandbytes 는 설치하지 말 것!
```
(또는 `env/requirements_frozen.txt` 참고. 단 그 env는 일부 꼬여 있으니 위 깨끗한 설치 권장.)

### (2) 데이터 준비
- 원 서버의 `voice_datas/datas/*.zip`(또는 `aihub/`)를 새 PC로 복사하거나 동일 경로에 마운트.
- `data_scripts/extract_aihub.sh` 의 `SRC/DST` 경로 수정 후 실행해 `aihub/Training` 생성.
- **manifest 재생성** (경로가 새 PC 기준으로 박힘):
  ```bash
  export AIHUB_DATA_DIR="<새경로>/aihub/Training"
  cd training
  python prepare_dataset.py --data-dir "$AIHUB_DATA_DIR" --output manifests/100p \
      --categories 환자 의사 간호사 --max-speakers 100
  ```
  ※ 동봉된 `manifests/100p` 를 그대로 쓰려면 데이터가 **정확히 같은 절대경로**여야 함.

### (3) 학습 재개
- 동봉된 `training/outputs/phase_b_small/checkpoint-1000` 을 그대로 두면 `--resume` 이 자동 인식.
- ⚠️ checkpoint 의 데이터 경로 의존성 때문에, 데이터 경로가 다르면 manifest 재생성 후
  **새로 시작**하는 편이 깔끔할 수 있음(step1000까지가 ~15분이라 손해 작음).
```bash
export AIHUB_DATA_DIR="<새경로>/aihub/Training"
export CUDA_VISIBLE_DEVICES=0,1,2,3        # 전원 이슈 의심되면 0,1 로 축소
cd training
torchrun --nproc_per_node=4 train_whisper.py \
  --phase b --manifest-dir manifests/100p \
  --base-model openai/whisper-medium --num-workers 8 --resume
# 모니터: tensorboard --logdir outputs/phase_b_small/tensorboard --host 0.0.0.0 --port 6006
```
**중요**: `pkill -f torchrun` 류로 프로세스 죽이지 말 것 — 명령줄에 "torchrun"이 들어가면
자기 셸까지 죽임. PID 로 죽이거나 `pkill -f train_whisper.py` 사용.

### (4) 평가
```bash
CONDA=python3   # (새 env의 python)
$CONDA evaluate.py --model-dir outputs/phase_b_small/checkpoint-XXXX \
  --base-model openai/whisper-medium --manifest-dir manifests/100p --split test --max-samples 300
# 베이스 비교: --model-dir openai/whisper-medium
```
⚠️ 알려진 표본 편향: `--max-samples 300` 은 test 앞 300개(전부 환자)만 봄. 의사/간호사 포함
정식 수치는 전체 test 또는 카테고리 균형 샘플로 평가할 것.

---

## 9. 다음 할 일 (TODO)

- [ ] **크래시 원인 확정**(IPMI/syslog) → 안전한 GPU 수로 재개
- [ ] Phase B 학습 완주(step 7236/3 epoch) → test 전체(9,883) 정식 평가(카테고리별 CER)
- [ ] 결과 좋으면 Phase C(전체 113만) 확장
- [ ] (선택) 학습된 LoRA 를 `baseline.py`/`benchmark.py` 에 통합해 베이스라인 대비 재측정

---

## 10. 파일 인덱스 (이 압축본)

```
HANDOFF.md                  ← 이 문서
env/requirements_frozen.txt ← 원 서버 pip freeze
env/ENVIRONMENT_NOTES.md    ← 환경 gotcha 상세
DATA_NOTES.md               ← AI Hub 데이터 위치/구조/재추출
data_scripts/extract_aihub.sh
claude_settings/settings.json ← Claude 권한 설정(새 PC .claude/ 에 복사하면 프롬프트 감소)
voice_emr_baseline/         ← 프로젝트 전체 (venv 제외)
  ├── training/             ← 수정된 학습 코드 + manifests + outputs(checkpoint-1000 포함)
  ├── STT_BASELINE_REPORT.md, bench_results*/, results/, accuracy_results/, sample_bench_results/
  └── baseline.py, app.py, benchmark.py, ...
```
