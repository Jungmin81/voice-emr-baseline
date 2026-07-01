# 이 PC에서 재개하기 — A6000 환경 follow-up

> `HANDOFF.md`(원 서버: 4×A100 80GB 기준)를 **이 머신에 맞게 다시 쓴 문서**입니다.
> 원본 인계 내용은 `docs/HANDOFF.md`, 데이터는 `docs/DATA_NOTES.md`, 환경 함정은
> `docs/ENVIRONMENT_NOTES.md` 를 그대로 참고하세요. 여기서는 **달라진 점과 실제 명령**만 정리합니다.

작성: 2026-06-25 · 대상 머신: `RTX A6000 48GB ×2`, driver 595(CUDA 13.2), Python 3.10, 32코어/376GB RAM

---

## 0. 원 서버와 무엇이 다른가 (핵심)

| 항목 | 원 서버(HANDOFF 기준) | 이 PC | 영향 |
|---|---|---|---|
| GPU | A100 80GB ×4 | **A6000 48GB ×2** | per-device batch ↓, grad-accum 으로 보전 |
| 데이터 | `/disk1/.../aihub/Training` (414GB) | **없음** | manifest 절대경로 무효 → 데이터 확보 후 **재생성 필수** |
| Python | conda `jungmin.cheon_39` (3.9) | conda base 3.10 (**torch 1.12 CPU**) | 학습 불가 → **별도 venv** (`scripts/setup_venv.sh`) |
| 크래시 이슈 | A100 서버 전원/열 추정 | 다른 HW | **재현 안 될 가능성 높음**(HANDOFF §3) — 단 2-GPU 풀로드 전력은 여전히 주시 |

> 한 줄: **코드/체크포인트/문서는 그대로 쓰되, ① venv 새로 설치 ② 데이터 확보 후 manifest 재생성
> ③ batch 를 A6000 에 맞게 조정** — 이 세 가지만 하면 됩니다.

---

## 1. 환경 구축 (1회)

```bash
cd /home/jungmin.cheon/jm_repo/voice_v2/voice_emr
bash scripts/setup_venv.sh        # venv_train 생성 + torch(cu124)/transformers/peft/... 설치 + 검증
source venv_train/bin/activate
```
- `torchvision` / `bitsandbytes` 는 **설치하지 않음** (원 서버에서 import 체인을 깨뜨린 이력 — ENVIRONMENT_NOTES §함정 1·2).
- torch 휠은 driver 595 와 호환되는 **cu124**.
- 검증 출력에 `cuda: True / ngpu: 2 / A6000 ×2` 가 보이면 정상.

## 2. 데이터 준비 (필수 — 이 PC엔 데이터 없음)

AI Hub "비대면 진료를 위한 의료진 및 환자 음성"을 확보해야 합니다(상세: `docs/DATA_NOTES.md`).

1. zip(272GB) 또는 추출본(414GB)을 이 PC로 복사/마운트.
   - zip 만 있으면 `data_scripts/extract_aihub.sh` 의 `SRC`/`DST` 를 이 PC 경로로 고쳐 실행.
   - ⚠️ 디스크 여유 확인: 현재 `/` 에 **~228GB** 만 남음 → 414GB 추출은 **추가 스토리지 필요**.
2. **manifest 재생성** (동봉된 `manifests/100p` 의 audio 경로가 `/disk1/...` 절대경로라 이 PC에선 무효):
   ```bash
   source venv_train/bin/activate
   export AIHUB_DATA_DIR="<데이터>/aihub/Training"
   cd voice_emr_baseline/training
   python prepare_dataset.py --data-dir "$AIHUB_DATA_DIR" \
       --output manifests/100p --categories 환자 의사 간호사 --max-speakers 100
   ```

## 3. 학습 실행 (Phase B)

```bash
cd /home/jungmin.cheon/jm_repo/voice_v2/voice_emr
export AIHUB_DATA_DIR="<데이터>/aihub/Training"
bash scripts/run_train_a6000.sh              # 처음부터
# 또는 (데이터 경로가 원본과 동일할 때만) 재개:
bash scripts/run_train_a6000.sh --resume
```
`run_train_a6000.sh` 가 적용하는 A6000용 설정:
- `CUDA_VISIBLE_DEVICES=0,1` → `--nproc_per_node=2`
- **`--batch-size 8 --grad-accum 4`** → effective batch `8×2×4 = 64` (= 원 서버와 동일 → step수 7236·LR 스케줄 그대로)
- `gradient_checkpointing` OFF 유지. **OOM 이면** `--batch-size 4 --grad-accum 8` 로 더 줄일 것.

> 참고: `--grad-accum` 은 이 PC용으로 `train_whisper.py` 에 **새로 추가한 CLI**(원본엔 config 고정값 1).

**프로세스 종료 시 주의(HANDOFF §8):** `pkill -f torchrun` 금지(자기 셸까지 죽음).
`pkill -f train_whisper.py` 또는 PID 로 종료.

## 4. 모니터링

```bash
source venv_train/bin/activate
tensorboard --logdir voice_emr_baseline/training/outputs/phase_b_small/tensorboard \
            --host 0.0.0.0 --port 6006
# SCALARS 탭: train loss, eval/loss, eval/cer_pct(직접 추가한 지표)
```
Gradio 대시보드(`training/monitor.py`)는 원격 경유 시 안 뜨는 이슈가 있어 TensorBoard 권장(HANDOFF §7).

## 5. 평가

```bash
# 파인튜닝 체크포인트
bash scripts/run_eval.sh outputs/phase_b_small/checkpoint-1000 --max-samples 300
# 베이스 비교
bash scripts/run_eval.sh openai/whisper-medium --max-samples 300
```
⚠️ `--max-samples 300` 은 test 앞 300개(전부 환자)만 보는 **표본 편향**. 정식 수치는 옵션을 빼고 전체 test 로.

---

## 6. 동봉된 중간 성과 (참고)

| 모델 | test 300샘플(환자) CER |
|---|---:|
| 베이스 whisper-medium | 5.61% |
| 파인튜닝 step1000 | **2.28%** (−59% 상대) |

- 마지막 상태: Phase B step ~1325/7236(epoch 0.55)에서 원 서버 크래시로 중단. 저장된 마지막 체크포인트는 `checkpoint-1000`.
- 체크포인트는 `voice_emr_baseline/training/outputs/phase_b_small/checkpoint-{500,1000}` 에 그대로 포함(LoRA 어댑터, 각 ~217MB).

## 7. TODO (HANDOFF §9 기준, 이 PC 관점)

- [ ] (이 PC) venv 설치 → `scripts/setup_venv.sh`
- [ ] (이 PC) AI Hub 데이터 확보 + 디스크 확보(414GB) → manifest 재생성
- [ ] Phase B 완주(step 7236/3epoch) → 전체 test 카테고리별 CER 정식 평가
- [ ] 결과 좋으면 Phase C(전체 113만) 확장
- [ ] (선택) 학습된 LoRA 를 `baseline.py`/`benchmark.py` 에 통합해 베이스라인 대비 재측정
- 크래시 원인 확정(IPMI/syslog)은 **원 서버 한정** 항목 — 이 PC에선 해당 없음(2-GPU 전력만 주시).
