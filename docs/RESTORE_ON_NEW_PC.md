# 다른 PC에서 복원하기 — clone + 산출물 압축본

> 이 저장소에는 **코드와 평가 지표만** 들어있습니다. 학습된 **가중치·음성 샘플·학습 로그**는
> 용량(`*.safetensors`, `*.pt`)과 데이터 성격(환자 음성) 때문에 `.gitignore` 로 제외돼 있습니다.
> 그 빠진 부분을 **별도 압축본**으로 들고 다니며 복원하는 절차입니다.

작성: 2026-08-18

---

## 0. 무엇이 git 에 있고 무엇이 없나

| 구분 | git 포함 | 압축본에만 |
|---|:---:|:---:|
| 앱·학습·평가 코드 (`*.py`, `*.sh`) | ✅ | (중복 포함) |
| 평가 지표 (`eval_*.json`, `compare.txt`) | ✅ | (중복 포함) |
| 문서 (`docs/`, `README.md`) | ✅ | — |
| **최종 LoRA 어댑터** (`final/`, 75MB) | ❌ | ✅ |
| 학습 로그 (`cer_log.jsonl`, `sample_predictions.jsonl`, tensorboard) | ❌ | ✅ |
| 데모 음성 (`samples/`, 61MB) | ❌ | ✅ |
| 벤치마크 결과 (`bench_results*`, `accuracy_results`) | ❌ | ✅ |
| 중간 체크포인트 (checkpoint-44000~52278, 3.6GB) | ❌ | ❌ |
| manifest (`manifests/`, 1.2GB) | ❌ | ❌ |
| 가상환경 (`venv_train/`, 4.9GB) | ❌ | ❌ |

압축본 파일명: `voice_emr_artifacts_<YYYYMMDD>.zip` (또는 `.tar.gz`) — 약 108MB.
GitHub 파일당 100MB 제한을 넘으므로 **저장소에 올리지 않고 USB/NAS 로 따로 보관**합니다.

---

## 1. 복원 순서

```bash
# ① 저장소 clone
git clone https://github.com/Jungmin81/voice-emr-baseline.git
cd voice-emr-baseline

# ② 압축본을 저장소 루트에서 해제 (경로가 상대경로라 제자리로 복원됨)
unzip /경로/voice_emr_artifacts_20260818.zip
#   또는
tar -xzf /경로/voice_emr_artifacts_20260818.tar.gz
```

압축본 내부 경로가 `results/...`, `voice_emr_baseline/...` 형태의 **저장소 루트 기준 상대경로**라
clone 폴더 안에서 풀면 각 파일이 원래 자리로 들어갑니다.

**해제 후 확인:**

```bash
ls -l voice_emr_baseline/training/outputs/full/phase_c_full/final/adapter_model.safetensors  # ~75MB
git status --short    # 깨끗해야 정상 (압축본 내용은 전부 .gitignore 대상 or 동일 코드)
```

`git status` 에 뭔가 뜨면 압축본이 추적 중인 코드를 덮어썼다는 뜻입니다 — `git diff` 로 확인하세요.

---

## 2. 실행 환경 구축

`venv_train/` 은 압축본에 **포함돼 있지 않습니다**(4.9GB). 새 PC에서 다시 설치합니다.

```bash
bash scripts/setup_venv.sh        # venv_train 생성 + torch(cu124)/transformers/peft 등 설치
source venv_train/bin/activate
```

- 버전 고정이 필요하면 [requirements_frozen.txt](requirements_frozen.txt) 참고.
- GPU·드라이버 조합, `torchvision`/`bitsandbytes` 미설치 이유 등은
  [SETUP_A6000.md](SETUP_A6000.md), [ENVIRONMENT_NOTES.md](ENVIRONMENT_NOTES.md) 참고.

## 3. 데모 앱 실행

```bash
source venv_train/bin/activate
cd voice_emr_baseline
DEMO_PORT=7860 python -u app_ft.py    # → http://<서버-IP>:7860
```

앱이 `training/outputs/full/phase_c_full/final/` 의 LoRA 어댑터를 읽으므로,
**1번 단계에서 압축본을 풀어야 정상 동작**합니다. 없으면 베이스 whisper-medium 으로 떨어지거나 로드 실패합니다.

---

## 4. 재학습이 필요할 때만

압축본에 없는 두 가지를 추가로 준비해야 합니다.

**manifest 재생성** — `/mnt/nas_raw/...` 절대경로 목록이라 PC가 바뀌면 무효입니다.

```bash
source venv_train/bin/activate
export AIHUB_DATA_DIR="<데이터>/aihub/Training"
cd voice_emr_baseline/training
python prepare_dataset.py --data-dir "$AIHUB_DATA_DIR" --output manifests/full \
    --categories 환자 의사 간호사
```

**중간 체크포인트** — `--resume` 로 학습을 이어가려면 원본 PC의
`training/outputs/full/phase_c_full/checkpoint-*` (3.6GB) 를 별도로 복사해야 합니다.
학습이 이미 끝난 상태라 보통은 불필요합니다.

데이터 확보·추출 절차는 [DATA_NOTES.md](DATA_NOTES.md), 학습 실행은 [SETUP_A6000.md](SETUP_A6000.md) §3 참고.

---

## 5. 압축본 다시 만들기

원본 PC에서 산출물이 갱신됐을 때, 저장소 루트에서 실행합니다.

```bash
cd <저장소 루트>
{
  find voice_emr_baseline/training/outputs/full/phase_c_full/final -type f
  find voice_emr_baseline/training/outputs \( -name '*.json' -o -name '*.jsonl' \) -type f \
       -not -path '*/final/*' -not -path '*/checkpoint-*/*'
  find voice_emr_baseline/training/outputs -path '*tensorboard*' -type f
  find voice_emr_baseline/training/outputs -path '*/checkpoint-*' -type f -size -2M
  find voice_emr_baseline/training -maxdepth 1 -type f
  find results voice_emr_baseline/bench_results voice_emr_baseline/bench_results_hub \
       voice_emr_baseline/accuracy_results voice_emr_baseline/results -type f
  find voice_emr_baseline/samples -type f
} | sort -u > /tmp/bundle_list.txt

tar -czf ../voice_emr_artifacts_$(date +%Y%m%d).tar.gz -T /tmp/bundle_list.txt
```

zip 이 필요하면 (`zip` 명령이 없는 환경 기준):

```bash
python3 -c "
import zipfile, os
files = [l.strip() for l in open('/tmp/bundle_list.txt', encoding='utf-8') if l.strip()]
with zipfile.ZipFile('../voice_emr_artifacts.zip', 'w', zipfile.ZIP_DEFLATED, allowZip64=True) as z:
    for f in files: z.write(f, f)
print(len(files), 'files', os.path.getsize('../voice_emr_artifacts.zip')/1048576, 'MB')
"
```

---

## ⚠️ 취급 주의

압축본에는 **AI Hub 의료진·환자 음성**(`samples/`, 61MB)이 들어있습니다.
저장소 `.gitignore` 가 `samples/`, `*.wav` 를 막아둔 것도 같은 이유입니다.

- 저장소·공개 클라우드·공유 폴더에 **업로드하지 마세요.**
- 배포가 필요하면 음성을 뺀 번들(약 69MB)을 쓰거나, 어댑터만(약 68MB) 전달하세요.
  둘 다 GitHub 100MB 제한 안에 들어가므로 Release 에셋으로 올릴 수 있습니다.
