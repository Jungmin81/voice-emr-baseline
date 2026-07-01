# 환경 노트 (작업 환경 재현용)

## 원 서버 요약
- 호스트: `nvidia-a100`, GPU: **NVIDIA A100 80GB PCIe ×4**, 128 cores
- OS: Ubuntu (kernel 6.2.0-37-generic)
- Python: conda env **`jungmin.cheon_39`** (Python 3.9.7)
  - 경로: `/usr/anaconda3/envs/jungmin.cheon_39/bin/python3`
  - ⚠️ 재부팅 후 셸 PATH 가 `mlflow-env` 로 바뀌어 `python3` 가 엉뚱한 곳을 가리킨 적 있음.
    학습 시엔 `source /usr/anaconda3/bin/activate jungmin.cheon_39` 또는 풀경로 사용.

## 동작 확인된 패키지 버전 (런타임 기준)
- torch **2.8.0+cu128** (CUDA 사용 가능) — ※ `pip freeze` 메타에는 2.5.1 로 보이는 꼬임이 있음
  (여러 torch 설치가 겹침). 새 PC에선 그냥 CUDA 맞는 단일 torch 깨끗이 설치 권장.
- transformers 4.57.6, peft 0.17.1, accelerate 1.10.1, datasets 4.5.0
- gradio 4.44.1, tensorboard 2.20.0
- librosa 0.11.0, soundfile 0.13.1, jiwer 4.0.0, faster-whisper 1.2.1, numpy 1.26.4
- 전체 목록: `requirements_frozen.txt`

## 반드시 피해야 할 함정 (원 서버에서 겪은 것)
1. **torchvision 설치 금지** — torch 와 버전 불일치 시 `torchvision::nms does not exist` 가
   transformers import 체인을 통째로 막음(`PreTrainedModel` import 실패). Whisper STT/LoRA 엔 불필요.
   → 원 서버에선 site-packages 에서 torchvision 을 옆으로 치워 비활성화함.
2. **bitsandbytes 설치 금지(또는 깨졌으면 제거)** — peft LoRA 모듈 생성 시 import 되는데,
   CUDA 셋업 실패하면 학습이 죽음. bf16 LoRA 엔 양자화 불필요 → 비활성화함.
3. **datasets 의 Audio 피처 쓰지 말 것** — datasets 4.x 는 오디오 디코딩에 torchcodec 요구.
   우리는 `Audio` 안 쓰고 preprocess 에서 librosa 로 직접 로드(코드에 반영됨).
4. **gradient_checkpointing + DDP 충돌** — reentrant/non-reentrant 모두 "mark a variable ready
   only once" 오류. medium 은 80GB 에서 checkpointing 불필요하므로 **끔**(config 기본 False).

## 깨끗한 설치 (새 PC 권장)
```bash
python3.10 -m venv venv_train && source venv_train/bin/activate
pip install -U pip
pip install torch --index-url https://download.pytorch.org/whl/cu121   # 드라이버에 맞는 CUDA
pip install "transformers>=4.45,<5" "peft>=0.13" "accelerate>=0.34" "datasets>=2.20" \
            tensorboard "gradio>=4.44" librosa soundfile jiwer pandas psutil faster-whisper
# ❌ torchvision, bitsandbytes 는 설치하지 않는다.
# 검증:
python -c "import torch,transformers,peft,accelerate,datasets,librosa; print('OK', torch.cuda.is_available())"
```

## Claude 권한 설정
`claude_settings/settings.json` 을 새 PC 프로젝트의 `.claude/settings.json` 으로 복사하면
자주 쓰는 Bash 계열(python3/pip/torchrun/nvidia-smi 등)이 허용돼 권한 프롬프트가 줄어듦.
