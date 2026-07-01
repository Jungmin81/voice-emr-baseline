# Voice EMR Baseline 테스트

전이학습 전, **일반 모델의 한국어 의료 대화 처리 성능**을 측정하기 위한 베이스라인 프로그램.
이 결과를 기준으로 향후 AI Hub 데이터 파인튜닝 후 얼마나 개선됐는지 비교 가능.

## ✅ 모든 모델 상업적 사용 가능

| 구성요소 | 모델 / 라이브러리 | 라이선스 | 용도 |
|---|---|---|---|
| STT | **faster-whisper** | MIT | 음성 → 텍스트 |
| Whisper 모델 | OpenAI Whisper large-v3 | MIT | 한국어 인식 |
| 키워드 임베딩 | sentence-transformers | Apache 2.0 | 다국어 임베딩 |
| 키워드 추출 | KeyBERT | MIT | 핵심 어구 추출 |
| 임베딩 모델 | paraphrase-multilingual-MiniLM-L12-v2 | Apache 2.0 | 한국어 포함 |
| LLM | **Qwen2.5-Instruct** | Apache 2.0 | SOAP 구조 요약 |
| 웹 UI | Gradio | Apache 2.0 | 브라우저 UI |

→ **모두 자유롭게 상업적 사용·재배포 가능**합니다.

---

## 🚀 설치

### Step 1. Python 3.10+ 확인

```bash
python --version
# Python 3.10 이상이어야 함
```

### Step 2. 가상환경 만들기 (권장)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### Step 3. ffmpeg 설치 (음성 디코딩)

```bash
# Windows (관리자 PowerShell)
winget install ffmpeg
# 또는: choco install ffmpeg

# Linux
sudo apt install ffmpeg

# Mac
brew install ffmpeg
```

설치 확인:
```bash
ffmpeg -version
```

### Step 4. Python 패키지 설치

**GPU 있는 경우** (NVIDIA, A100/RTX 4090/3090 등):
```bash
# PyTorch GPU 버전 먼저 (CUDA 12.1 기준)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# 나머지 패키지
pip install -r requirements.txt
```

**GPU 없는 경우** (CPU 전용):
```bash
pip install -r requirements-cpu.txt
```

⚠️ CPU에서 LLM은 매우 느리므로 `--skip-llm` 옵션 사용 권장.

---

## 🎯 사용법

### A. CLI 버전 (권장 — 빠르고 안정적)

```bash
# 기본 사용 (STT + 키워드 + SOAP 요약)
python baseline.py 음성파일.wav

# 빠른 테스트 (LLM 건너뜀, 키워드만)
python baseline.py 음성파일.wav --skip-llm

# 작은 모델 (메모리 부족 시)
python baseline.py 음성파일.wav --whisper-model small --llm Qwen/Qwen2.5-3B-Instruct

# 결과를 파일로 저장
python baseline.py 음성파일.wav --output result.json
python baseline.py 음성파일.wav --output result.txt

# 전체 옵션
python baseline.py --help
```

### B. 웹 UI 버전 (브라우저에서 드래그-드롭)

```bash
python app.py
```

→ 브라우저에서 `http://localhost:7860` 접속 → 음성 파일 드래그-드롭 → ▶ 분석 시작

---

## 📊 출력 예시

```
======================================================================
  1. STT (음성 → 텍스트)
======================================================================

  음성 길이:     45.3초
  처리 시간:     8.2초
  RTF (실시간계수): 0.181x  (낮을수록 빠름)
  감지 언어:     ko (99.7% 확률)

  전사 결과:
  ──────────────────────────────────────────────────────────────────
  안녕하세요 선생님. 며칠 전부터 머리가 많이 아프고 어지러워서
  왔습니다. 잠도 잘 못자고요. 약을 먹어도 별로 효과가 없네요...

======================================================================
  2. 주요 키워드 (KeyBERT, 다국어 임베딩)
======================================================================

  머리 아프고                    0.642  ███████████████████
  어지러워서                     0.581  █████████████████
  잠도 잘 못자                   0.523  ███████████████
  약을 먹어도                    0.487  ██████████████
  ...

======================================================================
  3. 의료 카테고리 매핑 (룰 기반)
======================================================================

  • 증상: 두통, 어지러, 불면, 통증
  • 처방: 약, 복용
  • 환자정보: 잠

======================================================================
  4. SOAP 구조 요약 (LLM)
======================================================================

## S (Subjective)
- 주 증상: 두통, 어지러움
- 발생 시기: 며칠 전부터
- 동반 증상: 불면, 약물 효과 부족

## O (Objective)
- 활력 징후: 기재 없음
- 신체 검사: 기재 없음
...
```

---

## 🔧 트러블슈팅

### 1. `ffmpeg not found` 에러
→ Step 3 다시 확인. ffmpeg가 PATH에 등록되어야 함.

### 2. CUDA out of memory
→ 더 작은 모델 사용:
```bash
python baseline.py audio.wav --whisper-model small --llm Qwen/Qwen2.5-3B-Instruct
```

### 3. LLM 다운로드 너무 큼 (~15GB)
→ 첫 실행 1회만 다운로드. 이후 캐시됨 (`~/.cache/huggingface/`).
→ 작은 모델: `--llm Qwen/Qwen2.5-1.5B-Instruct` (3GB)

### 4. CPU에서 너무 느림
→ `--skip-llm` 사용. 또는 더 작은 Whisper 모델 (`small`/`base`).

### 5. 한국어 키워드 품질이 낮음
→ 베이스 모델 한계. 전이학습 후엔 의료 도메인 어휘 인식이 개선됩니다.

---

## 📈 다음 단계 (전이학습 진행)

이 베이스라인 측정 결과를 기록해두고, 동일한 음성으로 전이학습 후 모델 결과와 비교:

| 항목 | 베이스라인 (현재) | 목표 (전이학습 후) | 개선 |
|---|---|---|---|
| 한국어 CER | ? | 5~8% | -50% 이상 |
| 의료 용어 정확도 | ? | 95%+ | 대폭 |
| SOAP 구조 일치 | ? | 90%+ | 대폭 |
| RTF (GPU) | 0.1~0.3x | 0.05~0.1x | 2~3배 |

업무계획서대로:
1. AI Hub 1,450시간 데이터로 LoRA 파인튜닝
2. 진단과별 특화 모델 분리
3. 파일럿 클리닉 실데이터로 2차 학습

---

## 📁 파일 구조

```
voice_emr_baseline/
├── README.md               # 이 문서
├── requirements.txt        # GPU 환경 의존성
├── requirements-cpu.txt    # CPU 전용 의존성
├── baseline.py             # CLI 메인 스크립트
└── app.py                  # 웹 UI (Gradio)
```

## 📝 라이선스 / 출처 표기

이 프로그램으로 만든 결과물을 외부 공개·판매 시 사용 모델 출처를 표기해주세요:

```
이 결과물은 다음 오픈소스 모델로 생성되었습니다:
- Whisper (OpenAI, MIT License)
- faster-whisper (SYSTRAN, MIT License)
- Qwen2.5 (Alibaba Cloud, Apache 2.0 License)
- KeyBERT (Maarten Grootendorst, MIT License)
- sentence-transformers (UKP Lab, Apache 2.0 License)
```
