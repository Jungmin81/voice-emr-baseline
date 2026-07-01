# Voice EMR — 진료 대화 → 자동 의무기록(SOAP)

의사·환자의 **진료 대화 음성**을 받아, **받아쓰기(STT) → 화자분리 → 키워드 → SOAP 진료기록 초안**까지
자동으로 생성하는 파이프라인입니다.

> 📖 비전문가용 상세 개요: **[docs/PROJECT_OVERVIEW.md](docs/PROJECT_OVERVIEW.md)**

---

## 핵심 성과 — 전이학습 전/후 (테스트셋 2,400개, CER↓)

| 대상 | 학습 전 (일반 AI) | 학습 후 (의료 특화) | 개선 |
|---|---:|---:|---:|
| **전체** | 7.07% | **0.93%** | **-86.8%** |
| 의사 | 6.84% | 0.66% | -90.3% |
| 간호사 | 3.80% | 0.39% | -89.7% |
| 환자 | 11.51% | 1.89% | -83.6% |

*CER = 글자 오류율(낮을수록 정확). Whisper-medium + LoRA, AI Hub 의료음성 약 1,450시간 학습.*

---

## 파이프라인

```
🎙️ 진료 음성
  → 화자분리(pyannote) → 역할판별(의사/환자)
  → 받아쓰기(Whisper-medium + LoRA 전이학습)
  → 키워드(KeyBERT)
  → SOAP 생성(Gemma-2-27B) → 한자/띄어쓰기 후처리
📄 SOAP 진료기록 초안 (S/O/A/P)
```

## 폴더 구조

| 경로 | 내용 |
|---|---|
| `voice_emr_baseline/` | 앱·추론·벤치마크 코드 (`app_ft.py` = 데모 앱) |
| `voice_emr_baseline/training/` | 전이학습 코드·설정 (Phase A/B/C) |
| `docs/` | 프로젝트 개요·핸드오프·환경 노트 |
| `scripts/`, `data_scripts/` | 실험·데이터 준비 스크립트 |
| `results/` | 전/후 평가 결과 (CER 비교) |

> **가중치**: 학습된 LoRA 어댑터는 `voice_emr_baseline/training/outputs/full/phase_c_full/final/adapter_model.safetensors` (약 73MB).
> 용량 문제로 git에는 포함되지 않습니다(`.gitignore`). 로컬/별도 스토리지에 보관하세요.

## 실행 (데모 앱)

```bash
# venv_train 환경에서
cd voice_emr_baseline
DEMO_PORT=7860 python -u app_ft.py    # → http://<서버-IP>:7860
```

세부 설치/환경은 [docs/SETUP_A6000.md](docs/SETUP_A6000.md), [docs/ENVIRONMENT_NOTES.md](docs/ENVIRONMENT_NOTES.md) 참고.
