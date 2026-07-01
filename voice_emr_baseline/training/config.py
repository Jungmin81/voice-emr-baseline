# -*- coding: utf-8 -*-
"""
전이학습 설정 — 데이터 경로, 모델, Phase별 하이퍼파라미터

데이터 경로는 환경변수 AIHUB_DATA_DIR 로 설정하거나,
실행 시 --data-dir 인자로 넘기면 됩니다.

사용 예:
    export AIHUB_DATA_DIR="/path/to/Training"
    python train_whisper.py --phase a

또는:
    python train_whisper.py --data-dir /path/to/Training --phase a
"""
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


# ──────────────────────────────────────────────────────────────
# 데이터 경로 — 환경변수 또는 인자로 덮어쓰기
# ──────────────────────────────────────────────────────────────
DEFAULT_DATA_DIR = os.environ.get(
    "AIHUB_DATA_DIR",
    # 기본값 (서버에서 실제 경로로 바꿔주세요)
    "/data/aihub/Training"
)

# 출력 폴더 — 학습 산출물 저장
DEFAULT_OUTPUT_ROOT = os.environ.get(
    "TRAINING_OUTPUT_DIR",
    "./outputs"
)


@dataclass
class TrainingPhase:
    """단계별 학습 설정."""
    name: str                           # 단계 이름
    max_speakers: Optional[int]         # 사용할 화자 수 (None = 전체)
    speaker_categories: List[str]       # 환자/의사/간호사 중 어느 것
    num_epochs: int
    batch_size: int                     # GPU 1장 기준
    learning_rate: float
    description: str


# ──────────────────────────────────────────────────────────────
# Phase 정의 — A/B/C 단계적 진행
# ──────────────────────────────────────────────────────────────
PHASES = {
    "a": TrainingPhase(
        name="phase_a_quick_verify",
        max_speakers=1,                 # 1명만 (~5시간 음성)
        speaker_categories=["환자"],
        num_epochs=2,
        batch_size=8,
        learning_rate=1e-4,
        description="빠른 검증 — 코드 동작 확인 (2~3시간)",
    ),
    "b": TrainingPhase(
        name="phase_b_small",
        max_speakers=100,               # ~100명 (~50시간)
        speaker_categories=["환자", "의사", "간호사"],
        num_epochs=3,
        batch_size=16,
        learning_rate=1e-4,
        description="초기 성능 검증 (12~24시간)",
    ),
    "c": TrainingPhase(
        name="phase_c_full",
        max_speakers=None,              # 전체 (1,450시간)
        speaker_categories=["환자", "의사", "간호사"],
        num_epochs=3,
        batch_size=16,
        learning_rate=5e-5,
        description="본 학습 — 전체 데이터 (2~3일)",
    ),
}


@dataclass
class ModelConfig:
    """Whisper + LoRA 설정."""
    base_model: str = "openai/whisper-medium"   # medium 으로 전이학습
    language: str = "korean"
    task: str = "transcribe"

    # LoRA 설정
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj", "k_proj", "out_proj"]
    )

    # 학습 일반 설정
    warmup_steps: int = 500
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 1
    # medium 은 A100 80GB 에서 checkpointing 없이 충분 → 끄면 DDP 충돌 회피.
    # (large-v3 등 큰 모델 + 메모리 부족 시에만 True 로)
    gradient_checkpointing: bool = False
    fp16: bool = False                    # A100은 bf16 권장
    bf16: bool = True

    # 평가 / 체크포인트
    # loop(step)당 시간이 길므로 자주 저장하고 넉넉히 보관 (LoRA 어댑터라 용량 작음)
    eval_steps: int = 500
    save_steps: int = 500
    save_total_limit: int = 10            # 최근 N개 보관 (+ best 별도 유지)
    logging_steps: int = 25

    # 분할 비율 (화자별 분리, leakage 방지)
    train_ratio: float = 0.9
    val_ratio: float = 0.05
    test_ratio: float = 0.05

    # 데이터 필터링
    max_audio_seconds: float = 30.0       # Whisper 입력 한계
    min_audio_seconds: float = 0.5
    min_text_chars: int = 1
    max_text_chars: int = 448             # Whisper 토크나이저 한계


@dataclass
class MonitoringConfig:
    """모니터링 설정."""
    tensorboard_enabled: bool = True
    gradio_dashboard_enabled: bool = True
    gradio_port: int = 7861               # baseline app.py가 7860 쓰니까

    # 로그 파일
    train_log_file: str = "train_log.jsonl"
    eval_log_file: str = "eval_log.jsonl"
    sample_pred_file: str = "sample_predictions.jsonl"
    gpu_stats_file: str = "gpu_stats.jsonl"
    status_file: str = "status.json"

    # 샘플 예측 (검증 시 보여줄 개수)
    num_sample_predictions: int = 5
    gpu_poll_interval_sec: int = 30


def get_phase_config(phase: str) -> TrainingPhase:
    if phase not in PHASES:
        raise ValueError(f"Unknown phase: {phase}. Options: {list(PHASES.keys())}")
    return PHASES[phase]


def get_output_dir(phase_name: str, output_root: str = None) -> Path:
    root = Path(output_root or DEFAULT_OUTPUT_ROOT)
    return root / phase_name
