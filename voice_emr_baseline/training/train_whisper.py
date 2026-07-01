#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Whisper LoRA 전이학습 — AI Hub 의료 음성 데이터

사용:
    # Phase A (빠른 검증, 1명 화자만)
    python train_whisper.py --phase a --manifest-dir manifests/quick

    # Phase B (100명, 12~24시간)
    python train_whisper.py --phase b --manifest-dir manifests/100p

    # Phase C (전체 1,450시간, 2~3일)
    python train_whisper.py --phase c --manifest-dir manifests/full

    # 멀티 GPU (A100 × 4)
    torchrun --nproc_per_node=4 train_whisper.py --phase b --manifest-dir ...

모니터링:
    TensorBoard:  tensorboard --logdir outputs/<phase>/tensorboard
    Gradio:       python monitor.py --output-dir outputs/<phase>
"""
import argparse
import json
import os
import sys
import time
import threading
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    DEFAULT_DATA_DIR, DEFAULT_OUTPUT_ROOT,
    PHASES, ModelConfig, MonitoringConfig,
    get_phase_config, get_output_dir,
)
from utils import normalize_korean, batch_cer


def load_manifest(path: Path) -> list:
    """JSONL manifest 로드."""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


# ─────────────────────────────────────────────────────────────
# GPU 모니터링 (백그라운드 스레드)
# ─────────────────────────────────────────────────────────────
def gpu_stats_logger(output_dir: Path, stop_event: threading.Event,
                     interval_sec: int = 30):
    """nvidia-smi 폴링해서 JSONL로 저장."""
    import subprocess
    log_path = output_dir / "gpu_stats.jsonl"
    while not stop_event.is_set():
        try:
            r = subprocess.run(
                ["nvidia-smi",
                 "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            )
            if r.returncode == 0:
                gpus = []
                for line in r.stdout.strip().split("\n"):
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 6:
                        gpus.append({
                            "idx": int(parts[0]),
                            "name": parts[1],
                            "util_pct": int(parts[2]),
                            "mem_used_mb": int(parts[3]),
                            "mem_total_mb": int(parts[4]),
                            "temp_c": int(parts[5]),
                        })
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({
                        "ts": time.time(),
                        "gpus": gpus,
                    }) + "\n")
        except Exception:
            pass
        stop_event.wait(interval_sec)


# ─────────────────────────────────────────────────────────────
# 데이터 콜레이터 (Whisper용)
# ─────────────────────────────────────────────────────────────
def make_collator(processor):
    import torch

    def collate(features):
        # 음성 입력
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = processor.feature_extractor.pad(input_features, return_tensors="pt")
        # 모델 가중치(bf16)와 dtype 일치 (학습/평가 generate 모두 안전)
        batch["input_features"] = batch["input_features"].to(torch.bfloat16)

        # 라벨 (텍스트)
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        # decoder_start_token_id 처리
        if (labels[:, 0] == processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch
    return collate


# ─────────────────────────────────────────────────────────────
# 대시보드 콜백 — 실시간 진행률/ETA + 샘플 예측 (main 안에서 생성)
# ─────────────────────────────────────────────────────────────
def build_dashboard_callback(processor, val_dataset, output_dir: Path,
                             num_samples: int = 5, cer_eval_n: int = 64):
    """transformers TrainerCallback 생성.

    - on_log: progress.json 갱신 (step/epoch/loss/속도/ETA) → 대시보드 실시간 표시
    - on_evaluate: val 일부(cer_eval_n)로 generate → CER 계산(cer_log.jsonl) +
                   샘플 N개 GT vs PRED 저장. (dtype 을 bf16 로 통제해 generate)
    메인 프로세스(world_process_zero)에서만 파일 기록.
    """
    import torch
    from transformers import TrainerCallback

    progress_path = output_dir / "progress.json"
    sample_path = output_dir / "sample_predictions.jsonl"
    cer_path = output_dir / "cer_log.jsonl"

    # CER 을 TensorBoard 에도 기록 (Gradio 가 막힌 환경에서 TB 로 확인 가능)
    try:
        from torch.utils.tensorboard import SummaryWriter
        tb_writer = SummaryWriter(str(output_dir / "tensorboard"))
    except Exception:
        tb_writer = None

    class DashboardCallback(TrainerCallback):
        def __init__(self):
            self.t_start = None

        def on_train_begin(self, args, state, control, **kwargs):
            self.t_start = time.time()

        def _write_progress(self, state, extra=None):
            if not state.is_world_process_zero:
                return
            cur = state.global_step
            mx = state.max_steps or 0
            elapsed = time.time() - (self.t_start or time.time())
            steps_per_sec = cur / elapsed if elapsed > 0 and cur > 0 else 0.0
            remaining = mx - cur
            eta_sec = remaining / steps_per_sec if steps_per_sec > 0 else 0.0
            data = {
                "ts": time.time(),
                "global_step": cur,
                "max_steps": mx,
                "epoch": round(state.epoch or 0, 3),
                "num_train_epochs": state.num_train_epochs,
                "progress_pct": round(cur / mx * 100, 2) if mx else 0,
                "elapsed_sec": round(elapsed, 1),
                "steps_per_sec": round(steps_per_sec, 4),
                "sec_per_step": round(1 / steps_per_sec, 3) if steps_per_sec > 0 else None,
                "remaining_steps": remaining,
                "eta_sec": round(eta_sec, 1),
                "best_metric": state.best_metric,
            }
            if extra:
                data.update(extra)
            try:
                progress_path.write_text(
                    json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            except Exception:
                pass

        def on_log(self, args, state, control, logs=None, **kwargs):
            extra = {}
            if logs:
                if "loss" in logs:
                    extra["last_train_loss"] = logs["loss"]
                if "eval_loss" in logs:
                    extra["last_eval_loss"] = logs["eval_loss"]
                if "eval_cer" in logs:
                    extra["last_eval_cer"] = logs["eval_cer"]
            self._write_progress(state, extra)

        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            if not state.is_world_process_zero:
                self._write_progress(state)
                return
            model = kwargs.get("model")
            if model is None or val_dataset is None:
                self._write_progress(state)
                return
            model.eval()
            device = next(model.parameters()).device
            n = min(cer_eval_n, len(val_dataset))
            refs, hyps, samples = [], [], []
            try:
                with torch.no_grad():
                    for i in range(n):
                        item = val_dataset[i]
                        feats = torch.tensor(item["input_features"]).unsqueeze(0).to(
                            device, dtype=torch.bfloat16)
                        pred_ids = model.generate(feats, max_new_tokens=200)
                        pred_text = processor.tokenizer.batch_decode(
                            pred_ids, skip_special_tokens=True)[0]
                        gt_text = processor.tokenizer.decode(
                            [t for t in item["labels"] if t != -100],
                            skip_special_tokens=True)
                        refs.append(gt_text)
                        hyps.append(pred_text)
                        if i < num_samples:
                            samples.append({"gt": gt_text, "pred": pred_text})
                cer_score = batch_cer(refs, hyps)
                if tb_writer is not None:
                    tb_writer.add_scalar("eval/cer_pct", cer_score * 100, state.global_step)
                    tb_writer.flush()
                with open(sample_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({
                        "ts": time.time(), "step": state.global_step,
                        "samples": samples,
                    }, ensure_ascii=False) + "\n")
                with open(cer_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({
                        "ts": time.time(), "step": state.global_step,
                        "epoch": round(state.epoch or 0, 3),
                        "eval_cer": cer_score, "n": n,
                        "eval_loss": (metrics or {}).get("eval_loss"),
                    }, ensure_ascii=False) + "\n")
                self._write_progress(state, {
                    "last_eval_cer": cer_score,
                    "last_eval_loss": (metrics or {}).get("eval_loss"),
                })
            except Exception as e:
                print(f"  ⚠ CER/샘플 평가 실패: {e}")
                self._write_progress(state)
            finally:
                model.train()

    return DashboardCallback()


# ─────────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Whisper LoRA 전이학습")
    parser.add_argument("--phase", required=True, choices=["a", "b", "c"],
                        help="학습 단계: a=빠른검증, b=100명, c=전체")
    parser.add_argument("--manifest-dir", required=True,
                        help="prepare_dataset.py로 만든 manifest 폴더")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT,
                        help="학습 산출물 루트 폴더")
    parser.add_argument("--base-model", default="openai/whisper-large-v3",
                        help="베이스 모델 (기본: large-v3)")
    parser.add_argument("--resume", action="store_true",
                        help="기존 체크포인트에서 재개")
    # loop당 시간이 길므로 저장/평가 주기를 직접 조정 가능
    parser.add_argument("--eval-steps", type=int, default=None,
                        help="평가 주기 (기본: config 값)")
    parser.add_argument("--save-steps", type=int, default=None,
                        help="체크포인트 저장 주기 (기본: config 값)")
    parser.add_argument("--save-total-limit", type=int, default=None,
                        help="보관할 체크포인트 수 (기본: config 값)")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="최대 step (스모크 테스트용, 지정 시 epoch 무시)")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="GPU당 batch (기본: phase 값)")
    parser.add_argument("--grad-accum", type=int, default=None,
                        help="gradient accumulation steps (기본: config=1). "
                             "GPU 메모리가 작아 per-device batch 를 줄였을 때 "
                             "effective batch 를 보전하려면 키운다 "
                             "(effective = batch × GPU수 × grad-accum).")
    parser.add_argument("--num-workers", type=int, default=8,
                        help="DataLoader 워커 수 (on-the-fly 오디오 로드)")
    parser.add_argument("--early-stop-patience", type=int, default=None,
                        help="eval_loss 가 N번 연속 개선 없으면 조기중단 "
                             "(미지정=off). 전체 데이터 장기 런에 권장. "
                             "eval_steps 와 함께 조정.")
    args = parser.parse_args()

    # 설정 로드
    phase_cfg = get_phase_config(args.phase)
    model_cfg = ModelConfig(base_model=args.base_model)
    mon_cfg = MonitoringConfig()

    # CLI 오버라이드
    if args.eval_steps is not None:
        model_cfg.eval_steps = args.eval_steps
    if args.save_steps is not None:
        model_cfg.save_steps = args.save_steps
    if args.save_total_limit is not None:
        model_cfg.save_total_limit = args.save_total_limit
    if args.batch_size is not None:
        phase_cfg.batch_size = args.batch_size
    if args.grad_accum is not None:
        model_cfg.gradient_accumulation_steps = args.grad_accum

    # DDP: main()이 GPU 수만큼 실행됨 → 파일 기록은 rank 0 에서만
    is_main = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", 0))) == 0

    output_dir = get_output_dir(phase_cfg.name, args.output_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Whisper LoRA 전이학습 — Phase {args.phase.upper()}")
    print(f"{'='*60}")
    print(f"  설정: {phase_cfg.description}")
    print(f"  출력: {output_dir}")
    print(f"  베이스 모델: {model_cfg.base_model}")

    # ─── manifest 로드 ───
    manifest_dir = Path(args.manifest_dir)
    train_data = load_manifest(manifest_dir / "train.jsonl")
    val_data = load_manifest(manifest_dir / "val.jsonl")
    print(f"\n📊 데이터: train {len(train_data)} / val {len(val_data)}")

    # ─── 라이브러리 import (느리니까 검증 후) ───
    print(f"\n📚 라이브러리 로딩 중...")
    import torch
    import librosa
    from torch.utils.data import Dataset as TorchDataset
    from transformers import (
        WhisperForConditionalGeneration, WhisperProcessor,
        Seq2SeqTrainer, Seq2SeqTrainingArguments,
    )
    from peft import LoraConfig, get_peft_model

    # ─── Processor + 모델 ───
    print(f"🤖 모델 로딩: {model_cfg.base_model}")
    processor = WhisperProcessor.from_pretrained(
        model_cfg.base_model,
        language=model_cfg.language,
        task=model_cfg.task,
    )
    # 가중치·입력 모두 bf16 로 통일 (collator 가 input_features 를 bf16 로 캐스팅).
    # → 학습/평가(generate) 경로 모두 dtype 일관 → conv dtype 충돌 없음.
    model = WhisperForConditionalGeneration.from_pretrained(
        model_cfg.base_model, torch_dtype=torch.bfloat16)
    # 한국어로 강제
    model.generation_config.language = model_cfg.language
    model.generation_config.task = model_cfg.task
    model.generation_config.forced_decoder_ids = None

    # ─── LoRA 적용 ───
    print(f"⚙  LoRA 어댑터 적용 (r={model_cfg.lora_r}, "
          f"target={model_cfg.lora_target_modules})")
    peft_config = LoraConfig(
        r=model_cfg.lora_r,
        lora_alpha=model_cfg.lora_alpha,
        target_modules=model_cfg.lora_target_modules,
        lora_dropout=model_cfg.lora_dropout,
        bias="none",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    if model_cfg.gradient_checkpointing:
        # gradient checkpointing 과 use_cache 는 양립 불가 → 학습 중 캐시 끔
        model.config.use_cache = False
        # non-reentrant checkpointing: DDP 와 호환 (reentrant 는 "mark ready once" 오류)
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
        # LoRA(동결 베이스) + checkpointing 시 입력 임베딩에 grad 가 흐르도록
        # 해줘야 backward 가 동작 (없으면 "does not require grad" 오류)
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    # ─── Dataset (on-the-fly 전처리) ───
    # 1.1M 샘플의 mel-spectrogram 을 미리 만들면 디스크 ~1TB → 불가.
    # DataLoader 워커에서 __getitem__ 시점에 로드/추출 (lazy, 메모리 안전).
    class WhisperAudioDataset(TorchDataset):
        def __init__(self, records):
            self.records = records

        def __len__(self):
            return len(self.records)

        def __getitem__(self, idx):
            r = self.records[idx]
            audio, _ = librosa.load(r["audio"], sr=16000, mono=True)
            feats = processor.feature_extractor(
                audio, sampling_rate=16000).input_features[0]
            text = normalize_korean(r["text"], remove_punct=False)
            tokens = processor.tokenizer(text).input_ids
            return {"input_features": feats, "labels": tokens}

    train_ds = WhisperAudioDataset(train_data)
    val_ds = WhisperAudioDataset(val_data)
    print(f"\n🔄 on-the-fly 전처리 — train {len(train_ds)} / val {len(val_ds)} "
          f"(DataLoader 워커에서 로드)")

    # ─── 평가 메트릭 (CER) ───
    def compute_metrics(eval_pred):
        pred_ids = eval_pred.predictions
        label_ids = eval_pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        cer_score = batch_cer(label_str, pred_str)
        return {"cer": cer_score}

    # ─── Training Arguments ───
    ta_kwargs = dict(
        output_dir=str(output_dir),
        per_device_train_batch_size=phase_cfg.batch_size,
        per_device_eval_batch_size=phase_cfg.batch_size,
        gradient_accumulation_steps=model_cfg.gradient_accumulation_steps,
        learning_rate=phase_cfg.learning_rate,
        warmup_steps=model_cfg.warmup_steps,
        num_train_epochs=phase_cfg.num_epochs,
        weight_decay=model_cfg.weight_decay,
        gradient_checkpointing=model_cfg.gradient_checkpointing,
        bf16=model_cfg.bf16,
        eval_steps=model_cfg.eval_steps,
        save_strategy="steps",
        save_steps=model_cfg.save_steps,
        save_total_limit=model_cfg.save_total_limit,
        logging_strategy="steps",
        logging_steps=model_cfg.logging_steps,
        logging_dir=str(output_dir / "tensorboard"),
        report_to=["tensorboard"],
        # 평가는 eval_loss 만 (generate 는 대시보드 콜백에서 dtype 통제하며 수행).
        predict_with_generate=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        push_to_hub=False,
        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        label_names=["labels"],
        # LoRA: grad 가 있는 건 어댑터뿐이고 전부 사용됨 → unused 탐색 불필요
        ddp_find_unused_parameters=False,
    )
    if args.num_workers > 0:
        ta_kwargs["dataloader_persistent_workers"] = True
    # 스모크 테스트: --max-steps 지정 시 epoch 무시하고 step 수로 제한
    if args.max_steps is not None:
        ta_kwargs["max_steps"] = args.max_steps
    # transformers 버전별 인자명: 4.46+ 는 eval_strategy, 그 이전은 evaluation_strategy
    import inspect
    ta_params = inspect.signature(Seq2SeqTrainingArguments.__init__).parameters
    if "eval_strategy" in ta_params:
        ta_kwargs["eval_strategy"] = "steps"
    else:
        ta_kwargs["evaluation_strategy"] = "steps"

    training_args = Seq2SeqTrainingArguments(**ta_kwargs)

    # ─── GPU 모니터링 백그라운드 시작 (rank 0 만) ───
    stop_event = threading.Event()
    gpu_thread = None
    status_path = output_dir / "status.json"
    if is_main:
        gpu_thread = threading.Thread(
            target=gpu_stats_logger,
            args=(output_dir, stop_event, mon_cfg.gpu_poll_interval_sec),
            daemon=True,
        )
        gpu_thread.start()
        status_path.write_text(json.dumps({
            "phase": args.phase,
            "phase_name": phase_cfg.name,
            "started_at": time.time(),
            "status": "training",
            "base_model": model_cfg.base_model,
            "train_size": len(train_data),
            "val_size": len(val_data),
        }, ensure_ascii=False, indent=2), encoding="utf-8")

    # ─── Trainer ───
    dashboard_cb = build_dashboard_callback(
        processor, val_ds, output_dir, mon_cfg.num_sample_predictions)

    callbacks = [dashboard_cb]
    if args.early_stop_patience is not None:
        from transformers import EarlyStoppingCallback
        callbacks.append(EarlyStoppingCallback(
            early_stopping_patience=args.early_stop_patience))
        if is_main:
            print(f"⏹  EarlyStopping 활성화: patience={args.early_stop_patience} "
                  f"(eval_loss 기준)")

    trainer_kwargs = dict(
        args=training_args,
        model=model,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=make_collator(processor),
        callbacks=callbacks,
    )
    # transformers 버전별 인자명: 신버전은 processing_class, 구버전은 tokenizer
    import inspect as _inspect
    if "processing_class" in _inspect.signature(Seq2SeqTrainer.__init__).parameters:
        trainer_kwargs["processing_class"] = processor.feature_extractor
    else:
        trainer_kwargs["tokenizer"] = processor.feature_extractor
    trainer = Seq2SeqTrainer(**trainer_kwargs)

    # ─── 학습 ───
    print(f"\n🚀 학습 시작!\n")
    try:
        trainer.train(resume_from_checkpoint=args.resume)
    except KeyboardInterrupt:
        print("\n⏸ 사용자 중단 — 현재 상태 저장")
    finally:
        stop_event.set()
        if gpu_thread is not None:
            gpu_thread.join(timeout=5)
        if is_main and status_path.exists():
            status_path.write_text(json.dumps({
                **json.loads(status_path.read_text(encoding="utf-8")),
                "status": "stopped",
                "ended_at": time.time(),
            }, ensure_ascii=False, indent=2), encoding="utf-8")

    # ─── 최종 모델 저장 ───
    final_dir = output_dir / "final"
    print(f"\n💾 최종 모델 저장: {final_dir}")
    trainer.save_model(str(final_dir))
    processor.save_pretrained(str(final_dir))

    print(f"\n✅ 학습 완료!")
    print(f"\n다음 단계:")
    print(f"  평가:        python evaluate.py --model-dir {final_dir} --manifest-dir {manifest_dir}")
    print(f"  벤치마크:    python ../benchmark.py --whisper-checkpoint {final_dir}")


if __name__ == "__main__":
    main()
