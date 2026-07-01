#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
학습된 Whisper LoRA 모델 평가

사용:
    python evaluate.py \\
        --model-dir outputs/phase_a_quick_verify/final \\
        --manifest-dir manifests/quick \\
        --split test

결과:
    - CER (전체 + 카테고리별 + 성별/연령별)
    - 샘플 N개 비교 출력
    - results.json 저장
"""
import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils import batch_cer, normalize_korean


def load_manifest(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True,
                        help="학습된 모델 폴더 (PEFT 어댑터 또는 full 모델)")
    parser.add_argument("--base-model", default="openai/whisper-large-v3",
                        help="베이스 모델 (LoRA의 경우 필요)")
    parser.add_argument("--manifest-dir", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--max-samples", type=int, default=None,
                        help="평가할 최대 샘플 수 (빠른 검증용)")
    parser.add_argument("--output", default=None,
                        help="결과 저장 경로 (기본: model_dir/eval_<split>.json)")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    manifest_path = Path(args.manifest_dir) / f"{args.split}.jsonl"
    data = load_manifest(manifest_path)
    if args.max_samples:
        data = data[:args.max_samples]
    print(f"📊 평가 대상: {len(data)} 샘플 ({args.split})")

    # 모델 로드
    print(f"🤖 모델 로딩...")
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    from peft import PeftModel
    import librosa

    processor = WhisperProcessor.from_pretrained(
        args.base_model, language="korean", task="transcribe"
    )

    base_model = WhisperForConditionalGeneration.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16
    ).to(args.device)

    # LoRA 어댑터인지 풀모델인지 자동 감지
    if (Path(args.model_dir) / "adapter_config.json").exists():
        print(f"   → LoRA 어댑터 로드: {args.model_dir}")
        model = PeftModel.from_pretrained(base_model, args.model_dir)
    else:
        print(f"   → Full 모델 로드: {args.model_dir}")
        model = WhisperForConditionalGeneration.from_pretrained(
            args.model_dir, torch_dtype=torch.bfloat16
        ).to(args.device)

    model.eval()
    model.generation_config.language = "korean"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None

    # 평가 루프
    print(f"\n🔄 평가 시작...")
    refs = []
    hyps = []
    per_category = defaultdict(lambda: {"refs": [], "hyps": []})
    per_gender = defaultdict(lambda: {"refs": [], "hyps": []})
    per_age = defaultdict(lambda: {"refs": [], "hyps": []})

    samples_log = []
    t_start = time.time()

    for i, item in enumerate(data):
        if (i + 1) % 50 == 0:
            elapsed = time.time() - t_start
            speed = (i + 1) / elapsed
            eta = (len(data) - i - 1) / speed if speed > 0 else 0
            print(f"  [{i+1}/{len(data)}] 진행 — 속도 {speed:.1f} samples/sec, "
                  f"남은 시간 ~{eta/60:.1f}분")

        try:
            audio, sr = librosa.load(item["audio"], sr=16000, mono=True)
            inputs = processor(
                audio, sampling_rate=16000, return_tensors="pt"
            ).input_features.to(args.device, dtype=torch.bfloat16)

            with torch.no_grad():
                pred_ids = model.generate(inputs, max_new_tokens=225)
            pred_text = processor.batch_decode(pred_ids, skip_special_tokens=True)[0]
        except Exception as e:
            print(f"  ⚠ 오류: {item.get('audio', '?')} — {e}")
            continue

        ref = normalize_korean(item["text"])
        hyp = normalize_korean(pred_text)
        refs.append(ref)
        hyps.append(hyp)

        cat = item.get("category", "?")
        per_category[cat]["refs"].append(ref)
        per_category[cat]["hyps"].append(hyp)

        gender = item.get("gender", "?")
        per_gender[gender]["refs"].append(ref)
        per_gender[gender]["hyps"].append(hyp)

        age = item.get("age", "?")
        per_age[age]["refs"].append(ref)
        per_age[age]["hyps"].append(hyp)

        if len(samples_log) < 20:
            samples_log.append({
                "audio": item["audio"],
                "ref": ref,
                "hyp": hyp,
                "match": ref.replace(" ", "") == hyp.replace(" ", ""),
            })

    # 결과 집계
    overall_cer = batch_cer(refs, hyps)

    print(f"\n{'='*60}")
    print(f"  📊 평가 결과")
    print(f"{'='*60}")
    print(f"\n  전체 CER: **{overall_cer*100:.2f}%**\n")

    print(f"  카테고리별 CER:")
    cat_results = {}
    for cat, d in per_category.items():
        if d["refs"]:
            c = batch_cer(d["refs"], d["hyps"])
            cat_results[cat] = c
            print(f"    - {cat}: {c*100:.2f}% (n={len(d['refs'])})")

    print(f"\n  성별 CER:")
    gender_results = {}
    for g, d in per_gender.items():
        if d["refs"]:
            c = batch_cer(d["refs"], d["hyps"])
            gender_results[g] = c
            print(f"    - {g}: {c*100:.2f}% (n={len(d['refs'])})")

    print(f"\n  연령별 CER:")
    age_results = {}
    for a, d in per_age.items():
        if d["refs"]:
            c = batch_cer(d["refs"], d["hyps"])
            age_results[a] = c
            print(f"    - {a}: {c*100:.2f}% (n={len(d['refs'])})")

    # 샘플 출력
    print(f"\n  📝 샘플 비교 (처음 5개):")
    for i, s in enumerate(samples_log[:5], 1):
        mark = "✅" if s["match"] else "⚠"
        print(f"\n  {i}. {mark}")
        print(f"     GT:   {s['ref']}")
        print(f"     PRED: {s['hyp']}")

    # 저장
    output_path = args.output or (Path(args.model_dir) / f"eval_{args.split}.json")
    output_path = Path(output_path)
    output_path.write_text(json.dumps({
        "model_dir": args.model_dir,
        "manifest_dir": args.manifest_dir,
        "split": args.split,
        "n_samples": len(refs),
        "overall_cer": overall_cer,
        "by_category": cat_results,
        "by_gender": gender_results,
        "by_age": age_results,
        "samples": samples_log,
        "elapsed_sec": time.time() - t_start,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n💾 저장: {output_path}")


if __name__ == "__main__":
    main()
