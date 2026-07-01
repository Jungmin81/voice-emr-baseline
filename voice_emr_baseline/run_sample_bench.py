#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sample_long / sample_short 에 대해 Whisper 모델별 STT 실행 + 정확도(CER) 산출.

정답(samples/sample_*.txt)에는 [의사]/[환자] 화자 태그가 있으나 음성엔 없으므로
CER 계산 시 태그를 제거한다.

사용:
  python run_sample_bench.py
  python run_sample_bench.py --models tiny small medium large-v3 --output-dir sample_bench_results
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "training"))
from utils import cer  # noqa: E402

SPEAKER_TAG_RE = re.compile(r"\[[^\]]+\]")  # [의사], [환자] 등 제거


def clean_reference(text: str) -> str:
    """정답에서 화자 태그 제거."""
    return SPEAKER_TAG_RE.sub(" ", text)


def transcribe(model, audio_path, language="ko"):
    segments, info = model.transcribe(
        audio_path, language=language, beam_size=5, vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
    )
    t0 = time.time()
    parts = [seg.text.strip() for seg in segments]
    elapsed = time.time() - t0
    text = " ".join(parts)
    return text, info.duration, elapsed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+",
                    default=["tiny", "small", "medium", "large-v3"])
    ap.add_argument("--samples", default="samples")
    ap.add_argument("--files", nargs="+",
                    default=["sample_short.wav", "sample_long.wav"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output-dir", default="sample_bench_results")
    args = ap.parse_args()

    from faster_whisper import WhisperModel

    samples_dir = Path(args.samples)
    compute = "int8" if args.device == "cpu" else "float16"

    results = []
    for model_name in args.models:
        print(f"\n{'='*70}\n  모델 로딩: {model_name} ({args.device}, {compute})\n{'='*70}")
        t_load = time.time()
        model = WhisperModel(model_name, device=args.device, compute_type=compute)
        load_sec = time.time() - t_load
        print(f"  로드 완료 ({load_sec:.1f}s)")

        for fname in args.files:
            wav = samples_dir / fname
            gt_path = wav.with_suffix(".txt")
            gt_raw = gt_path.read_text(encoding="utf-8")
            gt = clean_reference(gt_raw)

            hyp, dur, stt_sec = transcribe(model, str(wav))
            c = cer(gt, hyp)
            rtf = stt_sec / dur if dur else 0
            print(f"  [{fname}] {dur:.1f}s 음성 → {stt_sec:.2f}s "
                  f"(RTF {rtf:.3f}) | CER {c*100:.2f}%")
            results.append({
                "model": model_name,
                "file": fname,
                "audio_sec": round(dur, 2),
                "stt_sec": round(stt_sec, 2),
                "rtf": round(rtf, 4),
                "cer": round(c, 4),
                "accuracy": round(1 - c, 4),
                "load_sec": round(load_sec, 2),
                "hypothesis": hyp,
                "reference_clean": gt.strip(),
            })
        del model

    # 표 출력
    print(f"\n{'='*78}\n  모델별 STT 정확도/속도 — sample_long / sample_short\n{'='*78}")
    print(f"  {'model':<10}{'file':<20}{'CER':>8}{'정확도':>9}{'RTF':>8}{'처리(s)':>9}")
    print(f"  {'-'*72}")
    for r in sorted(results, key=lambda x: (x["file"], x["cer"])):
        print(f"  {r['model']:<10}{r['file']:<20}{r['cer']*100:>7.2f}%"
              f"{r['accuracy']*100:>8.2f}%{r['rtf']:>8.3f}{r['stt_sec']:>9.2f}")

    # 저장
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    import csv
    with open(out / "results.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "file", "audio_sec", "stt_sec", "rtf",
                    "cer", "accuracy", "load_sec"])
        for r in results:
            w.writerow([r["model"], r["file"], r["audio_sec"], r["stt_sec"],
                        r["rtf"], r["cer"], r["accuracy"], r["load_sec"]])
    print(f"\n💾 저장: {out/'results.json'}, {out/'results.csv'}")


if __name__ == "__main__":
    main()
