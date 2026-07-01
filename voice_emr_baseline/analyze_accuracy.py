#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
벤치마크 결과 → 모델별 STT 소요시간 + 정확도(CER) 산출

bench_results*/results.json 의 전사 결과를 samples/ 의 정답 텍스트(.txt)와
비교해서 모델(config)별 평균 CER / RTF / 처리시간을 표로 만든다.

사용 예:
  # 기본 (bench_results_hub 분석)
  python analyze_accuracy.py

  # 다른 벤치 폴더
  python analyze_accuracy.py --bench-dir bench_results --samples samples

  # 결과 저장
  python analyze_accuracy.py --output-dir accuracy_results
"""
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

# training/utils.py 의 CER 재사용
sys.path.insert(0, str(Path(__file__).parent / "training"))
from utils import cer, normalize_korean  # noqa: E402


def find_ground_truth(audio_name: str, samples_dir: Path):
    """audio 파일명(stem)에 대응하는 정답 .txt 를 samples/ 에서 찾는다."""
    stem = Path(audio_name).stem
    matches = list(samples_dir.rglob(f"{stem}.txt"))
    if not matches:
        return None
    return matches[0].read_text(encoding="utf-8")


def load_records(bench_dir: Path):
    """results.json 로드 (status == ok 만)."""
    path = bench_dir / "results.json"
    if not path.exists():
        raise FileNotFoundError(f"결과 파일 없음: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    return data


def analyze(bench_dir: Path, samples_dir: Path):
    records = load_records(bench_dir)

    # GT 캐시 (파일당 1회만 읽음)
    gt_cache = {}

    per_config = defaultdict(list)   # config_label -> [record dict]
    missing_gt = set()

    for r in records:
        if r.get("status") != "ok":
            continue
        audio = r["audio"]
        if audio not in gt_cache:
            gt_cache[audio] = find_ground_truth(audio, samples_dir)
        gt = gt_cache[audio]
        if gt is None:
            missing_gt.add(audio)
            continue
        hyp = r.get("transcription") or ""
        ref_norm = normalize_korean(gt).replace(" ", "")
        c = cer(gt, hyp)
        per_config[r["config_label"].strip()].append({
            "audio": audio,
            "audio_sec": float(r.get("audio_sec") or 0),
            "stt_sec": float(r.get("stt_sec") or 0),
            "stt_rtf": float(r.get("stt_rtf") or 0),
            "whisper_load_sec": float(r.get("whisper_load_sec") or 0),
            "ref_chars": len(ref_norm),
            "cer": c,
            "whisper_model": r["whisper_model"],
        })

    return per_config, missing_gt


def summarize(per_config: dict):
    """config 별 집계: char-weighted CER, 평균 RTF, 총 음성/처리 시간."""
    rows = []
    for cfg, items in per_config.items():
        n = len(items)
        total_ref = sum(i["ref_chars"] for i in items)
        # 문자 가중 CER (표준): sum(err) / sum(chars)
        weighted_cer = (
            sum(i["cer"] * i["ref_chars"] for i in items) / total_ref
            if total_ref else 0.0
        )
        simple_cer = sum(i["cer"] for i in items) / n if n else 0.0
        total_audio = sum(i["audio_sec"] for i in items)
        total_stt = sum(i["stt_sec"] for i in items)
        avg_rtf = total_stt / total_audio if total_audio else 0.0
        rows.append({
            "config": cfg,
            "whisper_model": items[0]["whisper_model"],
            "n_files": n,
            "weighted_cer": weighted_cer,
            "simple_cer": simple_cer,
            "accuracy": 1.0 - weighted_cer,
            "total_audio_sec": total_audio,
            "total_stt_sec": total_stt,
            "avg_rtf": avg_rtf,
            "whisper_load_sec": items[0]["whisper_load_sec"],
        })
    # CER 낮은 순 정렬
    rows.sort(key=lambda x: x["weighted_cer"])
    return rows


def print_table(rows, bench_dir):
    print(f"\n{'='*92}")
    print(f"  모델별 STT 정확도 / 속도 — {bench_dir}")
    print(f"{'='*92}")
    header = (f"  {'config':<34} {'CER':>7} {'정확도':>8} "
             f"{'평균RTF':>8} {'총처리(s)':>10} {'음성(s)':>9} {'파일':>5}")
    print(header)
    print(f"  {'-'*88}")
    for r in rows:
        print(f"  {r['config']:<34} "
              f"{r['weighted_cer']*100:>6.2f}% "
              f"{r['accuracy']*100:>7.2f}% "
              f"{r['avg_rtf']:>8.3f} "
              f"{r['total_stt_sec']:>10.2f} "
              f"{r['total_audio_sec']:>9.1f} "
              f"{r['n_files']:>5}")
    print(f"\n  * CER = 문자 가중 평균 (sum(오류문자)/sum(정답문자)), 낮을수록 정확")
    print(f"  * 정확도 = 1 - CER, RTF = 처리시간/음성길이 (낮을수록 빠름)")


def save_outputs(rows, per_config, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) config 요약 CSV
    with open(out_dir / "accuracy_summary.csv", "w",
              encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "config", "whisper_model", "n_files", "weighted_cer",
            "simple_cer", "accuracy", "total_audio_sec", "total_stt_sec",
            "avg_rtf", "whisper_load_sec"])
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 5) if isinstance(v, float) else v)
                        for k, v in r.items()})

    # 2) 파일별 상세 CSV
    with open(out_dir / "accuracy_per_file.csv", "w",
              encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["config", "audio", "audio_sec", "stt_sec",
                    "stt_rtf", "ref_chars", "cer"])
        for cfg, items in per_config.items():
            for i in items:
                w.writerow([cfg, i["audio"], i["audio_sec"], i["stt_sec"],
                            i["stt_rtf"], i["ref_chars"], round(i["cer"], 5)])

    # 3) 마크다운 리포트
    lines = ["# 모델별 STT 정확도 / 속도 비교\n"]
    lines.append("정답 텍스트(samples/) 대비 전사 결과의 CER 측정.\n")
    lines.append("| config | Whisper | CER | 정확도 | 평균 RTF | 총 처리(s) | 음성(s) | 파일 |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        lines.append(
            f"| {r['config']} | {r['whisper_model']} | "
            f"{r['weighted_cer']*100:.2f}% | {r['accuracy']*100:.2f}% | "
            f"{r['avg_rtf']:.3f} | {r['total_stt_sec']:.2f} | "
            f"{r['total_audio_sec']:.1f} | {r['n_files']} |")
    lines.append("\n- **CER**: 문자 가중 평균 (낮을수록 정확)")
    lines.append("- **정확도**: 1 − CER")
    lines.append("- **RTF**: 처리시간 / 음성길이 (낮을수록 빠름)")
    (out_dir / "accuracy_report.md").write_text(
        "\n".join(lines), encoding="utf-8")

    print(f"\n💾 저장 완료:")
    print(f"   - {out_dir / 'accuracy_summary.csv'}")
    print(f"   - {out_dir / 'accuracy_per_file.csv'}")
    print(f"   - {out_dir / 'accuracy_report.md'}")


def main():
    ap = argparse.ArgumentParser(description="벤치 결과 → 모델별 정확도/속도 산출")
    ap.add_argument("--bench-dir", default="bench_results_hub",
                    help="results.json 이 있는 벤치 폴더")
    ap.add_argument("--samples", default="samples",
                    help="정답 .txt 가 있는 폴더")
    ap.add_argument("--output-dir", default=None,
                    help="결과 저장 폴더 (지정 시 CSV/MD 생성)")
    args = ap.parse_args()

    bench_dir = Path(args.bench_dir)
    samples_dir = Path(args.samples)

    per_config, missing = analyze(bench_dir, samples_dir)
    if not per_config:
        print("❌ 분석할 데이터가 없습니다 (정답 매칭 실패 또는 ok 레코드 없음).")
        if missing:
            print(f"   정답 못 찾은 파일: {sorted(missing)}")
        sys.exit(1)

    rows = summarize(per_config)
    print_table(rows, bench_dir)

    if missing:
        print(f"\n⚠️  정답(.txt) 못 찾아 제외한 파일 {len(missing)}개: "
              f"{', '.join(sorted(missing))}")

    if args.output_dir:
        save_outputs(rows, per_config, Path(args.output_dir))


if __name__ == "__main__":
    main()
