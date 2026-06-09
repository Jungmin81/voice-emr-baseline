#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Voice EMR Baseline — 벤치마크 스크립트 (v2: warm-up + 정확한 시간 측정)

개선 사항:
- 모델 로딩 시간을 명시적으로 분리 측정
- Warm-up 호출 1회 후 본 측정 시작 (cold/warm 시간 모두 기록)
- 리포트에 "model_load_sec" / "warm_inference_sec" 명확히 표시

사용:
    python benchmark.py --samples samples/*.wav --preset full
"""
import argparse
import csv
import gc
import json
import sys
import time
from pathlib import Path
from datetime import datetime


def free_memory():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except ImportError:
        pass


# ─────────────────────────────────────────────────────────────
# 프리셋 설정
# ─────────────────────────────────────────────────────────────
def build_quick_configs(devices):
    """STT만 — 빠른 벤치마크."""
    configs = []
    models = ["tiny", "small", "medium"]
    if "cuda" in devices:
        models.append("large-v3")
    for model in models:
        for device in devices:
            if model == "large-v3" and device == "cpu":
                continue
            configs.append({
                "label": f"{model:>9s} | {device.upper():>4s} | STT only",
                "whisper": model,
                "device": device,
                "skip_llm": True,
            })
    return configs


def build_full_configs(devices):
    """STT + SOAP — 전체 벤치마크."""
    configs = build_quick_configs(devices)

    if "cuda" in devices:
        configs += [
            {
                "label": "   medium | CUDA | Qwen2.5-3B (GPU)",
                "whisper": "medium",
                "device": "cuda",
                "llm_name": "Qwen/Qwen2.5-3B-Instruct",
            },
            {
                "label": "large-v3 | CUDA | Qwen2.5-3B (GPU)",
                "whisper": "large-v3",
                "device": "cuda",
                "llm_name": "Qwen/Qwen2.5-3B-Instruct",
            },
            {
                "label": "large-v3 | CUDA | Qwen2.5-7B (GPU)",
                "whisper": "large-v3",
                "device": "cuda",
                "llm_name": "Qwen/Qwen2.5-7B-Instruct",
            },
        ]

    if "cpu" in devices:
        configs += [
            {
                "label": "    small | CPU  | GGUF 1.5B",
                "whisper": "small",
                "device": "cpu",
                "cpu_llm": "1.5B",
            },
            {
                "label": "   medium | CPU  | GGUF 3B",
                "whisper": "medium",
                "device": "cpu",
                "cpu_llm": "3B",
            },
        ]

    return configs


# ─────────────────────────────────────────────────────────────
# 벤치마크 실행 — Warm-up + 정확한 시간 측정
# ─────────────────────────────────────────────────────────────
def run_one_config(config, samples, results_list, save_partial_fn):
    import baseline as bl

    print(f"\n{'='*70}")
    print(f"  ▶ {config['label']}")
    print(f"{'='*70}")

    free_memory()

    try:
        # ─── Step 1. Processor 생성 ───
        processor = bl.Processor(
            whisper_model=config["whisper"],
            device=config["device"],
            llm_name=config.get("llm_name", "Qwen/Qwen2.5-7B-Instruct"),
            cpu_llm=config.get("cpu_llm"),
            skip_llm=config.get("skip_llm", False),
        )

        # ─── Step 2. 모델 명시적 사전 로딩 (시간 측정) ───
        print(f"  📦 모델 로딩...")
        t_load_start = time.time()

        # Whisper 모델 로드 트리거
        _ = processor.whisper
        whisper_load_sec = time.time() - t_load_start

        # LLM 모델 로드 (있는 경우)
        llm_load_sec = 0
        if not config.get("skip_llm", False):
            t_llm_load = time.time()
            processor._ensure_llm()
            llm_load_sec = time.time() - t_llm_load

        # KeyBERT 모델 로드
        t_kw = time.time()
        _ = processor.kw_model
        kw_load_sec = time.time() - t_kw

        total_load_sec = whisper_load_sec + llm_load_sec + kw_load_sec
        print(f"     Whisper: {whisper_load_sec:.2f}초 / "
              f"LLM: {llm_load_sec:.2f}초 / "
              f"KeyBERT: {kw_load_sec:.2f}초 "
              f"→ 총 로딩: {total_load_sec:.2f}초")

        # ─── Step 3. Warm-up 호출 (첫 sample 1회, 측정 X) ───
        if samples:
            print(f"  🔥 워밍업 ({Path(samples[0]).name})...")
            t_warm = time.time()
            _ = processor.process_file(str(samples[0]))
            warmup_sec = time.time() - t_warm
            print(f"     워밍업 완료: {warmup_sec:.2f}초")

        # ─── Step 4. 본 측정 (모든 sample, warm 상태) ───
        for j, sample in enumerate(samples):
            sample_name = Path(sample).name
            print(f"\n  [{j+1}/{len(samples)}] {sample_name}")

            t_total_start = time.time()
            result = processor.process_file(str(sample))
            total_sec = time.time() - t_total_start

            entry = {
                "config_label": config["label"],
                "whisper_model": config["whisper"],
                "device": config["device"],
                "llm": (
                    "transformers " + config.get("llm_name", "").split("/")[-1]
                    if config.get("llm_name") and not config.get("skip_llm") and not config.get("cpu_llm")
                    else ("GGUF " + config["cpu_llm"]) if config.get("cpu_llm")
                    else "(none)"
                ),
                "audio": sample_name,
                "audio_sec": result.get("stt_meta", {}).get("duration_sec", 0),
                "stt_sec": result.get("stt_meta", {}).get("processing_sec", 0),
                "stt_rtf": result.get("stt_meta", {}).get("rtf", 0),
                "llm_sec": (
                    result.get("llm_meta", {}).get("gen_sec")
                    or result.get("llm_meta", {}).get("llm_sec")
                    or 0
                ),
                "warm_total_sec": round(total_sec, 2),  # 워밍업 후 정확한 시간
                "model_load_sec": round(total_load_sec, 2),  # 1회만 측정됨
                "whisper_load_sec": round(whisper_load_sec, 2),
                "llm_load_sec": round(llm_load_sec, 2),
                "status": result.get("status", "ok"),
                "transcription": result.get("transcription", ""),
                "keywords": result.get("keywords", []),
                "categories": result.get("categories", {}),
                "soap": result.get("soap"),
                "error": result.get("error"),
            }

            # 출력 표시
            print(f"     음성 {entry['audio_sec']:.1f}초 → "
                  f"STT {entry['stt_sec']:.2f}초 (RTF {entry['stt_rtf']:.3f}) "
                  + (f"+ LLM {entry['llm_sec']:.2f}초" if entry['llm_sec'] > 0 else "")
                  + f" = 처리 {entry['warm_total_sec']:.2f}초")
            print(f"     텍스트: {entry['transcription'][:60]}"
                  + ("..." if len(entry['transcription']) > 60 else ""))

            results_list.append(entry)
            save_partial_fn(results_list)

        del processor

    except Exception as e:
        print(f"  ❌ 실패: {e}")
        for sample in samples:
            results_list.append({
                "config_label": config["label"],
                "whisper_model": config["whisper"],
                "device": config["device"],
                "llm": "(failed)",
                "audio": Path(sample).name,
                "status": "error",
                "error": str(e),
            })
        save_partial_fn(results_list)

    free_memory()


# ─────────────────────────────────────────────────────────────
# 리포트 — model_load_sec 컬럼 추가
# ─────────────────────────────────────────────────────────────
def save_csv(results, path):
    fieldnames = [
        "config_label", "whisper_model", "device", "llm",
        "audio", "audio_sec",
        "model_load_sec", "whisper_load_sec", "llm_load_sec",
        "stt_sec", "stt_rtf", "llm_sec", "warm_total_sec",
        "status", "error",
    ]
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            writer.writerow(r)


def save_json(results, path):
    serial = []
    for r in results:
        rr = dict(r)
        if "keywords" in rr and rr["keywords"]:
            rr["keywords"] = [[kw, float(s)] for kw, s in rr["keywords"]]
        serial.append(rr)
    path.write_text(json.dumps(serial, ensure_ascii=False, indent=2), encoding="utf-8")


def save_markdown(results, path, samples):
    lines = [
        f"# Voice EMR 벤치마크 리포트 (v2)",
        f"",
        f"**실행 시각**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**테스트 샘플**: " + ", ".join(Path(s).name for s in samples),
        f"**총 설정 수**: {len(set(r['config_label'] for r in results))}",
        f"",
        f"> 📌 **측정 방식 변경**: 모델 로딩 시간을 분리 측정하고, 워밍업 호출 1회 후 본 측정합니다.",
        f"> 따라서 **`warm_total_sec`이 실제 운영 환경에서의 처리 시간**입니다.",
        f"> 첫 실행 시 사용자가 기다리는 시간 = `model_load_sec` + `warm_total_sec`",
        f"",
        f"---",
        f"",
    ]

    # ─── 모델 로딩 시간 요약 (한 번만) ───
    lines.append(f"## 🔧 모델 로딩 시간 (1회성)\n")
    lines.append("설정마다 메모리에 모델 올리는 시간. **서버 재시작 시에만 발생**.\n")
    lines.append("| 설정 | Whisper 로드(초) | LLM 로드(초) | 총 로딩(초) |")
    lines.append("|------|-----------------|-------------|------------|")
    seen = set()
    for r in results:
        if r["config_label"] in seen or r["status"] != "ok":
            continue
        seen.add(r["config_label"])
        lines.append(
            f"| {r['config_label']} | "
            f"{r.get('whisper_load_sec', 0):.2f} | "
            f"{r.get('llm_load_sec', 0):.2f} | "
            f"**{r.get('model_load_sec', 0):.2f}** |"
        )
    lines.append("\n---\n")

    # ─── 실제 운영 시 처리 시간 (warm) ───
    lines.append(f"## ⚡ 실제 운영 시 처리 시간 (Warm)\n")
    lines.append("모델이 메모리에 이미 있을 때의 시간. **이게 사용자가 매 음성마다 기다리는 시간**.\n")

    samples_in_results = sorted(set(r["audio"] for r in results))
    for sample in samples_in_results:
        lines.append(f"### {sample}\n")
        lines.append("| 설정 | Whisper | Device | LLM | 음성(초) | STT(초) | RTF | LLM(초) | **처리(초)** |")
        lines.append("|------|---------|--------|-----|---------|---------|-----|---------|------------|")
        rows = [r for r in results if r["audio"] == sample]
        rows.sort(key=lambda x: x.get("warm_total_sec") or 999)
        for r in rows:
            if r["status"] == "error":
                lines.append(
                    f"| {r['config_label']} | {r['whisper_model']} | {r['device']} | "
                    f"{r['llm']} | - | - | - | - | ❌ {(r.get('error') or '')[:30]} |"
                )
            else:
                lines.append(
                    f"| {r['config_label']} | {r['whisper_model']} | {r['device']} | "
                    f"{r['llm']} | {r.get('audio_sec', 0):.1f} | "
                    f"{r.get('stt_sec', 0):.2f} | {r.get('stt_rtf', 0):.3f} | "
                    f"{r.get('llm_sec', 0):.2f} | **{r.get('warm_total_sec', 0):.2f}** |"
                )
        lines.append("")

    lines.append("---\n")

    # ─── STT 결과 비교 ───
    lines.append(f"## 📝 STT 텍스트 결과 비교\n")
    for sample in samples_in_results:
        lines.append(f"### {sample}\n")
        rows = [r for r in results if r["audio"] == sample and r["status"] == "ok"]
        for r in rows:
            lines.append(f"**[{r['config_label'].strip()}]**\n")
            lines.append(f"> {r.get('transcription', '')}\n")
        lines.append("---\n")

    # ─── SOAP 결과 비교 ───
    soap_rows = [r for r in results if r.get("soap") and r["status"] == "ok"]
    if soap_rows:
        lines.append(f"## 📋 SOAP 요약 결과 비교\n")
        for sample in samples_in_results:
            sample_soaps = [r for r in soap_rows if r["audio"] == sample]
            if not sample_soaps:
                continue
            lines.append(f"### {sample}\n")
            for r in sample_soaps:
                lines.append(f"#### [{r['config_label'].strip()}]\n")
                lines.append("```")
                lines.append(r["soap"])
                lines.append("```\n")
            lines.append("---\n")

    # ─── 통계 ───
    lines.append(f"## 📈 통계 요약\n")
    ok_results = [r for r in results if r["status"] == "ok"]
    err_results = [r for r in results if r["status"] != "ok"]
    lines.append(f"- 총 측정: {len(results)}건 (성공 {len(ok_results)}, 실패 {len(err_results)})")
    if ok_results:
        fastest = min(ok_results, key=lambda x: x.get("warm_total_sec", 999))
        slowest = max(ok_results, key=lambda x: x.get("warm_total_sec", 0))
        lines.append(f"\n- ⚡ **운영 시 가장 빠름**: {fastest['config_label'].strip()} on "
                     f"{fastest['audio']} — {fastest['warm_total_sec']:.2f}초")
        lines.append(f"- 🐢 **운영 시 가장 느림**: {slowest['config_label'].strip()} on "
                     f"{slowest['audio']} — {slowest['warm_total_sec']:.2f}초")
        if slowest['warm_total_sec'] > 0:
            ratio = slowest['warm_total_sec'] / max(fastest['warm_total_sec'], 0.01)
            lines.append(f"- 📊 속도 차이: **{ratio:.1f}배**")

        # 모델 로딩 통계
        max_load = max(ok_results, key=lambda x: x.get("model_load_sec", 0))
        lines.append(f"\n- 🔧 **가장 무거운 모델 로딩**: {max_load['config_label'].strip()} — "
                     f"{max_load.get('model_load_sec', 0):.2f}초 (1회만)")

    path.write_text("\n".join(lines), encoding="utf-8")


# ─────────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Voice EMR 벤치마크 v2 — Warm-up + 정확한 시간 측정",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--samples", nargs="+", required=True)
    parser.add_argument("--preset", choices=["quick", "full"], default="quick")
    parser.add_argument("--devices", nargs="+", default=None,
                        choices=["cpu", "cuda"])
    parser.add_argument("--output-dir", default="bench_results")
    args = parser.parse_args()

    if args.devices is None:
        devices = ["cpu"]
        try:
            import torch
            if torch.cuda.is_available():
                devices.append("cuda")
        except ImportError:
            pass
    else:
        devices = args.devices

    print(f"\n{'='*70}")
    print(f"  Voice EMR 벤치마크 v2 시작 (Warm-up 포함)")
    print(f"{'='*70}")
    print(f"  샘플: {len(args.samples)}개")
    print(f"  디바이스: {', '.join(devices)}")
    print(f"  프리셋: {args.preset}")

    valid_samples = []
    for s in args.samples:
        p = Path(s)
        if not p.exists():
            print(f"  ⚠ 파일 없음 (스킵): {s}")
            continue
        valid_samples.append(p)
    if not valid_samples:
        print("❌ 처리할 음성 파일 없음")
        sys.exit(1)

    configs = (build_quick_configs(devices) if args.preset == "quick"
               else build_full_configs(devices))

    print(f"  실행할 설정: {len(configs)}개\n")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "results.csv"
    json_path = out_dir / "results.json"
    md_path = out_dir / "report.md"

    def save_partial(results):
        try:
            save_csv(results, csv_path)
            save_json(results, json_path)
            save_markdown(results, md_path, valid_samples)
        except Exception as e:
            print(f"  (부분저장 실패: {e})")

    results = []
    t_overall = time.time()
    for i, config in enumerate(configs):
        print(f"\n>>> 진행 {i+1}/{len(configs)} (전체 {(time.time()-t_overall)/60:.1f}분 경과)")
        run_one_config(config, valid_samples, results, save_partial)

    elapsed_total = time.time() - t_overall

    save_csv(results, csv_path)
    save_json(results, json_path)
    save_markdown(results, md_path, valid_samples)

    print(f"\n{'='*70}")
    print(f"  ✅ 벤치마크 완료 — 총 {elapsed_total/60:.1f}분")
    print(f"{'='*70}")
    print(f"\n  📁 저장: {out_dir.resolve()}")
    print(f"     - results.csv   (엑셀)")
    print(f"     - report.md     (마크다운, 로딩·운영 시간 분리됨) ⭐")
    print(f"     - results.json  (raw 데이터)")
    print()


if __name__ == "__main__":
    main()
