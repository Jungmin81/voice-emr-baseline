#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Voice EMR Baseline — 벤치마크 스크립트

여러 설정(Whisper 모델 크기 × CPU/GPU × LLM 옵션)을 자동으로 돌려서
처리 시간 + 추론 결과 + SOAP 결과를 한 번에 비교.

사용:
    # 빠른 벤치마크 (STT만, 7개 설정)
    python benchmark.py --samples samples/sample_short.wav samples/sample_long.wav

    # 전체 벤치마크 (SOAP 포함, 12개 설정)
    python benchmark.py --samples samples/sample_short.wav samples/sample_long.wav --preset full

    # 커스텀 모델만
    python benchmark.py --samples samples/*.wav --models small medium large-v3 --devices cuda

출력:
    bench_results/
      ├── results.csv       (엑셀용 요약표)
      ├── results.json      (전체 데이터)
      └── report.md         (마크다운 비교 리포트, 모든 결과 포함)
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
    """모델 사이 메모리 정리."""
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
    """STT만 — 빠른 벤치마크 (LLM 없음)."""
    configs = []
    models = ["tiny", "small", "medium"]
    if "cuda" in devices:
        models.append("large-v3")  # large는 GPU에서만 (CPU에선 너무 느림)
    for model in models:
        for device in devices:
            # large-v3는 CPU에서 너무 느려 제외
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

    # GPU + transformers LLM 조합 (RTX 3060 12GB에 맞춤)
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
        ]

    # CPU + GGUF 조합
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
# 벤치마크 실행
# ─────────────────────────────────────────────────────────────
def run_one_config(config, samples, results_list, save_partial_fn):
    """한 설정으로 모든 샘플 처리."""
    import baseline as bl

    print(f"\n{'='*70}")
    print(f"  ▶ {config['label']}")
    print(f"{'='*70}")

    free_memory()

    try:
        # Processor 생성
        t_load_start = time.time()
        processor = bl.Processor(
            whisper_model=config["whisper"],
            device=config["device"],
            llm_name=config.get("llm_name", "Qwen/Qwen2.5-7B-Instruct"),
            cpu_llm=config.get("cpu_llm"),
            skip_llm=config.get("skip_llm", False),
        )

        # 첫 샘플 처리 시 모델 lazy 로드되므로 그 시간도 측정에 포함
        for j, sample in enumerate(samples):
            sample_name = Path(sample).name
            print(f"\n  [{j+1}/{len(samples)}] {sample_name}")

            t_total_start = time.time()
            result = processor.process_file(str(sample))
            total_sec = time.time() - t_total_start

            # 결과 정리
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
                "total_sec": round(total_sec, 2),
                "status": result.get("status", "ok"),
                "transcription": result.get("transcription", ""),
                "keywords": result.get("keywords", []),
                "categories": result.get("categories", {}),
                "soap": result.get("soap"),
                "error": result.get("error"),
            }

            print(f"     음성 {entry['audio_sec']:.1f}초 → "
                  f"STT {entry['stt_sec']:.1f}초 (RTF {entry['stt_rtf']}) "
                  + (f"+ LLM {entry['llm_sec']:.1f}초" if entry['llm_sec'] > 0 else "")
                  + f" = 총 {entry['total_sec']:.1f}초")
            print(f"     텍스트: {entry['transcription'][:60]}"
                  + ("..." if len(entry['transcription']) > 60 else ""))

            results_list.append(entry)
            save_partial_fn(results_list)  # 매번 저장 (크래시 대비)

        # Processor 정리
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
# 리포트 생성
# ─────────────────────────────────────────────────────────────
def save_csv(results, path):
    fieldnames = [
        "config_label", "whisper_model", "device", "llm",
        "audio", "audio_sec", "stt_sec", "stt_rtf", "llm_sec", "total_sec",
        "status", "error",
    ]
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            writer.writerow(r)


def save_json(results, path):
    # keywords의 tuple을 list로 변환
    serial = []
    for r in results:
        rr = dict(r)
        if "keywords" in rr and rr["keywords"]:
            rr["keywords"] = [[kw, float(s)] for kw, s in rr["keywords"]]
        serial.append(rr)
    path.write_text(json.dumps(serial, ensure_ascii=False, indent=2), encoding="utf-8")


def save_markdown(results, path, samples):
    lines = [
        f"# Voice EMR 벤치마크 리포트",
        f"",
        f"**실행 시각**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**테스트 샘플**: " + ", ".join(Path(s).name for s in samples),
        f"**총 설정 수**: {len(set(r['config_label'] for r in results))}",
        f"",
        f"---",
        f"",
    ]

    # ─── 1. 처리 시간 비교표 (샘플별) ───
    lines.append(f"## 📊 처리 시간 비교\n")

    samples_in_results = sorted(set(r["audio"] for r in results))
    for sample in samples_in_results:
        lines.append(f"### {sample}\n")
        lines.append("| 설정 | Whisper | Device | LLM | 음성(초) | STT(초) | RTF | LLM(초) | **총(초)** | 상태 |")
        lines.append("|------|---------|--------|-----|---------|---------|-----|---------|----------|------|")
        rows = [r for r in results if r["audio"] == sample]
        # 총 시간 순 정렬
        rows.sort(key=lambda x: x.get("total_sec") or 999)
        for r in rows:
            if r["status"] == "error":
                lines.append(
                    f"| {r['config_label']} | {r['whisper_model']} | {r['device']} | "
                    f"{r['llm']} | - | - | - | - | - | ❌ {(r.get('error') or '')[:30]} |"
                )
            else:
                lines.append(
                    f"| {r['config_label']} | {r['whisper_model']} | {r['device']} | "
                    f"{r['llm']} | {r.get('audio_sec', 0):.1f} | "
                    f"{r.get('stt_sec', 0):.2f} | {r.get('stt_rtf', 0):.3f} | "
                    f"{r.get('llm_sec', 0):.2f} | **{r.get('total_sec', 0):.2f}** | ✅ |"
                )
        lines.append("")

    lines.append("---\n")

    # ─── 2. STT 결과 비교 (음성별) ───
    lines.append(f"## 📝 STT 텍스트 결과 비교\n")

    for sample in samples_in_results:
        lines.append(f"### {sample}\n")
        rows = [r for r in results if r["audio"] == sample and r["status"] == "ok"]
        for r in rows:
            lines.append(f"**[{r['config_label'].strip()}]**\n")
            lines.append(f"> {r.get('transcription', '')}\n")

            if r.get("keywords"):
                top_kws = ", ".join(f"`{kw}`" for kw, _ in r["keywords"][:5])
                lines.append(f"- 키워드(Top5): {top_kws}")
            if r.get("categories"):
                cats = ", ".join(r["categories"].keys())
                lines.append(f"- 카테고리: {cats}")
            lines.append("")
        lines.append("---\n")

    # ─── 3. SOAP 결과 비교 (있는 것만) ───
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

    # ─── 4. 통계 요약 ───
    lines.append(f"## 📈 통계 요약\n")
    ok_results = [r for r in results if r["status"] == "ok"]
    err_results = [r for r in results if r["status"] != "ok"]
    lines.append(f"- 총 측정: {len(results)}건")
    lines.append(f"- 성공: {len(ok_results)}건")
    lines.append(f"- 실패: {len(err_results)}건")
    if ok_results:
        fastest = min(ok_results, key=lambda x: x.get("total_sec", 999))
        slowest = max(ok_results, key=lambda x: x.get("total_sec", 0))
        lines.append(f"\n- ⚡ **가장 빠름**: {fastest['config_label'].strip()} on "
                     f"{fastest['audio']} — {fastest['total_sec']:.2f}초")
        lines.append(f"- 🐢 **가장 느림**: {slowest['config_label'].strip()} on "
                     f"{slowest['audio']} — {slowest['total_sec']:.2f}초")
        if slowest['total_sec'] > 0:
            ratio = slowest['total_sec'] / max(fastest['total_sec'], 0.01)
            lines.append(f"- 📊 속도 차이: **{ratio:.1f}배**")

    path.write_text("\n".join(lines), encoding="utf-8")


# ─────────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Voice EMR 벤치마크 — 여러 설정 자동 비교",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--samples", nargs="+", required=True,
                        help="테스트할 음성 파일들")
    parser.add_argument("--preset", choices=["quick", "full"], default="quick",
                        help="quick: STT만 (빠름), full: SOAP 포함 (느림)")
    parser.add_argument("--devices", nargs="+", default=None,
                        choices=["cpu", "cuda"],
                        help="테스트할 디바이스 (기본: 사용 가능한 거 자동)")
    parser.add_argument("--output-dir", default="bench_results",
                        help="결과 저장 폴더 (기본: bench_results)")
    args = parser.parse_args()

    # 디바이스 자동 결정
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
    print(f"  Voice EMR 벤치마크 시작")
    print(f"{'='*70}")
    print(f"  샘플: {len(args.samples)}개")
    for s in args.samples:
        print(f"    - {s}")
    print(f"  디바이스: {', '.join(devices)}")
    print(f"  프리셋: {args.preset}")

    # 샘플 파일 검증
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

    # 설정 빌드
    configs = (build_quick_configs(devices) if args.preset == "quick"
               else build_full_configs(devices))

    # 예상 시간 안내
    print(f"\n  실행할 설정: {len(configs)}개")
    print(f"  ⏱  예상 소요: {len(configs) * 1.5:.0f}~{len(configs) * 5:.0f}분")
    print(f"     (모델 로드 + 추론 시간 포함, 첫 다운로드 별도)")
    print()

    # 출력 폴더
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "results.csv"
    json_path = out_dir / "results.json"
    md_path = out_dir / "report.md"

    # 부분 저장 함수
    def save_partial(results):
        try:
            save_csv(results, csv_path)
            save_json(results, json_path)
            save_markdown(results, md_path, valid_samples)
        except Exception as e:
            print(f"  (부분저장 실패: {e})")

    # 벤치마크 실행
    results = []
    t_overall = time.time()
    for i, config in enumerate(configs):
        print(f"\n>>> 진행 {i+1}/{len(configs)} (전체 {(time.time()-t_overall)/60:.1f}분 경과)")
        run_one_config(config, valid_samples, results, save_partial)

    elapsed_total = time.time() - t_overall

    # 최종 저장
    save_csv(results, csv_path)
    save_json(results, json_path)
    save_markdown(results, md_path, valid_samples)

    # 완료
    print(f"\n{'='*70}")
    print(f"  ✅ 벤치마크 완료 — 총 {elapsed_total/60:.1f}분")
    print(f"{'='*70}")
    print(f"\n  📁 저장 위치: {out_dir.resolve()}")
    print(f"     - results.csv   (엑셀에서 열기)")
    print(f"     - report.md     (마크다운, 모든 결과 포함) ⭐")
    print(f"     - results.json  (프로그래밍용 raw 데이터)")
    print()


if __name__ == "__main__":
    main()
