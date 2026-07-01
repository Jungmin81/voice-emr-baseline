#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
전이학습 전 베이스라인 STT 성능 통합 리포트 생성기.

입력:
  - bench_results_hub/results.json   (AI Hub 실데이터, config×파일)
  - sample_bench_results/results.json (합성 sample_long/short, 모델×파일)
  - samples/                          (정답 .txt)

출력:
  - STT_BASELINE_REPORT.md            (모델별 + 길이별 통합 리포트)

길이 카테고리: 파일명 끝 토큰 (10s / 30s / 60s / 180s)
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "training"))
from utils import cer, normalize_korean  # noqa: E402

ROOT = Path(__file__).parent
SPEAKER_TAG_RE = re.compile(r"\[[^\]]+\]")


def length_of(audio_name: str) -> str:
    """파일명에서 길이 카테고리 추출. 예 PA_0031_long_180s -> 180s"""
    return Path(audio_name).stem.split("_")[-1]


def find_gt(audio_name: str, strip_tags: bool = False):
    stem = Path(audio_name).stem
    matches = list((ROOT / "samples").rglob(f"{stem}.txt"))
    if not matches:
        return None
    text = matches[0].read_text(encoding="utf-8")
    if strip_tags:
        text = SPEAKER_TAG_RE.sub(" ", text)
    return text


def weighted(items):
    """char-weighted CER + 시간 집계."""
    total_ref = sum(i["ref_chars"] for i in items)
    total_audio = sum(i["audio_sec"] for i in items)
    total_stt = sum(i["stt_sec"] for i in items)
    return {
        "n": len(items),
        "cer": (sum(i["cer"] * i["ref_chars"] for i in items) / total_ref
                if total_ref else 0.0),
        "total_audio": total_audio,
        "total_stt": total_stt,
        "rtf": total_stt / total_audio if total_audio else 0.0,
    }


# ──────────────────────────────────────────────────────────────
# 1. AI Hub: STT-only config 만 사용 (모델별 순수 STT 비교)
# ──────────────────────────────────────────────────────────────
def analyze_aihub():
    data = json.loads((ROOT / "bench_results_hub" / "results.json").read_text("utf-8"))
    # whisper_model -> list of records (STT only config 만)
    per_model = defaultdict(list)
    for r in data:
        if r.get("status") != "ok":
            continue
        if "STT only" not in r["config_label"]:
            continue  # LLM config 는 전사 동일하므로 중복 제외
        gt = find_gt(r["audio"])
        if gt is None:
            continue
        ref = normalize_korean(gt).replace(" ", "")
        per_model[r["whisper_model"]].append({
            "audio": r["audio"],
            "length": length_of(r["audio"]),
            "audio_sec": float(r.get("audio_sec") or 0),
            "stt_sec": float(r.get("stt_sec") or 0),
            "ref_chars": len(ref),
            "cer": cer(gt, r.get("transcription") or ""),
        })
    return per_model


def analyze_samples():
    path = ROOT / "sample_bench_results" / "results.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text("utf-8"))
    per_model = defaultdict(list)
    for r in data:
        per_model[r["model"]].append(r)
    return per_model


# 모델 정렬 순서
MODEL_ORDER = ["tiny", "small", "medium", "large-v3"]
LEN_ORDER = ["10s", "30s", "60s", "180s"]


def sort_models(models):
    return sorted(models, key=lambda m: MODEL_ORDER.index(m)
                  if m in MODEL_ORDER else 99)


def build_report(aihub, samples):
    L = []
    L.append("# 전이학습 전 STT 베이스라인 성능 리포트\n")
    L.append("> 일반(미세조정 전) Whisper 모델의 한국어 의료 음성 STT 성능 측정.")
    L.append("> 향후 AI Hub 파인튜닝 후 동일 지표로 비교하기 위한 기준선.\n")
    L.append("**지표 정의**")
    L.append("- **CER** (Character Error Rate): 문자 가중 평균 `sum(오류문자)/sum(정답문자)`. 낮을수록 정확.")
    L.append("- **정확도**: `1 − CER`")
    L.append("- **RTF** (Real-Time Factor): `처리시간 / 음성길이`. 낮을수록 빠름.\n")
    L.append("---\n")

    # ── 1. AI Hub 모델별 종합 ──
    L.append("## 1. AI Hub 실데이터 — 모델별 종합\n")
    L.append("실제 환자 발화 18개 파일(총 음성 ~1,109초). STT-only 결과 기준.\n")
    L.append("| 모델 | CER | 정확도 | 평균 RTF | 총 처리(s) | 파일 |")
    L.append("|---|---:|---:|---:|---:|---:|")
    model_summary = {}
    for m in sort_models(aihub.keys()):
        s = weighted(aihub[m])
        model_summary[m] = s
        L.append(f"| **{m}** | {s['cer']*100:.2f}% | {(1-s['cer'])*100:.2f}% | "
                 f"{s['rtf']:.3f} | {s['total_stt']:.2f} | {s['n']} |")
    best = min(model_summary, key=lambda m: model_summary[m]["cer"])
    L.append(f"\n→ **최저 CER: `{best}` ({model_summary[best]['cer']*100:.2f}%)**\n")
    L.append("---\n")

    # ── 2. AI Hub 길이별 × 모델별 ──
    L.append("## 2. AI Hub 실데이터 — 길이별 × 모델별\n")
    L.append("음성 길이 카테고리(10s/30s/60s/180s)별 CER. 짧을수록 표본 문맥이 적어 변동 큼.\n")

    # 길이별 CER 표
    L.append("### 2-1. CER (%)\n")
    header = "| 모델 | " + " | ".join(LEN_ORDER) + " |"
    L.append(header)
    L.append("|---|" + "---:|" * len(LEN_ORDER))
    for m in sort_models(aihub.keys()):
        by_len = defaultdict(list)
        for i in aihub[m]:
            by_len[i["length"]].append(i)
        cells = []
        for ln in LEN_ORDER:
            if by_len.get(ln):
                cells.append(f"{weighted(by_len[ln])['cer']*100:.2f}%")
            else:
                cells.append("-")
        L.append(f"| **{m}** | " + " | ".join(cells) + " |")

    # 길이별 RTF 표
    L.append("\n### 2-2. 평균 RTF\n")
    L.append(header)
    L.append("|---|" + "---:|" * len(LEN_ORDER))
    for m in sort_models(aihub.keys()):
        by_len = defaultdict(list)
        for i in aihub[m]:
            by_len[i["length"]].append(i)
        cells = []
        for ln in LEN_ORDER:
            if by_len.get(ln):
                cells.append(f"{weighted(by_len[ln])['rtf']:.3f}")
            else:
                cells.append("-")
        L.append(f"| **{m}** | " + " | ".join(cells) + " |")

    # 길이별 파일 수
    counts = defaultdict(int)
    for i in next(iter(aihub.values())):
        counts[i["length"]] += 1
    L.append("\n*길이별 파일 수: " +
             ", ".join(f"{ln} = {counts.get(ln,0)}개" for ln in LEN_ORDER) + "*\n")
    L.append("> ⚠️ large-v3는 일부 60s 파일에서 환각·반복이 비결정적으로 발생해 "
             "60s 구간 CER이 불안정합니다(표본 small).\n")
    L.append("---\n")

    # ── 3. 합성 sample 결과 ──
    if samples:
        L.append("## 3. 합성 샘플 (sample_long / sample_short)\n")
        L.append("`generate_samples.py`로 만든 깨끗한 TTS 음성. 파이프라인 정상 동작 확인용.\n")
        # 파일별 표
        files = sorted({r["file"] for recs in samples.values() for r in recs})
        L.append("| 모델 | " + " | ".join(f.replace('.wav','') for f in files) +
                 " | (RTF long) |")
        L.append("|---|" + "---:|" * (len(files)+1))
        for m in sort_models(samples.keys()):
            recs = {r["file"]: r for r in samples[m]}
            cells = []
            rtf_long = "-"
            for f in files:
                if f in recs:
                    cells.append(f"{recs[f]['cer']*100:.2f}%")
                    if "long" in f:
                        rtf_long = f"{recs[f]['rtf']:.3f}"
                else:
                    cells.append("-")
            L.append(f"| **{m}** | " + " | ".join(cells) + f" | {rtf_long} |")
        L.append("\n*값 = CER. 마지막 열 = sample_long RTF.*\n")
        L.append("---\n")

    # ── 4. 대조 ──
    L.append("## 4. 실데이터 vs 합성 (medium 기준)\n")
    L.append("| 데이터 | medium CER | 성격 |")
    L.append("|---|---:|---|")
    if "medium" in model_summary:
        L.append(f"| AI Hub (실데이터) | {model_summary['medium']['cer']*100:.2f}% | "
                 f"실제 발화, 잡음·구어체 |")
    if samples.get("medium"):
        long_rec = next((r for r in samples["medium"] if "long" in r["file"]), None)
        if long_rec:
            L.append(f"| 합성 sample_long | {long_rec['cer']*100:.2f}% | "
                     f"깨끗한 TTS, 정형 문장 |")
    L.append("\n→ **현실적 임상 베이스라인은 AI Hub 수치**가 기준. "
             "합성 샘플은 상한(이상적 조건) 참고용.\n")
    L.append("---\n")

    # ── 5. 결론 ──
    L.append("## 5. 결론 / 권고\n")
    L.append(f"- **정확도·속도 균형**: `medium` 권장 "
             f"(AI Hub CER {model_summary.get('medium',{}).get('cer',0)*100:.2f}%, "
             f"RTF {model_summary.get('medium',{}).get('rtf',0):.3f}).")
    L.append("- `large-v3`는 더 무겁지만 실데이터에서 medium보다 정확하지 않고 불안정 → "
             "현 시점 비용 대비 이점 작음.")
    L.append("- `tiny`는 속도만 빠르고 정확도 부족, `small`은 경량 대안.")
    L.append("- 길이가 짧을수록(10s) CER 변동이 크므로 베이스라인 비교 시 "
             "길이 카테고리를 맞춰 비교할 것.")
    L.append("\n*생성: `make_report.py` / CER 함수 `training/utils.py` 재사용*")

    return "\n".join(L)


def main():
    aihub = analyze_aihub()
    samples = analyze_samples()
    report = build_report(aihub, samples)
    out = ROOT / "STT_BASELINE_REPORT.md"
    out.write_text(report, encoding="utf-8")
    print(f"💾 리포트 생성: {out}")
    print(f"   AI Hub 모델: {sorted(aihub.keys())}")
    print(f"   sample 모델: {sorted(samples.keys())}")


if __name__ == "__main__":
    main()
