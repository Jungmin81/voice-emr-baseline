#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Voice EMR Baseline — 음성 → 텍스트 → 주제어 추출 베이스라인 테스트

전이학습 진행 전 현재 일반 모델의 성능을 측정하기 위한 프로그램.

지원 기능:
- 단일 파일 처리
- 다중 파일 일괄 처리 (모델 한 번만 로드)
- 폴더 전체 일괄 처리
- 비교 리포트 자동 생성 (CSV/JSON/Markdown)

상업적 사용 가능한 라이선스만 사용:
- faster-whisper (MIT) + Whisper large-v3 (MIT)
- sentence-transformers (Apache 2.0) + multilingual MiniLM (Apache 2.0)
- KeyBERT (MIT)
- Qwen2.5-Instruct (Apache 2.0)
- llama-cpp-python (MIT) — CPU 최적화 백엔드

사용 예:
  # 단일 파일
  python baseline.py sample.wav

  # 여러 파일
  python baseline.py file1.wav file2.wav file3.wav --output-dir results/

  # 폴더 전체
  python baseline.py --folder samples/ --output-dir results/

  # CPU 최적화 (LLM 빠름)
  python baseline.py --folder samples/ --cpu-llm 3B

  # STT만 (가장 빠름)
  python baseline.py --folder samples/ --skip-llm
"""
import argparse
import csv
import json
import os
import sys
import textwrap
import time
from pathlib import Path
from typing import Optional


# ─────────────────────────────────────────────────────────────
# 의료 도메인 규칙 사전
# ─────────────────────────────────────────────────────────────
MEDICAL_LEXICON = {
    "증상": ["통증", "아프", "저리", "두통", "복통", "구토", "설사", "발열", "기침",
             "호흡곤란", "어지러", "현기증", "피로", "불면", "오심", "객담", "흉통",
             "혈변", "변비", "체중", "식욕"],
    "병력": ["고혈압", "당뇨", "간염", "결핵", "수술", "입원", "복용", "복약",
             "처방받", "치료받", "진단받", "병력", "기왕력", "과거력"],
    "검사": ["혈액", "엑스레이", "X-ray", "CT", "MRI", "초음파", "내시경",
             "심전도", "혈압", "혈당", "검사", "측정", "체크"],
    "처방": ["처방", "약", "복용", "하루", "복약", "투약", "용량", "용법",
             "정", "캡슐", "주사", "연고"],
    "계획": ["다음", "재방문", "경과", "관찰", "추적", "예약", "내원", "방문",
             "수술", "입원", "퇴원"],
    "환자정보": ["나이", "성별", "직업", "흡연", "음주", "알레르기", "가족력"],
}

SOAP_PROMPT = """당신은 한국 의료 차트 작성을 보조하는 AI입니다.
아래 의사-환자 진료 대화를 표준 SOAP 구조로 정리해주세요.
원문에 명시되지 않은 내용은 절대 추가하지 마세요. 추측·짐작 금지.
정보가 없는 항목은 "기재 없음"으로 적어주세요.

[대화 원문]
{text}

[SOAP 정리]
## S (Subjective)
- 주 증상:
- 발생 시기 / 기간:
- 동반 증상:

## O (Objective)
- 활력 징후:
- 신체 검사 / 관찰:
- 검사 결과:

## A (Assessment)
- 주 진단:
- 감별 진단:

## P (Plan)
- 처방:
- 추가 검사:
- 경과 관찰:
- 재방문:
"""


# ─────────────────────────────────────────────────────────────
# Processor — 모델을 한 번만 로드하고 여러 파일 재사용
# ─────────────────────────────────────────────────────────────
class Processor:
    """음성 → STT → 키워드 → SOAP 파이프라인.

    모델은 lazy 로딩이며 한 번 로드되면 같은 인스턴스에서 재사용됨.
    여러 파일 처리 시 매우 빠름 (모델 로딩 시간 1회만).
    """

    def __init__(
        self,
        whisper_model: str = "medium",
        device: str = "auto",
        language: str = "ko",
        llm_name: Optional[str] = None,
        cpu_llm: Optional[str] = None,
        skip_llm: bool = False,
        top_keywords: int = 12,
    ):
        self.whisper_model_name = whisper_model
        self.device = self._auto_device(device)
        self.language = language
        self.llm_name = llm_name or "Qwen/Qwen2.5-7B-Instruct"
        self.cpu_llm = cpu_llm
        self.skip_llm = skip_llm
        self.top_keywords = top_keywords

        # 모델 캐시
        self._whisper = None
        self._kw_model = None
        self._llm_bundle = None  # (tokenizer, model) 또는 llama Llama
        self._gguf_llm = None

        # CPU 환경에서 큰 LLM 자동 차단
        self._check_resources()

    # ─── 자동 디바이스 ───
    @staticmethod
    def _auto_device(device: str) -> str:
        if device != "auto":
            return device
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def _check_resources(self):
        """CPU 환경 + 큰 LLM이면 경고."""
        if self.skip_llm:
            return
        if self.device == "cpu" and not self.cpu_llm:
            # transformers + 7B on CPU = 메모리 부족 가능성 높음
            try:
                import psutil
                total_ram_gb = psutil.virtual_memory().total / 1024**3
            except Exception:
                total_ram_gb = 16  # 알 수 없으면 가정

            if "7B" in self.llm_name and total_ram_gb < 32:
                print(f"\n⚠️  경고: CPU 환경에서 {self.llm_name} 사용 — "
                      f"메모리 부족 가능성 높음 (RAM {total_ram_gb:.0f}GB)")
                print(f"   다음 옵션 권장:")
                print(f"   - CPU 최적화 GGUF:  --cpu-llm 3B   (메모리 ~2.5GB, 빠름)")
                print(f"   - 작은 transformers: --llm Qwen/Qwen2.5-1.5B-Instruct")
                print(f"   - LLM 건너뜀:        --skip-llm")
                print()

    # ─── Whisper (lazy) ───
    @property
    def whisper(self):
        if self._whisper is None:
            from faster_whisper import WhisperModel
            compute = "int8" if self.device == "cpu" else "float16"
            print(f"  [STT] 모델 로딩 ({self.whisper_model_name}, "
                  f"device={self.device}, compute={compute})...")
            t0 = time.time()
            self._whisper = WhisperModel(
                self.whisper_model_name,
                device=self.device,
                compute_type=compute,
            )
            print(f"  [STT] 모델 로드 완료 ({time.time()-t0:.1f}초)")
        return self._whisper

    # ─── KeyBERT (lazy) ───
    @property
    def kw_model(self):
        if self._kw_model is None:
            from keybert import KeyBERT
            from sentence_transformers import SentenceTransformer
            print(f"  [KW] 임베딩 모델 로딩...")
            t0 = time.time()
            embed = SentenceTransformer(
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            )
            self._kw_model = KeyBERT(model=embed)
            print(f"  [KW] 모델 로드 완료 ({time.time()-t0:.1f}초)")
        return self._kw_model

    # ─── LLM (lazy) ───
    def _ensure_llm(self):
        if self.skip_llm:
            return None

        if self.cpu_llm:
            # GGUF 백엔드
            if self._gguf_llm is None:
                from llm_cpu import load_gguf_llm
                print(f"  [LLM] GGUF 로딩 ({self.cpu_llm})...")
                self._gguf_llm = load_gguf_llm(self.cpu_llm)
            return self._gguf_llm

        # transformers 백엔드
        if self._llm_bundle is None:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch

            dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
            print(f"  [LLM] transformers 로딩 ({self.llm_name}, device={self.device})...")
            t0 = time.time()
            tokenizer = AutoTokenizer.from_pretrained(self.llm_name)
            model = AutoModelForCausalLM.from_pretrained(
                self.llm_name,
                torch_dtype=dtype,
                device_map="auto" if self.device == "cuda" else None,
                low_cpu_mem_usage=True,
            )
            if self.device == "cpu":
                model = model.to("cpu")
            self._llm_bundle = (tokenizer, model)
            print(f"  [LLM] 모델 로드 완료 ({time.time()-t0:.1f}초)")
        return self._llm_bundle

    # ─── 파이프라인 단계 ───
    def _stt(self, audio_path: str):
        segments, info = self.whisper.transcribe(
            audio_path,
            language=self.language,
            beam_size=5,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
        )
        t0 = time.time()
        parts = []
        timeline = []
        for seg in segments:
            parts.append(seg.text.strip())
            timeline.append({
                "start": round(seg.start, 2),
                "end": round(seg.end, 2),
                "text": seg.text.strip(),
            })
        elapsed = time.time() - t0
        duration = info.duration
        rtf = elapsed / duration if duration > 0 else 0
        text = " ".join(parts)
        meta = {
            "duration_sec": round(duration, 2),
            "processing_sec": round(elapsed, 2),
            "rtf": round(rtf, 3),
            "language": info.language,
            "language_probability": round(info.language_probability, 3),
        }
        return text, timeline, meta

    def _keywords(self, text: str):
        if len(text.strip()) < 10:
            return []
        return self.kw_model.extract_keywords(
            text,
            keyphrase_ngram_range=(1, 3),
            top_n=self.top_keywords,
            use_mmr=True,
            diversity=0.5,
            stop_words=None,
        )

    def _categories(self, text: str):
        found = {}
        for category, keywords in MEDICAL_LEXICON.items():
            matched = sorted({kw for kw in keywords if kw in text})
            if matched:
                found[category] = matched
        return found

    def _soap(self, text: str):
        llm = self._ensure_llm()
        if llm is None:
            return None, {}

        if self.cpu_llm:
            # GGUF chat
            from llm_cpu import run_soap_gguf
            return run_soap_gguf(llm, text, SOAP_PROMPT)

        # transformers
        tokenizer, model = llm
        import torch

        prompt = SOAP_PROMPT.format(text=text)
        messages = [{"role": "user", "content": prompt}]
        inputs = tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True
        ).to(model.device)

        t0 = time.time()
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=1024,
                temperature=0.2,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        elapsed = time.time() - t0
        response = tokenizer.decode(
            outputs[0][inputs.shape[1]:], skip_special_tokens=True
        )
        return response.strip(), {"llm_sec": round(elapsed, 2),
                                   "backend": "transformers"}

    # ─── 한 파일 처리 ───
    def process_file(self, audio_path) -> dict:
        audio_path = Path(audio_path)
        try:
            text, timeline, stt_meta = self._stt(str(audio_path))
            kws = self._keywords(text)
            cats = self._categories(text)
            soap, llm_meta = (None, {})
            if not self.skip_llm:
                try:
                    soap, llm_meta = self._soap(text)
                except Exception as e:
                    soap = f"[LLM 실패: {e}]"
                    llm_meta = {"error": str(e)}
            return {
                "audio_path": str(audio_path),
                "audio_name": audio_path.name,
                "transcription": text,
                "timeline": timeline,
                "stt_meta": stt_meta,
                "keywords": [(kw, float(s)) for kw, s in kws],
                "categories": cats,
                "soap": soap,
                "llm_meta": llm_meta,
                "status": "ok",
            }
        except Exception as e:
            return {
                "audio_path": str(audio_path),
                "audio_name": audio_path.name,
                "status": "error",
                "error": str(e),
            }

    # ─── 배치 처리 ───
    def process_batch(self, audio_paths) -> list:
        results = []
        n = len(audio_paths)
        for i, path in enumerate(audio_paths, 1):
            print(f"\n{'='*70}")
            print(f"  [{i}/{n}] {Path(path).name}")
            print(f"{'='*70}")
            r = self.process_file(path)
            results.append(r)
            self._print_quick_summary(r)
        return results

    @staticmethod
    def _print_quick_summary(r: dict):
        if r["status"] != "ok":
            print(f"  ❌ 실패: {r.get('error', '')}")
            return
        m = r["stt_meta"]
        print(f"  음성 {m['duration_sec']}초 → 처리 {m['processing_sec']}초 "
              f"(RTF {m['rtf']}x)")
        snippet = r["transcription"][:80] + ("..." if len(r["transcription"]) > 80 else "")
        print(f"  텍스트: {snippet}")
        if r["keywords"]:
            top3 = ", ".join(kw for kw, _ in r["keywords"][:3])
            print(f"  키워드(상위3): {top3}")
        if r["categories"]:
            cats = ", ".join(r["categories"].keys())
            print(f"  카테고리: {cats}")


# ─────────────────────────────────────────────────────────────
# 리포트 생성 — CSV, JSON, Markdown
# ─────────────────────────────────────────────────────────────
def save_csv_summary(results: list, path: Path):
    """모든 파일의 요약을 CSV로 저장."""
    fieldnames = [
        "file", "status", "duration_sec", "processing_sec", "rtf",
        "language", "lang_prob", "text_chars",
        "top_keywords", "categories",
        "has_soap",
    ]
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            if r["status"] != "ok":
                writer.writerow({
                    "file": r["audio_name"],
                    "status": "error",
                })
                continue
            m = r["stt_meta"]
            top_kws = " | ".join(kw for kw, _ in r["keywords"][:5])
            cats = ", ".join(r["categories"].keys())
            writer.writerow({
                "file": r["audio_name"],
                "status": "ok",
                "duration_sec": m["duration_sec"],
                "processing_sec": m["processing_sec"],
                "rtf": m["rtf"],
                "language": m["language"],
                "lang_prob": m["language_probability"],
                "text_chars": len(r["transcription"]),
                "top_keywords": top_kws,
                "categories": cats,
                "has_soap": "Y" if r.get("soap") else "N",
            })


def save_json_full(results: list, path: Path):
    """전체 결과를 JSON으로 저장."""
    path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )


def save_markdown_report(results: list, path: Path):
    """비교 가능한 마크다운 리포트."""
    lines = ["# Voice EMR Baseline — 일괄 처리 리포트\n"]
    lines.append(f"총 **{len(results)}**개 파일 처리\n")
    ok_count = sum(1 for r in results if r["status"] == "ok")
    err_count = len(results) - ok_count
    lines.append(f"- 성공: {ok_count}\n- 실패: {err_count}\n")
    lines.append("\n---\n")

    # 요약 표
    lines.append("## 📊 요약 비교표\n")
    lines.append("| # | 파일 | 음성(초) | 처리(초) | RTF | 텍스트 | 상위 키워드 | 카테고리 |")
    lines.append("|---|------|---------|---------|-----|--------|-------------|----------|")
    for i, r in enumerate(results, 1):
        if r["status"] != "ok":
            lines.append(f"| {i} | {r['audio_name']} | - | - | - | ❌ 오류 | - | - |")
            continue
        m = r["stt_meta"]
        text_preview = r["transcription"][:30].replace("|", " ").replace("\n", " ")
        if len(r["transcription"]) > 30:
            text_preview += "…"
        top_kws = ", ".join(kw for kw, _ in r["keywords"][:3])
        cats = ", ".join(r["categories"].keys()) or "-"
        lines.append(
            f"| {i} | {r['audio_name']} | {m['duration_sec']} | "
            f"{m['processing_sec']} | {m['rtf']} | {text_preview} | "
            f"{top_kws} | {cats} |"
        )

    lines.append("\n---\n")

    # 파일별 상세
    lines.append("## 📁 파일별 상세\n")
    for i, r in enumerate(results, 1):
        lines.append(f"### [{i}] {r['audio_name']}\n")
        if r["status"] != "ok":
            lines.append(f"**❌ 오류**: {r.get('error', '')}\n")
            continue
        m = r["stt_meta"]
        lines.append(
            f"- 음성 길이: **{m['duration_sec']}초** / "
            f"처리: {m['processing_sec']}초 (RTF {m['rtf']}x)\n"
            f"- 언어 감지: {m['language']} ({m['language_probability']*100:.1f}%)\n"
        )
        lines.append(f"\n**전사 텍스트:**\n")
        lines.append(f"> {r['transcription']}\n")

        lines.append(f"\n**키워드 (Top {len(r['keywords'])}):**\n")
        for kw, sc in r["keywords"]:
            lines.append(f"- `{kw}` ({sc:.3f})")
        lines.append("")

        if r["categories"]:
            lines.append(f"\n**의료 카테고리:**\n")
            for cat, ws in r["categories"].items():
                lines.append(f"- **{cat}**: {', '.join(ws)}")
            lines.append("")

        if r.get("soap"):
            lines.append(f"\n**SOAP 요약:**\n")
            lines.append("```")
            lines.append(r["soap"])
            lines.append("```\n")

        lines.append("\n---\n")

    path.write_text("\n".join(lines), encoding="utf-8")


# ─────────────────────────────────────────────────────────────
# 파일 수집 헬퍼
# ─────────────────────────────────────────────────────────────
AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac", ".wma", ".opus"}


def collect_audio_files(paths_or_files):
    """파일 경로 또는 폴더 경로를 받아서 오디오 파일 리스트 반환."""
    files = []
    for p in paths_or_files:
        p = Path(p)
        if p.is_dir():
            for f in sorted(p.rglob("*")):
                if f.suffix.lower() in AUDIO_EXTS:
                    files.append(f)
        elif p.is_file():
            if p.suffix.lower() in AUDIO_EXTS:
                files.append(p)
            else:
                print(f"⚠️  지원하지 않는 형식: {p}")
        else:
            print(f"⚠️  파일 없음: {p}")
    return files


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Voice EMR Baseline — STT + 주제어 추출 (다중 파일 지원)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        예시:
          # 단일 파일
          python baseline.py audio.wav

          # 여러 파일
          python baseline.py a.wav b.wav c.wav

          # 폴더 일괄
          python baseline.py --folder samples/ --output-dir results/

          # CPU 환경 (LLM 빠름)
          python baseline.py --folder samples/ --cpu-llm 3B

          # STT 정확도만 (베이스라인 측정용)
          python baseline.py --folder samples/ --skip-llm
        """),
    )
    parser.add_argument("audio", nargs="*",
                        help="음성 파일들 (여러 개 가능)")
    parser.add_argument("--folder", help="폴더 안 모든 음성 파일 일괄 처리")

    parser.add_argument("--whisper-model", default="medium",
                        choices=["tiny", "base", "small", "medium", "large-v3"],
                        help="Whisper 모델 크기 (기본: medium)")
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cpu", "cuda"])
    parser.add_argument("--language", default="ko")

    parser.add_argument("--llm", default="Qwen/Qwen2.5-7B-Instruct",
                        help="transformers LLM (GPU 환경)")
    parser.add_argument("--cpu-llm", default=None,
                        choices=["1.5B", "3B", "7B"],
                        help="CPU 최적화 GGUF LLM (CPU 환경 권장)")
    parser.add_argument("--skip-llm", action="store_true",
                        help="LLM 요약 건너뜀 (가장 빠름)")
    parser.add_argument("--top-keywords", type=int, default=12)

    parser.add_argument("--output-dir", default=None,
                        help="결과 저장 폴더 (CSV/JSON/MD 자동 생성)")

    args = parser.parse_args()

    # 파일 수집
    inputs = list(args.audio)
    if args.folder:
        inputs.append(args.folder)
    if not inputs:
        parser.error("음성 파일 또는 --folder 지정 필요")

    audio_files = collect_audio_files(inputs)
    if not audio_files:
        print("❌ 처리할 음성 파일이 없습니다.")
        sys.exit(1)

    print(f"📂 처리 대상: {len(audio_files)}개 파일")
    for f in audio_files:
        print(f"   - {f}")

    # Processor 생성
    print(f"\n🚀 모델 로딩 (한 번만 수행)...")
    processor = Processor(
        whisper_model=args.whisper_model,
        device=args.device,
        language=args.language,
        llm_name=args.llm,
        cpu_llm=args.cpu_llm,
        skip_llm=args.skip_llm,
        top_keywords=args.top_keywords,
    )

    # 배치 처리
    t_total = time.time()
    results = processor.process_batch([str(f) for f in audio_files])
    total_sec = time.time() - t_total

    # 통계
    print(f"\n{'='*70}")
    print(f"  ✅ 완료 — 총 {len(results)}개 파일, {total_sec:.1f}초 소요")
    print(f"{'='*70}")
    ok = [r for r in results if r["status"] == "ok"]
    if ok:
        total_audio = sum(r["stt_meta"]["duration_sec"] for r in ok)
        total_proc = sum(r["stt_meta"]["processing_sec"] for r in ok)
        avg_rtf = total_proc / total_audio if total_audio > 0 else 0
        print(f"  📊 STT 통계:")
        print(f"     - 처리한 음성 총 길이: {total_audio:.1f}초")
        print(f"     - 평균 RTF: {avg_rtf:.3f}x")

    # 결과 저장
    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        save_csv_summary(results, out_dir / "summary.csv")
        save_json_full(results, out_dir / "all_results.json")
        save_markdown_report(results, out_dir / "report.md")

        print(f"\n💾 저장 완료:")
        print(f"   - {out_dir / 'summary.csv'}     (엑셀 호환)")
        print(f"   - {out_dir / 'all_results.json'}  (전체 데이터)")
        print(f"   - {out_dir / 'report.md'}        (비교 리포트)")
    else:
        print(f"\n💡 결과 저장하려면: --output-dir results/")


# ─────────────────────────────────────────────────────────────
# 하위 호환 함수 (app.py에서 사용)
# ─────────────────────────────────────────────────────────────
def transcribe(audio_path, model_size="medium", device="auto",
               compute_type="auto", language="ko"):
    p = Processor(whisper_model=model_size, device=device,
                  language=language, skip_llm=True)
    text, timeline, meta = p._stt(audio_path)
    return text, timeline, meta


def extract_keywords(text, top_k=12):
    p = Processor(skip_llm=True, top_keywords=top_k)
    return p._keywords(text)


def categorize_topics(text):
    found = {}
    for category, keywords in MEDICAL_LEXICON.items():
        matched = sorted({kw for kw in keywords if kw in text})
        if matched:
            found[category] = matched
    return found


def summarize_to_soap(text, model_name="Qwen/Qwen2.5-7B-Instruct",
                     device="auto", max_new_tokens=1024):
    p = Processor(llm_name=model_name, device=device)
    return p._soap(text)


if __name__ == "__main__":
    main()
