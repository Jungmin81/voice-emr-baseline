#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Voice EMR Baseline — Web UI (Gradio, 다중 파일 지원)

브라우저에서 음성 파일들을 한꺼번에 드래그-드롭으로 테스트.
모델은 한 번만 로드되고 모든 파일에 재사용됨.

실행:  python app.py
접속:  http://localhost:7860
"""
import json
from pathlib import Path

import gradio as gr
import pandas as pd

import baseline as bl


# ─────────────────────────────────────────────────────────────
# 전역 Processor — 설정 바뀔 때만 재생성
# ─────────────────────────────────────────────────────────────
_processor = None
_last_config = None


def get_processor(whisper_size, run_llm, llm_backend, llm_model_size, top_keywords):
    """동일 설정이면 기존 Processor 재사용, 아니면 새로 생성."""
    global _processor, _last_config

    config = (whisper_size, run_llm, llm_backend, llm_model_size, top_keywords)
    if _processor is not None and _last_config == config:
        return _processor, "(캐시된 모델 재사용)"

    skip_llm = not run_llm
    cpu_llm = None
    llm_name = "Qwen/Qwen2.5-7B-Instruct"

    if run_llm and llm_backend == "CPU 최적화 (GGUF)":
        cpu_llm = llm_model_size

    _processor = bl.Processor(
        whisper_model=whisper_size,
        device="auto",
        llm_name=llm_name,
        cpu_llm=cpu_llm,
        skip_llm=skip_llm,
        top_keywords=int(top_keywords),
    )
    _last_config = config
    return _processor, "(새 모델 로드)"


# ─────────────────────────────────────────────────────────────
# 메인 처리
# ─────────────────────────────────────────────────────────────
def process_files(files, whisper_size, run_llm, llm_backend,
                  llm_model_size, top_keywords, progress=gr.Progress()):
    if not files:
        return (
            pd.DataFrame(),
            "❌ 음성 파일을 업로드해주세요",
            gr.update(choices=[], value=None),
            {},
            "",
        )

    progress(0, desc="모델 로딩...")
    processor, cache_msg = get_processor(
        whisper_size, run_llm, llm_backend, llm_model_size, top_keywords
    )

    file_paths = [f.name if hasattr(f, "name") else f for f in files]
    n = len(file_paths)

    results = []
    for i, path in enumerate(file_paths):
        progress((i + 1) / (n + 1),
                 desc=f"[{i+1}/{n}] {Path(path).name} 처리 중...")
        r = processor.process_file(path)
        results.append(r)

    progress(1.0, desc="✅ 완료")

    # 요약 DataFrame
    rows = []
    for r in results:
        if r["status"] != "ok":
            rows.append({
                "파일": r["audio_name"],
                "음성(초)": "-",
                "처리(초)": "-",
                "RTF": "-",
                "글자수": 0,
                "키워드(Top3)": "❌ " + r.get("error", "오류")[:30],
                "카테고리": "-",
            })
            continue
        m = r["stt_meta"]
        top3 = ", ".join(kw for kw, _ in r["keywords"][:3])
        cats = ", ".join(r["categories"].keys()) or "-"
        rows.append({
            "파일": r["audio_name"],
            "음성(초)": m["duration_sec"],
            "처리(초)": m["processing_sec"],
            "RTF": m["rtf"],
            "글자수": len(r["transcription"]),
            "키워드(Top3)": top3,
            "카테고리": cats,
        })

    df = pd.DataFrame(rows)

    # 통계 요약
    ok = [r for r in results if r["status"] == "ok"]
    err_count = len(results) - len(ok)
    total_audio = sum(r["stt_meta"]["duration_sec"] for r in ok) if ok else 0
    total_proc = sum(r["stt_meta"]["processing_sec"] for r in ok) if ok else 0
    avg_rtf = (total_proc / total_audio) if total_audio > 0 else 0

    summary = f"""✅ **완료** {cache_msg}

- 총 파일: **{len(results)}**개 (성공 {len(ok)} / 실패 {err_count})
- 처리한 음성 총 길이: **{total_audio:.1f}초**
- 처리 총 시간: **{total_proc:.1f}초**
- 평균 RTF (실시간계수): **{avg_rtf:.3f}x**  *(낮을수록 빠름)*
"""

    # 파일별 선택 옵션
    file_choices = [r["audio_name"] for r in results]
    first_file = file_choices[0] if file_choices else None

    # 결과 저장 (전역에 보관해서 상세보기·다운로드 사용)
    global _last_results
    _last_results = results

    detail = render_detail(first_file) if first_file else ""

    return df, summary, gr.update(choices=file_choices, value=first_file), \
           {"results_count": len(results)}, detail


_last_results = []


def render_detail(file_name: str) -> str:
    """선택된 파일 상세 정보 마크다운으로."""
    if not file_name or not _last_results:
        return ""
    r = next((x for x in _last_results if x["audio_name"] == file_name), None)
    if r is None:
        return f"파일 '{file_name}' 결과를 찾을 수 없습니다."

    if r["status"] != "ok":
        return f"❌ **오류**: {r.get('error', '')}"

    m = r["stt_meta"]
    parts = []
    parts.append(f"### 📁 {r['audio_name']}\n")
    parts.append(f"- **음성 길이**: {m['duration_sec']}초")
    parts.append(f"- **처리 시간**: {m['processing_sec']}초")
    parts.append(f"- **RTF**: {m['rtf']}x")
    parts.append(f"- **언어**: {m['language']} ({m['language_probability']*100:.1f}%)\n")

    parts.append("#### 📝 전사 텍스트")
    parts.append(f"> {r['transcription']}\n")

    parts.append("#### 🔑 키워드")
    if r["keywords"]:
        for kw, sc in r["keywords"]:
            parts.append(f"- `{kw}` — {sc:.3f}")
    else:
        parts.append("(추출된 키워드 없음)")
    parts.append("")

    if r["categories"]:
        parts.append("#### 🏥 의료 카테고리")
        for cat, ws in r["categories"].items():
            parts.append(f"- **{cat}**: {', '.join(ws)}")
        parts.append("")

    if r.get("soap"):
        parts.append("#### 📋 SOAP 요약 (LLM)")
        parts.append("```")
        parts.append(r["soap"])
        parts.append("```")

    return "\n".join(parts)


def download_results(format_type: str):
    """결과를 CSV/JSON/MD로 저장하고 경로 반환."""
    if not _last_results:
        return None
    out_dir = Path.cwd() / "results"
    out_dir.mkdir(exist_ok=True)

    if format_type == "CSV":
        path = out_dir / "summary.csv"
        bl.save_csv_summary(_last_results, path)
    elif format_type == "JSON":
        path = out_dir / "all_results.json"
        bl.save_json_full(_last_results, path)
    else:  # Markdown
        path = out_dir / "report.md"
        bl.save_markdown_report(_last_results, path)

    return str(path)


# ─────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────
with gr.Blocks(title="Voice EMR Baseline", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🎙️ Voice EMR Baseline 테스트 (다중 파일 지원)

    **여러 음성 파일을 한 번에 업로드** 해서 STT 정확도·키워드·SOAP 결과를 비교합니다.
    모델은 첫 번째 처리 시 한 번만 로드되고 같은 설정이면 이후엔 재사용됩니다.

    **사용 모델** (모두 상업적 사용 가능):
    `faster-whisper (MIT)` + `KeyBERT (MIT)` + `Qwen2.5-Instruct (Apache 2.0)`
    """)

    with gr.Row():
        with gr.Column(scale=2):
            audio_input = gr.File(
                label="🎵 음성 파일 업로드 (여러 개 가능, WAV/MP3/M4A/FLAC)",
                file_count="multiple",
                file_types=["audio"],
                type="filepath",
            )

            with gr.Accordion("⚙️ 처리 옵션", open=True):
                whisper_size = gr.Dropdown(
                    label="Whisper 모델",
                    choices=["tiny", "base", "small", "medium", "large-v3"],
                    value="medium",
                )
                top_keywords = gr.Slider(
                    label="키워드 개수",
                    minimum=5, maximum=20, value=12, step=1,
                )
                run_llm = gr.Checkbox(
                    label="LLM SOAP 요약 실행",
                    value=False,
                    info="끄면 STT + 키워드만 (훨씬 빠름)",
                )
                llm_backend = gr.Radio(
                    label="LLM 백엔드",
                    choices=["CPU 최적화 (GGUF)", "transformers (GPU)"],
                    value="CPU 최적화 (GGUF)",
                    info="CPU 환경이면 GGUF 권장. GPU 있으면 transformers.",
                )
                llm_model_size = gr.Dropdown(
                    label="LLM 모델 크기",
                    choices=["1.5B", "3B", "7B"],
                    value="3B",
                    info="RAM에 맞게: 8GB→1.5B, 16GB→3B, 32GB+→7B",
                )

            submit_btn = gr.Button("▶ 일괄 분석 시작", variant="primary", size="lg")

            summary_md = gr.Markdown(label="📊 처리 요약")

        with gr.Column(scale=3):
            gr.Markdown("### 📋 비교 요약 표")
            results_table = gr.Dataframe(
                headers=["파일", "음성(초)", "처리(초)", "RTF", "글자수",
                         "키워드(Top3)", "카테고리"],
                interactive=False,
                wrap=True,
            )

            gr.Markdown("### 🔍 파일별 상세 (클릭해서 선택)")
            with gr.Row():
                file_selector = gr.Dropdown(
                    label="파일 선택",
                    choices=[],
                    interactive=True,
                )
            detail_md = gr.Markdown()

            with gr.Row():
                gr.Markdown("**📥 결과 저장**")
                csv_btn = gr.Button("CSV", size="sm")
                json_btn = gr.Button("JSON", size="sm")
                md_btn = gr.Button("Markdown", size="sm")
            downloaded_file = gr.File(label="저장된 파일", interactive=False)

    raw_state = gr.JSON(visible=False)

    # 이벤트
    submit_btn.click(
        process_files,
        inputs=[audio_input, whisper_size, run_llm, llm_backend,
                llm_model_size, top_keywords],
        outputs=[results_table, summary_md, file_selector, raw_state, detail_md],
    )

    file_selector.change(
        render_detail,
        inputs=[file_selector],
        outputs=[detail_md],
    )

    csv_btn.click(lambda: download_results("CSV"), outputs=[downloaded_file])
    json_btn.click(lambda: download_results("JSON"), outputs=[downloaded_file])
    md_btn.click(lambda: download_results("Markdown"), outputs=[downloaded_file])

    gr.Markdown("""
    ---
    💡 **활용 팁**
    - **여러 파일을 한 번에**: 모델 로드 시간(보통 10~30초) 절약 — 첫 파일 후엔 빠름
    - **CPU 환경**: LLM 백엔드를 `CPU 최적화 (GGUF)`로 + 모델 크기 1.5B 또는 3B
    - **베이스라인 측정 목적이면**: LLM 끄고 STT 정확도만 측정 (가장 빠르고 정확)
    - **결과 다운로드**: CSV는 엑셀에서, Markdown은 깃허브·노션에서 보기 좋음
    """)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
