#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gradio 실시간 학습 모니터링 대시보드

학습이 진행되는 중에 별도 터미널에서 실행:
    python monitor.py --output-dir outputs/phase_a_quick_verify

브라우저에서: http://localhost:7861

표시 내용:
- 현재 진행 (epoch / step / 진행률 / ETA)
- Loss / CER 그래프
- 샘플 예측 (GT vs PRED) 비교
- GPU 사용량 (4장 동시)
- 학습 상태 (running / stopped / done)
"""
import argparse
import json
import os
import sys
from pathlib import Path

import gradio as gr
import pandas as pd


def safe_read_jsonl(path: Path, last_n: int = None) -> list:
    """JSONL 파일 읽기 (없으면 빈 리스트)."""
    if not path.exists():
        return []
    rows = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    except Exception:
        return []
    if last_n:
        return rows[-last_n:]
    return rows


def safe_read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def format_eta(seconds: float) -> str:
    if seconds <= 0:
        return "?"
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    if h > 0:
        return f"{h}시간 {m}분"
    return f"{m}분"


# ──────────────────────────────────────────────────────────────
# 데이터 로딩 함수 (대시보드에서 주기적 호출)
# ──────────────────────────────────────────────────────────────
def collect_data(output_dir: Path):
    """모든 로그·상태 모아서 dict로 반환."""
    status = safe_read_json(output_dir / "status.json")
    # 학습 콜백이 매 logging step 마다 갱신하는 실시간 진행 파일
    progress = safe_read_json(output_dir / "progress.json")

    # Trainer가 기본으로 trainer_state.json 저장 — train loss 그래프용
    trainer_state = {}
    for ckpt in sorted(output_dir.glob("checkpoint-*"),
                       key=lambda p: int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else -1,
                       reverse=True):
        state_file = ckpt / "trainer_state.json"
        if state_file.exists():
            trainer_state = safe_read_json(state_file)
            break

    log_history = trainer_state.get("log_history", [])
    gpu_logs = safe_read_jsonl(output_dir / "gpu_stats.jsonl", last_n=1)
    sample_preds = safe_read_jsonl(output_dir / "sample_predictions.jsonl", last_n=1)
    cer_logs = safe_read_jsonl(output_dir / "cer_log.jsonl")

    return {
        "status": status,
        "progress": progress,
        "trainer_state": trainer_state,
        "log_history": log_history,
        "cer_logs": cer_logs,
        "gpu_latest": gpu_logs[-1] if gpu_logs else None,
        "sample_preds_latest": sample_preds[-1] if sample_preds else None,
    }


def render_dashboard(output_dir_str: str):
    """대시보드 UI 컴포넌트들에 들어갈 값 생성."""
    output_dir = Path(output_dir_str)
    if not output_dir.exists():
        return ("❌ output 폴더 없음", "", "", None, None, "", "")

    data = collect_data(output_dir)

    # ─── 상태 패널 (progress.json 우선 — 실시간) ───
    status = data["status"]
    prog = data["progress"] or {}
    ts = data["trainer_state"]
    log_history = data["log_history"]

    cur_epoch = prog.get("epoch", ts.get("epoch", 0))
    cur_step = prog.get("global_step", ts.get("global_step", 0))
    max_steps = prog.get("max_steps", ts.get("max_steps", 0))
    total_epochs = prog.get("num_train_epochs", 0)

    pct = prog.get("progress_pct") or ((cur_step / max_steps * 100) if max_steps > 0 else 0)
    filled = int(pct // 5)
    progress_bar = f"{'█' * filled}{'░' * (20 - filled)} {pct:.1f}%"

    # ETA: 콜백이 계산한 값 우선, 없으면 started_at 기반 추정
    eta_text = "?"
    if prog.get("eta_sec") is not None and cur_step > 0:
        eta_text = format_eta(prog["eta_sec"])
    elif status.get("started_at") and cur_step > 0 and max_steps:
        import time
        elapsed = time.time() - status["started_at"]
        eta_text = format_eta(elapsed * (max_steps - cur_step) / cur_step)

    sec_per_step = prog.get("sec_per_step")
    speed_text = f"{sec_per_step:.2f}초/step" if sec_per_step else "(측정 중)"
    remaining_steps = prog.get("remaining_steps", max(0, max_steps - cur_step))
    best = prog.get("best_metric")
    best_text = f"{best:.4f}" if isinstance(best, (int, float)) else "-"

    status_md = f"""### 📊 학습 진행 현황

**Phase**: {status.get('phase_name', '?')}  |  **상태**: {status.get('status', '?')}
**Epoch**: {cur_epoch:.3f} / {total_epochs or '?'}
**Step**: {cur_step} / {max_steps}  (남은 step **{remaining_steps}**)
**진행**: `{progress_bar}`
**예상 남은 시간(ETA)**: **{eta_text}**
**속도**: {speed_text}
**최저 eval_loss(best)**: {best_text}

**학습 데이터**: {status.get('train_size', 0)} 샘플  |  **검증**: {status.get('val_size', 0)} 샘플
**베이스 모델**: {status.get('base_model', '?')}
"""

    # ─── Train Loss 그래프 (trainer_state) ───
    train_logs = []
    for l in log_history:
        if "loss" in l and "eval_loss" not in l:
            train_logs.append({"step": l.get("step", 0), "loss": l["loss"]})
    loss_df = pd.DataFrame(train_logs) if train_logs else pd.DataFrame(columns=["step", "loss"])

    # ─── Eval CER 그래프 (cer_log.jsonl — 콜백이 기록) ───
    cer_rows = [{"step": c.get("step", 0), "eval_cer": (c.get("eval_cer") or 0) * 100}
                for c in data["cer_logs"] if c.get("eval_cer") is not None]
    eval_df = pd.DataFrame(cer_rows) if cer_rows else pd.DataFrame(columns=["step", "eval_cer"])

    # ─── GPU 사용량 ───
    gpu_text = "GPU 정보 없음"
    if data["gpu_latest"]:
        lines = []
        for g in data["gpu_latest"].get("gpus", []):
            mem_pct = g["mem_used_mb"] / g["mem_total_mb"] * 100 if g["mem_total_mb"] > 0 else 0
            lines.append(
                f"**GPU {g['idx']}** ({g['name']}): "
                f"사용률 {g['util_pct']}% / "
                f"메모리 {g['mem_used_mb']}/{g['mem_total_mb']} MB ({mem_pct:.0f}%) / "
                f"온도 {g['temp_c']}°C"
            )
        gpu_text = "\n\n".join(lines)

    # ─── 샘플 예측 (GT vs PRED) ───
    pred_md = "_샘플 예측 아직 없음 (첫 검증 step 이후 표시)_"
    if data["sample_preds_latest"]:
        sp = data["sample_preds_latest"]
        lines = [f"### 🎤 샘플 예측 (Step {sp.get('step', '?')})\n"]
        for i, s in enumerate(sp.get("samples", []), 1):
            gt = s.get("gt", "").strip()
            pred = s.get("pred", "").strip()
            match = "✅" if gt == pred else "⚠"
            lines.append(f"**{i}. {match}**")
            lines.append(f"- GT:   `{gt}`")
            lines.append(f"- PRED: `{pred}`\n")
        pred_md = "\n".join(lines)

    # ─── 최근 메트릭 요약 ───
    last_metric = ""
    if data["cer_logs"]:
        latest = data["cer_logs"][-1]
        el = latest.get("eval_loss")
        last_metric = (
            f"### 📉 최근 평가 (Step {latest.get('step', '?')})\n"
            f"- Eval Loss: **{el:.4f}**\n" if isinstance(el, (int, float)) else "### 📉 최근 평가\n"
        )
        last_metric += (
            f"- Eval CER: **{(latest.get('eval_cer') or 0) * 100:.2f}%** "
            f"(val {latest.get('n', '?')}개)\n"
        )

    # ─── plotly 차트 (드래그 확대/패닝 + 로그스케일) ───
    import plotly.graph_objects as go

    fig_loss = go.Figure()
    if not loss_df.empty:
        fig_loss.add_trace(go.Scatter(
            x=loss_df["step"], y=loss_df["loss"], mode="lines", name="train loss"))
    fig_loss.update_layout(
        title="Train Loss (Y=로그스케일 · 드래그=확대 · 더블클릭=초기화)",
        height=320, margin=dict(l=50, r=20, t=50, b=40),
        xaxis_title="step", yaxis_title="loss")
    fig_loss.update_yaxes(type="log")   # 초반 큰 값에 안 눌리게 로그스케일

    fig_eval = go.Figure()
    if not eval_df.empty:
        fig_eval.add_trace(go.Scatter(
            x=eval_df["step"], y=eval_df["eval_cer"], mode="lines+markers", name="eval CER"))
    fig_eval.update_layout(
        title="Eval CER (%) · 드래그=확대",
        height=320, margin=dict(l=50, r=20, t=50, b=40),
        xaxis_title="step", yaxis_title="CER (%)")

    return (
        status_md,
        last_metric,
        gpu_text,
        fig_loss,
        fig_eval,
        pred_md,
        f"마지막 새로고침: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
    )


# ──────────────────────────────────────────────────────────────
# UI
# ──────────────────────────────────────────────────────────────
def build_ui(default_output_dir: str):
    with gr.Blocks(title="Voice EMR 학습 모니터", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🎓 Voice EMR 전이학습 실시간 모니터")
        gr.Markdown("Whisper LoRA 학습 진행 상황을 실시간으로 표시합니다.")

        with gr.Row():
            output_dir_input = gr.Textbox(
                label="📁 학습 output 폴더",
                value=default_output_dir,
                placeholder="예: outputs/phase_a_quick_verify",
                scale=4,
            )
            refresh_btn = gr.Button("🔄 새로고침", scale=1, variant="primary")

        timestamp_text = gr.Markdown("")

        with gr.Row():
            with gr.Column(scale=1):
                status_md = gr.Markdown("로딩 중...")
                last_metric_md = gr.Markdown("")

            with gr.Column(scale=1):
                gpu_md = gr.Markdown("### 🖥 GPU 사용량\n_정보 없음_")

        with gr.Row():
            with gr.Column():
                gr.Markdown("### 📉 Train Loss (드래그로 확대 가능)")
                loss_plot = gr.Plot(label="Train Loss")
            with gr.Column():
                gr.Markdown("### 📊 Eval CER (드래그로 확대 가능)")
                eval_plot = gr.Plot(label="Eval CER")

        pred_md = gr.Markdown("_샘플 예측 로딩 중..._")

        # queue=False → 이벤트를 WS/SSE 큐 대신 일반 HTTP POST 로 처리.
        # (tailscale/프록시 경유 시 WS 스트림이 막혀 "로딩중"에서 멈추는 문제 회피)
        _out = [status_md, last_metric_md, gpu_md, loss_plot, eval_plot, pred_md, timestamp_text]
        demo.load(render_dashboard, inputs=[output_dir_input], outputs=_out, queue=False)
        refresh_btn.click(render_dashboard, inputs=[output_dir_input], outputs=_out, queue=False)

        # 15초마다 자동 새로고침
        timer = gr.Timer(15)
        timer.tick(render_dashboard, inputs=[output_dir_input], outputs=_out, queue=False)

    return demo


def main():
    parser = argparse.ArgumentParser(description="Whisper 학습 모니터 (Gradio)")
    parser.add_argument("--output-dir", default="outputs/phase_a_quick_verify",
                        help="학습 출력 폴더 (status.json, log_history 등 읽음)")
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    demo = build_ui(args.output_dir)
    # 큐 비활성: 모든 이벤트를 HTTP POST 로 → tailscale 등 WS 막힌 환경에서도 동작
    demo.launch(server_name=args.host, server_port=args.port,
                share=False, max_threads=8)


if __name__ == "__main__":
    main()
