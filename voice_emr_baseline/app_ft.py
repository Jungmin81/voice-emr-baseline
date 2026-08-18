# -*- coding: utf-8 -*-
"""
Voice EMR 데모 — 파인튜닝 Whisper(LoRA) STT + Qwen2.5 SOAP

입력 두 가지: (1) 녹음 파일 업로드  (2) 마이크 녹음(브라우저)
출력 두 가지: STT 결과 / SOAP 결과
마이크 장치 유무는 브라우저(navigator.mediaDevices)에서 확인.

실행:
    source ../venv_train/bin/activate
    python app_ft.py
    # http://localhost:7860

환경변수(선택):
    FT_CKPT     파인튜닝 체크포인트 (기본: 중규모 best = checkpoint-2000)
    BASE_MODEL  베이스 (기본 openai/whisper-medium)
    LLM_MODEL   SOAP LLM (기본 Qwen/Qwen2.5-3B-Instruct, 더 좋게: 7B)
    DEMO_DEVICE cuda / cpu (기본 cuda, 학습과 GPU 공유)
"""
import os
import time
from pathlib import Path

import gradio as gr

# ── 설정 ──────────────────────────────────────────────────────────
HERE = Path(__file__).resolve().parent
FT_CKPT = os.environ.get(
    "FT_CKPT",
    # 전체 데이터 학습 best 모델 (CER 0.93%). 없으면 중규모로 폴백.
    str(HERE / "training/outputs/full/phase_c_full/final"),
)
if not os.path.exists(FT_CKPT):
    FT_CKPT = str(HERE / "training/outputs/exp1/phase_b_small/checkpoint-2000")
BASE_MODEL = os.environ.get("BASE_MODEL", "openai/whisper-medium")
LLM_MODEL = os.environ.get("LLM_MODEL", "google/gemma-2-27b-it")  # SOAP: 환각·한자 적음
DEVICE = os.environ.get("DEMO_DEVICE", "cuda")

# SOAP 프롬프트 (baseline.py 와 동일 — 여기서 입맛대로 수정 가능)
SOAP_PROMPT = """당신은 한국 의료 차트 작성을 보조하는 AI입니다.
아래 의사-환자 진료 대화를 표준 SOAP 구조로 정리해주세요.

[작성 규칙 — 반드시 지킬 것]
- 모든 내용은 **한국어(한글)로만** 작성합니다.
- **한자(漢字)·중국어·일본어를 절대 쓰지 마세요.** 의학용어도 한글로 풀어 씁니다.
  (예: 處方→처방, 護理→간호, 增加檢査→추가 검사, 比較診斷→감별 진단, 心電圖→심전도)
- 영어는 꼭 필요한 고유명사만 허용(예: X-ray). 그 외엔 한글.
- 단어를 이상하게 띄우지 마세요(예: "심 전 도"(X) → "심전도"(O)).
- 원문에 명시되지 않은 내용은 절대 추가하지 마세요. 추측·짐작 금지.
- 정보가 없는 항목은 "기재 없음"으로 적어주세요.

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

# ── 모델 (지연 로딩) ─────────────────────────────────────────────
_asr = None      # transformers ASR pipeline (LoRA merge 된 whisper)
_llm = None      # (tokenizer, model)
_vad = None      # silero VAD 모델
_diar = None     # pyannote 화자분리 파이프라인
_cjk_bad = None  # 한자/중국어 토큰 금지 리스트(SOAP 한글 강제)
_kiwi = None     # 한국어 띄어쓰기 교정기


def _resolve_device():
    try:
        import torch
        if DEVICE == "cuda" and torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def get_asr():
    """파인튜닝 Whisper 로드 (LoRA를 베이스에 merge → 표준 모델 → 30초 청킹 파이프라인)."""
    global _asr
    if _asr is not None:
        return _asr
    import torch
    from transformers import (
        WhisperForConditionalGeneration, WhisperProcessor, pipeline,
    )
    from peft import PeftModel

    dev = _resolve_device()
    dtype = torch.bfloat16 if dev == "cuda" else torch.float32
    print(f"[ASR] base={BASE_MODEL} ckpt={FT_CKPT} device={dev}")
    processor = WhisperProcessor.from_pretrained(BASE_MODEL, language="korean", task="transcribe")
    base = WhisperForConditionalGeneration.from_pretrained(BASE_MODEL, torch_dtype=dtype)
    model = PeftModel.from_pretrained(base, FT_CKPT)
    model = model.merge_and_unload()        # LoRA 가중치를 베이스에 합쳐 표준 모델로
    model.to(dev)
    model.generation_config.language = "korean"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None
    _asr = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=dtype,
        device=0 if dev == "cuda" else -1,
        chunk_length_s=30,                  # 긴 녹음도 30초 청크로 처리
        stride_length_s=5,
    )
    return _asr


def get_llm():
    global _llm
    if _llm is not None:
        return _llm
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = _resolve_device()
    dtype = torch.bfloat16 if dev == "cuda" else torch.float32
    print(f"[LLM] {LLM_MODEL} device={dev}")
    tok = AutoTokenizer.from_pretrained(LLM_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL, torch_dtype=dtype,
        device_map=("auto" if dev == "cuda" else None),  # 27B 등 대형은 다중 GPU 분산
    )
    if dev != "cuda":
        model.to(dev)
    _llm = (tok, model)
    return _llm


def transcribe(audio_path: str) -> str:
    asr = get_asr()
    out = asr(
        audio_path,
        generate_kwargs={"language": "korean", "task": "transcribe"},
        return_timestamps=False,
    )
    return (out.get("text") or "").strip()


def get_vad():
    global _vad
    if _vad is None:
        from silero_vad import load_silero_vad
        _vad = load_silero_vad()
    return _vad


def transcribe_vad(audio_path: str, max_chunk_s: float = 28.0, pad_s: float = 0.2) -> str:
    """Silero VAD로 발화 경계에서 잘라 청크별 STT → 묵음 제거(환각↓)·경계 정렬(긴 음성용)."""
    import numpy as np, librosa, torch
    from silero_vad import get_speech_timestamps
    audio, _ = librosa.load(audio_path, sr=16000, mono=True)
    vad = get_vad()
    ts = get_speech_timestamps(torch.from_numpy(audio), vad,
                               sampling_rate=16000, return_seconds=False)
    if not ts:                         # 발화 미검출 → 일반 경로로 폴백
        return transcribe(audio_path)
    SR = 16000
    maxs = int(max_chunk_s * SR)
    pad = int(pad_s * SR)
    # 발화 구간들을 ≤max_chunk_s 윈도우로 greedy 병합 (경계는 발화 단위)
    windows = []
    cs, ce = ts[0]["start"], ts[0]["end"]
    for seg in ts[1:]:
        if seg["end"] - cs <= maxs:
            ce = seg["end"]
        else:
            windows.append((cs, ce)); cs, ce = seg["start"], seg["end"]
    windows.append((cs, ce))
    asr = get_asr()
    texts = []
    for s, e in windows:
        chunk = audio[max(0, s - pad):min(len(audio), e + pad)]
        out = asr({"raw": chunk, "sampling_rate": SR},
                  generate_kwargs={"language": "korean", "task": "transcribe"},
                  return_timestamps=False)
        t = (out.get("text") or "").strip()
        if t:
            texts.append(t)
    return " ".join(texts)


def get_diar():
    """pyannote 화자분리 파이프라인 (CPU — 학습 GPU 비간섭). 첫 호출 시 로드."""
    global _diar
    if _diar is None:
        import torch
        # torch 2.6 weights_only 기본 True → pyannote 체크포인트 로딩 실패. 신뢰 소스라 False.
        _orig = torch.load
        torch.load = lambda *a, **k: (k.update(weights_only=False) or _orig(*a, **k))
        from pyannote.audio import Pipeline
        tok = os.environ.get("HF_TOKEN") or True   # 캐시된 로그인 사용 가능
        _diar = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1", use_auth_token=tok)
        diar_dev = os.environ.get("DEMO_DIAR_DEVICE", "cuda")  # GPU 가용 시 GPU
        if diar_dev == "cuda" and not torch.cuda.is_available():
            diar_dev = "cpu"
        _diar.to(torch.device(diar_dev))
    return _diar


def transcribe_diarized(audio_path: str, num_speakers: int = 2):
    """화자분리 → 화자별 구간 STT. (speaker_label, text) 턴 리스트 반환."""
    import librosa
    audio, _ = librosa.load(audio_path, sr=16000, mono=True)
    pipe = get_diar()
    kw = {"num_speakers": num_speakers} if num_speakers and num_speakers > 0 else {}
    diar = pipe(audio_path, **kw)
    raw = sorted((t.start, t.end, s) for t, _, s in diar.itertracks(yield_label=True))
    # 인접한 동일 화자 구간 병합 (0.8초 이내)
    merged = []
    for st, en, s in raw:
        if merged and merged[-1][2] == s and st - merged[-1][1] < 0.8:
            merged[-1] = (merged[-1][0], en, s)
        else:
            merged.append([st, en, s])
    asr = get_asr()
    turns = []
    for st, en, s in merged:
        chunk = audio[int(st * 16000):int(en * 16000)]
        if len(chunk) < int(0.3 * 16000):
            continue
        out = asr({"raw": chunk, "sampling_rate": 16000},
                  generate_kwargs={"language": "korean", "task": "transcribe"},
                  return_timestamps=False)
        txt = (out.get("text") or "").strip()
        if txt:
            turns.append((s, txt))
    return turns


def label_roles(turns):
    """화자 라벨(SPEAKER_xx) → 의사/환자 매핑. LLM 우선, 실패 시 물음표 휴리스틱."""
    speakers = sorted({s for s, _ in turns})
    if len(speakers) < 2:
        return {s: "화자" for s in speakers}
    # 휴리스틱(폴백): 질문(?) 많은 화자 = 의사
    q = {s: 0 for s in speakers}
    for s, t in turns:
        q[s] += t.count("?") + t.count("나요") + t.count("세요")
    heur_doc = max(q, key=q.get)
    mapping = {s: ("의사" if s == heur_doc else "환자") for s in speakers}
    # LLM 정밀 판정
    try:
        import json as _json
        tok, model = get_llm()
        import torch
        transcript = "\n".join(f"{s}: {t}" for s, t in turns[:40])
        prompt = ("다음은 진료 대화의 화자별 발화입니다. 각 화자 라벨이 '의사'인지 '환자'인지 "
                  "판단하세요. 의사는 질문·지시·진단을, 환자는 증상·답변을 말합니다.\n\n"
                  f"{transcript}\n\n"
                  "반드시 JSON 한 줄로만 답하세요. 예: {\"SPEAKER_00\":\"환자\",\"SPEAKER_01\":\"의사\"}")
        msgs = [{"role": "user", "content": prompt}]
        inp = tok.apply_chat_template(msgs, return_tensors="pt", add_generation_prompt=True).to(model.device)
        with torch.no_grad():
            out = model.generate(inp, max_new_tokens=120, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        resp = tok.decode(out[0][inp.shape[1]:], skip_special_tokens=True)
        st, en = resp.find("{"), resp.rfind("}")
        if st >= 0 and en > st:
            parsed = _json.loads(resp[st:en + 1])
            for s in speakers:
                v = parsed.get(s, "")
                if "의사" in v:
                    mapping[s] = "의사"
                elif "환자" in v:
                    mapping[s] = "환자"
    except Exception:
        pass
    return mapping


def render_dialogue_html(turns, roles):
    """화자분리 대화를 좌(의사)/우(환자) 채팅 말풍선 HTML로 렌더."""
    import html as _html
    rows = []
    for s, t in turns:
        role = roles.get(s, s)
        if role == "의사":
            side, cls = "left", "doc"
        elif role == "환자":
            side, cls = "right", "pat"
        else:
            side, cls = "left", "etc"
        rows.append(
            f'<div class="chat-row {side}">'
            f'<div class="chat-bubble {cls}">'
            f'<span class="who">{_html.escape(str(role))}</span>{_html.escape(t)}'
            f'</div></div>'
        )
    return '<div class="chat-wrap">' + "".join(rows) + "</div>"


def plain_to_html(text):
    """비화자분리(일반 STT) 텍스트를 줄바꿈 보존 HTML로 감쌈."""
    import html as _html
    if not text:
        return ""
    return f'<div class="stt-plain">{_html.escape(text)}</div>'


def analyze_diarized(audio_path, do_soap, num_speakers=2):
    """화자분리 → 역할라벨 → 대화 텍스트 → (선택)SOAP."""
    if not audio_path:
        return "", "", "⚠️ 오디오가 없습니다."
    t0 = time.time()
    try:
        turns = transcribe_diarized(audio_path, num_speakers=num_speakers)
    except Exception as e:
        return "", "", f"❌ 화자분리/STT 실패: {e}"
    if not turns:
        return "", "", "⚠️ 인식된 발화가 없습니다."
    roles = label_roles(turns)
    dialogue = "\n".join(f"{roles.get(s, s)}: {t}" for s, t in turns)  # SOAP 입력(평문)
    dialogue_html = render_dialogue_html(turns, roles)                 # 화면 표시(말풍선)
    t_diar = time.time() - t0
    nspk = len({roles.get(s, s) for s, _ in turns})
    if not do_soap:
        return dialogue_html, "", f"✅ 화자분리+STT {t_diar:.1f}s · 화자 {nspk}명 · SOAP 건너뜀"
    try:
        t1 = time.time()
        soap = run_soap(dialogue)
        meta = f"✅ 화자분리+STT {t_diar:.1f}s · SOAP {time.time()-t1:.1f}s · 화자 {nspk}명"
    except Exception as e:
        return dialogue_html, f"❌ SOAP 실패: {e}", f"✅ 화자분리 {t_diar:.1f}s · SOAP 실패"
    return dialogue_html, soap, meta


def get_cjk_bad_words(tok):
    """한자/중국어(CJK 한자) 글자를 포함한 토큰 id 목록 → 생성 시 금지.
    Qwen 계열이 의료용어를 한자로 뱉는 걸 디코딩 단계에서 원천 차단."""
    global _cjk_bad
    if _cjk_bad is not None:
        return _cjk_bad
    import re
    pat = re.compile(r'[㐀-䶿一-鿿豈-﫿]')  # CJK 한자(한글 제외)
    bad = []
    n = getattr(tok, "vocab_size", 0) or len(tok)
    for i in range(n):
        try:
            s = tok.decode([i])
        except Exception:
            continue
        if pat.search(s):
            bad.append([i])
    _cjk_bad = bad
    return bad


def run_soap(text: str) -> str:
    import torch
    tok, model = get_llm()
    prompt = SOAP_PROMPT.format(text=text)
    messages = [{"role": "user", "content": prompt}]
    inputs = tok.apply_chat_template(
        messages, return_tensors="pt", add_generation_prompt=True
    ).to(model.device)
    gen_kw = dict(
        max_new_tokens=1024,
        do_sample=False,                 # 결정적 디코딩 → 한글 깨짐·랜덤 토큰 감소
        repetition_penalty=1.05,         # 약하게만: 값이 크면 "답답함"→"답 dap함"처럼 로마자로 튐
        # no_repeat_ngram_size 제거: SOAP는 빈 항목마다 "기재 없음"이 정당하게 반복되는데
        # 3-gram 금지가 이를 막아 "기재없움/기여 없음/없읍" 같은 깨진 대체어를 유발함.
        pad_token_id=tok.eos_token_id,
    )
    # 한자 누출은 Qwen 계열 문제 → Qwen일 때만 토큰 금지(가드). Gemma 등은 불필요.
    if "qwen" in LLM_MODEL.lower():
        gen_kw["bad_words_ids"] = get_cjk_bad_words(tok)
    with torch.no_grad():
        out = model.generate(inputs, **gen_kw)
    result = tok.decode(out[0][inputs.shape[1]:], skip_special_tokens=True).strip()
    # 후처리1: bad_words 로도 새는 잔여 한자를 한글 음으로 변환 (果→과, 處方→처방)
    try:
        import hanja
        result = hanja.translate(result, "substitution")
    except Exception:
        pass
    # 후처리2: 띄어쓰기 교정 ("심 전 도"→"심전도"). 마크다운/영어는 보존됨.
    try:
        global _kiwi
        if _kiwi is None:
            from kiwipiepy import Kiwi
            _kiwi = Kiwi()
        result = "\n".join(
            _kiwi.space(ln, reset_whitespace=True) if ln.strip() else ln
            for ln in result.split("\n"))
    except Exception:
        pass
    return result


def analyze(audio_path, do_soap, use_vad=True):
    """오디오 → STT (+ 선택적 SOAP). (stt_text, soap_md, meta) 반환."""
    if not audio_path:
        return "", "", "⚠️ 오디오가 없습니다. 파일을 올리거나 마이크로 녹음하세요."
    t0 = time.time()
    try:
        stt = transcribe_vad(audio_path) if use_vad else transcribe(audio_path)
    except Exception as e:
        return "", "", f"❌ STT 실패: {e}"
    t_stt = time.time() - t0
    if not stt:
        return "", "", f"⚠️ 인식된 텍스트가 없습니다. (STT {t_stt:.1f}s)"
    if not do_soap:
        return stt, "", f"✅ STT {t_stt:.1f}s · SOAP 건너뜀"
    try:
        t1 = time.time()
        soap = run_soap(stt)
        t_soap = time.time() - t1
    except Exception as e:
        return stt, f"❌ SOAP 실패: {e}", f"✅ STT {t_stt:.1f}s · ❌ SOAP 실패"
    return stt, soap, f"✅ STT {t_stt:.1f}s · SOAP {t_soap:.1f}s"


def run_analyze(audio_path, do_soap, use_vad, use_diar, num_spk):
    """UI 라우터: 화자분리 켜면 의사/환자 대화로, 아니면 일반 STT."""
    if use_diar:
        ns = 0 if str(num_spk) in ("자동", "auto", "0") else int(num_spk)
        return analyze_diarized(audio_path, do_soap, num_speakers=ns)
    stt_txt, soap, meta = analyze(audio_path, do_soap, use_vad=use_vad)
    return plain_to_html(stt_txt), soap, meta


# 브라우저에서 마이크 유무 확인 (서버가 아니라 클라이언트 장치를 봐야 맞음)
MIC_CHECK_JS = """
async () => {
  try {
    if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) {
      return '❌ 이 브라우저는 장치 확인을 지원하지 않습니다.';
    }
    let stream = null;
    try { stream = await navigator.mediaDevices.getUserMedia({audio:true}); } catch (e) {
      return '⚠️ 마이크 권한이 거부되었거나 장치가 없습니다 (' + e.name + ').';
    }
    const devs = await navigator.mediaDevices.enumerateDevices();
    const mics = devs.filter(d => d.kind === 'audioinput');
    if (stream) stream.getTracks().forEach(t => t.stop());
    if (mics.length === 0) return '❌ 마이크 장치가 감지되지 않았습니다.';
    const names = mics.map(m => m.label || '(이름 비공개)').join(', ');
    return '✅ 마이크 ' + mics.length + '개 감지됨: ' + names;
  } catch (err) {
    return '⚠️ 확인 실패: ' + err;
  }
}
"""

APP_CSS = """
/* 오디오 플레이어의 길이/시간 표시가 잘리지 않도록 */
#aud_up, #aud_mic { overflow: visible !important; padding-right: 14px !important; }
#aud_up *, #aud_mic * { overflow: visible !important; text-overflow: clip !important; }
#aud_up time, #aud_mic time,
#aud_up [class*="time"], #aud_mic [class*="time"],
#aud_up [class*="duration"], #aud_mic [class*="duration"] {
    white-space: nowrap !important;
    min-width: max-content !important;
    flex-shrink: 0 !important;
}

/* 화자분리 대화 — 좌(의사)/우(환자) 채팅 말풍선 */
.chat-wrap { display:flex; flex-direction:column; gap:8px; padding:6px 2px;
             max-height:74vh; overflow-y:auto; }
.chat-row { display:flex; width:100%; }
.chat-row.left  { justify-content:flex-start; }
.chat-row.right { justify-content:flex-end; }
.chat-bubble { max-width:78%; padding:8px 12px; border-radius:14px;
               font-size:14px; line-height:1.45; word-break:break-word;
               box-shadow:0 1px 2px rgba(0,0,0,.08); }
.chat-bubble .who { display:block; font-size:11px; font-weight:700;
                    margin-bottom:2px; opacity:.75; }
.chat-bubble.doc { background:#e8f0fe; color:#1a3d6d; border-bottom-left-radius:4px; }
.chat-bubble.pat { background:#eafaf0; color:#1c5b34; border-bottom-right-radius:4px; }
.chat-bubble.etc { background:#f0f0f2; color:#333; }
.stt-plain { white-space:pre-wrap; font-size:14px; line-height:1.5; padding:6px 2px; }
"""

with gr.Blocks(title="Voice EMR (파인튜닝)") as demo:
    gr.Markdown(
        "# 🎙️ Voice EMR — 음성 → STT → SOAP\n"
        "파인튜닝 Whisper(medium+LoRA, 중규모 모델)로 받아쓰고 Qwen2.5로 SOAP 차트를 생성합니다."
    )
    with gr.Row(equal_height=False):
        # ── 좌측: 옵션(상) → 오디오(중) → SOAP(하) ──
        with gr.Column(scale=2, min_width=360):
            with gr.Group():
                gr.Markdown("#### ⚙️ 옵션")
                do_soap = gr.Checkbox(value=True, label="SOAP 생성 (끄면 STT만)")
                use_vad = gr.Checkbox(value=True, label="VAD 청킹 (긴 음성 권장)")
                use_diar = gr.Checkbox(value=False, label="🧑‍⚕️ 화자 분리(의사/환자)")
                num_spk = gr.Dropdown(choices=["2", "자동"], value="2", label="화자 수")

            gr.Markdown("#### 🎧 오디오 입력")
            with gr.Tabs():
                with gr.Tab("📁 녹음본 업로드"):
                    up_audio = gr.Audio(sources=["upload"], type="filepath",
                                        label="오디오 파일 (wav/mp3/m4a ...)",
                                        editable=False, elem_id="aud_up")
                    up_btn = gr.Button("▶ 분석", variant="primary")
                with gr.Tab("🎤 마이크 녹음"):
                    mic_check_btn = gr.Button("🔍 마이크 장치 확인")
                    mic_status = gr.Markdown("마이크 사용 전 '장치 확인'을 눌러보세요.")
                    mic_audio = gr.Audio(sources=["microphone"], type="filepath",
                                         label="마이크로 녹음", editable=False,
                                         elem_id="aud_mic")
                    mic_btn = gr.Button("▶ 분석", variant="primary")
            meta = gr.Markdown()

            gr.Markdown("### 🧾 SOAP 결과")
            soap_out = gr.Markdown()

        # ── 우측: STT 결과 전체 ──
        with gr.Column(scale=3):
            gr.Markdown("### 📝 STT 결과 (화자분리 시: 🩺의사 왼쪽 / 🧑환자 오른쪽)")
            stt_out = gr.HTML()

    mic_check_btn.click(None, inputs=None, outputs=mic_status, js=MIC_CHECK_JS)
    up_btn.click(run_analyze, inputs=[up_audio, do_soap, use_vad, use_diar, num_spk],
                 outputs=[stt_out, soap_out, meta])
    mic_btn.click(run_analyze, inputs=[mic_audio, do_soap, use_vad, use_diar, num_spk],
                  outputs=[stt_out, soap_out, meta])

    gr.Markdown(
        "---\n"
        "ℹ️ 마이크는 **브라우저**에서 녹음됩니다(서버 장치 아님). 첫 분석 시 모델 로딩에 수십 초 걸릴 수 있습니다."
    )

if __name__ == "__main__":
    port = int(os.environ.get("DEMO_PORT", "7860"))
    share = os.environ.get("DEMO_SHARE", "0") == "1"      # 외부 공개 URL(gradio.live)
    # 선택: 간단 인증 (DEMO_USER/DEMO_PASS 설정 시 로그인 요구)
    user = os.environ.get("DEMO_USER")
    pw = os.environ.get("DEMO_PASS")
    auth = (user, pw) if user and pw else None
    demo.queue().launch(server_name="0.0.0.0", server_port=port,
                        share=share, auth=auth,
                        theme=gr.themes.Soft(), css=APP_CSS)
