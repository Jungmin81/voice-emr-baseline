# -*- coding: utf-8 -*-
"""
CPU 최적화 LLM 백엔드 (llama-cpp-python + GGUF)

CPU에서 SOAP 요약을 빠르게 실행하기 위한 모듈.
4-bit 양자화 GGUF 모델 사용으로 메모리 75% 절약 + 속도 5~10배 향상.

설치:
    pip install llama-cpp-python

사용:
    from llm_cpu import load_gguf_llm, run_soap_gguf
    llm = load_gguf_llm("3B")              # 1회 로드
    result, meta = run_soap_gguf(llm, text, SOAP_PROMPT)  # 매번 호출
"""
import os
import time
from pathlib import Path
import urllib.request


# GGUF 모델 정보
GGUF_MODELS = {
    "1.5B": {
        "repo": "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
        "filename": "qwen2.5-1.5b-instruct-q4_k_m.gguf",
        "ram_gb": 1.5,
    },
    "3B": {
        "repo": "Qwen/Qwen2.5-3B-Instruct-GGUF",
        "filename": "qwen2.5-3b-instruct-q4_k_m.gguf",
        "ram_gb": 2.5,
    },
    "7B": {
        "repo": "Qwen/Qwen2.5-7B-Instruct-GGUF",
        "filename": "qwen2.5-7b-instruct-q4_k_m.gguf",
        "ram_gb": 5.0,
    },
}


def download_gguf(model_size: str = "3B", cache_dir: str = None) -> str:
    """HuggingFace에서 GGUF 모델 다운로드 (첫 1회만)."""
    if cache_dir is None:
        cache_dir = str(Path.home() / ".cache" / "voice_emr_gguf")
    os.makedirs(cache_dir, exist_ok=True)

    info = GGUF_MODELS[model_size]
    local_path = Path(cache_dir) / info["filename"]

    if local_path.exists():
        return str(local_path)

    url = f"https://huggingface.co/{info['repo']}/resolve/main/{info['filename']}"
    print(f"  [GGUF] 다운로드 중... ({info['ram_gb']}GB)")
    print(f"  URL: {url}")
    print(f"  → {local_path}")

    last_pct = [-1]
    def _hook(blocks, blocksize, total):
        if total <= 0:
            return
        done_gb = blocks * blocksize / 1024**3
        all_gb = total / 1024**3
        pct = int(blocks * blocksize / total * 100)
        if pct != last_pct[0]:
            last_pct[0] = pct
            print(f"\r  진행: {done_gb:.2f}GB / {all_gb:.2f}GB ({pct}%)",
                  end="", flush=True)

    urllib.request.urlretrieve(url, local_path, reporthook=_hook)
    print()
    return str(local_path)


def load_gguf_llm(model_size: str = "3B", n_threads: int = None,
                  n_ctx: int = 4096):
    """llama-cpp-python으로 GGUF 모델 로드.

    Returns:
        Llama 인스턴스 (반복 호출 가능)
    """
    try:
        from llama_cpp import Llama
    except ImportError:
        raise ImportError(
            "llama-cpp-python이 설치되지 않았습니다.\n"
            "설치: pip install llama-cpp-python"
        )

    model_path = download_gguf(model_size)

    if n_threads is None:
        n_threads = max(1, (os.cpu_count() or 4) - 2)

    print(f"  [GGUF LLM] 모델 로딩 ({model_size}, threads={n_threads})...")
    t0 = time.time()
    llm = Llama(
        model_path=model_path,
        n_ctx=n_ctx,
        n_threads=n_threads,
        n_gpu_layers=0,
        verbose=False,
    )
    print(f"  [GGUF LLM] 로드 완료 ({time.time()-t0:.1f}초)")
    return llm


def run_soap_gguf(llm, text: str, soap_prompt: str, max_tokens: int = 1024):
    """로드된 GGUF LLM으로 SOAP 요약 1회 실행.

    Returns:
        (soap_text, meta_dict)
    """
    prompt = soap_prompt.format(text=text)
    messages = [{"role": "user", "content": prompt}]

    t0 = time.time()
    output = llm.create_chat_completion(
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.2,
        top_p=0.9,
    )
    elapsed = time.time() - t0

    response = output["choices"][0]["message"]["content"].strip()
    tokens = output["usage"]["completion_tokens"]
    tps = tokens / elapsed if elapsed > 0 else 0

    meta = {
        "backend": "llama-cpp (CPU GGUF Q4)",
        "gen_sec": round(elapsed, 1),
        "tokens_generated": tokens,
        "tokens_per_sec": round(tps, 1),
    }
    return response, meta


# 하위 호환 — 단일 호출 함수
def summarize_to_soap_cpu(text: str, model_size: str = "3B", **kwargs):
    """단일 호출용 wrapper. 반복 호출엔 load_gguf_llm + run_soap_gguf 사용."""
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
    llm = load_gguf_llm(model_size)
    return run_soap_gguf(llm, text, SOAP_PROMPT)


if __name__ == "__main__":
    sample = """안녕하세요 선생님. 사흘 전부터 머리가 많이 아프고 어지러워요.
잠도 잘 못 자고요. 약을 먹어도 별로 효과가 없어요.
혈압은 평소에 140정도였고, 당뇨는 없습니다."""

    soap, meta = summarize_to_soap_cpu(sample, model_size="1.5B")
    print("\n" + "=" * 60)
    print(soap)
    print("=" * 60)
    print(f"메타: {meta}")
