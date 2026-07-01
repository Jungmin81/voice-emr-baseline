# -*- coding: utf-8 -*-
"""공통 유틸리티 — CER, 텍스트 정규화, 파일 매칭."""
import json
import re
import unicodedata
from pathlib import Path
from typing import Optional


# ──────────────────────────────────────────────────────────────
# 텍스트 정규화
# ──────────────────────────────────────────────────────────────
PUNCT_RE = re.compile(r"[.,!?·…\"'\(\)\[\]:;-]+")
WHITESPACE_RE = re.compile(r"\s+")


def normalize_korean(text: str, remove_punct: bool = True) -> str:
    """학습/평가용 텍스트 정규화."""
    if not text:
        return ""
    # 유니코드 정규화 (NFC = 자모 결합형)
    text = unicodedata.normalize("NFC", text)
    if remove_punct:
        text = PUNCT_RE.sub(" ", text)
    text = WHITESPACE_RE.sub(" ", text).strip()
    return text


# ──────────────────────────────────────────────────────────────
# CER (Character Error Rate)
# ──────────────────────────────────────────────────────────────
def edit_distance(s1: str, s2: str) -> int:
    """Levenshtein edit distance."""
    if len(s1) < len(s2):
        return edit_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    prev_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        cur_row = [i + 1]
        for j, c2 in enumerate(s2):
            ins = prev_row[j + 1] + 1
            dele = cur_row[j] + 1
            sub = prev_row[j] + (c1 != c2)
            cur_row.append(min(ins, dele, sub))
        prev_row = cur_row
    return prev_row[-1]


def cer(reference: str, hypothesis: str, normalize: bool = True) -> float:
    """Character Error Rate.

    예: ref='안녕하세요', hyp='안녕하새요' → 1/5 = 0.2
    """
    if normalize:
        reference = normalize_korean(reference)
        hypothesis = normalize_korean(hypothesis)
    if not reference:
        return 1.0 if hypothesis else 0.0
    # 공백 제거 (CER 표준 — 공백 차이 무시)
    ref = reference.replace(" ", "")
    hyp = hypothesis.replace(" ", "")
    if not ref:
        return 1.0 if hyp else 0.0
    return edit_distance(ref, hyp) / len(ref)


def batch_cer(refs, hyps, normalize: bool = True) -> float:
    """여러 샘플의 평균 CER."""
    if not refs:
        return 0.0
    total_chars = 0
    total_errs = 0
    for r, h in zip(refs, hyps):
        if normalize:
            r = normalize_korean(r).replace(" ", "")
            h = normalize_korean(h).replace(" ", "")
        if not r:
            continue
        total_chars += len(r)
        total_errs += edit_distance(r, h)
    return total_errs / total_chars if total_chars > 0 else 0.0


# ──────────────────────────────────────────────────────────────
# AI Hub 파일 매칭
# ──────────────────────────────────────────────────────────────
# 실제 데이터의 화자 prefix → 카테고리 매핑
#   환자: PA·PB·PC·PD·PE·PF  (zip 1~6)
#   간호사: HA
#   의사: HB  (※ 코드 원본은 DR로 잘못 가정 → 의사 전체 누락하던 버그 수정)
SPEAKER_PREFIX_TO_CATEGORY = {
    "PA": "환자", "PB": "환자", "PC": "환자",
    "PD": "환자", "PE": "환자", "PF": "환자",
    "HA": "간호사",
    "HB": "의사",
}
VALID_SPEAKER_PREFIXES = set(SPEAKER_PREFIX_TO_CATEGORY)


def prefix_to_category(prefix: str):
    """화자 prefix(PA/HA/HB...) → 한글 카테고리. 모르면 None."""
    return SPEAKER_PREFIX_TO_CATEGORY.get(prefix)


def parse_filename(filename: str) -> dict:
    """파일명에서 메타정보 추출.
    예: HA_0010-1-01-02-M-04-A.wav
        → {'category': 'HA', 'speaker_id': 'HA_0010', 'utt_num': '1',
           'cat1': '01', 'cat2': '02', 'gender': 'M', 'age': '04', 'sub': 'A'}
    """
    name = Path(filename).stem
    parts = name.split("-")
    if len(parts) < 7:
        return {}
    prefix = parts[0].split("_")[0]
    return {
        "speaker_id": parts[0],
        "prefix": prefix,                              # PA~PF / HA / HB
        "category": prefix_to_category(prefix) or "?",  # 환자 / 간호사 / 의사
        "utt_num": parts[1],
        "cat1": parts[2],
        "cat2": parts[3],
        "gender": parts[4],
        "age": parts[5],
        "sub": parts[6],
    }


def load_label_json(json_path: Path) -> dict:
    """라벨 JSON 로드."""
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_label_text(json_path: Path) -> Optional[str]:
    """LabelText 추출 (없으면 None)."""
    try:
        data = load_label_json(json_path)
        text = data.get("전사정보", {}).get("LabelText", "").strip()
        return text if text else None
    except Exception:
        return None


def find_label_for_wav(wav_path: Path, label_root: Path) -> Optional[Path]:
    """원천 WAV에 대응하는 라벨 JSON 찾기.

    구조:
      원천: [T원천]의료진_간호사_X/1/HA_0010/HA_0010-1-...wav
      라벨: [T]라벨링데이터/medsub/간호사/HA_0010/HA_0010-1-...json
    """
    name = wav_path.stem
    speaker_id = name.split("-")[0]
    prefix = speaker_id.split("_")[0]

    cat_kor = SPEAKER_PREFIX_TO_CATEGORY.get(prefix)
    if not cat_kor:
        return None

    # 가능한 라벨 경로들
    candidates = [
        label_root / "medsub" / cat_kor / speaker_id / f"{name}.json",
        label_root / cat_kor / speaker_id / f"{name}.json",
        label_root / speaker_id / f"{name}.json",
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    return None
