#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI Hub 데이터 → 학습용 manifest 변환

WAV ↔ JSON 매칭, 메타정보 추출, train/val/test 분할 (화자별).

사용:
    # 환자 전체
    python prepare_dataset.py --data-dir /data/aihub/Training --output manifests/full

    # 환자만 100명
    python prepare_dataset.py --data-dir ... --output manifests/100p \\
        --categories 환자 --max-speakers 100

    # 빠른 검증 (1명)
    python prepare_dataset.py --data-dir ... --output manifests/quick \\
        --categories 환자 --max-speakers 1

출력:
    manifests/<name>/
      ├── train.jsonl    (학습 manifest)
      ├── val.jsonl      (검증 manifest)
      ├── test.jsonl     (테스트 manifest)
      └── stats.json     (통계)
"""
import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import sys as _sys
_sys.path.insert(0, str(Path(__file__).parent))
from utils import (  # noqa
    get_label_text, parse_filename, find_label_for_wav,
    VALID_SPEAKER_PREFIXES, prefix_to_category,
)


def find_speaker_folders(data_dir: Path, categories: list) -> dict:
    """data_dir 전체를 훑어 화자 폴더(PA~PF / HA / HB)를 카테고리별로 모은다.

    실제 압축 해제 구조(예):
        [T원천]환자_1/1/PA_0016/*.wav
        [T원천]의료진_의사_1/1/HB_0062/*.wav
    폴더 이름(대괄호/숫자)에 의존하지 않고 화자 prefix로 분류하므로
    어떤 레이아웃이든 동작한다.

    Returns:
        {category: {speaker_id: [folder_path, ...]}, ...}
    """
    wanted = set(categories)
    speakers = defaultdict(lambda: defaultdict(list))
    seen = set()

    # 화자 폴더는 <원천>/<번호>/<화자> 깊이(=data_dir/*/*/*)에 있음.
    # 1.1M wav 전체를 훑지 않도록 고정 깊이 글롭을 우선 사용하고,
    # 비면 안전망으로 rglob 으로 폴백.
    candidates = list(data_dir.glob("*/*/*"))
    if not any(c.is_dir() for c in candidates):
        candidates = [p for p in data_dir.rglob("*") if p.is_dir()]

    for inner in candidates:
        if not inner.is_dir():
            continue
        name = inner.name
        if "_" not in name:
            continue
        prefix = name.split("_")[0]
        if prefix not in VALID_SPEAKER_PREFIXES:
            continue
        cat = prefix_to_category(prefix)
        if cat not in wanted:
            continue
        key = str(inner)
        if key in seen:
            continue
        seen.add(key)
        speakers[cat][name].append(inner)

    return speakers


def collect_wav_label_pairs(speaker_folder: Path, label_root: Path) -> list:
    """한 화자 폴더의 WAV ↔ 라벨 페어 수집."""
    pairs = []
    for wav in sorted(speaker_folder.glob("*.wav")):
        label_path = find_label_for_wav(wav, label_root)
        if label_path is None:
            continue
        text = get_label_text(label_path)
        if not text:
            continue
        pairs.append({
            "audio": str(wav),
            "text": text,
            "label_json": str(label_path),
            **parse_filename(wav.name),
        })
    return pairs


def split_by_speaker(speaker_data: dict, train_ratio: float, val_ratio: float,
                     seed: int = 42):
    """화자 단위로 train/val/test 분할 (data leakage 방지)."""
    rng = random.Random(seed)

    all_speakers = []
    for cat, spk_map in speaker_data.items():
        for spk, samples in spk_map.items():
            all_speakers.append((cat, spk, samples))
    rng.shuffle(all_speakers)

    n = len(all_speakers)
    n_val = int(n * val_ratio)
    n_test = int(n * (1 - train_ratio - val_ratio))
    # 화자가 적어도 val/test 가 비지 않도록 최소 1명 보장 (n>=3 일 때)
    if n >= 3:
        n_val = max(1, n_val)
        n_test = max(1, n_test)
    n_train = n - n_val - n_test
    if n_train < 1:  # 극단적으로 적은 경우 train 우선 확보
        n_train, n_val, n_test = max(1, n - 2), min(1, n - 1), max(0, n - 2)

    train_spk = all_speakers[:n_train]
    val_spk = all_speakers[n_train:n_train + n_val]
    test_spk = all_speakers[n_train + n_val:]

    def flatten(speakers):
        out = []
        for cat, spk, samples in speakers:
            for s in samples:
                s["category"] = cat
                s["speaker"] = spk
                out.append(s)
        return out

    return flatten(train_spk), flatten(val_spk), flatten(test_spk)


def _split_speaker_list(spk_items, train_ratio, val_ratio):
    """화자 리스트(셔플됨)를 ratio 로 train/val/test 인덱스 분할 (최소 보장 포함)."""
    n = len(spk_items)
    n_val = int(n * val_ratio)
    n_test = int(n * (1 - train_ratio - val_ratio))
    if n >= 3:
        n_val = max(1, n_val)
        n_test = max(1, n_test)
    n_train = n - n_val - n_test
    if n_train < 1:
        n_train, n_val, n_test = max(1, n - 2), min(1, n - 1), max(0, n - 2)
    return (spk_items[:n_train],
            spk_items[n_train:n_train + n_val],
            spk_items[n_train + n_val:])


def split_by_speaker_stratified(speaker_data: dict, train_ratio: float,
                                val_ratio: float, seed: int = 42):
    """카테고리별로 독립 분할 후 합침 → val/test 에 모든 카테고리 포함 보장.

    (전역 분할은 화자를 통째로 섞어 test 가 특정 카테고리에 쏠릴 수 있음.
     before/after CER 을 카테고리별로 공정 비교하려면 층화 분할이 필요.)
    """
    rng = random.Random(seed)
    train_all, val_all, test_all = [], [], []
    for cat in sorted(speaker_data.keys()):
        spk_items = [(cat, spk, samples) for spk, samples in speaker_data[cat].items()]
        rng.shuffle(spk_items)
        tr, va, te = _split_speaker_list(spk_items, train_ratio, val_ratio)
        for bucket, dest in [(tr, train_all), (va, val_all), (te, test_all)]:
            for c, spk, samples in bucket:
                for s in samples:
                    s["category"] = c
                    s["speaker"] = spk
                    dest.append(s)
    rng.shuffle(train_all)
    return train_all, val_all, test_all


def save_manifest(samples: list, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")


def compute_stats(samples: list) -> dict:
    """기본 통계 계산."""
    cat_counts = Counter(s.get("category", "?") for s in samples)
    gender_counts = Counter(s.get("gender", "?") for s in samples)
    age_counts = Counter(s.get("age", "?") for s in samples)
    speaker_counts = len(set(s.get("speaker", "?") for s in samples))
    text_lens = [len(s.get("text", "")) for s in samples]
    return {
        "n_samples": len(samples),
        "n_speakers": speaker_counts,
        "by_category": dict(cat_counts),
        "by_gender": dict(gender_counts),
        "by_age": dict(age_counts),
        "text_chars_avg": round(sum(text_lens) / max(len(text_lens), 1), 1),
        "text_chars_max": max(text_lens) if text_lens else 0,
        "text_chars_min": min(text_lens) if text_lens else 0,
    }


def main():
    parser = argparse.ArgumentParser(description="AI Hub → Whisper 학습 manifest")
    parser.add_argument("--data-dir", required=True, help="Training 폴더 경로")
    parser.add_argument("--output", required=True, help="manifest 출력 폴더")
    parser.add_argument("--categories", nargs="+",
                        default=["환자", "의사", "간호사"],
                        choices=["환자", "의사", "간호사"])
    parser.add_argument("--max-speakers", type=int, default=None,
                        help="카테고리별 최대 화자 수 (검증용)")
    parser.add_argument("--max-utts-per-speaker", type=int, default=None,
                        help="화자당 최대 발화 수 (카테고리 균형·I/O 절감용; "
                             "의사처럼 발화 많은 화자의 지배 방지)")
    parser.add_argument("--stratify", action="store_true",
                        help="카테고리별로 독립 분할(층화) → val/test 에 모든 "
                             "카테고리가 균형 있게 포함됨 (공정한 before/after 비교용)")
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"❌ Data dir 없음: {data_dir}")
        sys.exit(1)

    # 라벨 루트 찾기
    label_root = None
    for cand in [
        data_dir / "[T]라벨링데이터" / "medsub",
        data_dir / "[T]라벨링데이터",
    ]:
        if cand.exists():
            label_root = cand
            break
    if label_root is None:
        print(f"❌ 라벨 폴더 없음 — {data_dir} 안에 [T]라벨링데이터 풀어주세요")
        sys.exit(1)

    print(f"📂 Data: {data_dir}")
    print(f"🏷  Label: {label_root}")
    print(f"📋 Categories: {args.categories}")
    if args.max_speakers:
        print(f"🎯 화자 수 제한: {args.max_speakers}명/카테고리")

    # 화자 폴더 모으기
    print(f"\n🔍 화자 폴더 탐색 중...")
    speakers = find_speaker_folders(data_dir, args.categories)
    for cat, spk_map in speakers.items():
        print(f"  - {cat}: {len(spk_map)}명")

    # max-speakers 적용
    if args.max_speakers:
        rng = random.Random(args.seed)
        for cat in list(speakers.keys()):
            spk_list = list(speakers[cat].keys())
            rng.shuffle(spk_list)
            speakers[cat] = {k: speakers[cat][k] for k in spk_list[:args.max_speakers]}
            print(f"  → {cat}: {len(speakers[cat])}명으로 제한")

    # WAV-라벨 페어 수집 (화자당 상한을 라벨 읽기 전에 적용 → NFS I/O 절감)
    print(f"\n🎵 WAV ↔ 라벨 매칭 중...")
    if args.max_utts_per_speaker:
        print(f"   (화자당 발화 상한: {args.max_utts_per_speaker})")
    rng_cap = random.Random(args.seed + 1)
    speaker_data = defaultdict(lambda: defaultdict(list))
    total = 0
    for cat, spk_map in speakers.items():
        for spk, folders in spk_map.items():
            wavs = []
            for folder in folders:
                wavs.extend(sorted(folder.glob("*.wav")))
            if args.max_utts_per_speaker and len(wavs) > args.max_utts_per_speaker:
                wavs = rng_cap.sample(wavs, args.max_utts_per_speaker)
            samples = []
            for wav in wavs:
                label_path = find_label_for_wav(wav, label_root)
                if label_path is None:
                    continue
                text = get_label_text(label_path)
                if not text:
                    continue
                samples.append({
                    "audio": str(wav),
                    "text": text,
                    "label_json": str(label_path),
                    **parse_filename(wav.name),
                })
            if samples:
                speaker_data[cat][spk] = samples
                total += len(samples)
        print(f"  - {cat}: {sum(len(s) for s in speaker_data[cat].values())} 페어")

    print(f"\n📊 총 {total}개 WAV-라벨 페어")

    # 분할
    mode = "층화(카테고리별)" if args.stratify else "전역"
    print(f"\n✂  화자 단위 분할 중 ({mode}, leakage 방지)...")
    if args.stratify:
        train, val, test = split_by_speaker_stratified(
            speaker_data, args.train_ratio, args.val_ratio, args.seed
        )
    else:
        train, val, test = split_by_speaker(
            speaker_data, args.train_ratio, args.val_ratio, args.seed
        )
    print(f"  - train: {len(train)} 샘플")
    print(f"  - val:   {len(val)} 샘플")
    print(f"  - test:  {len(test)} 샘플")

    # 저장
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    save_manifest(train, out / "train.jsonl")
    save_manifest(val, out / "val.jsonl")
    save_manifest(test, out / "test.jsonl")

    stats = {
        "train": compute_stats(train),
        "val": compute_stats(val),
        "test": compute_stats(test),
        "config": vars(args),
    }
    (out / "stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"\n💾 저장 완료: {out.resolve()}")
    print(f"\n📈 통계 미리보기:")
    print(json.dumps(stats["train"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
