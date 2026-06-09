#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI Hub 의료 음성 데이터 → 테스트 샘플 준비

기능:
- AI Hub 단일 발화 WAV들을 환자별로 묶어 연속 대화 파일 생성
- 정답 텍스트 (LabelText) 추출 및 저장
- 다양한 길이(10초/30초/60초)의 샘플 자동 구성

Windows에서 실행:
    python prepare_aihub_samples.py --source "D:\\aihubb\\비대면 진료를 위한 의료진 및 환자 음성\\Training" --output samples\\aihub --max-per-patient 20

출력:
    samples/aihub/
      ├── PA_0016_short.wav    (약 10초)
      ├── PA_0016_short.txt    (정답)
      ├── PA_0016_long.wav     (약 60초)
      ├── PA_0016_long.txt
      └── ...
"""
import argparse
import json
import random
from pathlib import Path
from typing import Optional

try:
    from pydub import AudioSegment
except ImportError:
    print("❌ pydub 설치 필요: pip install pydub")
    raise SystemExit(1)


def find_patient_folders(source_dir: Path):
    """AI Hub 환자 폴더 구조 탐색.

    예상 구조:
      Training/1/PA_0016/PA_0016-10-01-...wav
      Training/[T]라벨링데이터/medsub/환자/PA_0016/PA_0016-10-01-...json
    """
    # 원천 음성 폴더 탐색
    candidates = []
    for sub in source_dir.iterdir():
        if not sub.is_dir():
            continue
        # Training/1/PA_xxx 패턴
        for inner in sub.iterdir():
            if inner.is_dir() and inner.name.startswith("PA_"):
                candidates.append(inner)
    return sorted(candidates)


def find_label_for(wav_path: Path, label_root: Path) -> Optional[Path]:
    """WAV 파일에 대응하는 JSON 라벨 찾기."""
    name = wav_path.stem  # PA_0016-10-01-03-M-09-A
    patient_id = name.split("-")[0]  # PA_0016
    candidate = label_root / patient_id / f"{name}.json"
    if candidate.exists():
        return candidate
    # 다른 위치도 탐색
    for cat in ["환자", "의사", "간호사"]:
        candidate2 = label_root / cat / patient_id / f"{name}.json"
        if candidate2.exists():
            return candidate2
    return None


def read_label_text(json_path: Path) -> str:
    """JSON에서 LabelText 추출."""
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("전사정보", {}).get("LabelText", "").strip()
    except Exception:
        return ""


def build_sample(wav_paths_with_text, target_duration_sec, output_wav, output_txt,
                 gap_ms=300):
    """여러 WAV와 텍스트를 묶어 target 길이의 대화 파일 생성."""
    combined_audio = None
    combined_text = []
    silence = AudioSegment.silent(duration=gap_ms, frame_rate=16000)

    for wav_path, text in wav_paths_with_text:
        try:
            audio = AudioSegment.from_wav(str(wav_path))
            audio = audio.set_channels(1).set_frame_rate(16000).set_sample_width(2)
            if combined_audio is None:
                combined_audio = audio
            else:
                combined_audio = combined_audio + silence + audio
            combined_text.append(text)

            # 목표 길이 도달하면 중단
            if len(combined_audio) / 1000 >= target_duration_sec:
                break
        except Exception as e:
            print(f"  ⚠ 스킵: {wav_path.name} ({e})")
            continue

    if combined_audio is None or len(combined_audio) < 1000:
        return False

    output_wav.parent.mkdir(parents=True, exist_ok=True)
    combined_audio.export(str(output_wav), format="wav")

    with open(output_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(combined_text))

    duration = len(combined_audio) / 1000
    print(f"  ✅ {output_wav.name} ({duration:.1f}초, {len(combined_text)}개 발화)")
    return True


def main():
    parser = argparse.ArgumentParser(description="AI Hub 의료 음성 → 테스트 샘플 변환")
    parser.add_argument("--source", required=True,
                        help="AI Hub Training 폴더 경로")
    parser.add_argument("--output", default="samples/aihub",
                        help="출력 폴더 (기본: samples/aihub)")
    parser.add_argument("--patients", type=int, default=5,
                        help="대상 환자 수 (기본: 5명)")
    parser.add_argument("--lengths", type=str, default="10,30,60",
                        help="생성할 길이(초) 쉼표구분 (기본: 10,30,60)")
    parser.add_argument("--max-per-patient", type=int, default=30,
                        help="환자당 사용할 최대 발화 수 (기본: 30)")
    parser.add_argument("--seed", type=int, default=42,
                        help="환자 선택 랜덤 시드 (재현 가능)")
    args = parser.parse_args()

    source = Path(args.source)
    if not source.exists():
        print(f"❌ source 폴더 없음: {source}")
        return

    print(f"📂 source: {source}")
    print(f"🎯 출력: {args.output}\n")

    # 환자 폴더 찾기
    patient_folders = find_patient_folders(source)
    print(f"발견된 환자 폴더: {len(patient_folders)}개")

    if not patient_folders:
        print("❌ 환자 폴더 없음 — source 경로 확인 필요")
        return

    # 라벨 루트 찾기
    label_root = None
    for candidate in [
        source / "[T]라벨링데이터" / "medsub" / "환자",
        source / "[T]라벨링데이터",
    ]:
        if candidate.exists():
            label_root = candidate
            break

    if label_root is None:
        print("⚠️  라벨 폴더 못 찾음 — 정답 텍스트 없이 진행")
    else:
        print(f"📋 라벨 폴더: {label_root}\n")

    # 환자 선택
    random.seed(args.seed)
    selected = random.sample(patient_folders, min(args.patients, len(patient_folders)))

    lengths = [int(x.strip()) for x in args.lengths.split(",")]
    output_dir = Path(args.output)

    print(f"선택된 환자: {[p.name for p in selected]}\n")

    success_count = 0
    for patient_dir in selected:
        print(f"🩺 {patient_dir.name}")
        wavs = sorted(patient_dir.glob("*.wav"))[:args.max_per_patient]

        # WAV + 텍스트 매칭
        wav_text_pairs = []
        for wav in wavs:
            if label_root:
                label_path = find_label_for(wav, label_root)
                if label_path:
                    text = read_label_text(label_path)
                else:
                    # .txt 같은 폴더에 있는지 확인
                    txt_path = wav.with_suffix(".txt")
                    text = txt_path.read_text(encoding="utf-8").strip() if txt_path.exists() else ""
            else:
                text = ""
            wav_text_pairs.append((wav, text))

        if not wav_text_pairs:
            print(f"  ⚠ WAV 없음 — 스킵")
            continue

        # 각 길이별로 샘플 생성
        for length in lengths:
            label = "short" if length <= 15 else ("mid" if length <= 45 else "long")
            output_wav = output_dir / f"{patient_dir.name}_{label}_{length}s.wav"
            output_txt = output_dir / f"{patient_dir.name}_{label}_{length}s.txt"
            if build_sample(wav_text_pairs, length, output_wav, output_txt):
                success_count += 1

        print()

    print(f"🎉 완료 — {success_count}개 샘플 생성")
    print(f"📁 위치: {output_dir.resolve()}")
    print(f"\n다음 단계:")
    print(f"  1. (선택) 합성 샘플과 비교: 두 폴더 모두 벤치마크 가능")
    print(f"  2. 서버로 전송:")
    print(f"     scp -r {output_dir} user@server:~/voice-emr-baseline/samples/")
    print(f"  3. 서버에서 벤치마크:")
    print(f"     python benchmark.py --samples samples/aihub/*.wav --preset full")


if __name__ == "__main__":
    main()
