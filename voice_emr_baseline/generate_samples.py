#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
한국어 의료 대화 테스트 샘플 생성 — Microsoft Edge TTS (무료, 상업적 사용 가능)

생성 파일:
    samples/sample_short.wav  (약 10초, 짧은 진료 대화)
    samples/sample_long.wav   (약 60초, 긴 진료 대화)
    samples/sample_short.txt  (전사 텍스트 — 정답 비교용)
    samples/sample_long.txt   (전사 텍스트 — 정답 비교용)

설치:
    pip install edge-tts pydub
    # ffmpeg도 필요 (winget install ffmpeg)

실행:
    python generate_samples.py
"""
import asyncio
import sys
from pathlib import Path

try:
    import edge_tts
    from pydub import AudioSegment
except ImportError:
    print("❌ 패키지 없음. 설치: pip install edge-tts pydub")
    sys.exit(1)


# Edge TTS 한국어 보이스 (마이크로소프트 무료)
VOICE_DOCTOR = "ko-KR-InJoonNeural"     # 남성 (의사 역할)
VOICE_PATIENT = "ko-KR-SunHiNeural"     # 여성 (환자 역할)
# 다른 한국어 보이스: ko-KR-HyunsuNeural (남), ko-KR-GookMinNeural (남)


# ─── 짧은 샘플 (10초 내외) ───
SHORT_DIALOGUE = [
    ("의사", "어디가 불편하셔서 오셨어요?"),
    ("환자", "일주일 전부터 머리가 자주 아파요. 약을 먹어도 잘 안 들어요."),
]

# ─── 긴 샘플 (60초 내외, 흉통 호소 시나리오) ───
LONG_DIALOGUE = [
    ("의사", "안녕하세요. 어떻게 오셨어요?"),
    ("환자", "며칠 전부터 가슴이 답답하고 호흡이 좀 가빠요. 계단을 올라갈 때 특히 심해요."),
    ("의사", "언제부터 그런 증상이 시작되셨나요?"),
    ("환자", "한 일주일 정도 됐어요. 처음엔 그냥 피곤한가 했는데 점점 심해지는 것 같아서요."),
    ("의사", "평소에 운동은 하시나요? 기저질환은 있으세요?"),
    ("환자", "운동은 거의 안 하고요. 작년에 고혈압 진단받아서 약 먹고 있어요. 그 외엔 특별한 건 없어요."),
    ("의사", "흡연은 하시나요?"),
    ("환자", "오 년 전에 끊었습니다."),
    ("의사", "알겠습니다. 일단 심전도 검사하고 흉부 엑스레이 한 번 찍어볼게요. 혈액검사도 해서 콜레스테롤 수치도 확인해보겠습니다."),
    ("환자", "네, 알겠습니다."),
]


async def tts_save_segment(text, voice, output_path):
    """단일 발화를 wav로."""
    communicate = edge_tts.Communicate(text, voice, rate="+0%", volume="+0%")
    tmp_mp3 = output_path.with_suffix(".mp3")
    await communicate.save(str(tmp_mp3))
    audio = AudioSegment.from_mp3(tmp_mp3)
    audio = audio.set_channels(1).set_frame_rate(16000).set_sample_width(2)
    audio.export(output_path, format="wav")
    tmp_mp3.unlink()


async def generate(dialogue, output_path, gap_ms=400):
    """전체 대화를 한 파일로 합성."""
    print(f"\n📝 [{output_path.name}] 생성 중...")
    tmp_dir = output_path.parent / "_tmp"
    tmp_dir.mkdir(exist_ok=True)

    segments = []
    for i, (speaker, text) in enumerate(dialogue):
        voice = VOICE_DOCTOR if speaker == "의사" else VOICE_PATIENT
        tmp_path = tmp_dir / f"seg_{i:02d}.wav"
        print(f"   [{speaker}] {text}")
        await tts_save_segment(text, voice, tmp_path)
        segments.append(AudioSegment.from_wav(tmp_path))

    # 발화 사이 무음 추가
    silence = AudioSegment.silent(duration=gap_ms, frame_rate=16000)
    combined = segments[0]
    for seg in segments[1:]:
        combined += silence + seg

    combined = combined.set_channels(1).set_frame_rate(16000).set_sample_width(2)
    combined.export(output_path, format="wav")
    duration_sec = len(combined) / 1000
    print(f"   ✅ {output_path.name} ({duration_sec:.1f}초)")

    # 전사 텍스트 같이 저장 (정답 비교용)
    txt_path = output_path.with_suffix(".txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        for speaker, text in dialogue:
            f.write(f"[{speaker}] {text}\n")
    print(f"   ✅ {txt_path.name} (정답 전사)")

    # 정리
    for f in tmp_dir.glob("*"):
        f.unlink()
    tmp_dir.rmdir()


async def main():
    out_dir = Path("samples")
    out_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("  한국어 의료 대화 테스트 샘플 생성")
    print("=" * 60)

    try:
        await generate(SHORT_DIALOGUE, out_dir / "sample_short.wav")
        await generate(LONG_DIALOGUE, out_dir / "sample_long.wav")
    except Exception as e:
        print(f"\n❌ 오류: {e}")
        print("인터넷 연결 확인 후 다시 시도하세요.")
        sys.exit(1)

    print(f"\n🎉 완료 — {out_dir.resolve()} 폴더 확인")
    print(f"\n다음 단계:")
    print(f"  python baseline.py samples/sample_short.wav --skip-llm")
    print(f"  python baseline.py samples/sample_long.wav --cpu-llm 3B")


if __name__ == "__main__":
    asyncio.run(main())
