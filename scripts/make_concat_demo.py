# -*- coding: utf-8 -*-
"""한 화자의 발화들을 이어붙여 long-form 음성 + 정답 텍스트 파일 생성 (추론 X).
사용: python scripts/make_concat_demo.py <category> <speaker> <target_sec>
출력: voice_emr_baseline/samples/concat_demo/<cat>_<spk>_concat_<sec>s.wav (+ .gt.txt)
"""
import sys, json
from pathlib import Path
import numpy as np, librosa, soundfile as sf

TR = Path("/home/jungmin.cheon/jm_repo/voice_v2/voice_emr/voice_emr_baseline/training")
OUTDIR = Path("/home/jungmin.cheon/jm_repo/voice_v2/voice_emr/voice_emr_baseline/samples/concat_demo")
OUTDIR.mkdir(parents=True, exist_ok=True)

cat = sys.argv[1]; spk = sys.argv[2]
target = float(sys.argv[3]) if len(sys.argv) > 3 else 90.0
SR = 16000; GAP = int(0.3 * SR)   # 발화 사이 0.3초 묵음

recs = [json.loads(l) for l in open(TR / "manifests/exp1/test.jsonl", encoding="utf-8")]
mine = [r for r in recs if r["speaker"] == spk and r["category"] == cat]
mine.sort(key=lambda r: int(r.get("utt_num", 0)))

parts, texts, used, dur = [], [], 0, 0.0
for r in mine:
    try:
        a, _ = librosa.load(r["audio"], sr=SR, mono=True)
    except Exception:
        continue
    parts += [a, np.zeros(GAP, np.float32)]
    texts.append(r["text"]); used += 1
    dur += len(a) / SR + 0.3
    if dur >= target:
        break

wav = np.concatenate(parts).astype(np.float32)
base = f"{cat}_{spk}_concat_{int(dur)}s"
sf.write(OUTDIR / f"{base}.wav", wav, SR)
(OUTDIR / f"{base}.gt.txt").write_text(" ".join(texts), encoding="utf-8")
print(f"[OK] {base}.wav  ({dur:.1f}s, {used}발화)  +  {base}.gt.txt")
