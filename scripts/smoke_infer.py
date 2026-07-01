# -*- coding: utf-8 -*-
"""동봉 체크포인트가 이 PC에서 로드/추론되는지 확인하는 스모크 테스트.
사용: python scripts/smoke_infer.py <wav> [checkpoint_dir]"""
import sys, torch, librosa
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import PeftModel

wav = sys.argv[1]
ckpt = sys.argv[2] if len(sys.argv) > 2 else \
    "voice_emr_baseline/training/outputs/phase_b_small/checkpoint-1000"
base = "openai/whisper-medium"
dev = "cuda" if torch.cuda.is_available() else "cpu"

print(f"device={dev}  base={base}  ckpt={ckpt}")
proc = WhisperProcessor.from_pretrained(base, language="korean", task="transcribe")
model = WhisperForConditionalGeneration.from_pretrained(base, torch_dtype=torch.bfloat16)
model = PeftModel.from_pretrained(model, ckpt).to(dev).eval()
model.generation_config.language = "korean"
model.generation_config.task = "transcribe"
model.generation_config.forced_decoder_ids = None

audio, _ = librosa.load(wav, sr=16000, mono=True)
feats = proc.feature_extractor(audio, sampling_rate=16000).input_features
feats = torch.tensor(feats, dtype=torch.bfloat16, device=dev)
with torch.no_grad():
    ids = model.generate(feats, max_new_tokens=200)
print("HYP:", proc.tokenizer.batch_decode(ids, skip_special_tokens=True)[0])
print("OK — 체크포인트 로드/추론 성공")
