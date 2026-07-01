# 모델별 STT 정확도 / 속도 비교

정답 텍스트(samples/) 대비 전사 결과의 CER 측정.

| config | Whisper | CER | 정확도 | 평균 RTF | 총 처리(s) | 음성(s) | 파일 |
|---|---|---:|---:|---:|---:|---:|---:|
| medium | CUDA | STT only | medium | 3.35% | 96.65% | 0.028 | 30.98 | 1108.8 | 18 |
| medium | CUDA | Qwen2.5-3B (GPU) | medium | 3.35% | 96.65% | 0.029 | 32.04 | 1108.8 | 18 |
| small | CUDA | STT only | small | 5.78% | 94.22% | 0.017 | 19.25 | 1108.8 | 18 |
| large-v3 | CUDA | STT only | large-v3 | 6.81% | 93.19% | 0.077 | 85.84 | 1108.8 | 18 |
| large-v3 | CUDA | Qwen2.5-7B (GPU) | large-v3 | 9.84% | 90.16% | 0.074 | 82.33 | 1108.8 | 18 |
| large-v3 | CUDA | Qwen2.5-3B (GPU) | large-v3 | 11.83% | 88.17% | 0.071 | 78.81 | 1108.8 | 18 |
| tiny | CUDA | STT only | tiny | 15.71% | 84.29% | 0.012 | 13.54 | 1108.8 | 18 |

- **CER**: 문자 가중 평균 (낮을수록 정확)
- **정확도**: 1 − CER
- **RTF**: 처리시간 / 음성길이 (낮을수록 빠름)