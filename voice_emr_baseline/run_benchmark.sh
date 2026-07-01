#!/usr/bin/env bash
# Voice EMR Baseline 벤치마크 실행 (Linux/Ubuntu용)
#
# 사용: bash run_benchmark.sh [quick|full|gpu|cpu]
#       또는 chmod +x run_benchmark.sh && ./run_benchmark.sh

set -e

echo "======================================================"
echo "  Voice EMR Benchmark"
echo "======================================================"
echo ""

# venv 활성화
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# 샘플 확인
if [ ! -f "samples/sample_short.wav" ] || [ ! -f "samples/sample_long.wav" ]; then
    echo "❌ samples/ 폴더에 sample_short.wav / sample_long.wav 없음"
    echo "   먼저: python generate_samples.py"
    exit 1
fi

# 모드 선택
MODE="${1:-quick}"

case "$MODE" in
    quick)
        python benchmark.py \
            --samples samples/sample_short.wav samples/sample_long.wav \
            --preset quick
        ;;
    full)
        python benchmark.py \
            --samples samples/sample_short.wav samples/sample_long.wav \
            --preset full
        ;;
    gpu)
        python benchmark.py \
            --samples samples/sample_short.wav samples/sample_long.wav \
            --preset full --devices cuda
        ;;
    cpu)
        python benchmark.py \
            --samples samples/sample_short.wav samples/sample_long.wav \
            --preset full --devices cpu
        ;;
    *)
        echo "사용법: $0 [quick|full|gpu|cpu]"
        echo ""
        echo "  quick : STT만 비교 (빠름, 약 5~10분)"
        echo "  full  : SOAP 요약 포함 (약 20~40분)"
        echo "  gpu   : GPU만 사용한 전체 비교"
        echo "  cpu   : CPU만 사용한 전체 비교"
        exit 1
        ;;
esac

echo ""
echo "======================================================"
echo "  완료 — bench_results/report.md 확인"
echo "======================================================"
