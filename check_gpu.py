#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPU 환경 진단 스크립트

실행: python check_gpu.py
"""
import os
import platform
import subprocess
import sys


def section(title):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def check_command(cmd):
    """명령어 실행해서 결과 반환."""
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=10, shell=True
        )
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)


def main():
    section("1. 시스템 정보")
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Architecture: {platform.machine()}")

    # ─── NVIDIA GPU 확인 ───
    section("2. NVIDIA GPU 확인")
    ok, stdout, stderr = check_command("nvidia-smi")
    if ok:
        # 첫 20줄만 출력
        print("✅ nvidia-smi 실행 가능")
        for line in stdout.split("\n")[:20]:
            print(f"   {line}")
    else:
        print("❌ nvidia-smi 실행 실패")
        print("   원인 후보:")
        print("   - NVIDIA GPU 없음 (Intel/AMD만 있음)")
        print("   - NVIDIA 드라이버 설치 안 됨")
        print("   - PATH 문제")
        print("\n   해결: https://www.nvidia.com/Download/index.aspx 에서 드라이버 설치")

    # ─── Windows: 그래픽카드 종류 ───
    if platform.system() == "Windows":
        section("3. 그래픽카드 정보 (Windows)")
        ok, stdout, _ = check_command(
            'powershell -Command "Get-WmiObject Win32_VideoController | '
            'Select-Object Name, AdapterRAM, DriverVersion | Format-List"'
        )
        if ok:
            print(stdout.strip())
        else:
            print("정보 가져오기 실패")

    # ─── PyTorch CUDA 지원 ───
    section("4. PyTorch GPU 지원")
    try:
        import torch
        print(f"PyTorch 버전: {torch.__version__}")
        print(f"CUDA 버전 (PyTorch 빌드): {torch.version.cuda}")
        print(f"cuDNN 버전: {torch.backends.cudnn.version() if torch.cuda.is_available() else 'N/A'}")
        print()

        if torch.cuda.is_available():
            print(f"✅ CUDA 사용 가능!")
            print(f"   GPU 개수: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"   GPU #{i}: {props.name}")
                print(f"          메모리: {props.total_memory / 1024**3:.1f} GB")
                print(f"          Compute Capability: {props.major}.{props.minor}")
        else:
            print(f"❌ CUDA 사용 불가")
            if torch.version.cuda is None:
                print(f"   ⚠ PyTorch가 CPU 전용 빌드로 설치됨")
                print(f"     해결: pip uninstall torch torchvision torchaudio")
                print(f"           pip install torch --index-url https://download.pytorch.org/whl/cu121")
            else:
                print(f"   ⚠ PyTorch는 CUDA 빌드인데 GPU 못 잡음")
                print(f"     원인: NVIDIA 드라이버 / GPU 자체 문제")
    except ImportError:
        print("❌ PyTorch 설치 안 됨")
        print("   해결: pip install torch --index-url https://download.pytorch.org/whl/cu121")

    # ─── faster-whisper GPU 지원 ───
    section("5. faster-whisper GPU 테스트")
    try:
        from faster_whisper import WhisperModel
        try:
            print("GPU 모드 시도 (tiny 모델로 가볍게)...")
            model = WhisperModel("tiny", device="cuda", compute_type="float16")
            print("✅ faster-whisper GPU 모드 OK")
            del model
        except Exception as e:
            print(f"❌ faster-whisper GPU 모드 실패")
            print(f"   에러: {e}")
            print(f"   → CPU 모드로 폴백 권장")
    except ImportError:
        print("⚠ faster-whisper 설치 안 됨 (선택사항)")

    # ─── 메모리 정보 ───
    section("6. 시스템 메모리")
    try:
        import psutil
        vm = psutil.virtual_memory()
        print(f"전체 RAM: {vm.total / 1024**3:.1f} GB")
        print(f"사용 가능: {vm.available / 1024**3:.1f} GB")
        print(f"사용 중: {vm.percent}%")
    except ImportError:
        print("psutil 없음 — 메모리 정보 생략")

    # ─── 결론 / 추천 ───
    section("📋 진단 결론 및 권장 사항")

    try:
        import torch
        has_cuda = torch.cuda.is_available()
        if has_cuda:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"✅ GPU 사용 가능: {gpu_name} ({gpu_mem_gb:.1f}GB)")
            print(f"\n권장 설정:")
            if gpu_mem_gb >= 16:
                print(f"  python baseline.py audio.wav --whisper-model large-v3 --llm Qwen/Qwen2.5-7B-Instruct")
            elif gpu_mem_gb >= 8:
                print(f"  python baseline.py audio.wav --whisper-model medium --llm Qwen/Qwen2.5-3B-Instruct")
            else:
                print(f"  python baseline.py audio.wav --whisper-model small --llm Qwen/Qwen2.5-1.5B-Instruct")
        else:
            print(f"⚠️ GPU 사용 불가 — CPU 모드로 동작")
            print(f"\n권장 설정 (CPU 환경):")
            print(f"  python baseline.py audio.wav --device cpu --whisper-model small --cpu-llm 3B")
            print(f"  또는 LLM 끄고:")
            print(f"  python baseline.py audio.wav --device cpu --whisper-model small --skip-llm")
    except ImportError:
        print("⚠️ PyTorch가 없어서 진단 불완전.")
        print("\n먼저 PyTorch 설치 필요:")
        print("  pip install torch --index-url https://download.pytorch.org/whl/cu121")

    print()


if __name__ == "__main__":
    main()
