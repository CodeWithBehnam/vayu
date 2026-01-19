#!/usr/bin/env python3
"""Benchmark script for decoder performance.

Measures transcription throughput before and after mx.compile optimization.
"""

import time
from pathlib import Path
import mlx.core as mx
from whisper_mlx import transcribe, load_model
from whisper_mlx.audio import log_mel_spectrogram, N_FRAMES, pad_or_trim


def benchmark_transcription(
    audio_path: str,
    model_path: str = "mlx-community/whisper-small",
    num_runs: int = 3,
    warmup_runs: int = 1,
) -> dict:
    """Benchmark transcription throughput.

    Args:
        audio_path: Path to audio file
        model_path: HuggingFace model path
        num_runs: Number of benchmark runs
        warmup_runs: Number of warmup runs (not counted)

    Returns:
        dict with timing statistics
    """
    # Warmup runs to trigger JIT compilation
    print(f"Running {warmup_runs} warmup run(s)...")
    for i in range(warmup_runs):
        _ = transcribe(audio_path, path_or_hf_repo=model_path)
        mx.eval()  # Ensure all operations complete

    # Timed runs
    print(f"Running {num_runs} benchmark run(s)...")
    times = []
    for i in range(num_runs):
        start = time.perf_counter()
        result = transcribe(audio_path, path_or_hf_repo=model_path)
        mx.eval()  # Ensure all operations complete
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        print(f"  Run {i+1}: {elapsed:.3f}s")

    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)

    # Get audio duration for throughput calculation
    from whisper_mlx.audio import load_audio, SAMPLE_RATE
    audio = load_audio(audio_path)
    audio_duration = len(audio) / SAMPLE_RATE

    return {
        "audio_duration_sec": audio_duration,
        "avg_time_sec": avg_time,
        "min_time_sec": min_time,
        "max_time_sec": max_time,
        "speedup_ratio": audio_duration / avg_time,  # Real-time factor
        "text": result["text"][:200] + "..." if len(result["text"]) > 200 else result["text"],
    }


def main():
    audio_path = "data/test3.mp4"
    model_path = "mlx-community/whisper-turbo"  # Use cached model

    if not Path(audio_path).exists():
        print(f"Error: Audio file not found: {audio_path}")
        return

    print("=" * 60)
    print("Decoder Benchmark")
    print("=" * 60)
    print(f"Audio: {audio_path}")
    print(f"Model: {model_path}")
    print()

    results = benchmark_transcription(
        audio_path=audio_path,
        model_path=model_path,
        num_runs=5,
        warmup_runs=3,  # More warmup for JIT compilation
    )

    print()
    print("=" * 60)
    print("Results")
    print("=" * 60)
    print(f"Audio duration: {results['audio_duration_sec']:.2f}s")
    print(f"Avg transcription time: {results['avg_time_sec']:.3f}s")
    print(f"Min time: {results['min_time_sec']:.3f}s")
    print(f"Max time: {results['max_time_sec']:.3f}s")
    print(f"Real-time factor: {results['speedup_ratio']:.2f}x")
    print()
    print("Sample output:")
    print(results["text"])


if __name__ == "__main__":
    main()
