#!/usr/bin/env python3
"""
Time sequential (batch_size=1) against batched decoding on one audio file.

Usage:
    python scripts/benchmark.py audio.mp3 --model distil-large-v3 --batch-size 12

The model is loaded and warmed up before timing, and the audio is decoded once
up front, so the numbers cover transcription only. Use a few minutes of real
speech: short clips fit in one window and can't benefit from batching.
"""

import argparse
import platform
import time

import mlx.core as mx

from whisper_mlx import SAMPLE_RATE, load_audio, transcribe
from whisper_mlx.utils import resolve_model_path


def time_transcription(audio: mx.array, repo: str, batch_size: int, language):
    start = time.perf_counter()
    result = transcribe(
        audio,
        path_or_hf_repo=repo,
        batch_size=batch_size,
        language=language,
        verbose=None,
    )
    return time.perf_counter() - start, result


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    parser.add_argument("audio", help="Audio file to transcribe")
    parser.add_argument(
        "--model", default="distil-large-v3", help="Model name or HuggingFace repo"
    )
    parser.add_argument("--quant", default=None, choices=["4bit", "8bit"])
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument(
        "--language",
        default=None,
        help="Language code; set it to leave language detection out of the timings",
    )
    parser.add_argument("--runs", type=int, default=3, help="Timed runs per setting")
    args = parser.parse_args()

    repo = resolve_model_path(args.model, args.quant)
    audio = load_audio(args.audio)
    duration = audio.shape[0] / SAMPLE_RATE

    print(f"Model:  {repo}")
    print(f"Audio:  {args.audio} ({duration:.1f}s)")
    print(f"System: {platform.platform()}, MLX {mx.__version__}")
    print()

    # Load the model and compile kernels outside the timed runs
    time_transcription(audio[: 30 * SAMPLE_RATE], repo, 1, args.language)

    best = {}
    for batch_size in dict.fromkeys([1, args.batch_size]):
        times = [
            time_transcription(audio, repo, batch_size, args.language)[0]
            for _ in range(args.runs)
        ]
        best[batch_size] = min(times)
        print(
            f"batch_size={batch_size:<3} best {best[batch_size]:7.2f}s  "
            f"({duration / best[batch_size]:5.1f}x real time)"
        )

    if args.batch_size != 1:
        print(f"\nSpeed-up: {best[1] / best[args.batch_size]:.2f}x")


if __name__ == "__main__":
    main()
