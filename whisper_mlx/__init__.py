# Copyright © 2023-2024 Apple Inc. and contributors

"""
Vayu (وایو) - The fastest Whisper implementation on Apple Silicon

Named after the ancient Persian god of wind — the swiftest force in nature.

A unified implementation combining ml-explore/mlx-examples/whisper features
with lightning-whisper-mlx batched decoding for optimal performance.

Features:
- Batched decoding for higher throughput
- CLI with multiple output formats (txt, vtt, srt, tsv, json)
- Word-level timestamps
- Multiple model support (tiny, base, small, medium, large, turbo, distil)
- Quantization support (4bit, 8bit)
- Simple API via LightningWhisperMLX wrapper

Quick Start:
    # Simple API
    from whisper_mlx import LightningWhisperMLX

    whisper = LightningWhisperMLX(model="distil-large-v3", batch_size=12)
    result = whisper.transcribe("audio.mp3")
    print(result["text"])

    # Full API
    from whisper_mlx import transcribe

    result = transcribe(
        "audio.mp3",
        path_or_hf_repo="mlx-community/whisper-turbo",
        batch_size=6,
        language="en",
    )

CLI Usage:
    vayu audio.mp3 --model mlx-community/whisper-turbo --batch-size 6
"""

from .audio import (
    CHUNK_LENGTH,
    FRAMES_PER_SECOND,
    HOP_LENGTH,
    N_FFT,
    N_FRAMES,
    N_SAMPLES,
    SAMPLE_RATE,
    TOKENS_PER_SECOND,
    load_audio,
    log_mel_spectrogram,
    pad_or_trim,
)
from .decoding import DecodingOptions, DecodingResult, decode, detect_language
from .lightning import LightningWhisperMLX
from .load_models import load_model
from .tokenizer import LANGUAGES, Tokenizer, get_tokenizer
from .transcribe import transcribe
from .whisper import ModelDimensions, Whisper
from .writers import (
    ResultWriter,
    WriteJSON,
    WriteSRT,
    WriteTSV,
    WriteTXT,
    WriteVTT,
    get_writer,
)
from .speculative import (
    SpeculativeDecoder,
    speculative_transcribe,
    VADProcessor,
    parallel_chunk_transcribe,
)

__version__ = "1.0.0"

__all__ = [
    # Main transcription function
    "transcribe",
    # Simple API wrapper
    "LightningWhisperMLX",
    # Model loading
    "load_model",
    "Whisper",
    "ModelDimensions",
    # Decoding
    "decode",
    "detect_language",
    "DecodingOptions",
    "DecodingResult",
    # Audio processing
    "load_audio",
    "log_mel_spectrogram",
    "pad_or_trim",
    "SAMPLE_RATE",
    "N_FFT",
    "HOP_LENGTH",
    "CHUNK_LENGTH",
    "N_SAMPLES",
    "N_FRAMES",
    "FRAMES_PER_SECOND",
    "TOKENS_PER_SECOND",
    # Tokenizer
    "get_tokenizer",
    "Tokenizer",
    "LANGUAGES",
    # Writers
    "get_writer",
    "ResultWriter",
    "WriteTXT",
    "WriteVTT",
    "WriteSRT",
    "WriteTSV",
    "WriteJSON",
    # Novel optimizations
    "SpeculativeDecoder",
    "speculative_transcribe",
    "VADProcessor",
    "parallel_chunk_transcribe",
]
