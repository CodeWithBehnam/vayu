# Copyright © 2024 Apple Inc.
# Centralized constants for whisper_mlx

from typing import Tuple

# ==============================================================================
# Audio Processing Constants (also defined in audio.py for backward compat)
# ==============================================================================
SAMPLE_RATE: int = 16000  # Hz - Whisper expects 16kHz audio
N_FFT: int = 400  # FFT window size
HOP_LENGTH: int = 160  # Samples between STFT frames
CHUNK_LENGTH: int = 30  # Seconds per audio chunk
N_SAMPLES: int = CHUNK_LENGTH * SAMPLE_RATE  # 480000 samples in a 30-second chunk
N_FRAMES: int = N_SAMPLES // HOP_LENGTH  # 3000 frames in a mel spectrogram

# Derived audio constants
N_SAMPLES_PER_TOKEN: int = HOP_LENGTH * 2  # Initial convolutions have stride 2
FRAMES_PER_SECOND: int = SAMPLE_RATE // HOP_LENGTH  # 100 fps (10ms per frame)
TOKENS_PER_SECOND: int = SAMPLE_RATE // N_SAMPLES_PER_TOKEN  # 50 tps (20ms per token)

# ==============================================================================
# Decoding Quality Thresholds
# These values were empirically determined by OpenAI for Whisper
# ==============================================================================

# Compression ratio threshold for detecting hallucination
# A ratio > 2.4 indicates repetitive text (hallucination)
# Calculated as: len(text) / len(zlib.compress(text))
COMPRESSION_RATIO_THRESHOLD: float = 2.4

# Log probability threshold for detecting low confidence output
# Values below -1.0 suggest the model is uncertain about its output
LOGPROB_THRESHOLD: float = -1.0

# No-speech probability threshold
# If no_speech_prob > 0.6 AND avg_logprob < LOGPROB_THRESHOLD, treat as silence
NO_SPEECH_THRESHOLD: float = 0.6

# ==============================================================================
# Temperature Schedule for Fallback Decoding
# ==============================================================================

# When decoding fails quality thresholds, retry with higher temperatures
# Starting at 0.0 (greedy) and increasing to 1.0 (maximum randomness)
TEMPERATURE_SCHEDULE: Tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)

# ==============================================================================
# Token Limits
# ==============================================================================

# Maximum tokens per segment (Whisper's text context window)
# This is the max number of text tokens the decoder can produce per chunk
MAX_TOKENS_PER_SEGMENT: int = 224

# Default number of draft tokens for speculative decoding
# Balance between draft overhead and verification cost
DEFAULT_DRAFT_TOKENS: int = 5

# ==============================================================================
# Length Penalty Constants (from Google NMT paper)
# Used in beam search: penalty = ((5 + length) / 6) ** alpha
# ==============================================================================
LENGTH_PENALTY_BASE: int = 5
LENGTH_PENALTY_DIVISOR: int = 6
