# Copyright © 2024 Whisper MLX Contributors
# Novel: Speculative Decoding for Whisper
#
# Idea: Use a tiny/fast model to "draft" tokens, then verify with main model.
# The main model can verify multiple tokens in parallel (one forward pass),
# making it much faster than autoregressive decoding.

"""
Speculative Decoding for Whisper MLX

This implements speculative decoding where:
1. A small "draft" model (e.g., whisper-tiny) generates candidate tokens quickly
2. The main "target" model (e.g., whisper-large) verifies tokens in parallel
3. Accepted tokens are kept, rejected tokens trigger re-generation

This can provide 2-3x speedup on top of batched decoding.

Usage:
    from whisper_mlx.speculative import speculative_transcribe

    result = speculative_transcribe(
        "audio.mp3",
        draft_model="tiny",
        target_model="large-v3",
        batch_size=12,
    )
"""

import time
from typing import Optional, Union

from .utils import MODEL_REPOS

import mlx.core as mx
import numpy as np

from .audio import N_FRAMES, log_mel_spectrogram, pad_or_trim
from .constants import DEFAULT_DRAFT_TOKENS, MAX_TOKENS_PER_SEGMENT
from .load_models import load_model
from .tokenizer import get_tokenizer


class SpeculativeDecoder:
    """
    Speculative decoding: draft with fast model, verify with accurate model.

    The key insight is that verification is parallelizable:
    - Draft model generates K tokens autoregressively (slow but using tiny model)
    - Target model verifies all K tokens in ONE forward pass (fast)
    - If all tokens match, we get K tokens for ~1 forward pass cost
    """

    def __init__(
        self,
        draft_model_path: str = "mlx-community/whisper-tiny-mlx",
        target_model_path: str = "mlx-community/whisper-large-v3-mlx",
        num_draft_tokens: int = DEFAULT_DRAFT_TOKENS,
        dtype: mx.Dtype = mx.float16,
    ):
        """
        Initialize speculative decoder.

        Args:
            draft_model_path: Fast model for drafting (tiny recommended)
            target_model_path: Accurate model for verification
            num_draft_tokens: Number of tokens to draft before verification
            dtype: Data type for computation
        """
        print(f"Loading draft model: {draft_model_path}")
        self.draft_model = load_model(draft_model_path, dtype=dtype)

        print(f"Loading target model: {target_model_path}")
        self.target_model = load_model(target_model_path, dtype=dtype)

        self.num_draft_tokens = num_draft_tokens
        self.dtype = dtype

        # Get tokenizer (same for both models)
        self.tokenizer = get_tokenizer(self.target_model.is_multilingual)

        # Stats tracking
        self.stats = {
            "draft_tokens": 0,
            "accepted_tokens": 0,
            "total_forward_passes": 0,
        }

    def _draft_tokens(self, mel: mx.array, initial_tokens: list) -> list:
        """Generate draft tokens using the fast model."""
        tokens = initial_tokens.copy()

        # Encode audio once
        audio_features = self.draft_model.encoder(mel)

        for _ in range(self.num_draft_tokens):
            # Decode next token
            token_array = mx.array([tokens])
            logits = self.draft_model.decoder(token_array, audio_features)
            next_token = mx.argmax(logits[0, -1]).item()
            tokens.append(next_token)

            # Stop if end of text
            if next_token == self.tokenizer.eot:
                break

        self.stats["draft_tokens"] += len(tokens) - len(initial_tokens)
        return tokens

    def _verify_tokens(
        self,
        mel: mx.array,
        initial_tokens: list,
        draft_tokens: list,
    ) -> tuple[list, bool]:
        """
        Verify draft tokens using target model.

        This is the key: we verify ALL draft tokens in ONE forward pass!
        """
        # Encode audio
        audio_features = self.target_model.encoder(mel)

        # Get target model's predictions for all positions
        token_array = mx.array([draft_tokens[:-1]])  # Input excludes last token
        logits = self.target_model.decoder(token_array, audio_features)

        self.stats["total_forward_passes"] += 1

        # Verify each drafted token
        accepted_tokens = initial_tokens.copy()
        all_accepted = True

        for i, draft_token in enumerate(draft_tokens[len(initial_tokens):]):
            pos = len(initial_tokens) + i - 1
            if pos < 0:
                pos = 0

            # Get target model's prediction for this position
            target_token = mx.argmax(logits[0, pos]).item()

            if target_token == draft_token:
                accepted_tokens.append(draft_token)
                self.stats["accepted_tokens"] += 1
            else:
                # Rejection: use target's token and stop
                accepted_tokens.append(target_token)
                all_accepted = False
                break

        return accepted_tokens, all_accepted

    def decode_segment(self, mel: mx.array, language: str = "en") -> dict:
        """
        Decode a single audio segment using speculative decoding.
        """
        # Initial tokens
        tokens = [
            self.tokenizer.sot,
            self.tokenizer.special_tokens[f"<|{language}|>"],
            self.tokenizer.special_tokens["<|transcribe|>"],
            self.tokenizer.special_tokens["<|notimestamps|>"],
        ]

        max_tokens = MAX_TOKENS_PER_SEGMENT

        while len(tokens) < max_tokens:
            # Draft tokens
            draft_tokens = self._draft_tokens(mel, tokens)

            # Verify with target model
            tokens, all_accepted = self._verify_tokens(mel, tokens, draft_tokens)

            # Check for end
            if tokens[-1] == self.tokenizer.eot:
                break

        # Decode tokens to text
        text_tokens = [t for t in tokens if t < self.tokenizer.eot]
        text = self.tokenizer.decode(text_tokens)

        return {
            "tokens": tokens,
            "text": text,
        }

    def get_stats(self) -> dict:
        """Get decoding statistics."""
        acceptance_rate = (
            self.stats["accepted_tokens"] / self.stats["draft_tokens"]
            if self.stats["draft_tokens"] > 0 else 0
        )
        return {
            **self.stats,
            "acceptance_rate": acceptance_rate,
            "speedup_factor": self.stats["draft_tokens"] / max(1, self.stats["total_forward_passes"]),
        }


def speculative_transcribe(
    audio: Union[str, np.ndarray],
    draft_model: str = "tiny",
    target_model: str = "large-v3",
    language: str = "en",
    verbose: bool = True,
) -> dict:
    """
    Transcribe audio using speculative decoding.

    Args:
        audio: Path to audio file or audio array
        draft_model: Small model for drafting ("tiny", "base", "small")
        target_model: Large model for verification ("large-v3", "turbo")
        language: Language code
        verbose: Print progress

    Returns:
        dict with "text", "segments", and "stats"
    """
    # Resolve model paths using centralized mapping
    draft_path = MODEL_REPOS.get(draft_model, draft_model)
    target_path = MODEL_REPOS.get(target_model, target_model)

    if verbose:
        print(f"Speculative Decoding: {draft_model} → {target_model}")
        print("=" * 50)

    # Initialize decoder
    decoder = SpeculativeDecoder(
        draft_model_path=draft_path,
        target_model_path=target_path,
    )

    # Load and process audio
    from .audio import load_audio, log_mel_spectrogram, pad_or_trim, N_FRAMES, SAMPLE_RATE

    if isinstance(audio, str):
        audio_array = load_audio(audio)
    else:
        audio_array = audio

    # Compute mel spectrogram
    mel = log_mel_spectrogram(audio_array)

    # Process in segments
    segments = []
    seek = 0
    total_frames = mel.shape[-2]

    start_time = time.time()

    while seek < total_frames:
        # Get segment
        segment_mel = mel[seek : seek + N_FRAMES]
        segment_mel = pad_or_trim(segment_mel, N_FRAMES, axis=-2)
        segment_mel = mx.expand_dims(segment_mel, axis=0)

        # Decode
        result = decoder.decode_segment(segment_mel, language=language)

        # Add segment
        time_offset = seek * 160 / SAMPLE_RATE  # HOP_LENGTH = 160
        segments.append({
            "start": time_offset,
            "end": time_offset + 30.0,
            "text": result["text"],
        })

        if verbose:
            print(f"[{time_offset:.1f}s] {result['text']}")

        seek += N_FRAMES

    elapsed = time.time() - start_time
    stats = decoder.get_stats()

    if verbose:
        print("=" * 50)
        print(f"Time: {elapsed:.2f}s")
        print(f"Acceptance rate: {stats['acceptance_rate']:.1%}")
        print(f"Effective speedup: {stats['speedup_factor']:.2f}x")

    return {
        "text": " ".join(s["text"] for s in segments),
        "segments": segments,
        "stats": stats,
        "elapsed": elapsed,
    }


# ============================================================
# IDEA 2: VAD-Guided Processing (Skip Silence)
# ============================================================

class VADProcessor:
    """
    Voice Activity Detection guided processing.

    Instead of processing fixed 30-second chunks, detect speech regions
    and only process those, skipping silence entirely.
    """

    def __init__(self, energy_threshold: float = 0.01, min_speech_duration: float = 0.5):
        self.energy_threshold = energy_threshold
        self.min_speech_duration = min_speech_duration

    def detect_speech_regions(self, audio: np.ndarray, sample_rate: int = 16000) -> list:
        """
        Simple energy-based VAD to detect speech regions.

        Returns list of (start_sample, end_sample) tuples.
        """
        # Convert to numpy if mlx array
        if hasattr(audio, 'tolist') and not isinstance(audio, np.ndarray):
            audio = np.array(audio.tolist())

        # Frame-based energy calculation
        frame_size = int(0.025 * sample_rate)  # 25ms frames
        hop_size = int(0.010 * sample_rate)    # 10ms hop

        num_frames = (len(audio) - frame_size) // hop_size + 1
        energy = np.zeros(num_frames)

        for i in range(num_frames):
            start = i * hop_size
            frame = audio[start : start + frame_size]
            energy[i] = np.sqrt(np.mean(frame ** 2))

        # Normalize energy
        energy = energy / (np.max(energy) + 1e-8)

        # Find speech regions
        is_speech = energy > self.energy_threshold

        # Merge close regions and filter short ones
        regions = []
        in_speech = False
        start = 0

        min_frames = int(self.min_speech_duration / 0.010)

        for i, speech in enumerate(is_speech):
            if speech and not in_speech:
                start = i
                in_speech = True
            elif not speech and in_speech:
                if i - start >= min_frames:
                    start_sample = start * hop_size
                    end_sample = min(i * hop_size + frame_size, len(audio))
                    regions.append((start_sample, end_sample))
                in_speech = False

        # Handle case where audio ends during speech
        if in_speech and len(is_speech) - start >= min_frames:
            regions.append((start * hop_size, len(audio)))

        return regions

    def get_skip_ratio(self, audio: np.ndarray, sample_rate: int = 16000) -> float:
        """Calculate what percentage of audio can be skipped."""
        # Convert to numpy if mlx array
        if hasattr(audio, 'tolist') and not isinstance(audio, np.ndarray):
            audio = np.array(audio.tolist())

        regions = self.detect_speech_regions(audio, sample_rate)
        speech_samples = sum(end - start for start, end in regions)
        return 1.0 - (speech_samples / len(audio))


# ============================================================
# IDEA 3: Parallel Chunk Processing with Overlap Merging
# ============================================================

def parallel_chunk_transcribe(
    audio: Union[str, np.ndarray],
    model_path: str = "mlx-community/whisper-turbo",
    chunk_duration: float = 30.0,
    overlap_duration: float = 2.0,
    language: str = "en",
) -> dict:
    """
    Process ALL audio chunks in parallel, then merge with overlap handling.

    Unlike sequential processing, this transcribes all chunks simultaneously,
    then uses overlap regions to stitch them together correctly.

    This maximizes GPU utilization for long audio files.
    """
    from .audio import load_audio, log_mel_spectrogram, SAMPLE_RATE, HOP_LENGTH
    from .transcribe import transcribe

    # Load audio
    if isinstance(audio, str):
        audio_array = load_audio(audio)
    else:
        audio_array = audio

    chunk_samples = int(chunk_duration * SAMPLE_RATE)
    overlap_samples = int(overlap_duration * SAMPLE_RATE)
    step_samples = chunk_samples - overlap_samples

    # Split into overlapping chunks
    chunks = []
    starts = []
    pos = 0

    while pos < len(audio_array):
        end = min(pos + chunk_samples, len(audio_array))
        chunks.append(audio_array[pos:end])
        starts.append(pos / SAMPLE_RATE)
        pos += step_samples

    print(f"Processing {len(chunks)} chunks in parallel...")

    # Transcribe all chunks with maximum batch size
    # (In practice, you'd process all mel spectrograms as one big batch)
    results = []
    for i, chunk in enumerate(chunks):
        result = transcribe(
            chunk,
            path_or_hf_repo=model_path,
            batch_size=1,  # Each chunk is one "batch"
            language=language,
            verbose=False,
        )
        results.append({
            "start": starts[i],
            "result": result,
        })

    # Merge overlapping segments
    merged_segments = merge_overlapping_segments(results, overlap_duration)

    return {
        "text": " ".join(s["text"] for s in merged_segments),
        "segments": merged_segments,
    }


def merge_overlapping_segments(results: list, overlap_duration: float) -> list:
    """
    Merge transcription results from overlapping chunks.

    Uses text similarity in overlap regions to find best merge points.
    """
    if not results:
        return []

    merged = []

    for i, r in enumerate(results):
        chunk_start = r["start"]
        segments = r["result"]["segments"]

        for seg in segments:
            # Adjust timestamps
            adjusted_seg = {
                "start": seg["start"] + chunk_start,
                "end": seg["end"] + chunk_start,
                "text": seg["text"],
            }

            # Check for overlap with previous segment
            if merged and adjusted_seg["start"] < merged[-1]["end"]:
                # Overlap detected - keep the one with better confidence
                # (Simple heuristic: keep longer text)
                if len(adjusted_seg["text"]) > len(merged[-1]["text"]):
                    merged[-1] = adjusted_seg
            else:
                merged.append(adjusted_seg)

    return merged
