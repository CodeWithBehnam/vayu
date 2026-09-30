# Copyright © 2024 Whisper MLX Contributors
# Speculative decoding for Whisper

"""
Speculative Decoding for Whisper MLX

A small "draft" model proposes a few tokens at a time, and the main "target"
model checks all of them in a single forward pass:

1. The draft model decodes `num_draft_tokens` tokens greedily.
2. The target model scores those tokens in one pass over its key/value cache.
3. Drafted tokens are kept up to the first one the target disagrees with; the
   target's own choice is used there (or appended, if every draft matched).

The output is exactly the target model's greedy (temperature 0, no timestamps)
transcript. Whether it is faster depends on how often the draft agrees with the
target and how cheap the draft is, so measure it on your own audio. The two
models must share a vocabulary, e.g. distil-large-v3 drafting for large-v3.

Usage:
    from whisper_mlx.speculative import speculative_transcribe

    result = speculative_transcribe(
        "audio.mp3",
        draft_model="distil-large-v3",
        target_model="large-v3",
    )
"""

import time
import warnings
from typing import Dict, List, Optional, Sequence, Tuple, Union

import mlx.core as mx
import numpy as np

from .audio import (
    HOP_LENGTH,
    N_FRAMES,
    N_SAMPLES,
    SAMPLE_RATE,
    load_audio,
    log_mel_spectrogram,
    pad_or_trim,
)
from .constants import DEFAULT_DRAFT_TOKENS
from .decoding import DecodingOptions, DecodingTask, LogitFilter
from .load_models import load_model
from .tokenizer import get_tokenizer
from .utils import MODEL_REPOS
from .whisper import Whisper

KVCache = List[Tuple[Tuple[mx.array, mx.array], Tuple[mx.array, mx.array]]]


def _cache_length(cache: Optional[KVCache]) -> int:
    return 0 if cache is None else cache[0][0][0].shape[1]


def _trim_cache(cache: KVCache, length: int) -> KVCache:
    """Drop self-attention entries past `length`; cross-attention is unchanged."""
    return [((k[:, :length], v[:, :length]), cross_kv) for (k, v), cross_kv in cache]


def _forward(
    model: Whisper, tokens: Sequence[int], features: mx.array, cache: Optional[KVCache]
) -> Tuple[mx.array, KVCache]:
    """Feed `tokens` after the cached ones; returns one row of logits per token."""
    logits, cache, _ = model.decoder(mx.array([tokens]), features, kv_cache=cache)
    return logits[0].astype(mx.float32), cache


def _greedy(logits: mx.array, context: List[int], filters: List[LogitFilter]) -> int:
    """Pick the next token after `context` the way DecodingTask does at T=0."""
    logits = logits[None]
    tokens = mx.array([context])
    for logit_filter in filters:
        logits = logit_filter.apply(logits, tokens)
    return mx.argmax(logits, axis=-1).item()


class SpeculativeDecoder:
    """
    Speculative decoding: draft with a fast model, verify with an accurate one.

    Both models keep a key/value cache, so each draft step costs one draft
    decoder pass for one token, and each verification costs one target decoder
    pass over all drafted tokens. Audio is encoded once per model per window.
    """

    def __init__(
        self,
        draft_model_path: str = "mlx-community/distil-whisper-large-v3",
        target_model_path: str = "mlx-community/whisper-large-v3-mlx",
        num_draft_tokens: int = DEFAULT_DRAFT_TOKENS,
        dtype: mx.Dtype = mx.float16,
        *,
        draft_model: Optional[Whisper] = None,
        target_model: Optional[Whisper] = None,
        max_tokens: Optional[int] = None,
    ):
        """
        Initialize speculative decoder.

        Args:
            draft_model_path: Fast model for drafting
            target_model_path: Accurate model whose output is reproduced
            num_draft_tokens: Number of tokens to draft before each verification
            dtype: Data type for computation
            draft_model, target_model: Already-loaded models to use instead of
                loading from the paths
            max_tokens: Maximum tokens per 30-second window (default: the model's
                limit, 224)
        """
        if num_draft_tokens < 1:
            raise ValueError(f"num_draft_tokens must be >= 1, got {num_draft_tokens}")

        if draft_model is None:
            draft_model = load_model(draft_model_path, dtype=dtype)
        if target_model is None:
            target_model = load_model(target_model_path, dtype=dtype)
        self.draft_model = draft_model
        self.target_model = target_model

        draft_vocab = self.draft_model.dims.n_vocab
        target_vocab = self.target_model.dims.n_vocab
        if draft_vocab != target_vocab:
            raise ValueError(
                f"Draft and target models must share a vocabulary, but have "
                f"{draft_vocab} and {target_vocab} tokens. Pair models from the same "
                "family, e.g. distil-large-v3 with large-v3, or tiny with large-v2."
            )

        self.num_draft_tokens = num_draft_tokens
        self.dtype = dtype
        self.max_tokens = max_tokens

        self.tokenizer = get_tokenizer(
            self.target_model.is_multilingual,
            num_languages=self.target_model.num_languages,
        )
        self._tasks: Dict[Tuple[int, str], DecodingTask] = {}

        # Stats tracking
        self.stats = {
            "draft_tokens": 0,
            "accepted_tokens": 0,
            "generated_tokens": 0,
            "draft_forward_passes": 0,
            "total_forward_passes": 0,  # target decoder passes
        }

    def _task(self, model: Whisper, language: str) -> DecodingTask:
        """Greedy, timestamp-free decoding setup: prompt tokens and logit filters."""
        key = (id(model), language)
        if key not in self._tasks:
            options = DecodingOptions(
                language=language,
                without_timestamps=True,
                sample_len=self.max_tokens,
                fp16=self.dtype == mx.float16,
            )
            self._tasks[key] = DecodingTask(model, options)
        return self._tasks[key]

    def _encode(self, model: Whisper, mel: mx.array) -> mx.array:
        if mel.ndim == 2:
            mel = mel[None]
        return model.encoder(mel.astype(self.dtype))

    def decode_segment(
        self,
        mel: mx.array,
        language: str = "en",
        *,
        target_mel: Optional[mx.array] = None,
    ) -> dict:
        """
        Decode one 30-second window with speculative decoding.

        Args:
            mel: Log-mel spectrogram of the window, shape (3000, n_mels), for the
                draft model (and the target model, unless target_mel is given)
            language: Language code; ignored by English-only models
            target_mel: Spectrogram for the target model when it expects a
                different number of mel bands than the draft (e.g. 128 vs 80)

        Returns:
            dict with the sampled "tokens" (without end-of-text) and "text"
        """
        if target_mel is None:
            target_mel = mel

        draft_task = self._task(self.draft_model, language)
        target_task = self._task(self.target_model, language)
        tokenizer = target_task.tokenizer
        eot = tokenizer.eot
        sample_begin = target_task.sample_begin
        max_length = sample_begin + target_task.sample_len

        draft_features = self._encode(self.draft_model, mel)
        target_features = self._encode(self.target_model, target_mel)
        draft_cache: Optional[KVCache] = None
        target_cache: Optional[KVCache] = None

        tokens = list(target_task.initial_tokens)
        while len(tokens) < max_length and tokens[-1] != eot:
            # 1. Draft up to num_draft_tokens tokens, one cached pass each
            context = list(tokens)
            drafts: List[int] = []
            n_draft = min(self.num_draft_tokens, max_length - len(tokens))
            while len(drafts) < n_draft:
                cached = _cache_length(draft_cache)
                logits, draft_cache = _forward(
                    self.draft_model, context[cached:], draft_features, draft_cache
                )
                self.stats["draft_forward_passes"] += 1
                token = _greedy(logits[-1], context, draft_task.logit_filters)
                drafts.append(token)
                context.append(token)
                if token == eot:
                    break

            # 2. Score every drafted position with one target pass;
            #    row j predicts the token after context[cached + j]
            cached = _cache_length(target_cache)
            logits, target_cache = _forward(
                self.target_model, context[cached:], target_features, target_cache
            )
            self.stats["total_forward_passes"] += 1

            # 3. Keep drafts while the target agrees, then take the target's token
            accepted: List[int] = []
            for i, draft in enumerate(drafts + [None]):
                position = len(tokens) + i
                token = _greedy(
                    logits[position - 1 - cached],
                    context[:position],
                    target_task.logit_filters,
                )
                accepted.append(token)
                if token != draft or token == eot:
                    break

            matched = 0
            while matched < len(drafts) and accepted[matched] == drafts[matched]:
                matched += 1
            self.stats["draft_tokens"] += len(drafts)
            self.stats["accepted_tokens"] += matched

            # 4. Roll both caches back to the tokens that were kept
            valid = len(tokens) + matched
            tokens = (tokens + accepted)[:max_length]
            target_cache = _trim_cache(target_cache, min(valid, len(tokens) - 1))
            draft_cache = _trim_cache(draft_cache, min(valid, len(tokens) - 1))

        sampled = tokens[sample_begin:]
        if eot in sampled:
            sampled = sampled[: sampled.index(eot)]
        self.stats["generated_tokens"] += len(sampled)

        return {
            "tokens": sampled,
            "text": tokenizer.decode(sampled).strip(),
        }

    def transcribe(
        self,
        audio: Union[str, np.ndarray, mx.array],
        language: str = "en",
        verbose: bool = False,
    ) -> dict:
        """Transcribe audio window by window (30 s each) with speculative decoding."""
        if isinstance(audio, str):
            audio = load_audio(audio)
        elif not isinstance(audio, mx.array):
            audio = mx.array(audio)

        draft_mels = self.draft_model.dims.n_mels
        target_mels = self.target_model.dims.n_mels
        # Pad 30 seconds of silence, as transcribe() does, for slicing
        mels = {
            n_mels: log_mel_spectrogram(audio, n_mels=n_mels, padding=N_SAMPLES)
            for n_mels in {draft_mels, target_mels}
        }
        content_frames = mels[draft_mels].shape[0] - N_FRAMES

        segments = []
        start_time = time.time()
        for seek in range(0, content_frames, N_FRAMES):
            segment_size = min(N_FRAMES, content_frames - seek)
            window = {
                n_mels: pad_or_trim(mel[seek : seek + segment_size], N_FRAMES, axis=-2)
                for n_mels, mel in mels.items()
            }
            result = self.decode_segment(
                window[draft_mels], language, target_mel=window[target_mels]
            )
            start = seek * HOP_LENGTH / SAMPLE_RATE
            segments.append(
                {
                    "start": start,
                    "end": (seek + segment_size) * HOP_LENGTH / SAMPLE_RATE,
                    "text": result["text"],
                    "tokens": result["tokens"],
                }
            )
            if verbose:
                print(f"[{start:.1f}s] {result['text']}")

        return {
            "text": " ".join(s["text"] for s in segments if s["text"]),
            "segments": segments,
            "stats": self.get_stats(),
            "elapsed": time.time() - start_time,
        }

    def get_stats(self) -> dict:
        """
        Get decoding statistics.

        `tokens_per_target_pass` is the number of tokens produced per target
        decoder pass (1.0 for ordinary greedy decoding); it bounds, but does not
        measure, the wall-clock speed-up.
        """
        draft_tokens = self.stats["draft_tokens"]
        passes = max(1, self.stats["total_forward_passes"])
        tokens_per_pass = self.stats["generated_tokens"] / passes
        return {
            **self.stats,
            "acceptance_rate": (
                self.stats["accepted_tokens"] / draft_tokens if draft_tokens else 0
            ),
            "tokens_per_target_pass": tokens_per_pass,
            "speedup_factor": tokens_per_pass,  # kept for backward compatibility
        }


def speculative_transcribe(
    audio: Union[str, np.ndarray],
    draft_model: str = "distil-large-v3",
    target_model: str = "large-v3",
    language: str = "en",
    verbose: bool = True,
    num_draft_tokens: int = DEFAULT_DRAFT_TOKENS,
) -> dict:
    """
    Transcribe audio using speculative decoding.

    Args:
        audio: Path to audio file or audio array
        draft_model: Small model for drafting; must share the target's vocabulary
        target_model: Model whose greedy output is reproduced
        language: Language code
        verbose: Print progress
        num_draft_tokens: Tokens drafted per verification pass

    Returns:
        dict with "text", "segments", "stats" and "elapsed"
    """
    # Resolve model paths using centralized mapping
    draft_path = MODEL_REPOS.get(draft_model, draft_model)
    target_path = MODEL_REPOS.get(target_model, target_model)

    if verbose:
        print(f"Speculative Decoding: {draft_model} → {target_model}")
        print("=" * 50)

    decoder = SpeculativeDecoder(
        draft_model_path=draft_path,
        target_model_path=target_path,
        num_draft_tokens=num_draft_tokens,
    )
    result = decoder.transcribe(audio, language=language, verbose=verbose)

    if verbose:
        stats = result["stats"]
        print("=" * 50)
        print(f"Time: {result['elapsed']:.2f}s")
        print(f"Acceptance rate: {stats['acceptance_rate']:.1%}")
        print(f"Tokens per target pass: {stats['tokens_per_target_pass']:.2f}")

    return result


# ============================================================
# IDEA 2: VAD-Guided Processing (Skip Silence)
# ============================================================


class VADProcessor:
    """
    Voice Activity Detection guided processing.

    Instead of processing fixed 30-second chunks, detect speech regions
    and only process those, skipping silence entirely.
    """

    def __init__(
        self, energy_threshold: float = 0.01, min_speech_duration: float = 0.5
    ):
        self.energy_threshold = energy_threshold
        self.min_speech_duration = min_speech_duration

    def detect_speech_regions(
        self, audio: np.ndarray, sample_rate: int = 16000
    ) -> list:
        """
        Simple energy-based VAD to detect speech regions.

        Returns list of (start_sample, end_sample) tuples.
        """
        audio = np.asarray(audio, dtype=np.float64).ravel()

        # Frame-based energy calculation
        frame_size = int(0.025 * sample_rate)  # 25ms frames
        hop_size = int(0.010 * sample_rate)  # 10ms hop
        if len(audio) < frame_size:
            return []

        # RMS energy per frame from a running sum of squares
        num_frames = (len(audio) - frame_size) // hop_size + 1
        cumulative = np.concatenate([[0.0], np.cumsum(audio**2)])
        starts = np.arange(num_frames) * hop_size
        energy = np.sqrt(
            np.maximum(cumulative[starts + frame_size] - cumulative[starts], 0.0)
            / frame_size
        )

        # Normalize energy
        energy = energy / (np.max(energy) + 1e-8)

        # Find runs of speech frames long enough to keep
        is_speech = np.concatenate([[False], energy > self.energy_threshold, [False]])
        edges = np.flatnonzero(is_speech[1:] != is_speech[:-1])
        run_starts, run_ends = edges[::2], edges[1::2]  # run_ends is exclusive

        min_frames = int(self.min_speech_duration / 0.010)
        regions = []
        for start, end in zip(run_starts, run_ends):
            if end - start < min_frames:
                continue
            if end == num_frames:
                # audio ends during speech
                end_sample = len(audio)
            else:
                end_sample = min(end * hop_size + frame_size, len(audio))
            regions.append((int(start * hop_size), int(end_sample)))

        return regions

    def get_skip_ratio(self, audio: np.ndarray, sample_rate: int = 16000) -> float:
        """Calculate what percentage of audio can be skipped."""
        audio = np.asarray(audio)
        regions = self.detect_speech_regions(audio, sample_rate)
        speech_samples = sum(end - start for start, end in regions)
        return 1.0 - (speech_samples / len(audio))


# ============================================================
# IDEA 3: Chunk Processing with Overlap Merging
# ============================================================


def parallel_chunk_transcribe(
    audio: Union[str, np.ndarray],
    model_path: str = "mlx-community/whisper-turbo",
    chunk_duration: float = 30.0,
    overlap_duration: float = 2.0,
    language: str = "en",
) -> dict:
    """
    Deprecated: transcribes overlapping chunks one after another, then merges them.

    Despite the name, chunks are not processed in parallel. Use
    `transcribe(audio, batch_size=N)`, which decodes N windows at once.
    """
    warnings.warn(
        "parallel_chunk_transcribe processes chunks sequentially and is deprecated; "
        "use transcribe(audio, batch_size=N) for batched decoding.",
        DeprecationWarning,
        stacklevel=2,
    )
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

    results = []
    for i, chunk in enumerate(chunks):
        result = transcribe(
            chunk,
            path_or_hf_repo=model_path,
            batch_size=1,
            language=language,
            verbose=False,
        )
        results.append(
            {
                "start": starts[i],
                "result": result,
            }
        )

    # Merge overlapping segments
    merged_segments = merge_overlapping_segments(results, overlap_duration)

    return {
        "text": " ".join(s["text"] for s in merged_segments),
        "segments": merged_segments,
    }


def merge_overlapping_segments(results: list, overlap_duration: float) -> list:
    """
    Merge transcription results from overlapping chunks.

    Where segments overlap, keeps the one with the longer text.
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
