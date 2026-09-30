"""Speculative decoding, VAD and the deprecated chunk helper."""

import importlib
from dataclasses import replace

import mlx.core as mx
import numpy as np
import pytest

from tests.conftest import TINY_EN_DIMS, random_whisper
from whisper_mlx.decoding import DecodingOptions
from whisper_mlx.speculative import (
    SpeculativeDecoder,
    VADProcessor,
    parallel_chunk_transcribe,
)

speculative = importlib.import_module("whisper_mlx.speculative")

MAX_TOKENS = 40


@pytest.fixture(scope="module")
def target():
    return random_whisper(seed=1, dtype=mx.float32)


@pytest.fixture(scope="module")
def draft():
    return random_whisper(seed=2, dtype=mx.float32)


def mel(seed: int, n_mels: int = 80) -> mx.array:
    mx.random.seed(seed)
    return mx.random.normal((3000, n_mels))


def greedy(model, window, language="en"):
    options = DecodingOptions(
        language=language, without_timestamps=True, sample_len=MAX_TOKENS, fp16=False
    )
    return model.decode(window, options).tokens


def speculative_decoder(draft, target, **kwargs):
    return SpeculativeDecoder(
        draft_model=draft,
        target_model=target,
        dtype=mx.float32,
        max_tokens=MAX_TOKENS,
        **kwargs,
    )


@pytest.mark.parametrize("seed, num_draft_tokens", [(0, 1), (0, 4), (1, 3)])
def test_output_matches_target_greedy_decoding(draft, target, seed, num_draft_tokens):
    decoder = speculative_decoder(draft, target, num_draft_tokens=num_draft_tokens)

    result = decoder.decode_segment(mel(seed))

    assert result["tokens"] == greedy(target, mel(seed))
    stats = decoder.get_stats()
    assert stats["draft_tokens"] > 0
    assert stats["total_forward_passes"] > 0


def test_identical_draft_is_always_accepted(target):
    decoder = speculative_decoder(target, target, num_draft_tokens=4)

    result = decoder.decode_segment(mel(0))

    stats = decoder.get_stats()
    assert result["tokens"] == greedy(target, mel(0))
    assert stats["acceptance_rate"] == 1.0
    # every pass yields the drafts plus one extra token from the target
    assert stats["tokens_per_target_pass"] > 4


def test_models_with_different_vocabularies_are_rejected(target):
    other = random_whisper(seed=3, dims=replace(TINY_EN_DIMS, n_vocab=51865))

    with pytest.raises(ValueError, match="share a vocabulary"):
        SpeculativeDecoder(draft_model=other, target_model=target)


def test_transcribe_gives_each_model_its_own_mel_bands(target):
    wide_target = random_whisper(seed=4, dims=replace(TINY_EN_DIMS, n_mels=128))
    decoder = SpeculativeDecoder(
        draft_model=random_whisper(seed=5), target_model=wide_target, max_tokens=8
    )
    audio = np.zeros(16000 * 35, dtype=np.float32)

    result = decoder.transcribe(audio)

    assert [(s["start"], s["end"]) for s in result["segments"]] == [
        (0.0, 30.0),
        (30.0, 35.0),
    ]
    assert all(len(s["tokens"]) <= 8 for s in result["segments"])


def test_multi_token_step_on_cache_matches_full_pass(target):
    """The causal mask must line up when several tokens follow a kv cache."""
    features = target.encoder(mel(0)[None])
    tokens = mx.array([[50257, 50362, 1000, 2000, 3000, 4000, 5000, 6000]])

    full, _, _ = target.decoder(tokens, features)
    _, cache, _ = target.decoder(tokens[:, :3], features)
    rest, _, _ = target.decoder(tokens[:, 3:], features, kv_cache=cache)

    assert np.allclose(np.array(full[:, 3:]), np.array(rest), atol=1e-4)


def reference_vad(audio, energy_threshold=0.01, min_speech_duration=0.5, sr=16000):
    """The original frame-by-frame implementation, for comparison."""
    frame_size, hop_size = int(0.025 * sr), int(0.010 * sr)
    num_frames = (len(audio) - frame_size) // hop_size + 1
    energy = np.zeros(num_frames)
    for i in range(num_frames):
        frame = audio[i * hop_size : i * hop_size + frame_size]
        energy[i] = np.sqrt(np.mean(frame**2))
    energy = energy / (np.max(energy) + 1e-8)
    is_speech = energy > energy_threshold
    regions, in_speech, start = [], False, 0
    min_frames = int(min_speech_duration / 0.010)
    for i, speech in enumerate(is_speech):
        if speech and not in_speech:
            start, in_speech = i, True
        elif not speech and in_speech:
            if i - start >= min_frames:
                regions.append(
                    (start * hop_size, min(i * hop_size + frame_size, len(audio)))
                )
            in_speech = False
    if in_speech and len(is_speech) - start >= min_frames:
        regions.append((start * hop_size, len(audio)))
    return regions


@pytest.mark.parametrize("ends_in_speech", [False, True])
def test_vad_matches_reference_implementation(ends_in_speech):
    rng = np.random.default_rng(0)
    audio = np.zeros(16000 * 6, dtype=np.float32)
    for start, length in [(0.5, 1.0), (2.0, 0.3), (3.0, 1.2)]:
        a, b = int(start * 16000), int((start + length) * 16000)
        audio[a:b] = rng.standard_normal(b - a) * 0.3
    if ends_in_speech:
        audio[-16000:] = rng.standard_normal(16000) * 0.3

    vad = VADProcessor()

    assert vad.detect_speech_regions(audio) == reference_vad(audio)
    assert len(vad.detect_speech_regions(audio)) == (3 if ends_in_speech else 2)


def test_vad_handles_audio_shorter_than_a_frame():
    assert VADProcessor().detect_speech_regions(np.zeros(100)) == []


def test_parallel_chunk_transcribe_is_deprecated(monkeypatch):
    transcribe_module = importlib.import_module("whisper_mlx.transcribe")
    monkeypatch.setattr(
        transcribe_module,
        "transcribe",
        lambda *a, **k: {"text": "", "segments": [], "language": "en"},
    )

    with pytest.deprecated_call():
        parallel_chunk_transcribe(np.zeros(16000 * 5, dtype=np.float32))
