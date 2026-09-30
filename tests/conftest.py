"""Shared fixtures: stub and random-weight models for tests without downloads."""

import importlib
from types import SimpleNamespace
from typing import Callable, List

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_map

from whisper_mlx.decoding import DecodingOptions, DecodingResult
from whisper_mlx.tokenizer import get_tokenizer
from whisper_mlx.whisper import ModelDimensions, Whisper

# transcribe() is re-exported by the package, so fetch the module itself
transcribe_module = importlib.import_module("whisper_mlx.transcribe")

SAMPLE_RATE = 16000

# English-only vocabulary with a deliberately small network: fast on CPU
TINY_EN_DIMS = ModelDimensions(
    n_mels=80,
    n_audio_ctx=1500,
    n_audio_state=64,
    n_audio_head=2,
    n_audio_layer=2,
    n_vocab=51864,
    n_text_ctx=448,
    n_text_state=64,
    n_text_head=2,
    n_text_layer=2,
)


def to_float16(model: Whisper) -> Whisper:
    """Cast weights to float16, as the published MLX checkpoints are."""
    model.update(
        tree_map(
            lambda p: (
                p.astype(mx.float16) if mx.issubdtype(p.dtype, mx.floating) else p
            ),
            model.parameters(),
        )
    )
    mx.eval(model.parameters())
    return model


def random_whisper(
    seed: int = 0, dims: ModelDimensions = TINY_EN_DIMS, dtype: mx.Dtype = mx.float16
) -> Whisper:
    """A Whisper model with random weights, deterministic for a given seed."""
    mx.random.seed(seed)
    model = Whisper(dims, dtype)
    if dtype == mx.float16:
        return to_float16(model)
    mx.eval(model.parameters())
    return model


@pytest.fixture
def tokenizer():
    return get_tokenizer(False)  # English-only (gpt2) tokenizer


class StubModel:
    """Stands in for Whisper in transcribe(): decode() returns canned tokens.

    `respond(options, window)` returns the token list for one window, where
    `window` numbers the distinct mel segments in the order they are first seen,
    so a re-decode of the same segment gets the same number. Use `noise=True`
    in run_transcribe to make every window distinct.
    """

    dims = SimpleNamespace(n_mels=80, n_audio_ctx=1500)
    is_multilingual = False
    num_languages = 99  # what an English-only checkpoint reports (n_vocab 51864)
    no_speech_prob = 0.0

    def __init__(
        self,
        respond: Callable[[DecodingOptions, int], List[int]],
        compression_ratio: Callable[[DecodingOptions, int], float] = lambda o, w: 1.0,
    ):
        self.respond = respond
        self.compression_ratio = compression_ratio
        self.calls: List[DecodingOptions] = []
        self.batch_sizes: List[int] = []
        self._windows = {}

    def decode(self, mel, options: DecodingOptions):
        single = mel.ndim == 2
        items = [mel] if single else [mel[i] for i in range(mel.shape[0])]
        self.calls.append(options)
        self.batch_sizes.append(len(items))
        results = []
        for item in items:
            key = np.array(item).tobytes()
            window = self._windows.setdefault(key, len(self._windows))
            results.append(
                DecodingResult(
                    audio_features=None,
                    language="en",
                    tokens=self.respond(options, window),
                    avg_logprob=-0.1,
                    no_speech_prob=self.no_speech_prob,
                    temperature=options.temperature,
                    compression_ratio=self.compression_ratio(options, window),
                )
            )
        return results[0] if single else results


@pytest.fixture
def run_transcribe(monkeypatch):
    """Run transcribe() on `seconds` of silence (or noise) against a stub model."""

    def run(model: StubModel, seconds: float, noise: bool = False, **kwargs):
        monkeypatch.setattr(
            transcribe_module.ModelHolder, "get_model", lambda *a, **k: model
        )
        n_samples = int(SAMPLE_RATE * seconds)
        if noise:
            rng = np.random.default_rng(0)
            audio = (0.1 * rng.standard_normal(n_samples)).astype(np.float32)
        else:
            audio = np.zeros(n_samples, dtype=np.float32)
        kwargs.setdefault("language", "en")
        return transcribe_module.transcribe(audio, **kwargs)

    return run


@pytest.fixture
def ts(tokenizer):
    """Token id of the timestamp `seconds` into a window, e.g. ts(2.0) -> <|2.00|>."""
    return lambda seconds: tokenizer.timestamp_begin + round(seconds / 0.02)
