"""Shared fixtures: a stub model for driving transcribe() without weights."""

import importlib
from types import SimpleNamespace
from typing import Callable, List

import numpy as np
import pytest

from whisper_mlx.decoding import DecodingOptions, DecodingResult
from whisper_mlx.tokenizer import get_tokenizer

# transcribe() is re-exported by the package, so fetch the module itself
transcribe_module = importlib.import_module("whisper_mlx.transcribe")

SAMPLE_RATE = 16000


@pytest.fixture
def tokenizer():
    return get_tokenizer(False)  # English-only (gpt2) tokenizer


class StubModel:
    """Stands in for Whisper in transcribe(): decode() returns canned tokens.

    `respond(options, window)` returns the token list for one window, where
    `window` is the index of the call's segment within the whole run.
    """

    dims = SimpleNamespace(n_mels=80, n_audio_ctx=1500)
    is_multilingual = False
    num_languages = 99  # what an English-only checkpoint reports (n_vocab 51864)

    def __init__(
        self,
        respond: Callable[[DecodingOptions, int], List[int]],
        compression_ratio: Callable[[DecodingOptions, int], float] = lambda o, w: 1.0,
    ):
        self.respond = respond
        self.compression_ratio = compression_ratio
        self.calls: List[DecodingOptions] = []
        self.batch_sizes: List[int] = []
        self._window = 0

    def decode(self, mel, options: DecodingOptions):
        single = mel.ndim == 2
        n = 1 if single else mel.shape[0]
        self.calls.append(options)
        self.batch_sizes.append(n)
        results = []
        for _ in range(n):
            results.append(
                DecodingResult(
                    audio_features=None,
                    language="en",
                    tokens=self.respond(options, self._window),
                    avg_logprob=-0.1,
                    no_speech_prob=0.0,
                    temperature=options.temperature,
                    compression_ratio=self.compression_ratio(options, self._window),
                )
            )
            self._window += 1
        return results[0] if single else results


@pytest.fixture
def run_transcribe(monkeypatch):
    """Run transcribe() on `seconds` of silence against a StubModel."""

    def run(model: StubModel, seconds: float, **kwargs):
        monkeypatch.setattr(
            transcribe_module.ModelHolder, "get_model", lambda *a, **k: model
        )
        audio = np.zeros(int(SAMPLE_RATE * seconds), dtype=np.float32)
        kwargs.setdefault("language", "en")
        return transcribe_module.transcribe(audio, **kwargs)

    return run


@pytest.fixture
def ts(tokenizer):
    """Token id of the timestamp `seconds` into a window, e.g. ts(2.0) -> <|2.00|>."""
    return lambda seconds: tokenizer.timestamp_begin + round(seconds / 0.02)
