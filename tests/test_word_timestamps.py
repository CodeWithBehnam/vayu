"""Word-level timestamps from transcribe(), sequential and batched."""

import warnings

import mlx.core as mx
import pytest

from tests.conftest import TINY_EN_DIMS, StubModel
from whisper_mlx.whisper import Whisper


class CannedWhisper(Whisper):
    """Random-weight Whisper: decode() returns canned tokens, alignment runs for real."""

    def __init__(self, stub: StubModel):
        super().__init__(TINY_EN_DIMS, mx.float16)
        mx.eval(self.parameters())
        self.stub = stub

    def decode(self, mel, options):
        return self.stub.decode(mel, options)


@pytest.fixture
def model(tokenizer, ts):
    tokens = [
        ts(0),
        *tokenizer.encode(" hello there"),
        ts(2),
        ts(2),
        *tokenizer.encode(" general"),
        ts(3),
    ]
    mx.random.seed(0)
    return CannedWhisper(StubModel(lambda options, window: tokens))


@pytest.mark.parametrize("batch_size", [1, 4])
def test_segments_get_words_within_their_window(model, run_transcribe, batch_size):
    result = run_transcribe(
        model, seconds=45, batch_size=batch_size, word_timestamps=True
    )

    segments = [s for s in result["segments"] if s["text"]]
    assert segments
    for segment in segments:
        assert "".join(w["word"] for w in segment["words"]) == segment["text"]
        window_start = segment["seek"] / 100
        for word in segment["words"]:
            assert window_start <= word["start"] <= word["end"] <= window_start + 30


def test_batched_words_use_each_windows_offset(model, run_transcribe):
    result = run_transcribe(model, seconds=65, batch_size=4, word_timestamps=True)

    first_word_starts = sorted(
        {s["seek"]: s["words"][0]["start"] for s in result["segments"]}.items()
    )
    assert [seek for seek, _ in first_word_starts] == [0, 3000, 6000]
    for seek, start in first_word_starts:
        assert seek / 100 <= start < seek / 100 + 30


def test_batched_warns_that_hallucination_threshold_is_ignored(model, run_transcribe):
    with pytest.warns(UserWarning, match="hallucination_silence_threshold"):
        run_transcribe(
            model,
            seconds=30,
            batch_size=2,
            word_timestamps=True,
            hallucination_silence_threshold=2.0,
        )


def test_sequential_does_not_warn_about_hallucination_threshold(model, run_transcribe):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        run_transcribe(
            model,
            seconds=30,
            batch_size=1,
            word_timestamps=True,
            hallucination_silence_threshold=2.0,
        )
