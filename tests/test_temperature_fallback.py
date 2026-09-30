"""Temperature fallback in transcribe(), sequential and batched."""

import pytest

from tests.conftest import StubModel

SCHEDULE = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)


@pytest.fixture
def tokens(tokenizer, ts):
    return [ts(0), *tokenizer.encode(" hello"), ts(2)]


@pytest.mark.parametrize("batch_size", [1, 3])
def test_single_temperature_means_no_fallback(tokens, run_transcribe, batch_size):
    model = StubModel(lambda o, w: tokens, compression_ratio=lambda o, w: 3.0)

    run_transcribe(
        model, seconds=90, noise=True, batch_size=batch_size, temperature=0.0
    )

    assert {c.temperature for c in model.calls} == {0.0}


def test_batched_fallback_walks_schedule_for_failing_segments_only(
    tokens, run_transcribe
):
    # window 1 is too repetitive until t=0.4; the others pass straight away
    def ratio(options, window):
        return 3.0 if window == 1 and options.temperature < 0.4 else 1.0

    model = StubModel(lambda o, w: tokens, compression_ratio=ratio)

    result = run_transcribe(model, seconds=90, noise=True, batch_size=3)

    assert [(c.temperature, n) for c, n in zip(model.calls, model.batch_sizes)] == [
        (0.0, 3),
        (0.2, 1),
        (0.4, 1),
    ]
    assert [s["temperature"] for s in result["segments"]] == [0.0, 0.4, 0.0]


def test_batched_fallback_keeps_last_result_when_schedule_runs_out(
    tokens, run_transcribe
):
    model = StubModel(lambda o, w: tokens, compression_ratio=lambda o, w: 3.0)

    result = run_transcribe(model, seconds=60, noise=True, batch_size=2)

    assert [c.temperature for c in model.calls] == list(SCHEDULE)
    assert [s["temperature"] for s in result["segments"]] == [1.0, 1.0]


def test_silent_segments_do_not_fall_back(tokens, run_transcribe):
    model = StubModel(lambda o, w: tokens, compression_ratio=lambda o, w: 3.0)
    model.no_speech_prob = 0.9

    run_transcribe(model, seconds=60, noise=True, batch_size=2, logprob_threshold=None)

    assert [c.temperature for c in model.calls] == [0.0]


def test_sequential_fallback_walks_schedule(tokens, run_transcribe):
    def ratio(options, window):
        return 3.0 if options.temperature < 0.4 else 1.0

    model = StubModel(lambda o, w: tokens, compression_ratio=ratio)

    result = run_transcribe(model, seconds=30, noise=True, batch_size=1)

    assert [c.temperature for c in model.calls] == [0.0, 0.2, 0.4]
    assert [s["temperature"] for s in result["segments"]] == [0.4]
