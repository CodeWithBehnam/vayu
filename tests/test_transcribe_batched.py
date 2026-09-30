"""Batched decoding (batch_size > 1) in transcribe()."""

from tests.conftest import StubModel


def test_keeps_text_after_last_timestamp_pair(tokenizer, ts, run_transcribe):
    # Each window ends mid-sentence: "<|0.00|> hello <|2.00|><|2.00|> world"
    tokens = [
        ts(0),
        *tokenizer.encode(" hello"),
        ts(2),
        ts(2),
        *tokenizer.encode(" world"),
    ]
    model = StubModel(lambda options, window: tokens)

    result = run_transcribe(model, seconds=65, batch_size=4)

    assert result["text"].count("world") == 3
    world = [s for s in result["segments"] if "world" in s["text"]]
    # the unfinished segment runs from its timestamp to the end of its window
    assert [(s["start"], s["end"]) for s in world] == [
        (2.0, 30.0),
        (32.0, 60.0),
        (62.0, 65.0),
    ]


def test_segments_record_their_own_window_seek(tokenizer, ts, run_transcribe):
    tokens = [
        ts(0),
        *tokenizer.encode(" hello"),
        ts(2),
        ts(2),
        *tokenizer.encode(" world"),
        ts(3),
    ]
    model = StubModel(lambda options, window: tokens)

    result = run_transcribe(model, seconds=65, batch_size=4)

    assert [s["seek"] for s in result["segments"]] == [0, 0, 3000, 3000, 6000, 6000]
    assert [s["id"] for s in result["segments"]] == list(range(6))


def test_single_timestamp_ending_is_unchanged(tokenizer, ts, run_transcribe):
    tokens = [
        ts(0),
        *tokenizer.encode(" hello"),
        ts(2),
        ts(2),
        *tokenizer.encode(" world"),
        ts(3),
    ]
    model = StubModel(lambda options, window: tokens)

    result = run_transcribe(model, seconds=30, batch_size=2)

    assert [(s["start"], s["end"], s["text"]) for s in result["segments"]] == [
        (0.0, 2.0, " hello"),
        (2.0, 3.0, " world"),
    ]


def test_trailing_timestamp_without_text_adds_no_segment(tokenizer, ts, run_transcribe):
    tokens = [ts(0), *tokenizer.encode(" hello"), ts(2), ts(2)]
    model = StubModel(lambda options, window: tokens)

    result = run_transcribe(model, seconds=30, batch_size=2)

    assert [(s["start"], s["end"], s["text"]) for s in result["segments"]] == [
        (0.0, 2.0, " hello")
    ]


def test_text_without_timestamp_pairs_spans_the_window(tokenizer, ts, run_transcribe):
    tokens = [ts(0), *tokenizer.encode(" hello world")]
    model = StubModel(lambda options, window: tokens)

    result = run_transcribe(model, seconds=45, batch_size=2)

    assert [(s["start"], s["end"]) for s in result["segments"]] == [
        (0.0, 30.0),
        (30.0, 45.0),
    ]
    assert result["text"] == " hello world hello world"
