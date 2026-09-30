"""ApplyTimestampRules: timestamp tokens must come in order and in pairs."""

import mlx.core as mx
import numpy as np
import pytest

from tests.conftest import random_whisper
from whisper_mlx.decoding import ApplyTimestampRules, DecodingOptions

N_VOCAB = 51864


@pytest.fixture
def allowed(tokenizer):
    """Which tokens the rules leave open after `sampled` (sot is the only prompt)."""
    rules = ApplyTimestampRules(
        tokenizer, sample_begin=1, max_initial_timestamp_index=None
    )

    def run(sampled):
        tokens = mx.array([[tokenizer.sot, *sampled]])
        return ~np.isneginf(np.array(rules.apply(mx.zeros((1, N_VOCAB)), tokens))[0])

    return run


def test_timestamp_after_text_must_be_later(allowed, tokenizer, ts):
    ok = allowed([ts(1.0), *tokenizer.encode(" hi")])

    assert not ok[ts(0.2)]
    assert not ok[ts(1.0)]  # a segment can't end where it started
    assert ok[ts(1.02)]


def test_single_closing_timestamp_may_repeat_to_open_next_segment(
    allowed, tokenizer, ts
):
    ok = allowed([ts(1.0), *tokenizer.encode(" hi"), ts(2.0)])

    assert not ok[ts(1.98)]
    assert ok[ts(2.0)]
    assert ok[ts(2.5)]


def test_no_timestamp_straight_after_a_pair(allowed, tokenizer, ts):
    ok = allowed([ts(1.0), *tokenizer.encode(" hi"), ts(2.0), ts(2.0)])

    assert not ok[tokenizer.timestamp_begin :].any()


def test_decoded_timestamps_never_go_backwards(tokenizer):
    model = random_whisper(seed=3)
    mx.random.seed(0)
    mel = mx.random.normal((2, 3000, 80)).astype(mx.float16)

    results = model.decode(mel, DecodingOptions(language="en", sample_len=96))

    for result in results:
        previous, text_since = None, False
        for token in result.tokens:
            if token >= tokenizer.timestamp_begin:
                if previous is not None:
                    assert token >= previous  # never earlier than the last one
                    if text_since:
                        assert token > previous  # a segment has nonzero length
                previous, text_since = token, False
            elif token < tokenizer.eot:
                text_since = True
