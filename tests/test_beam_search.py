"""Beam search is not implemented: say so instead of failing inconsistently."""

import pytest

from tests.conftest import StubModel
from whisper_mlx import LightningWhisperMLX
from whisper_mlx.cli import build_parser


@pytest.fixture
def stub(tokenizer, ts):
    return StubModel(lambda o, w: [ts(0), *tokenizer.encode(" hello"), ts(2)])


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("option", [{"beam_size": 5}, {"patience": 1.0}])
def test_transcribe_rejects_beam_search_options(
    stub, run_transcribe, batch_size, option
):
    with pytest.raises(NotImplementedError, match="beam search"):
        run_transcribe(stub, seconds=30, batch_size=batch_size, **option)


def test_unset_beam_search_options_are_fine(stub, run_transcribe):
    result = run_transcribe(stub, seconds=30, beam_size=None, patience=None)

    assert result["text"] == " hello"


def test_lightning_wrapper_rejects_beam_size():
    whisper = LightningWhisperMLX(model="tiny")

    with pytest.raises(NotImplementedError, match="beam search"):
        whisper.transcribe("audio.mp3", beam_size=5)


def test_cli_has_no_patience_option(capsys):
    with pytest.raises(SystemExit):
        build_parser().parse_args(["audio.mp3", "--patience", "1.0"])

    assert "--patience" in capsys.readouterr().err
