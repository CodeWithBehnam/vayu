"""Command-line interface: output naming and error reporting."""

import importlib
import shutil
import sys

import numpy as np
import pytest

from whisper_mlx.audio import AudioLoadError, load_audio

cli = importlib.import_module("whisper_mlx.cli")


@pytest.fixture
def run_cli(monkeypatch, tmp_path):
    """Run `vayu <args>` with transcription and writing stubbed out.

    Returns (exit_code, names written, inputs passed to transcribe()).
    """

    def run(*args, transcribe=None):
        written, transcribed = [], []

        def fake_transcribe(audio, **kwargs):
            transcribed.append(audio)
            return {"text": "", "segments": [], "language": "en"}

        monkeypatch.setattr(cli, "transcribe", transcribe or fake_transcribe)
        monkeypatch.setattr(
            cli,
            "get_writer",
            lambda fmt, out: lambda r, name, **k: written.append(name),
        )
        monkeypatch.setattr(sys, "argv", ["vayu", *args, "-o", str(tmp_path)])
        try:
            cli.main()
            code = 0
        except SystemExit as e:
            code = e.code
        return code, written, transcribed

    return run


def test_each_input_gets_its_own_output_name(run_cli):
    code, written, _ = run_cli("talks/a.mp3", "b.wav")

    assert code == 0
    assert written == ["a", "b"]


def test_output_name_applies_to_a_single_input(run_cli):
    code, written, _ = run_cli("a.mp3", "--output-name", "custom")

    assert (code, written) == (0, ["custom"])


def test_output_name_with_several_inputs_is_rejected(run_cli, capsys):
    code, written, _ = run_cli("a.mp3", "b.mp3", "--output-name", "custom")

    assert code == 2
    assert written == []
    assert "--output-name" in capsys.readouterr().err


def test_inputs_with_the_same_stem_are_rejected(run_cli, capsys):
    code, written, _ = run_cli("day1/talk.mp3", "day2/talk.mp3")

    assert code == 2
    assert written == []
    assert "talk" in capsys.readouterr().err


def test_stdin_is_read_inside_error_handling(run_cli, monkeypatch, capsys):
    def broken_stdin(**kwargs):
        raise AudioLoadError("Failed to load audio: invalid data")

    monkeypatch.setattr(cli.audio, "load_audio", broken_stdin)

    code, written, _ = run_cli("-")

    assert code == 1
    assert "  - -: AudioLoadError" in capsys.readouterr().out


def test_stdin_output_is_named_content(run_cli, monkeypatch):
    monkeypatch.setattr(cli.audio, "load_audio", lambda **k: np.zeros(16000))

    code, written, transcribed = run_cli("-")

    assert (code, written) == (0, ["content"])
    assert isinstance(transcribed[0], np.ndarray)


def test_missing_file_is_reported_and_others_still_run(run_cli, capsys):
    def transcribe(audio, **kwargs):
        if audio == "missing.wav":
            raise FileNotFoundError(f"Audio file not found: {audio}")
        return {"text": "", "segments": [], "language": "en"}

    code, written, _ = run_cli("missing.wav", "present.wav", transcribe=transcribe)

    assert code == 1
    assert written == ["present"]
    out = capsys.readouterr().out
    assert "1 file(s) failed" in out
    assert "missing.wav: FileNotFoundError" in out


def test_load_audio_reports_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="Audio file not found"):
        load_audio(str(tmp_path / "nope.wav"))


def test_load_audio_reports_missing_ffmpeg(tmp_path, monkeypatch):
    clip = tmp_path / "clip.wav"
    clip.write_bytes(b"not audio")
    monkeypatch.setenv("PATH", str(tmp_path))  # no ffmpeg in here

    with pytest.raises(AudioLoadError, match="ffmpeg was not found"):
        load_audio(str(clip))


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_load_audio_reports_undecodable_file(tmp_path):
    clip = tmp_path / "clip.mp3"
    clip.write_text("this is not an mp3")

    with pytest.raises(AudioLoadError, match="Failed to load audio"):
        load_audio(str(clip))


def test_transcribe_checks_the_file_before_loading_the_model(monkeypatch, tmp_path):
    module = importlib.import_module("whisper_mlx.transcribe")

    def no_model(*args, **kwargs):
        raise AssertionError("model should not be loaded")

    monkeypatch.setattr(module.ModelHolder, "get_model", no_model)

    with pytest.raises(FileNotFoundError):
        module.transcribe(str(tmp_path / "missing.mp3"))
