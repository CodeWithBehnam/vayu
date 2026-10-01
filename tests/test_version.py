"""The package's __version__ must match the version in pyproject.toml."""

from importlib.metadata import version

import whisper_mlx


def test_version_matches_installed_metadata():
    assert whisper_mlx.__version__ == version("vayu-whisper")
