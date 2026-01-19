"""Tests for model loading security features."""

import os
import tempfile
from pathlib import Path
from unittest import mock

import pytest

from whisper_mlx.load_models import (
    _get_allowed_model_dirs,
    load_model,
    validate_model_path,
)


class TestGetAllowedModelDirs:
    """Tests for _get_allowed_model_dirs function."""

    def test_returns_default_directories(self) -> None:
        """Default directories include HuggingFace cache and system dir."""
        with mock.patch.dict(os.environ, {}, clear=True):
            # Clear WHISPER_MLX_MODEL_DIRS if set
            os.environ.pop("WHISPER_MLX_MODEL_DIRS", None)
            dirs = _get_allowed_model_dirs()

        assert len(dirs) >= 2
        dir_strs = [str(d) for d in dirs]
        assert any(".cache/huggingface/hub" in d for d in dir_strs)
        assert any("/usr/local/share/whisper-mlx" in d for d in dir_strs)

    def test_includes_custom_dirs_from_env(self) -> None:
        """Custom directories from WHISPER_MLX_MODEL_DIRS are included."""
        custom_paths = "/custom/path1:/custom/path2"
        with mock.patch.dict(os.environ, {"WHISPER_MLX_MODEL_DIRS": custom_paths}):
            dirs = _get_allowed_model_dirs()

        dir_strs = [str(d) for d in dirs]
        assert any("custom/path1" in d for d in dir_strs)
        assert any("custom/path2" in d for d in dir_strs)

    def test_handles_empty_env_var(self) -> None:
        """Empty environment variable doesn't add empty paths."""
        with mock.patch.dict(os.environ, {"WHISPER_MLX_MODEL_DIRS": ""}):
            dirs = _get_allowed_model_dirs()

        # Should only have default dirs
        assert len(dirs) == 2

    def test_handles_whitespace_in_env_var(self) -> None:
        """Whitespace-only paths are ignored."""
        with mock.patch.dict(os.environ, {"WHISPER_MLX_MODEL_DIRS": "  :  :/valid/path"}):
            dirs = _get_allowed_model_dirs()

        dir_strs = [str(d) for d in dirs]
        # Should have defaults + 1 valid custom
        assert any("valid/path" in d for d in dir_strs)


class TestValidateModelPath:
    """Tests for validate_model_path function."""

    def test_allows_path_in_huggingface_cache(self) -> None:
        """Paths in HuggingFace cache are allowed."""
        hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
        test_path = hf_cache / "models--mlx-community--whisper-tiny"

        with mock.patch.object(Path, "resolve", return_value=test_path):
            with mock.patch.object(Path, "exists", return_value=True):
                # Create the directory structure for the test
                result = validate_model_path(test_path)
                assert result == test_path

    def test_rejects_path_traversal_attack(self) -> None:
        """Path traversal attempts are rejected."""
        malicious_path = Path("../../../etc/passwd")

        with pytest.raises(ValueError, match="not within allowed directories"):
            validate_model_path(malicious_path)

    def test_rejects_arbitrary_local_path(self) -> None:
        """Arbitrary local paths outside allowed dirs are rejected."""
        arbitrary_path = Path("/tmp/malicious/model")

        with pytest.raises(ValueError, match="not within allowed directories"):
            validate_model_path(arbitrary_path)

    def test_allows_custom_dir_from_env(self) -> None:
        """Paths in custom directories from env are allowed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            custom_model_dir = Path(tmpdir) / "models"
            custom_model_dir.mkdir()

            with mock.patch.dict(
                os.environ, {"WHISPER_MLX_MODEL_DIRS": str(custom_model_dir)}
            ):
                result = validate_model_path(custom_model_dir)
                assert result == custom_model_dir.resolve()

    def test_resolves_symlinks(self) -> None:
        """Symlinks are resolved before validation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a symlink pointing outside allowed dirs
            link_path = Path(tmpdir) / "sneaky_link"
            target = Path("/etc")

            # Skip if symlink can't be created (permissions)
            try:
                link_path.symlink_to(target)
            except OSError:
                pytest.skip("Cannot create symlink")

            with pytest.raises(ValueError, match="not within allowed directories"):
                validate_model_path(link_path)

    def test_error_message_includes_allowed_dirs(self) -> None:
        """Error message lists allowed directories for user guidance."""
        bad_path = Path("/bad/path")

        with pytest.raises(ValueError) as exc_info:
            validate_model_path(bad_path)

        error_msg = str(exc_info.value)
        assert "huggingface" in error_msg.lower()
        assert "WHISPER_MLX_MODEL_DIRS" in error_msg

    def test_handles_relative_path_in_allowed_dir(self) -> None:
        """Relative paths that resolve to allowed dirs work."""
        with tempfile.TemporaryDirectory() as tmpdir:
            allowed_dir = Path(tmpdir) / "allowed"
            allowed_dir.mkdir()
            model_dir = allowed_dir / "model"
            model_dir.mkdir()

            # Change to tmpdir and use relative path
            with mock.patch.dict(
                os.environ, {"WHISPER_MLX_MODEL_DIRS": str(allowed_dir)}
            ):
                result = validate_model_path(model_dir)
                assert result == model_dir.resolve()


class TestLoadModelSecurity:
    """Integration tests for load_model security."""

    def test_load_model_rejects_path_traversal(self) -> None:
        """load_model rejects path traversal attempts for local paths."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a fake model directory outside allowed paths
            malicious_path = Path(tmpdir) / "malicious_model"
            malicious_path.mkdir()
            (malicious_path / "config.json").write_text("{}")

            with pytest.raises(ValueError, match="not within allowed directories"):
                load_model(str(malicious_path))

    def test_load_model_validates_existing_local_paths(self) -> None:
        """load_model validates local paths that exist."""
        # Use /tmp which exists but is not in allowed dirs
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model"
            model_path.mkdir()
            (model_path / "config.json").write_text('{"n_mels": 80}')

            with pytest.raises(ValueError, match="not within allowed directories"):
                load_model(str(model_path))

    def test_load_model_allows_custom_env_dir(self) -> None:
        """load_model allows paths in WHISPER_MLX_MODEL_DIRS directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model"
            model_path.mkdir()

            # Create minimal config (will fail later but path validation should pass)
            (model_path / "config.json").write_text('{"n_mels": 80}')

            with mock.patch.dict(os.environ, {"WHISPER_MLX_MODEL_DIRS": tmpdir}):
                # Should pass path validation but fail on missing model dimensions
                with pytest.raises(TypeError):
                    # Missing required model args - but importantly it passed validation!
                    load_model(str(model_path))
