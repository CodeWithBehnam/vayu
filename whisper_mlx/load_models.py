# Copyright © 2023 Apple Inc.

import json
import os
from pathlib import Path
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
from huggingface_hub import snapshot_download
from mlx.utils import tree_unflatten

from . import whisper


def _get_allowed_model_dirs() -> list[Path]:
    """Get list of allowed directories for loading models.

    Returns directories that are considered safe for model loading:
    - HuggingFace cache directory (~/.cache/huggingface/hub)
    - System model directory (/usr/local/share/whisper-mlx)
    - Custom directories from WHISPER_MLX_MODEL_DIRS environment variable
    """
    allowed = [
        Path.home() / ".cache" / "huggingface" / "hub",
        Path("/usr/local/share/whisper-mlx"),
    ]

    # Allow additional directories via environment variable
    custom_dirs = os.environ.get("WHISPER_MLX_MODEL_DIRS", "")
    if custom_dirs:
        for dir_path in custom_dirs.split(os.pathsep):
            if dir_path.strip():
                allowed.append(Path(dir_path.strip()))

    return allowed


def validate_model_path(path: Path) -> Path:
    """Validate that a model path is within allowed directories.

    Args:
        path: The path to validate

    Returns:
        The resolved absolute path if valid

    Raises:
        ValueError: If the path is not within allowed directories
    """
    resolved_path = path.resolve()
    allowed_dirs = _get_allowed_model_dirs()

    for allowed_dir in allowed_dirs:
        try:
            # Check if resolved_path is within allowed_dir
            resolved_path.relative_to(allowed_dir.resolve())
            return resolved_path
        except ValueError:
            continue

    allowed_str = ", ".join(str(d) for d in allowed_dirs)
    raise ValueError(
        f"Model path '{resolved_path}' is not within allowed directories. "
        f"Allowed directories: {allowed_str}. "
        f"Set WHISPER_MLX_MODEL_DIRS environment variable to add custom directories."
    )


def load_model(
    path_or_hf_repo: str,
    dtype: mx.Dtype = mx.float32,
) -> whisper.Whisper:
    # Validate inputs
    if not path_or_hf_repo or not isinstance(path_or_hf_repo, str):
        raise ValueError("path_or_hf_repo must be a non-empty string")

    if not isinstance(dtype, mx.Dtype):
        raise TypeError(f"dtype must be an mx.Dtype, got {type(dtype).__name__}")

    model_path = Path(path_or_hf_repo)
    if model_path.exists():
        # Validate local paths to prevent path traversal attacks
        model_path = validate_model_path(model_path)
    else:
        # Download from HuggingFace Hub (automatically goes to allowed cache dir)
        model_path = Path(snapshot_download(repo_id=path_or_hf_repo))

    with open(str(model_path / "config.json"), "r") as f:
        config = json.loads(f.read())
        config.pop("model_type", None)
        quantization = config.pop("quantization", None)

    model_args = whisper.ModelDimensions(**config)

    # Prefer model.safetensors, fall back to weights.safetensors, then weights.npz
    wf = model_path / "model.safetensors"
    if not wf.exists():
        wf = model_path / "weights.safetensors"
    if not wf.exists():
        wf = model_path / "weights.npz"
    weights = mx.load(str(wf))

    model = whisper.Whisper(model_args, dtype)

    if quantization is not None:
        class_predicate = (
            lambda p, m: isinstance(m, (nn.Linear, nn.Embedding))
            and f"{p}.scales" in weights
        )
        nn.quantize(model, **quantization, class_predicate=class_predicate)

    weights = tree_unflatten(list(weights.items()))
    model.update(weights)
    mx.eval(model.parameters())
    return model


class ModelHolder:
    """Singleton cache for loaded Whisper models."""

    model: Optional[whisper.Whisper] = None
    model_path: Optional[str] = None

    @classmethod
    def get_model(cls, model_path: str, dtype: mx.Dtype) -> whisper.Whisper:
        """Get a cached model or load a new one."""
        if cls.model is None or model_path != cls.model_path:
            cls.model = load_model(model_path, dtype=dtype)
            cls.model_path = model_path
        return cls.model
