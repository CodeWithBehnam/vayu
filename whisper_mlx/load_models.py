# Copyright © 2023 Apple Inc.

import json
from pathlib import Path
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
from huggingface_hub import snapshot_download
from mlx.utils import tree_unflatten

from . import whisper


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
    if not model_path.exists():
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
