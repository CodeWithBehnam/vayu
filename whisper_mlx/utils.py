# Copyright © 2024 Apple Inc.
# Centralized utilities for whisper_mlx

from typing import Optional


def format_timestamp(
    seconds: float, always_include_hours: bool = False, decimal_marker: str = "."
) -> str:
    """
    Format seconds as a timestamp string.

    Parameters
    ----------
    seconds : float
        Time in seconds (must be non-negative)
    always_include_hours : bool
        If True, always include hours even if zero
    decimal_marker : str
        Character to use before milliseconds (default ".")

    Returns
    -------
    str
        Formatted timestamp like "01:23.456" or "00:01:23.456"
    """
    assert seconds >= 0, "non-negative timestamp expected"
    milliseconds = round(seconds * 1000.0)

    hours = milliseconds // 3_600_000
    milliseconds -= hours * 3_600_000

    minutes = milliseconds // 60_000
    milliseconds -= minutes * 60_000

    seconds_int = milliseconds // 1_000
    milliseconds -= seconds_int * 1_000

    hours_marker = f"{hours:02d}:" if always_include_hours or hours > 0 else ""
    return f"{hours_marker}{minutes:02d}:{seconds_int:02d}{decimal_marker}{milliseconds:03d}"


# Canonical mapping of model names to HuggingFace repos
MODEL_REPOS = {
    "tiny": "mlx-community/whisper-tiny-mlx",
    "tiny.en": "mlx-community/whisper-tiny.en-mlx",
    "base": "mlx-community/whisper-base-mlx",
    "base.en": "mlx-community/whisper-base.en-mlx",
    "small": "mlx-community/whisper-small-mlx",
    "small.en": "mlx-community/whisper-small.en-mlx",
    "medium": "mlx-community/whisper-medium-mlx",
    "medium.en": "mlx-community/whisper-medium.en-mlx",
    "large": "mlx-community/whisper-large-v3-mlx",
    "large-v2": "mlx-community/whisper-large-v2-mlx",
    "large-v3": "mlx-community/whisper-large-v3-mlx",
    "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
    "turbo": "mlx-community/whisper-turbo",
    "distil-large-v2": "mlx-community/distil-whisper-large-v2",
    "distil-large-v3": "mlx-community/distil-whisper-large-v3",
}

# Quantized model repos
QUANT_REPOS = {
    "tiny": {
        "4bit": "mlx-community/whisper-tiny-mlx-4bit",
        "8bit": "mlx-community/whisper-tiny-mlx-8bit",
    },
    "small": {
        "4bit": "mlx-community/whisper-small-mlx-4bit",
        "8bit": "mlx-community/whisper-small-mlx-8bit",
    },
    "medium": {
        "4bit": "mlx-community/whisper-medium-mlx-4bit",
        "8bit": "mlx-community/whisper-medium-mlx-8bit",
    },
    "large-v3": {
        "4bit": "mlx-community/whisper-large-v3-mlx-4bit",
        "8bit": "mlx-community/whisper-large-v3-mlx-8bit",
    },
    "distil-large-v3": {
        "4bit": "mlx-community/distil-whisper-large-v3-4bit",
        "8bit": "mlx-community/distil-whisper-large-v3-8bit",
    },
}


def resolve_model_path(
    model: str, quant: Optional[str] = None
) -> str:
    """
    Resolve a model name to its HuggingFace repo path.

    Parameters
    ----------
    model : str
        Model name (e.g., "tiny", "turbo") or HuggingFace repo path
    quant : str, optional
        Quantization level: "4bit" or "8bit"

    Returns
    -------
    str
        HuggingFace repo path

    Raises
    ------
    ValueError
        If model name is unknown
    """
    # Check quantized repos first
    if quant and model in QUANT_REPOS and quant in QUANT_REPOS[model]:
        return QUANT_REPOS[model][quant]

    # Check standard repos
    if model in MODEL_REPOS:
        return MODEL_REPOS[model]

    # Assume it's a HuggingFace repo path if contains "/"
    if "/" in model:
        return model

    raise ValueError(
        f"Unknown model: {model}. Available models: {list(MODEL_REPOS.keys())}"
    )
