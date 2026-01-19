# Copyright © 2024 Mustafa Aljadery & Siddharth Sharma
# Simple API wrapper for quick transcription

from typing import Any, Dict, Optional, Union

import mlx.core as mx
import numpy as np

from .transcribe import transcribe as transcribe_audio
from .utils import MODEL_REPOS, QUANT_REPOS, resolve_model_path


class LightningWhisperMLX:
    """
    Simple API wrapper for Whisper MLX transcription.

    This is a thin wrapper around transcribe() that provides:
    - Friendly model names (e.g., "turbo" instead of "mlx-community/whisper-turbo")
    - Quantization support via the quant parameter
    - All transcribe() parameters are supported via **kwargs

    Example usage:
        whisper = LightningWhisperMLX(model="distil-large-v3", batch_size=12)
        result = whisper.transcribe("audio.mp3")
        print(result["text"])

        # All transcribe() options work:
        result = whisper.transcribe(
            "audio.mp3",
            word_timestamps=True,
            condition_on_previous_text=False,
            compression_ratio_threshold=2.0,
        )
    """

    def __init__(
        self,
        model: str = "distil-large-v3",
        batch_size: int = 12,
        quant: Optional[str] = None,
    ) -> None:
        """
        Initialize the LightningWhisperMLX transcriber.

        Parameters
        ----------
        model : str
            Model name or HuggingFace repo path. Common options:
            - "tiny", "base", "small", "medium", "large-v3"
            - "turbo", "large-v3-turbo"
            - "distil-large-v2", "distil-large-v3"

        batch_size : int
            Number of audio segments to process in parallel.
            Higher values use more memory but improve throughput.
            Recommended: 12 for distil models, 6 for large models.

        quant : str, optional
            Quantization level: "4bit" or "8bit". Only supported for some models.
        """
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")

        self.batch_size = batch_size
        self.quant = quant
        self.name = model
        self.model_path = resolve_model_path(model, quant)

    def transcribe(
        self,
        audio: Union[str, np.ndarray, mx.array],
        language: Optional[str] = None,
        task: str = "transcribe",
        verbose: Optional[bool] = None,
        word_timestamps: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Transcribe an audio file.

        This is a thin wrapper around transcribe() - all parameters from
        transcribe() are supported via **kwargs.

        Parameters
        ----------
        audio : str | np.ndarray | mx.array
            Path to the audio file or audio waveform array.

        language : str, optional
            Language code (e.g., "en", "es", "fr"). If None, auto-detects.

        task : str
            "transcribe" for speech recognition or "translate" for translation to English.

        verbose : bool, optional
            Whether to print progress during transcription.

        word_timestamps : bool
            Whether to include word-level timestamps.

        **kwargs
            All transcribe() parameters are supported:
            - temperature, compression_ratio_threshold, logprob_threshold
            - no_speech_threshold, condition_on_previous_text, initial_prompt
            - prepend_punctuations, append_punctuations, clip_timestamps
            - hallucination_silence_threshold, fp16, beam_size, patience, etc.

        Returns
        -------
        dict
            Dictionary containing:
            - "text": The full transcription text
            - "segments": List of segment dictionaries with timestamps
            - "language": Detected or specified language
        """
        return transcribe_audio(
            audio,
            path_or_hf_repo=self.model_path,
            batch_size=self.batch_size,
            language=language,
            task=task,
            verbose=verbose,
            word_timestamps=word_timestamps,
            **kwargs,
        )

    def __repr__(self) -> str:
        return f"LightningWhisperMLX(model='{self.name}', batch_size={self.batch_size}, quant={self.quant})"
