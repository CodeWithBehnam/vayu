# Copyright © 2024 Mustafa Aljadery & Siddharth Sharma
# Simple API wrapper for quick transcription

from .transcribe import transcribe as transcribe_audio
from .utils import MODEL_REPOS, QUANT_REPOS, resolve_model_path


class LightningWhisperMLX:
    """
    Simple API wrapper for Whisper MLX transcription.

    Example usage:
        whisper = LightningWhisperMLX(model="distil-large-v3", batch_size=12)
        result = whisper.transcribe("audio.mp3")
        print(result["text"])
    """

    def __init__(
        self,
        model: str = "distil-large-v3",
        batch_size: int = 12,
        quant: str = None,
    ):
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
        audio_path: str,
        language: str = None,
        task: str = "transcribe",
        verbose: bool = False,
        word_timestamps: bool = False,
        **kwargs,
    ) -> dict:
        """
        Transcribe an audio file.

        Parameters
        ----------
        audio_path : str
            Path to the audio file to transcribe.

        language : str, optional
            Language code (e.g., "en", "es", "fr"). If None, auto-detects.

        task : str
            "transcribe" for speech recognition or "translate" for translation to English.

        verbose : bool
            Whether to print progress during transcription.

        word_timestamps : bool
            Whether to include word-level timestamps.

        **kwargs
            Additional arguments passed to the transcribe function.

        Returns
        -------
        dict
            Dictionary containing:
            - "text": The full transcription text
            - "segments": List of segment dictionaries with timestamps
            - "language": Detected or specified language
        """
        result = transcribe_audio(
            audio_path,
            path_or_hf_repo=self.model_path,
            batch_size=self.batch_size,
            language=language,
            task=task,
            verbose=verbose,
            word_timestamps=word_timestamps,
            **kwargs,
        )
        return result

    def __repr__(self):
        return f"LightningWhisperMLX(model='{self.name}', batch_size={self.batch_size}, quant={self.quant})"
