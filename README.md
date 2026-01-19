# Vayu (وایو)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Platform: macOS](https://img.shields.io/badge/platform-macOS-lightgrey.svg)](https://www.apple.com/macos/)
[![Apple Silicon](https://img.shields.io/badge/Apple%20Silicon-M1%2FM2%2FM3%2FM4-orange.svg)](https://support.apple.com/en-us/HT211814)
[![MLX](https://img.shields.io/badge/MLX-0.11+-purple.svg)](https://github.com/ml-explore/mlx)
[![Version](https://img.shields.io/badge/version-1.0.0-brightgreen.svg)](https://github.com/ml-explore/mlx-examples)

**The fastest Whisper implementation on Apple Silicon.**

> **Vayu** (وایو) is the ancient Persian god of wind — the swiftest force in nature. In Zoroastrian mythology, Vayu represents the divine wind that moves faster than any earthly creature. We chose this name because this implementation outperforms even "lightning-fast" alternatives, making Vayu the most fitting name for the fastest Whisper on Apple Silicon.

## Acknowledgments

This project builds upon the excellent work of others. We are grateful to:

- **[Apple MLX Team](https://github.com/ml-explore/mlx-examples)** - For the MLX framework and the original Whisper MLX implementation with CLI support, output writers, and numerical stability improvements
- **[Mustafa Aljadery](https://github.com/mustafaaljadery/lightning-whisper-mlx)** - For the lightning-fast batched decoding implementation that significantly improves throughput
- **[Siddharth Sharma](https://github.com/mustafaaljadery/lightning-whisper-mlx)** - Co-author of lightning-whisper-mlx
- **[OpenAI](https://github.com/openai/whisper)** - For creating the original Whisper model and making it open source

This unified implementation combines the best of both worlds:
- **ml-explore/mlx-examples/whisper** - Newer APIs, CLI support, output writers, numerical stability
- **lightning-whisper-mlx** - Batched decoding for higher throughput

## Features

- **Batched decoding** - Process multiple audio segments in parallel for 3-5x faster transcription
- **Multiple output formats** - txt, vtt, srt, tsv, json
- **Word-level timestamps** - Extract precise word timings
- **Multiple model support** - tiny, base, small, medium, large-v3, turbo, distil variants
- **Quantization** - 4-bit and 8-bit quantized models for reduced memory usage
- **Simple API** - Easy-to-use `LightningWhisperMLX` wrapper class

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd vayu

# Install the package
pip install -e .

# Download required assets (mel filters and tokenizer vocabularies)
python -m whisper_mlx.assets.download_assets
```

### Requirements

- macOS with Apple Silicon (M1/M2/M3)
- Python 3.10+
- MLX 0.11+

## Quick Start

### Simple API

```python
from whisper_mlx import LightningWhisperMLX

# Initialize with batched decoding
whisper = LightningWhisperMLX(model="distil-large-v3", batch_size=12)

# Transcribe
result = whisper.transcribe("audio.mp3")
print(result["text"])

# With options
result = whisper.transcribe(
    "audio.mp3",
    language="en",
    word_timestamps=True,
)
```

### Full API

```python
from whisper_mlx import transcribe

result = transcribe(
    "audio.mp3",
    path_or_hf_repo="mlx-community/whisper-turbo",
    batch_size=6,
    language="en",
    word_timestamps=True,
)

print(result["text"])
for segment in result["segments"]:
    print(f"[{segment['start']:.2f} -> {segment['end']:.2f}] {segment['text']}")
```

### CLI

```bash
# Basic transcription
vayu audio.mp3

# With batched decoding (faster)
vayu audio.mp3 --batch-size 12

# Specify model and output format
vayu audio.mp3 --model mlx-community/distil-whisper-large-v3 --output-format srt

# Multiple files
vayu audio1.mp3 audio2.mp3 --output-dir ./transcripts

# With word timestamps
vayu audio.mp3 --word-timestamps True

# Translate to English
vayu audio.mp3 --task translate
```

## Available Models

| Model | HuggingFace Repo | Size | Speed |
|-------|------------------|------|-------|
| tiny | mlx-community/whisper-tiny-mlx | 39M | Fastest |
| base | mlx-community/whisper-base-mlx | 74M | Fast |
| small | mlx-community/whisper-small-mlx | 244M | Medium |
| medium | mlx-community/whisper-medium-mlx | 769M | Slow |
| large-v3 | mlx-community/whisper-large-v3-mlx | 1.5B | Slowest |
| turbo | mlx-community/whisper-turbo | 809M | Fast |
| distil-large-v3 | mlx-community/distil-whisper-large-v3 | 756M | Fast |

### Quantized Models

For reduced memory usage, use quantized models:

```python
whisper = LightningWhisperMLX(model="distil-large-v3", quant="4bit")
```

## Batch Size Recommendations

| Model | Recommended batch_size | Memory Usage |
|-------|------------------------|--------------|
| tiny/base | 24-32 | Low |
| small | 16-24 | Medium |
| medium | 8-12 | High |
| large/turbo | 4-8 | High |
| distil-large-v3 | 12-16 | Medium |

Higher batch sizes improve throughput but require more memory. Start with the recommended values and adjust based on your hardware.

## API Reference

### transcribe()

```python
def transcribe(
    audio: Union[str, np.ndarray, mx.array],
    *,
    path_or_hf_repo: str = "mlx-community/whisper-turbo",
    batch_size: int = 1,
    verbose: Optional[bool] = None,
    temperature: Union[float, Tuple[float, ...]] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    compression_ratio_threshold: Optional[float] = 2.4,
    logprob_threshold: Optional[float] = -1.0,
    no_speech_threshold: Optional[float] = 0.6,
    condition_on_previous_text: bool = True,
    initial_prompt: Optional[str] = None,
    word_timestamps: bool = False,
    **decode_options,
) -> dict
```

### LightningWhisperMLX

```python
class LightningWhisperMLX:
    def __init__(
        self,
        model: str = "distil-large-v3",
        batch_size: int = 12,
        quant: str = None,
    )

    def transcribe(
        self,
        audio_path: str,
        language: str = None,
        task: str = "transcribe",
        verbose: bool = False,
        word_timestamps: bool = False,
        **kwargs,
    ) -> dict
```

## License

MIT License - see LICENSE file for details.

## Author

**Behnam Ebrahimi** - Unified implementation, security improvements, and maintenance

## Credits

This project would not be possible without:

| Project | Author(s) | Contribution |
|---------|-----------|--------------|
| [mlx-examples/whisper](https://github.com/ml-explore/mlx-examples) | Apple Inc. | MLX framework, Whisper port, CLI, output writers |
| [lightning-whisper-mlx](https://github.com/mustafaaljadery/lightning-whisper-mlx) | Mustafa Aljadery, Siddharth Sharma | Batched decoding for 3-5x speedup |
| [Whisper](https://github.com/openai/whisper) | OpenAI | Original model architecture and weights |

Thank you to all contributors who make open source AI accessible and fast on Apple Silicon.
