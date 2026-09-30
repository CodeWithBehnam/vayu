<p align="center">
  <img src="assets/banner.png" alt="Vayu - The Fastest Whisper on Apple Silicon" width="100%">
</p>

# Vayu - The Fastest Whisper on Apple Silicon

[![PyPI](https://img.shields.io/pypi/v/vayu-whisper.svg)](https://pypi.org/project/vayu-whisper/)
[![Downloads](https://img.shields.io/pypi/dm/vayu-whisper.svg)](https://pypi.org/project/vayu-whisper/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Platform: macOS](https://img.shields.io/badge/platform-macOS-lightgrey.svg)](https://www.apple.com/macos/)
[![Apple Silicon](https://img.shields.io/badge/Apple%20Silicon-M1%2FM2%2FM3%2FM4-orange.svg)](https://support.apple.com/en-us/HT211814)
[![MLX](https://img.shields.io/badge/MLX-0.11+-purple.svg)](https://github.com/ml-explore/mlx)
[![Stars](https://img.shields.io/github/stars/CodeWithBehnam/vayu?style=social)](https://github.com/CodeWithBehnam/vayu/stargazers)

**Vayu** (وایو) is the fastest Whisper speech-to-text implementation optimized for Apple Silicon Macs.
It combines [MLX Whisper](https://github.com/ml-explore/mlx-examples) with [Lightning Whisper MLX](https://github.com/mustafaaljadery/lightning-whisper-mlx) batched decoding to deliver **3-5x faster transcription** than standard Whisper on M1/M2/M3/M4 chips.

> Named after the ancient Persian god of wind - the swiftest force in nature.

## Why Vayu?

- **3-5x faster** than standard Whisper via batched decoding on Apple Silicon
- **One-line install** - `pip install vayu-whisper` and you're transcribing
- **All Whisper models** - tiny through large-v3, plus turbo and distil variants
- **Multiple output formats** - txt, vtt, srt, tsv, json
- **Word-level timestamps** - precise word timings for subtitles and alignment
- **Low memory options** - 4-bit and 8-bit quantization for constrained environments
- **Simple Python API + CLI** - use from code or the command line

## Installation

```bash
pip install vayu-whisper
```

Or install from source:

```bash
git clone https://github.com/CodeWithBehnam/vayu.git
cd vayu
pip install -e .
```

### Requirements

- macOS with Apple Silicon (M1/M2/M3/M4)
- Python 3.10+
- MLX 0.11+

## Quick Start

### Python API

```python
from whisper_mlx import LightningWhisperMLX

# Initialize with batched decoding for maximum speed
whisper = LightningWhisperMLX(model="distil-large-v3", batch_size=12)

# Transcribe audio
result = whisper.transcribe("audio.mp3")
print(result["text"])

# With word-level timestamps
result = whisper.transcribe("audio.mp3", language="en", word_timestamps=True)
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

# Batched decoding (3-5x faster)
vayu audio.mp3 --batch-size 12

# Specify model and output format
vayu audio.mp3 --model mlx-community/distil-whisper-large-v3 --output-format srt

# Multiple files
vayu audio1.mp3 audio2.mp3 --output-dir ./transcripts

# Word-level timestamps
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

`quant` accepts `"4bit"` or `"8bit"` and is available for `tiny`, `small`, `medium`, `large-v3` and `distil-large-v3`. Other models raise a `ValueError` rather than silently loading full precision.

## Batch Size Recommendations

| Model | Recommended batch_size | Memory Usage |
|-------|------------------------|--------------|
| tiny/base | 24-32 | Low |
| small | 16-24 | Medium |
| medium | 8-12 | High |
| large/turbo | 4-8 | High |
| distil-large-v3 | 12-16 | Medium |

Higher batch sizes improve throughput but require more memory. Start with the recommended values and adjust based on your hardware.

### Measuring the speed-up

The speed-up from batching depends on the model, batch size, chip and audio. To measure it on your Mac, time sequential against batched decoding on one of your own files:

```bash
python scripts/benchmark.py audio.mp3 --model distil-large-v3 --batch-size 12
```

The script loads the model and warms it up first, then reports the best of `--runs` timings for `batch_size=1` and for the batch size you chose.

## Batched vs Sequential Decoding

With `batch_size=1`, Vayu decodes like OpenAI's Whisper. Each 30-second window starts where the previous segment ended and is conditioned on the text so far.

With `batch_size > 1`, fixed 30-second windows are decoded together. This is much faster, with some trade-offs:

- Every window in a batch is conditioned on the text from *before* the batch, so `condition_on_previous_text` applies between batches, not between windows.
- Windows don't move to follow the speech. Text cut off at a window boundary is kept as a segment that ends at the boundary, so a word spanning two windows can be split.
- `hallucination_silence_threshold` needs window-by-window seeking and is ignored (with a warning).
- `best_of` is not used. Temperature fallback re-decodes only the windows that fail the quality checks.

Word-level timestamps work in both modes.

## Loading Local Models

`load_model` and `--model` accept a local directory with MLX weights (`config.json` plus `weights.safetensors`, `model.safetensors` or `weights.npz`). For safety, local directories are only loaded from:

- the HuggingFace cache (`~/.cache/huggingface/hub`)
- `/usr/local/share/whisper-mlx`
- directories listed in `WHISPER_MLX_MODEL_DIRS` (separated by `:`)

```bash
export WHISPER_MLX_MODEL_DIRS=~/models:/Volumes/External/whisper
vayu audio.mp3 --model ~/models/whisper-large-v3-mlx
```

Anything else fails with `Model path '...' is not within allowed directories`. HuggingFace repo names (`mlx-community/whisper-turbo`) are downloaded to the cache and are not affected.

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

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

```bash
# Development setup
git clone https://github.com/CodeWithBehnam/vayu.git
cd vayu
pip install -e ".[dev]"
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

This project builds on the excellent work of:

| Project | Author(s) | Contribution |
|---------|-----------|--------------|
| [mlx-examples/whisper](https://github.com/ml-explore/mlx-examples) | Apple Inc. | MLX framework, Whisper port, CLI, output writers |
| [lightning-whisper-mlx](https://github.com/mustafaaljadery/lightning-whisper-mlx) | Mustafa Aljadery, Siddharth Sharma | Batched decoding for 3-5x speedup |
| [Whisper](https://github.com/openai/whisper) | OpenAI | Original model architecture and weights |

## Star History

<a href="https://star-history.com/#CodeWithBehnam/vayu&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=CodeWithBehnam/vayu&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=CodeWithBehnam/vayu&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=CodeWithBehnam/vayu&type=Date" width="100%" />
 </picture>
</a>

## Author

**Behnam Ebrahimi** - Unified implementation, security improvements, and maintenance
