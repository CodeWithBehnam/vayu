---
status: closed
priority: p3
issue_id: API-001
tags: [api-design, documentation, usability]
dependencies: []
resolution: implemented
---

# Dual API Confusion Between transcribe() and LightningWhisperMLX

## Problem Statement

The library exposes two main APIs (`transcribe()` function and `LightningWhisperMLX` class) with different interfaces and capabilities. This causes confusion about which to use and when.

## Findings

### API Differences

| Feature | transcribe() | LightningWhisperMLX |
|---------|-------------|---------------------|
| Model path | `path_or_hf_repo` | `model` (friendly name) |
| Batch size | `batch_size` | `batch_size` |
| Quantization | Not supported | `quant` parameter |
| Model caching | Manual | Automatic |
| Word timestamps | `word_timestamps` | Not exposed |

### Inconsistent Parameter Names
```python
# transcribe() uses:
transcribe(audio, path_or_hf_repo="mlx-community/whisper-turbo")

# LightningWhisperMLX uses:
LightningWhisperMLX(model="turbo")  # Different parameter name!
```

### Missing Features in Simple API
```python
# LightningWhisperMLX doesn't expose:
# - word_timestamps
# - condition_on_previous_text
# - compression_ratio_threshold
# - logprob_threshold
# - no_speech_threshold
```

## Proposed Solutions

### Option 1: Unify parameter names
```python
# Make both accept same parameters
def transcribe(
    audio,
    model: str = "turbo",  # Friendly name OR full path
    batch_size: int = 12,
    quantization: str | None = None,  # Renamed from 'quant'
    ...
)

class LightningWhisperMLX:
    def __init__(
        self,
        model: str = "turbo",
        batch_size: int = 12,
        quantization: str | None = None,
        ...
    )
```

### Option 2: Make LightningWhisperMLX a thin wrapper
```python
class LightningWhisperMLX:
    """Simple wrapper around transcribe() for common use cases."""

    def __init__(self, model: str = "tiny", batch_size: int = 12):
        self.model = model
        self.batch_size = batch_size

    def transcribe(self, audio: str, **kwargs) -> dict:
        return transcribe(
            audio,
            path_or_hf_repo=resolve_model_path(self.model),
            batch_size=self.batch_size,
            **kwargs  # Pass through all other options
        )
```

### Option 3: Deprecate LightningWhisperMLX
```python
import warnings

class LightningWhisperMLX:
    def __init__(self, ...):
        warnings.warn(
            "LightningWhisperMLX is deprecated. Use transcribe() directly.",
            DeprecationWarning
        )
```

## Verification Steps

1. Choose unification strategy
2. Update both APIs for consistency
3. Update documentation and examples
4. Add deprecation warnings if removing features
