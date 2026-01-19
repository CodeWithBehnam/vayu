---
status: pending
priority: p3
issue_id: QUAL-001
tags: [code-quality, type-hints, maintainability]
dependencies: []
---

# Missing Type Hints on Function Parameters

## Problem Statement

Many functions lack complete type hints, making the code harder to understand and preventing IDE autocompletion and static analysis tools from catching bugs.

## Findings

### Functions Missing Type Hints

**whisper_mlx/transcribe.py:**
```python
# Line 62 - Missing return type
def transcribe(audio, path_or_hf_repo=..., ...):  # -> dict missing

# Various internal functions
def decode_with_fallback(segment, ...):  # No type hints
```

**whisper_mlx/lightning.py:**
```python
# Line 59 - quant should be Optional[str]
def __init__(self, model: str = "tiny", quant: str = None):
    # quant: str = None is incorrect, should be Optional[str] = None
```

**whisper_mlx/audio.py:**
```python
# Line 44 - Return type missing
def load_audio(file, sr=SAMPLE_RATE, from_stdin=False):
    # Should be: def load_audio(file: str, sr: int = ..., ...) -> np.ndarray:
```

**whisper_mlx/decoding.py:**
```python
# Various callback functions lack types
def _get_suppress_tokens(...):  # No type hints
```

## Proposed Solutions

### Fix: Add comprehensive type hints
```python
# transcribe.py
from typing import Literal, TypedDict

class TranscriptionResult(TypedDict):
    text: str
    segments: list[dict]
    language: str

def transcribe(
    audio: str | np.ndarray,
    path_or_hf_repo: str = "mlx-community/whisper-turbo",
    batch_size: int = 12,
    dtype: mx.Dtype = mx.float16,
    verbose: bool | None = None,
    language: str | None = None,
    task: Literal["transcribe", "translate"] = "transcribe",
    word_timestamps: bool = False,
    **kwargs,
) -> TranscriptionResult:
    ...

# lightning.py
from typing import Optional

class LightningWhisperMLX:
    def __init__(
        self,
        model: str = "tiny",
        batch_size: int = 12,
        quant: Optional[str] = None,  # Fixed!
    ) -> None:
        ...

    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
    ) -> dict[str, Any]:
        ...
```

### Add py.typed marker
```bash
# Create marker file for PEP 561
touch whisper_mlx/py.typed
```

### Run mypy for validation
```bash
uv run mypy whisper_mlx/ --strict
```

## Verification Steps

1. Add type hints to all public functions
2. Add py.typed marker
3. Run mypy in strict mode
4. Fix any type errors
5. Update documentation with types
