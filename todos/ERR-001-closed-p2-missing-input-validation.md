---
status: closed
priority: p2
issue_id: ERR-001
tags: [error-handling, validation, robustness]
dependencies: []
resolution: fixed
---

# Missing Input Validation Across Functions

## Problem Statement

Many public functions lack proper input validation, leading to cryptic errors when invalid inputs are provided. Users get unhelpful stack traces instead of clear error messages.

## Findings

### 1. load_model() - No path validation
**Location:** `whisper_mlx/load_models.py:14-49`
```python
def load_model(path_or_hf_repo: str, dtype: mx.Dtype = mx.float16):
    # No validation of path_or_hf_repo
    # Empty string, None, or invalid paths cause confusing errors
```

### 2. transcribe() - No audio validation
**Location:** `whisper_mlx/transcribe.py:62`
```python
def transcribe(audio, ...):
    # No validation that audio is valid file path or array
    # Fails deep in ffmpeg with cryptic error
```

### 3. LightningWhisperMLX - No model name validation
**Location:** `whisper_mlx/lightning.py:59`
```python
def __init__(self, model: str = "tiny", ...):
    # Invalid model name causes KeyError deep in code
```

### 4. Batch size validation
**Location:** Various
```python
# batch_size=0 or negative causes index errors
# No upper bound check for memory safety
```

## Proposed Solutions

### Fix 1: Add validation decorator
```python
# In whisper_mlx/validation.py (new file)
from functools import wraps

def validate_audio_input(func):
    @wraps(func)
    def wrapper(audio, *args, **kwargs):
        if isinstance(audio, str):
            if not audio:
                raise ValueError("Audio path cannot be empty")
            if not Path(audio).exists():
                raise FileNotFoundError(f"Audio file not found: {audio}")
        elif isinstance(audio, np.ndarray):
            if audio.size == 0:
                raise ValueError("Audio array cannot be empty")
        else:
            raise TypeError(f"Expected str or ndarray, got {type(audio)}")
        return func(audio, *args, **kwargs)
    return wrapper
```

### Fix 2: Validate at public API boundaries
```python
def transcribe(
    audio: str | np.ndarray,
    path_or_hf_repo: str = "mlx-community/whisper-turbo",
    batch_size: int = 12,
    ...
) -> dict:
    # Validate inputs upfront
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    if batch_size > 64:
        raise ValueError(f"batch_size {batch_size} may cause OOM, max is 64")

    if isinstance(audio, str) and not Path(audio).exists():
        raise FileNotFoundError(f"Audio file not found: {audio}")
```

### Fix 3: Add model name validation
```python
VALID_MODELS = {"tiny", "base", "small", "medium", "large-v3", "turbo", ...}

def validate_model_name(name: str) -> str:
    if name in VALID_MODELS or name.startswith("mlx-community/"):
        return name
    raise ValueError(
        f"Unknown model: {name}. "
        f"Valid options: {', '.join(sorted(VALID_MODELS))}"
    )
```

## Verification Steps

1. Add validation to each public function
2. Write tests for invalid inputs
3. Verify error messages are helpful
4. Test edge cases (empty strings, None, negative numbers)
