---
status: pending
priority: p2
issue_id: ARCH-003
tags: [architecture, dry, code-duplication]
dependencies: []
---

# Duplicate Code Across Modules

## Problem Statement

Several pieces of code are duplicated across modules, violating DRY (Don't Repeat Yourself) principles. This makes maintenance harder and risks inconsistencies when one copy is updated but not others.

## Findings

### 1. Timestamp Formatting (Duplicated)
**Locations:**
- `whisper_mlx/transcribe.py` - `_format_timestamp()`
- `whisper_mlx/writers.py` - `format_timestamp()`

### 2. MODEL_REPOS Mapping (Duplicated)
**Locations:**
- `whisper_mlx/lightning.py:20-35` - MODEL_REPOS dict
- `whisper_mlx/speculative.py:217-225` - model_map dict

### 3. Audio Constants (Potentially Scattered)
**Locations:**
- `whisper_mlx/audio.py` - SAMPLE_RATE, HOP_LENGTH, etc.
- Various files import and re-define these

### 4. Model Holder / Caching Logic
**Locations:**
- `whisper_mlx/transcribe.py:50-59` - ModelHolder class
- Should be in `load_models.py` with the model loading logic

## Proposed Solutions

### Fix 1: Centralize timestamp formatting
```python
# In whisper_mlx/utils.py (new file)
def format_timestamp(seconds: float, include_ms: bool = True) -> str:
    """Format seconds as HH:MM:SS,mmm or HH:MM:SS"""
    ...

# Import everywhere else
from .utils import format_timestamp
```

### Fix 2: Centralize model mappings
```python
# In whisper_mlx/models.py (new file)
MODEL_REPOS = {
    "tiny": "mlx-community/whisper-tiny-mlx",
    "base": "mlx-community/whisper-base-mlx",
    ...
}

QUANT_REPOS = {
    "tiny-8bit": "mlx-community/whisper-tiny-mlx-8bit",
    ...
}

def resolve_model_path(name_or_path: str, quantized: bool = False) -> str:
    """Resolve friendly name to HuggingFace repo path."""
    repos = QUANT_REPOS if quantized else MODEL_REPOS
    return repos.get(name_or_path, name_or_path)
```

### Fix 3: Move ModelHolder to load_models.py
```python
# In whisper_mlx/load_models.py
class ModelCache:
    """Singleton cache for loaded models."""
    _instance = None
    _models: dict[str, Whisper] = {}

    @classmethod
    def get(cls, path: str, dtype: mx.Dtype) -> Whisper:
        key = f"{path}:{dtype}"
        if key not in cls._models:
            cls._models[key] = load_model(path, dtype)
        return cls._models[key]
```

## Verification Steps

1. Create centralized utilities
2. Update all imports
3. Remove duplicate code
4. Run full test suite to verify behavior unchanged
