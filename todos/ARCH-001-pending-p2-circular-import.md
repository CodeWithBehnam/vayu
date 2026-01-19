---
status: pending
priority: p2
issue_id: ARCH-001
tags: [architecture, circular-import, refactoring]
dependencies: []
---

# Circular Import Between whisper.py and decoding.py

## Problem Statement

There's a circular import dependency between `whisper.py` and `decoding.py`. The `Whisper` class imports decoding functions, while decoding imports from whisper. This makes the codebase fragile and can cause import errors.

## Findings

**Location:**
- `whisper_mlx/whisper.py:13-14` imports from decoding
- `whisper_mlx/decoding.py:265-266` imports Whisper

**Current Structure:**
```
whisper.py ──imports──> decoding.py
    ^                       │
    └───────imports─────────┘
```

**Impact:**
- Import order matters (fragile)
- IDE autocompletion may fail
- Makes testing individual modules harder

## Proposed Solutions

### Option 1: Extract shared types to separate module
```
whisper_mlx/
├── types.py          # NEW: DecodingOptions, DecodingResult, ModelDimensions
├── whisper.py        # Model classes, imports types.py
├── decoding.py       # Decoding logic, imports types.py
└── __init__.py       # Public API
```

### Option 2: Use TYPE_CHECKING for type hints only
```python
# In decoding.py
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .whisper import Whisper

def decode(model: "Whisper", ...) -> DecodingResult:
    ...
```

### Option 3: Dependency injection
```python
# In decoding.py - don't import Whisper, use Protocol
from typing import Protocol

class WhisperModel(Protocol):
    def encoder(self, mel: mx.array) -> mx.array: ...
    def decoder(self, tokens: mx.array, audio: mx.array) -> mx.array: ...

def decode(model: WhisperModel, ...) -> DecodingResult:
    ...
```

## Verification Steps

1. Apply the refactoring
2. Run: `python -c "from whisper_mlx import transcribe, Whisper, decode"`
3. Verify all imports work in fresh Python session
4. Run full test suite
