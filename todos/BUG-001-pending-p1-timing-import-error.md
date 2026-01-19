---
status: completed
priority: p1
issue_id: BUG-001
tags: [bug, import-error, critical]
dependencies: []
---

# Import Error in timing.py

## Problem Statement

The `timing.py` module imports from `.model` which doesn't exist. This will cause an ImportError when the timing module is used for word-level timestamps.

## Findings

**Location:** `whisper_mlx/timing.py:16`

**Broken Code:**
```python
from .model import Whisper  # Module .model doesn't exist!
```

**Actual Module:** The `Whisper` class is defined in `whisper_mlx/whisper.py`

## Proposed Solutions

### Fix (Simple)
```python
# Change line 16 from:
from .model import Whisper

# To:
from .whisper import Whisper
```

## Verification Steps

1. Apply the fix
2. Run: `python -c "from whisper_mlx.timing import find_alignment"`
3. Test word-level timestamp generation on sample audio
4. Run any existing timing tests
