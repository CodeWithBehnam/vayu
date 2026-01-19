---
status: pending
priority: p3
issue_id: QUAL-002
tags: [code-quality, magic-numbers, readability]
dependencies: []
---

# Magic Numbers Throughout Codebase

## Problem Statement

The codebase contains numerous magic numbers without explanation, making the code harder to understand and maintain. Changes to these values require finding all occurrences.

## Findings

### Examples of Magic Numbers

**whisper_mlx/decoding.py:**
```python
# Line 180 - What is 224?
max_tokens = 224  # Max tokens per segment (but why 224?)

# Line 352 - Temperature values
temperatures = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]  # Why these specific values?
```

**whisper_mlx/transcribe.py:**
```python
# Line 340 - Compression threshold
if compression_ratio > 2.4:  # What does 2.4 mean?

# Line 345 - Log probability threshold
if avg_logprob < -1.0:  # Why -1.0?
```

**whisper_mlx/speculative.py:**
```python
# Line 54 - Draft tokens
num_draft_tokens: int = 5  # Why 5? What's the tradeoff?
```

**whisper_mlx/timing.py:**
```python
# Line 25 - Frame duration
frame_duration = 0.02  # 20ms - should be derived from constants
```

## Proposed Solutions

### Fix 1: Create constants module
```python
# whisper_mlx/constants.py

# Tokenization limits
MAX_TOKENS_PER_SEGMENT = 224  # Whisper's context window for text
MAX_INITIAL_TIMESTAMP = 1.0   # Seconds

# Quality thresholds (empirically determined)
COMPRESSION_RATIO_THRESHOLD = 2.4  # Higher = likely hallucination
AVG_LOGPROB_THRESHOLD = -1.0       # Lower = low confidence
NO_SPEECH_THRESHOLD = 0.6          # Probability threshold

# Temperature schedule for decoding fallback
TEMPERATURE_SCHEDULE = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)

# Speculative decoding
DEFAULT_DRAFT_TOKENS = 5  # Balance between draft overhead and verification
```

### Fix 2: Add explanatory comments
```python
# If compression ratio exceeds threshold, the model is likely repeating itself
# (hallucination). A ratio of 2.4 means the text is 2.4x more compressible
# than expected, indicating repetition.
COMPRESSION_RATIO_THRESHOLD = 2.4
```

### Fix 3: Derive values from other constants
```python
# In audio.py
FRAME_DURATION = HOP_LENGTH / SAMPLE_RATE  # 0.01 seconds (10ms)
FRAMES_PER_SECOND = SAMPLE_RATE // HOP_LENGTH  # 100 frames/sec
```

## Verification Steps

1. Identify all magic numbers
2. Create constants.py with documented values
3. Replace magic numbers with named constants
4. Ensure no semantic changes (same values)
5. Run tests to verify behavior unchanged
