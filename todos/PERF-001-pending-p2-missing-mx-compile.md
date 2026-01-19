---
status: pending
priority: p2
issue_id: PERF-001
tags: [performance, mlx, optimization]
dependencies: []
---

# Missing @mx.compile on Hot Decoding Loop

## Problem Statement

The decoding loop in `decoding.py` doesn't use MLX's `@mx.compile` decorator for JIT compilation. This causes repeated Python overhead on each forward pass, significantly reducing throughput.

## Findings

**Location:** `whisper_mlx/decoding.py:340-380`

**Current Implementation:**
```python
def _main_loop(self, audio_features: mx.array, tokens: mx.array) -> DecodingResult:
    # Hot loop without JIT compilation
    for i in range(self.n_ctx):
        logits = self.inference.logits(tokens, audio_features)
        # ... token selection ...
```

**Impact:**
- Each iteration incurs Python interpreter overhead
- Graph construction repeated every step
- Estimated 20-40% throughput loss

## Proposed Solutions

### Option 1: Add @mx.compile to inference methods
```python
class DecodingTask:
    def __init__(self, ...):
        # Compile the forward pass
        self._compiled_logits = mx.compile(self.inference.logits)

    def _main_loop(self, audio_features: mx.array, tokens: mx.array):
        for i in range(self.n_ctx):
            logits = self._compiled_logits(tokens, audio_features)
```

### Option 2: Compile entire decoding step
```python
@mx.compile
def _decode_step(model, tokens, audio_features):
    """Single compiled decoding step."""
    logits = model.decoder(tokens, audio_features)
    return logits[:, -1, :]
```

### Option 3: Use mx.compile with static shapes
```python
# Pre-compile for known batch sizes
_compiled_decode = {
    bs: mx.compile(lambda t, a: model.decoder(t, a), shapeless=False)
    for bs in [1, 4, 8, 12, 16]
}
```

## Verification Steps

1. Apply the fix
2. Benchmark before/after on 1-minute audio
3. Verify output quality is unchanged
4. Test with different batch sizes
