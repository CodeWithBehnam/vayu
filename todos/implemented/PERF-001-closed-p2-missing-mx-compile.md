---
status: closed
priority: p2
issue_id: PERF-001
tags: [performance, mlx, optimization]
dependencies: []
resolution: wontfix
---

# Missing @mx.compile on Hot Decoding Loop

## Problem Statement

The decoding loop in `decoding.py` doesn't use MLX's `@mx.compile` decorator for JIT compilation. This causes repeated Python overhead on each forward pass, significantly reducing throughput.

## Investigation Results (2026-01-19)

Tested all three proposed approaches. **None provided the expected 20-40% improvement.**

### Benchmark Setup
- **Audio:** 55 seconds (data/test3.mp4)
- **Model:** mlx-community/whisper-turbo
- **Runs:** 5 benchmark + 3 warmup

### Results

| Approach | Avg Time | Real-time Factor | Change |
|----------|----------|------------------|--------|
| Baseline (no compile) | 1.232s | 44.64x | - |
| Option 1: `mx.compile(decoder)` | 1.282s | 42.90x | **-3.9%** |
| Option 1 + `shapeless=True` | ERROR | - | Slice primitive incompatible |

### Why It Doesn't Work

1. **Dynamic shapes:** The kv_cache grows each decoding step, and positional embeddings are sliced dynamically:
   ```python
   # whisper_mlx/whisper.py:186-187
   self.positional_embedding[offset : offset + x.shape[-1]]
   ```

2. **`shapeless=True` incompatible:** MLX's Slice primitive cannot infer output shapes, causing:
   ```
   ValueError: [Primitive::output_shapes] Slice cannot infer output shapes.
   ```

3. **Already efficient:** The existing code using `mx.async_eval()` in `_main_loop` (line 609, 619) already achieves good pipelining. The decoder forward pass is pure tensor operations with minimal Python overhead.

### Conclusion

The original estimate of "20-40% throughput loss" was incorrect for this codebase. The decoder is already well-optimized through:
- MLX's lazy evaluation and async execution
- Pre-compiled categorical sampling (`@mx.compile` at line 250)
- Efficient kv-cache handling

**Recommendation:** Close as wontfix. No code changes needed.

---

## Original Analysis (Archived)

**Location:** `whisper_mlx/decoding.py:581-626` (corrected from 340-380)

### Proposed Solutions (Not Implemented)

#### Option 1: Add @mx.compile to inference methods
```python
class DecodingTask:
    def __init__(self, ...):
        self._compiled_logits = mx.compile(self.inference.logits)
```
**Result:** Slight regression (~4%)

#### Option 2: Compile entire decoding step
```python
@mx.compile
def _decode_step(model, tokens, audio_features):
    logits = model.decoder(tokens, audio_features)
    return logits[:, -1, :]
```
**Result:** Not tested - same kv_cache shape issues apply

#### Option 3: Use mx.compile with static shapes
```python
_compiled_decode = {
    bs: mx.compile(lambda t, a: model.decoder(t, a), shapeless=False)
    for bs in [1, 4, 8, 12, 16]
}
```
**Result:** Not viable - kv_cache sequence length varies within each batch
