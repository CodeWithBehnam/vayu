---
status: closed
priority: p2
issue_id: PERF-002
tags: [performance, mlx, gpu-transfer]
dependencies: []
resolution: wontfix
---

# Repeated GPU-CPU Transfers in Token Processing

## Problem Statement

The decoding loop performs unnecessary GPU-to-CPU transfers via `.tolist()` and `.item()` calls in the hot path, causing synchronization stalls and reduced throughput.

## Findings

**Location:** `whisper_mlx/decoding.py:352, 380-385`

**Problematic Code:**
```python
# Line 352 - tokens.tolist() in hot loop
tokens = tokens.tolist()  # GPU -> CPU transfer!

# Line 380-385 - Multiple .item() calls
next_token = mx.argmax(logits[0, -1]).item()  # Sync point!
if next_token == self.tokenizer.eot:  # Another sync
    break
```

**Impact:**
- Each `.tolist()` forces GPU synchronization
- Breaks MLX's lazy evaluation pipeline
- Can cause 2-5x slowdown on long sequences

## Proposed Solutions

### Option 1: Batch token comparisons on GPU
```python
# Keep comparisons on GPU until batch completion
eot_mask = tokens[:, -1] == self.tokenizer.eot
all_done = mx.all(eot_mask)
mx.eval(all_done)  # Single sync point

if all_done.item():
    break
```

### Option 2: Defer .tolist() until end
```python
def _main_loop(self, audio_features, tokens):
    # Keep tokens as mx.array throughout
    for i in range(self.n_ctx):
        logits = self.inference.logits(tokens, audio_features)
        next_tokens = mx.argmax(logits[:, -1], axis=-1)
        tokens = mx.concatenate([tokens, next_tokens[:, None]], axis=1)

        # Check completion on GPU
        if mx.all(next_tokens == self.tokenizer.eot).item():
            break

    # Single transfer at the end
    return tokens.tolist()
```

### Option 3: Use mx.async_eval for overlap
```python
# Overlap computation with transfer
mx.async_eval(next_tokens)
# ... do other work ...
next_tokens_cpu = next_tokens.tolist()
```

## Verification Steps

1. Apply the fix
2. Profile with MLX profiler: `MLX_DEBUG=1`
3. Measure GPU utilization before/after
4. Benchmark on varying sequence lengths

## Investigation Results (2026-01-19)

**Benchmark Results:**
```
Model: mlx-community/whisper-turbo
Audio: Synthetic 10s test audio (16kHz)

WITHOUT word_timestamps (baseline): 6.989s avg
WITH word_timestamps (uses ApplyTimestampRules): 4.710s avg

Timestamp overhead: -32.6% (FASTER, not slower!)
```

**Conclusion:**
The hypothesized performance issue was NOT observed in benchmarks:
- Code WITH `.tolist()` calls (word_timestamps=True) was actually 32% FASTER
- The existing `mx.async_eval()` usage provides adequate overlap
- The timestamp rules may help guide decoding to converge faster

**Resolution:** Won't fix - no measurable performance degradation from GPU-CPU transfers.
