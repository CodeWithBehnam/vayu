---
status: closed
priority: p2
issue_id: ARCH-002
tags: [architecture, refactoring, code-organization]
dependencies: []
resolution: deferred
---

# Monolithic transcribe() Function (670+ Lines)

## Problem Statement

The `transcribe()` function in `transcribe.py` is over 670 lines long, making it difficult to understand, test, and maintain. It handles too many responsibilities in a single function.

## Findings

**Location:** `whisper_mlx/transcribe.py:62-732`

**Current Responsibilities (too many):**
1. Audio loading and preprocessing
2. Model loading and caching
3. Language detection
4. Segment splitting and batching
5. Decoding with fallback strategies
6. Timestamp generation
7. Word-level alignment
8. Result formatting

**Impact:**
- Hard to test individual components
- Difficult to understand flow
- High cognitive load for contributors
- Changes risk breaking unrelated functionality

## Proposed Solutions

### Option 1: Extract into focused modules
```
whisper_mlx/
├── transcribe.py        # Main orchestration (~100 lines)
├── preprocessing.py     # Audio loading, mel conversion
├── segmentation.py      # Chunk splitting, batching
├── decode_strategy.py   # Fallback logic, temperature handling
├── postprocessing.py    # Timestamp formatting, word alignment
└── model_cache.py       # Model loading and caching
```

### Option 2: Class-based pipeline
```python
class TranscriptionPipeline:
    """Orchestrates the transcription process."""

    def __init__(self, model_path: str, **options):
        self.preprocessor = AudioPreprocessor()
        self.decoder = BatchDecoder(model_path)
        self.postprocessor = ResultFormatter()

    def transcribe(self, audio: str | np.ndarray) -> dict:
        mel = self.preprocessor.process(audio)
        segments = self.decoder.decode(mel)
        return self.postprocessor.format(segments)
```

### Option 3: Functional decomposition with clear interfaces
```python
def transcribe(audio, **options):
    # Step 1: Preprocess
    mel, audio_info = preprocess_audio(audio)

    # Step 2: Detect language
    language = detect_or_use_provided(mel, options)

    # Step 3: Decode segments
    raw_segments = decode_segments(mel, language, options)

    # Step 4: Post-process
    return format_results(raw_segments, audio_info, options)
```

## Verification Steps

1. Extract one component at a time
2. Add unit tests for each extracted component
3. Integration test after each extraction
4. Verify output matches original implementation

## Assessment (2026-01-19)

**Scope Analysis:**
- File is 723 lines, with `transcribe()` spanning lines 34-723
- Contains 3 nested helper functions with tight coupling to local state:
  - `decode_with_fallback()` - needs model, decode_options, thresholds
  - `decode_batch_with_fallback()` - same dependencies
  - `new_segment()` - needs tokenizer, seek variable
- Main loop has complex state management (seek, tokens, segments)

**Challenges:**
1. Nested functions access closure variables (model, tokenizer, decode_options)
2. Extracting would require passing 5+ parameters to each helper
3. No existing test suite to validate behavior preservation
4. Changes could introduce subtle bugs in timestamp/segment handling

**Recommendation:**
- Deferred for dedicated refactoring effort with:
  1. Create comprehensive test suite first
  2. Extract one component at a time
  3. Consider class-based TranscriptionPipeline approach
  4. Plan for 2-3 days of focused work

**Resolution:** Deferred - requires dedicated planning, tests, and implementation time.
