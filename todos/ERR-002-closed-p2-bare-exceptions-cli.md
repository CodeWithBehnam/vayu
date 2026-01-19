---
status: closed
priority: p2
issue_id: ERR-002
tags: [error-handling, cli, logging]
dependencies: []
resolution: fixed
---

# Bare Exceptions in CLI and Error Suppression

## Problem Statement

The CLI and other modules catch broad `Exception` types, potentially hiding bugs and making debugging difficult. Some errors are silently swallowed.

## Findings

### 1. CLI catches generic Exception
**Location:** `whisper_mlx/cli.py:200-210`
```python
try:
    result = transcribe(audio_path, ...)
except Exception as e:
    print(f"Error processing {audio_path}: {e}")
    continue  # Silently continues to next file
```

### 2. Silent failures in model loading
**Location:** `whisper_mlx/load_models.py`
```python
# If config.json is malformed, error may be cryptic
with open(config_path) as f:
    config = json.load(f)  # JSONDecodeError not caught
```

### 3. Audio loading swallows FFmpeg errors
**Location:** `whisper_mlx/audio.py:44-70`
```python
process = subprocess.run(cmd, ...)
# FFmpeg errors may not be properly surfaced
```

## Proposed Solutions

### Fix 1: Catch specific exceptions with proper logging
```python
import logging

logger = logging.getLogger(__name__)

def main():
    for audio_path in args.audio:
        try:
            result = transcribe(audio_path, ...)
        except FileNotFoundError:
            logger.error(f"File not found: {audio_path}")
            continue
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg failed for {audio_path}: {e.stderr}")
            continue
        except MemoryError:
            logger.error(f"Out of memory processing {audio_path}")
            raise  # Re-raise - can't recover
        except Exception:
            logger.exception(f"Unexpected error processing {audio_path}")
            if not args.continue_on_error:
                raise
```

### Fix 2: Add --strict flag to CLI
```python
parser.add_argument(
    "--strict",
    action="store_true",
    help="Exit on first error instead of continuing"
)

# In processing loop
if args.strict:
    raise
```

### Fix 3: Structured error reporting
```python
@dataclass
class TranscriptionError:
    file: str
    error_type: str
    message: str
    traceback: str | None = None

errors: list[TranscriptionError] = []

# After processing all files
if errors:
    print(f"\n{len(errors)} file(s) failed:")
    for err in errors:
        print(f"  - {err.file}: {err.error_type}: {err.message}")
```

## Verification Steps

1. Replace bare exceptions with specific ones
2. Add logging configuration
3. Test error scenarios:
   - Invalid audio file
   - Missing model
   - Out of memory
4. Verify errors are properly reported
