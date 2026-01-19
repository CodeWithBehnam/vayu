---
status: completed
priority: p1
issue_id: SEC-001
tags: [security, high-severity, command-injection]
dependencies: []
---

# Command Injection via FFmpeg File Path

## Problem Statement

The `load_audio()` function in `audio.py:44-59` passes user-supplied file paths directly to FFmpeg via subprocess without proper sanitization. This creates a command injection vulnerability where malicious filenames could execute arbitrary shell commands.

## Findings

**Location:** `whisper_mlx/audio.py:44-59`

**Vulnerable Code:**
```python
cmd = [
    "ffmpeg",
    "-nostdin",
    "-threads", "0",
    "-i", file,  # <-- Unsanitized user input
    "-f", "s16le",
    "-ac", "1",
    "-acodec", "pcm_s16le",
    "-ar", str(sr),
    "-",
]
```

**Attack Vector:** A crafted filename like `"; rm -rf /; echo "` could inject shell commands when passed through the subprocess.

## Proposed Solutions

### Option 1: Use shlex.quote() (Recommended)
```python
import shlex

cmd = [
    "ffmpeg",
    "-nostdin",
    "-threads", "0",
    "-i", shlex.quote(file),  # Properly escaped
    ...
]
```

### Option 2: Validate file path
```python
import os

def load_audio(file: str, sr: int = SAMPLE_RATE):
    # Validate file exists and is a regular file
    if not os.path.isfile(file):
        raise ValueError(f"Invalid audio file path: {file}")

    # Normalize path to prevent traversal
    file = os.path.realpath(file)
    ...
```

### Option 3: Use subprocess with shell=False (already done, but needs path validation)
The code already uses `shell=False`, but the file path should still be validated to prevent path traversal attacks.

## Verification Steps

1. Apply the fix
2. Test with malicious filenames: `test"; echo "pwned`
3. Test with path traversal: `../../etc/passwd`
4. Run existing audio loading tests
