---
name: Bug Report
about: Report a bug or unexpected behavior
title: ""
labels: bug
assignees: ""
---

**Describe the bug**
A clear description of what the bug is.

**To reproduce**
```python
# Minimal code to reproduce
from whisper_mlx import LightningWhisperMLX

whisper = LightningWhisperMLX(model="distil-large-v3", batch_size=12)
result = whisper.transcribe("audio.mp3")
```

**Expected behavior**
What you expected to happen.

**Error output**
```
Paste the full error/traceback here
```

**Environment**
- macOS version:
- Chip (M1/M2/M3/M4):
- Python version:
- Vayu version (`pip show vayu-whisper`):
- MLX version:

**Audio file details** (if relevant)
- Format (mp3/wav/m4a):
- Duration:
- Language:
