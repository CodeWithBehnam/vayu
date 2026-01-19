---
status: pending
priority: p1
issue_id: SEC-002
tags: [security, high-severity, path-traversal]
dependencies: []
---

# Path Traversal in Model Loading

## Problem Statement

The `load_model()` function in `load_models.py:14-49` accepts arbitrary paths without validation, allowing potential path traversal attacks. An attacker could potentially load malicious model files from unexpected locations.

## Findings

**Location:** `whisper_mlx/load_models.py:14-49`

**Vulnerable Code:**
```python
def load_model(
    path_or_hf_repo: str,
    dtype: mx.Dtype = mx.float16,
) -> Whisper:
    model_path = Path(path_or_hf_repo)
    if not model_path.exists():
        # Downloads from HuggingFace - OK
        model_path = Path(
            snapshot_download(
                repo_id=path_or_hf_repo,
                ...
            )
        )
    # No validation of the path before loading
```

**Attack Vector:**
- User provides `../../malicious/model` which could load untrusted pickle files
- Pickle deserialization can execute arbitrary code

## Proposed Solutions

### Option 1: Validate against allowed directories
```python
ALLOWED_MODEL_DIRS = [
    Path.home() / ".cache" / "huggingface",
    Path("/usr/local/share/whisper-mlx"),
]

def load_model(path_or_hf_repo: str, dtype: mx.Dtype = mx.float16) -> Whisper:
    model_path = Path(path_or_hf_repo).resolve()

    # Check if path is within allowed directories
    if model_path.exists():
        if not any(model_path.is_relative_to(allowed) for allowed in ALLOWED_MODEL_DIRS):
            raise ValueError(f"Model path not in allowed directories: {model_path}")
```

### Option 2: Restrict to HuggingFace repos only
```python
def load_model(repo_id: str, dtype: mx.Dtype = mx.float16) -> Whisper:
    """Load model from HuggingFace Hub only."""
    if not repo_id.startswith("mlx-community/"):
        raise ValueError("Only mlx-community models are supported")

    model_path = Path(snapshot_download(repo_id=repo_id, ...))
```

### Option 3: Add file integrity checks
```python
def load_model(path_or_hf_repo: str, dtype: mx.Dtype = mx.float16) -> Whisper:
    # ... existing code ...

    # Verify model files before loading
    expected_files = ["weights.npz", "config.json"]
    for f in expected_files:
        if not (model_path / f).exists():
            raise ValueError(f"Invalid model directory: missing {f}")
```

## Verification Steps

1. Apply the fix
2. Test with path traversal attempts: `../../../etc/passwd`
3. Test with symlinks pointing outside allowed dirs
4. Verify legitimate model loading still works
