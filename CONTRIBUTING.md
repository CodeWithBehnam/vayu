# Contributing to Vayu

Thanks for your interest in contributing!

## Setup

```bash
git clone https://github.com/CodeWithBehnam/vayu.git
cd vayu
pip install -e ".[dev]"      # or: uv sync --extra dev
```

The test suite also runs on Linux with MLX's CPU build: `pip install "mlx[cpu]"`.

## Making changes

1. Fork the repo and create a branch from `main`
2. Make your changes
3. Run tests: `pytest` (CI runs them on every pull request)
4. Submit a pull request

## Reporting bugs

Open an issue using the **Bug Report** template. Include your macOS version, chip (M1/M2/M3/M4), Python version, and a minimal code example to reproduce the problem.

## Feature requests

Open an issue using the **Feature Request** template. Describe the problem you're solving and your proposed API.
