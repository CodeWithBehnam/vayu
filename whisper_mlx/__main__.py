# Copyright © 2023 Apple Inc.

"""
Entry point for running whisper_mlx as a module.

Usage:
    python -m whisper_mlx audio.mp3 --model mlx-community/whisper-turbo --batch-size 6
"""

from .cli import main

if __name__ == "__main__":
    main()
