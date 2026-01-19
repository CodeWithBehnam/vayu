#!/usr/bin/env python3
"""
Download required assets for whisper_mlx.

This script downloads the mel_filters.npz and tiktoken vocabulary files
required for Whisper transcription.
"""

import os
import urllib.request
from pathlib import Path


def download_file(url: str, dest: Path) -> None:
    """Download a file from URL to destination."""
    print(f"Downloading {dest.name}...")
    urllib.request.urlretrieve(url, dest)
    print(f"  Saved to {dest}")


def main():
    assets_dir = Path(__file__).parent

    # URLs for assets from OpenAI Whisper repository
    base_url = "https://raw.githubusercontent.com/openai/whisper/main/whisper/assets"

    files = [
        "mel_filters.npz",
        "gpt2.tiktoken",
        "multilingual.tiktoken",
    ]

    for filename in files:
        dest = assets_dir / filename
        if dest.exists():
            print(f"{filename} already exists, skipping.")
            continue

        url = f"{base_url}/{filename}"
        try:
            download_file(url, dest)
        except Exception as e:
            print(f"Failed to download {filename}: {e}")
            print(f"Please manually download from: {url}")

    print("\nAssets download complete!")
    print("You can now use whisper_mlx.")


if __name__ == "__main__":
    main()
