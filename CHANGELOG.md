# Changelog

All notable changes to this project are documented here. The release workflow
publishes the section for each version as its GitHub release notes.

## [1.1.0] - 2026-10-01

### Fixed

- Batched decoding (`batch_size > 1`) no longer drops text at the end of each
  30-second window, and every segment records its own window's `seek` (#3).
- `word_timestamps=True` now works with batched decoding, the default for
  `LightningWhisperMLX` (#4).
- Batched temperature fallback follows the `temperature` schedule and
  re-checks each step; a single temperature means no fallback (#5).
- The rule that stops timestamps going backwards during decoding now applies
  (#6).
- CLI: each input file gets its own output name instead of every file
  overwriting the first one's output (#7).
- CLI: missing files, a missing ffmpeg and undecodable audio are reported
  clearly and listed in the failure summary (#7).
- Speculative decoding runs, uses a key/value cache and per-model
  spectrograms, and gives exactly the target model's greedy output (#8).

### Added

- `AudioLoadError`, raised when ffmpeg is missing or cannot decode the audio.
- `scripts/benchmark.py` to measure the batched speed-up on your own Mac (#11).
- README sections on loading local models and on batched vs sequential
  decoding (#11).
- CI that runs the test suite on Linux and Apple Silicon (#2).

### Changed

These may need a change in code that calls Vayu:

- `load_audio()` raises `FileNotFoundError` for a missing file (previously
  `ValueError`), and `transcribe()` checks the path before loading the model.
- `quant` raises `ValueError` when it can't be applied (an unknown level, a
  model without a quantised build, or a repo path) instead of silently loading
  full precision (#9).
- `beam_size` and `patience` raise `NotImplementedError` up front; beam search
  is not implemented. The CLI's `--patience` option is removed (#10).
- `hallucination_silence_threshold` with `batch_size > 1` now warns that it is
  ignored.
- Speculative decoding's default pair is distil-large-v3 → large-v3; draft and
  target models must share a vocabulary (#8).
- Decoded timestamps can differ slightly from 1.0.0 because the timestamp rule
  now applies (#6).
- Development dependencies live in the `dev` extra only: `pip install -e
  ".[dev]"` or `uv sync --extra dev` (#2).

### Deprecated

- `parallel_chunk_transcribe`: it never ran chunks in parallel. Use
  `transcribe(audio, batch_size=N)` (#8).

## [1.0.0] - 2026-01-19

First release on PyPI as `vayu-whisper`.

[1.1.0]: https://github.com/CodeWithBehnam/vayu/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/CodeWithBehnam/vayu/releases/tag/v1.0.0
