# Copyright © 2023 Apple Inc.

import argparse
import logging
import os
import pathlib
import sys
import warnings
from dataclasses import dataclass, field
from subprocess import CalledProcessError
from typing import List

from . import audio
from .transcribe import transcribe
from .writers import get_writer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class TranscriptionError:
    """Record of a failed transcription."""
    file: str
    error_type: str
    message: str


def str2bool(string):
    str2val = {"True": True, "False": False}
    if string in str2val:
        return str2val[string]
    raise ValueError(f"Expected one of {set(str2val.keys())}, got {string}")


def optional_int(string):
    return None if string == "None" else int(string)


def optional_float(string):
    return None if string == "None" else float(string)


def build_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "audio",
        nargs="+",
        type=str,
        help="Audio file(s) to transcribe or '-' for stdin",
    )
    parser.add_argument(
        "--model",
        default="mlx-community/whisper-turbo",
        type=str,
        help="The path to the Whisper model or a HuggingFace repo",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default=".",
        help="Directory to save the outputs",
    )
    parser.add_argument(
        "--output-format",
        "-f",
        type=str,
        default="all",
        choices=["txt", "vtt", "srt", "tsv", "json", "all"],
        help="Format of the output file",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default=None,
        help="Name of the output file, uses the audio file's name by default",
    )
    parser.add_argument(
        "--verbose",
        type=str2bool,
        default=True,
        help="Whether to print out progress and debug messages",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="transcribe",
        choices=["transcribe", "translate"],
        help="Whether to perform X->X speech recognition ('transcribe') or X->English translation ('translate')",
    )
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help="Language spoken in the audio, specify None to perform language detection",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of audio segments to process in parallel. Higher values use more memory but can significantly improve throughput.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0,
        help="Temperature to use for sampling",
    )
    parser.add_argument(
        "--best-of",
        type=optional_int,
        default=5,
        help="Number of candidates when sampling with non-zero temperature",
    )
    parser.add_argument(
        "--patience",
        type=float,
        default=None,
        help="Optional patience value to use in beam decoding, as in https://arxiv.org/abs/2204.05424, the default (1.0) is equivalent to conventional beam search",
    )
    parser.add_argument(
        "--length-penalty",
        type=float,
        default=None,
        help="Optional token length penalty coefficient (alpha) as in https://arxiv.org/abs/1609.08144, uses simple length normalization by default.",
    )
    parser.add_argument(
        "--suppress-tokens",
        type=str,
        default="-1",
        help="Comma-separated list of token ids to suppress during sampling; '-1' will suppress most special characters except common punctuations",
    )
    parser.add_argument(
        "--initial-prompt",
        type=str,
        default=None,
        help="Optional text to provide as a prompt for the first window.",
    )
    parser.add_argument(
        "--condition-on-previous-text",
        type=str2bool,
        default=True,
        help="If True, provide the previous output of the model as a prompt for the next window; disabling may make the text inconsistent across windows, but the model becomes less prone to getting stuck in a failure loop",
    )
    parser.add_argument(
        "--fp16",
        type=str2bool,
        default=True,
        help="Whether to perform inference in fp16",
    )
    parser.add_argument(
        "--compression-ratio-threshold",
        type=optional_float,
        default=2.4,
        help="if the gzip compression ratio is higher than this value, treat the decoding as failed",
    )
    parser.add_argument(
        "--logprob-threshold",
        type=optional_float,
        default=-1.0,
        help="If the average log probability is lower than this value, treat the decoding as failed",
    )
    parser.add_argument(
        "--no-speech-threshold",
        type=optional_float,
        default=0.6,
        help="If the probability of the token is higher than this value the decoding has failed due to `logprob_threshold`, consider the segment as silence",
    )
    parser.add_argument(
        "--word-timestamps",
        type=str2bool,
        default=False,
        help="Extract word-level timestamps and refine the results based on them",
    )
    parser.add_argument(
        "--prepend-punctuations",
        type=str,
        default="\"'\u201c\u00bf([{-",
        help="If --word-timestamps is True, merge these punctuation symbols with the next word",
    )
    parser.add_argument(
        "--append-punctuations",
        type=str,
        default="\"'.\u3002,\uff0c!\uff01?\uff1f:\uff1a\")\u300d]}\u3001",
        help="If --word-timestamps is True, merge these punctuation symbols with the previous word",
    )
    parser.add_argument(
        "--highlight-words",
        type=str2bool,
        default=False,
        help="(requires --word-timestamps True) underline each word as it is spoken in srt and vtt",
    )
    parser.add_argument(
        "--max-line-width",
        type=int,
        default=None,
        help="(requires --word-timestamps True) the maximum number of characters in a line before breaking the line",
    )
    parser.add_argument(
        "--max-line-count",
        type=int,
        default=None,
        help="(requires --word-timestamps True) the maximum number of lines in a segment",
    )
    parser.add_argument(
        "--max-words-per-line",
        type=int,
        default=None,
        help="(requires --word-timestamps True, no effect with --max-line-width) the maximum number of words in a segment",
    )
    parser.add_argument(
        "--hallucination-silence-threshold",
        type=optional_float,
        help="(requires --word-timestamps True) skip silent periods longer than this threshold (in seconds) when a possible hallucination is detected",
    )
    parser.add_argument(
        "--clip-timestamps",
        type=str,
        default="0",
        help="Comma-separated list start,end,start,end,... timestamps (in seconds) of clips to process, where the last end timestamp defaults to the end of the file",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit on first error instead of continuing to next file",
    )
    return parser


def main():
    parser = build_parser()
    args = vars(parser.parse_args())
    if args["verbose"] is True:
        print(f"Args: {args}")

    path_or_hf_repo: str = args.pop("model")
    output_dir: str = args.pop("output_dir")
    output_format: str = args.pop("output_format")
    output_name: str = args.pop("output_name")
    batch_size: int = args.pop("batch_size")
    strict: bool = args.pop("strict")
    os.makedirs(output_dir, exist_ok=True)

    writer = get_writer(output_format, output_dir)
    word_options = [
        "highlight_words",
        "max_line_count",
        "max_line_width",
        "max_words_per_line",
    ]
    writer_args = {arg: args.pop(arg) for arg in word_options}
    if not args["word_timestamps"]:
        for k, v in writer_args.items():
            if v:
                argop = k.replace("_", "-")
                parser.error(f"--{argop} requires --word-timestamps True")
    if writer_args["max_line_count"] and not writer_args["max_line_width"]:
        warnings.warn("--max-line-count has no effect without --max-line-width")
    if writer_args["max_words_per_line"] and writer_args["max_line_width"]:
        warnings.warn("--max-words-per-line has no effect with --max-line-width")

    errors: List[TranscriptionError] = []

    for audio_obj in args.pop("audio"):
        if audio_obj == "-":
            # receive the contents from stdin rather than read a file
            audio_obj = audio.load_audio(from_stdin=True)

            output_name = output_name or "content"
        else:
            output_name = output_name or pathlib.Path(audio_obj).stem
        try:
            result = transcribe(
                audio_obj,
                path_or_hf_repo=path_or_hf_repo,
                batch_size=batch_size,
                **args,
            )
            writer(result, output_name, **writer_args)
        except FileNotFoundError as e:
            logger.error(f"File not found: {audio_obj}")
            errors.append(TranscriptionError(audio_obj, "FileNotFoundError", str(e)))
            if strict:
                sys.exit(1)
        except CalledProcessError as e:
            # FFmpeg or other subprocess failures
            stderr_msg = e.stderr.decode() if e.stderr else str(e)
            logger.error(f"Audio processing failed for {audio_obj}: {stderr_msg}")
            errors.append(TranscriptionError(audio_obj, "CalledProcessError", stderr_msg))
            if strict:
                sys.exit(1)
        except ValueError as e:
            # Input validation errors
            logger.error(f"Invalid input for {audio_obj}: {e}")
            errors.append(TranscriptionError(audio_obj, "ValueError", str(e)))
            if strict:
                sys.exit(1)
        except MemoryError as e:
            # Out of memory - always fatal
            logger.error(f"Out of memory processing {audio_obj}. Try reducing --batch-size.")
            raise
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
            sys.exit(130)
        except Exception as e:
            # Catch-all for unexpected errors
            logger.exception(f"Unexpected error processing {audio_obj}")
            errors.append(TranscriptionError(audio_obj, type(e).__name__, str(e)))
            if strict:
                raise

    # Report summary of errors if any
    if errors:
        print(f"\n{len(errors)} file(s) failed:")
        for err in errors:
            print(f"  - {err.file}: {err.error_type}: {err.message}")
        sys.exit(1)


if __name__ == "__main__":
    main()
