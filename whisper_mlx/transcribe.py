# Copyright © 2023 Apple Inc.

import sys
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import mlx.core as mx
import numpy as np
import tqdm

from .audio import (
    FRAMES_PER_SECOND,
    HOP_LENGTH,
    N_FRAMES,
    N_SAMPLES,
    SAMPLE_RATE,
    log_mel_spectrogram,
    pad_or_trim,
)
from .constants import (
    COMPRESSION_RATIO_THRESHOLD,
    LOGPROB_THRESHOLD,
    NO_SPEECH_THRESHOLD,
    TEMPERATURE_SCHEDULE,
)
from .decoding import DecodingOptions, DecodingResult
from .load_models import ModelHolder
from .timing import add_word_timestamps
from .tokenizer import LANGUAGES, get_tokenizer
from .utils import format_timestamp


def _get_end(segments: List[dict]) -> Optional[float]:
    return next(
        (w["end"] for s in reversed(segments) for w in reversed(s["words"])),
        segments[-1]["end"] if segments else None,
    )


def transcribe(
    audio: Union[str, np.ndarray, mx.array],
    *,
    path_or_hf_repo: str = "mlx-community/whisper-turbo",
    batch_size: int = 1,
    verbose: Optional[bool] = None,
    temperature: Union[float, Tuple[float, ...]] = TEMPERATURE_SCHEDULE,
    compression_ratio_threshold: Optional[float] = COMPRESSION_RATIO_THRESHOLD,
    logprob_threshold: Optional[float] = LOGPROB_THRESHOLD,
    no_speech_threshold: Optional[float] = NO_SPEECH_THRESHOLD,
    condition_on_previous_text: bool = True,
    initial_prompt: Optional[str] = None,
    word_timestamps: bool = False,
    prepend_punctuations: str = "\"'\u201c\u00bf([{-",
    append_punctuations: str = "\"'.\u3002,\uff0c!\uff01?\uff1f:\uff1a\")\u300d]}\u3001",
    clip_timestamps: Union[str, List[float]] = "0",
    hallucination_silence_threshold: Optional[float] = None,
    **decode_options: Any,
) -> Dict[str, Any]:
    """
    Transcribe an audio file using Whisper

    Parameters
    ----------
    audio: Union[str, np.ndarray, mx.array]
        The path to the audio file to open, or the audio waveform

    path_or_hf_repo: str
        The localpath to the Whisper model or HF Hub repo with the MLX converted weights.

    batch_size: int
        Number of audio segments to process in parallel. Higher values use more memory
        but can significantly improve throughput. Default is 1 (sequential processing).

    verbose: bool
        Whether to display the text being decoded to the console. If True, displays all the details,
        If False, displays minimal details. If None, does not display anything

    temperature: Union[float, Tuple[float, ...]]
        Temperature for sampling. It can be a tuple of temperatures, which will be successively used
        upon failures according to either `compression_ratio_threshold` or `logprob_threshold`.

    compression_ratio_threshold: float
        If the gzip compression ratio is above this value, treat as failed

    logprob_threshold: float
        If the average log probability over sampled tokens is below this value, treat as failed

    no_speech_threshold: float
        If the no_speech probability is higher than this value AND the average log probability
        over sampled tokens is below `logprob_threshold`, consider the segment as silent

    condition_on_previous_text: bool
        if True, the previous output of the model is provided as a prompt for the next window;
        disabling may make the text inconsistent across windows, but the model becomes less prone to
        getting stuck in a failure loop, such as repetition looping or timestamps going out of sync.

    word_timestamps: bool
        Extract word-level timestamps using the cross-attention pattern and dynamic time warping,
        and include the timestamps for each word in each segment.

    prepend_punctuations: str
        If word_timestamps is True, merge these punctuation symbols with the next word

    append_punctuations: str
        If word_timestamps is True, merge these punctuation symbols with the previous word

    initial_prompt: Optional[str]
        Optional text to provide as a prompt for the first window. This can be used to provide, or
        "prompt-engineer" a context for transcription, e.g. custom vocabularies or proper nouns
        to make it more likely to predict those word correctly.

    decode_options: dict
        Keyword arguments to construct `DecodingOptions` instances

    clip_timestamps: Union[str, List[float]]
        Comma-separated list start,end,start,end,... timestamps (in seconds) of clips to process.
        The last end timestamp defaults to the end of the file.

    hallucination_silence_threshold: Optional[float]
        When word_timestamps is True, skip silent periods longer than this threshold (in seconds)
        when a possible hallucination is detected

    Returns
    -------
    A dictionary containing the resulting text ("text") and segment-level details ("segments"), and
    the spoken language ("language"), which is detected when `decode_options["language"]` is None.
    """
    # Validate inputs
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    if batch_size > 64:
        warnings.warn(
            f"batch_size={batch_size} may cause out-of-memory errors. "
            "Consider using batch_size <= 64."
        )

    if isinstance(audio, str):
        if not audio:
            raise ValueError("Audio path cannot be empty")
    elif isinstance(audio, (np.ndarray, mx.array)):
        if audio.size == 0:
            raise ValueError("Audio array cannot be empty")
    else:
        raise TypeError(
            f"audio must be a file path (str) or array, got {type(audio).__name__}"
        )

    dtype = mx.float16 if decode_options.get("fp16", True) else mx.float32
    model = ModelHolder.get_model(path_or_hf_repo, dtype)

    # Pad 30-seconds of silence to the input audio, for slicing
    mel = log_mel_spectrogram(audio, n_mels=model.dims.n_mels, padding=N_SAMPLES)
    content_frames = mel.shape[-2] - N_FRAMES
    content_duration = float(content_frames * HOP_LENGTH / SAMPLE_RATE)

    if verbose:
        system_encoding = sys.getdefaultencoding()
        if system_encoding != "utf-8":
            make_safe = lambda x: x.encode(system_encoding, errors="replace").decode(
                system_encoding
            )
        else:
            make_safe = lambda x: x

    if decode_options.get("language", None) is None:
        if not model.is_multilingual:
            decode_options["language"] = "en"
        else:
            if verbose:
                print(
                    "Detecting language using up to the first 30 seconds. "
                    "Use the `language` decoding option to specify the language"
                )
            mel_segment = pad_or_trim(mel, N_FRAMES, axis=-2).astype(dtype)
            _, probs = model.detect_language(mel_segment)
            decode_options["language"] = max(probs, key=probs.get)
            if verbose is not None:
                print(
                    f"Detected language: {LANGUAGES[decode_options['language']].title()}"
                )

    language: str = decode_options["language"]
    task: str = decode_options.get("task", "transcribe")
    tokenizer = get_tokenizer(
        model.is_multilingual,
        num_languages=model.num_languages,
        language=language,
        task=task,
    )

    if isinstance(clip_timestamps, str):
        clip_timestamps = [
            float(ts) for ts in (clip_timestamps.split(",") if clip_timestamps else [])
        ]
    seek_points: List[int] = [round(ts * FRAMES_PER_SECOND) for ts in clip_timestamps]
    if len(seek_points) == 0:
        seek_points.append(0)
    if len(seek_points) % 2 == 1:
        seek_points.append(content_frames)
    else:
        seek_points[-1] = min(content_frames, seek_points[-1])
    seek_clips: List[Tuple[int, int]] = list(zip(seek_points[::2], seek_points[1::2]))

    punctuation = "\"'\u201c\u00bf([{-\"'.\u3002,\uff0c!\uff01?\uff1f:\uff1a\")\u300d]}\u3001"

    if word_timestamps and task == "translate":
        warnings.warn("Word-level timestamps on translations may not be reliable.")

    def decode_with_fallback(segment: mx.array) -> DecodingResult:
        """Decode a single segment with temperature fallback."""
        temperatures = (
            [temperature] if isinstance(temperature, (int, float)) else temperature
        )
        decode_result = None

        for t in temperatures:
            kwargs = {**decode_options}
            if t > 0:
                # disable beam_size and patience when t > 0
                kwargs.pop("beam_size", None)
                kwargs.pop("patience", None)
            else:
                # disable best_of when t == 0
                kwargs.pop("best_of", None)

            options = DecodingOptions(**kwargs, temperature=t)
            decode_result = model.decode(segment, options)

            needs_fallback = False
            if (
                compression_ratio_threshold is not None
                and decode_result.compression_ratio > compression_ratio_threshold
            ):
                needs_fallback = True  # too repetitive
            if (
                logprob_threshold is not None
                and decode_result.avg_logprob < logprob_threshold
            ):
                needs_fallback = True  # average log probability is too low
            if (
                no_speech_threshold is not None
                and decode_result.no_speech_prob > no_speech_threshold
            ):
                needs_fallback = False  # silence
            if not needs_fallback:
                break

        return decode_result

    def decode_batch_with_fallback(segment_batch: mx.array) -> List[DecodingResult]:
        """Decode a batch of segments with per-segment temperature fallback.

        Optimized: Collects all segments needing fallback and batches them together
        instead of decoding each one individually (which destroys parallelism).
        """
        kwargs = {**decode_options}
        kwargs.pop("beam_size", None)
        kwargs.pop("patience", None)
        kwargs.pop("best_of", None)

        # First pass: decode all segments at temperature 0
        options = DecodingOptions(**kwargs, temperature=0.0)
        decode_results = model.decode(segment_batch, options)

        # Collect indices of segments needing fallback (instead of processing individually)
        fallback_indices = []
        for i, decode_result in enumerate(decode_results):
            needs_fallback = False
            if (
                compression_ratio_threshold is not None
                and decode_result.compression_ratio > compression_ratio_threshold
            ):
                needs_fallback = True
            if (
                logprob_threshold is not None
                and decode_result.avg_logprob < logprob_threshold
            ):
                needs_fallback = True
            if (
                no_speech_threshold is not None
                and decode_result.no_speech_prob > no_speech_threshold
            ):
                needs_fallback = False  # Silence, no fallback needed

            if needs_fallback:
                fallback_indices.append(i)

        # Batch all fallback segments together (instead of individual decoding)
        if fallback_indices:
            # Stack all segments needing fallback into one batch
            fallback_segments = mx.stack([segment_batch[i] for i in fallback_indices], axis=0)
            fallback_options = DecodingOptions(**kwargs, temperature=1.0)
            fallback_results = model.decode(fallback_segments, fallback_options)

            # Ensure fallback_results is a list
            if not isinstance(fallback_results, list):
                fallback_results = [fallback_results]

            # Update decode_results with fallback results
            for idx, fallback_result in zip(fallback_indices, fallback_results):
                decode_results[idx] = fallback_result

        return decode_results

    clip_idx = 0
    seek = seek_clips[clip_idx][0]
    input_stride = N_FRAMES // model.dims.n_audio_ctx  # mel frames per output token: 2
    time_precision = (
        input_stride * HOP_LENGTH / SAMPLE_RATE
    )  # time per output token: 0.02 (seconds)
    all_tokens = []
    all_segments = []
    prompt_reset_since = 0

    if initial_prompt is not None:
        initial_prompt_tokens = tokenizer.encode(" " + initial_prompt.strip())
        all_tokens.extend(initial_prompt_tokens)
    else:
        initial_prompt_tokens = []

    def new_segment(
        *, start: float, end: float, tokens: mx.array, result: DecodingResult
    ):
        tokens = tokens.tolist()
        text_tokens = [token for token in tokens if token < tokenizer.eot]
        return {
            "seek": seek,
            "start": start,
            "end": end,
            "text": tokenizer.decode(text_tokens),
            "tokens": tokens,
            "temperature": result.temperature,
            "avg_logprob": result.avg_logprob,
            "compression_ratio": result.compression_ratio,
            "no_speech_prob": result.no_speech_prob,
        }

    # show the progress bar when verbose is False (if True, transcribed text will be printed)
    with tqdm.tqdm(
        total=content_frames, unit="frames", disable=verbose is not False
    ) as pbar:
        last_speech_timestamp = 0.0
        for seek_clip_start, seek_clip_end in seek_clips:
            seek = seek_clip_start

            while seek < seek_clip_end:
                # Batched processing: collect multiple segments
                if batch_size > 1:
                    mel_segments = []
                    segment_seeks = []
                    segment_sizes = []

                    batch_seek = seek
                    for _ in range(batch_size):
                        if batch_seek >= seek_clip_end:
                            break

                        segment_size = min(
                            N_FRAMES, content_frames - batch_seek, seek_clip_end - batch_seek
                        )
                        mel_segment = mel[batch_seek : batch_seek + segment_size]
                        mel_segment = pad_or_trim(mel_segment, N_FRAMES, axis=-2).astype(dtype)

                        mel_segments.append(mel_segment)
                        segment_seeks.append(batch_seek)
                        segment_sizes.append(segment_size)

                        batch_seek += N_FRAMES

                    if not mel_segments:
                        break

                    # Stack segments into a batch
                    mel_batch = mx.stack(mel_segments, axis=0)
                    decode_options["prompt"] = all_tokens[prompt_reset_since:]
                    results = decode_batch_with_fallback(mel_batch)

                    # Process each result in the batch
                    for batch_idx, result in enumerate(results):
                        segment_seek = segment_seeks[batch_idx]
                        segment_size = segment_sizes[batch_idx]
                        time_offset = float(segment_seek * HOP_LENGTH / SAMPLE_RATE)
                        segment_duration = segment_size * HOP_LENGTH / SAMPLE_RATE

                        tokens = np.array(result.tokens)

                        if no_speech_threshold is not None:
                            should_skip = result.no_speech_prob > no_speech_threshold
                            if (
                                logprob_threshold is not None
                                and result.avg_logprob > logprob_threshold
                            ):
                                should_skip = False
                            if should_skip:
                                continue

                        current_segments = []
                        timestamp_tokens = tokens >= tokenizer.timestamp_begin
                        single_timestamp_ending = timestamp_tokens[-2:].tolist() == [False, True]

                        consecutive = np.where(
                            np.logical_and(timestamp_tokens[:-1], timestamp_tokens[1:])
                        )[0]
                        consecutive += 1

                        if len(consecutive) > 0:
                            slices = consecutive.tolist()
                            if single_timestamp_ending:
                                slices.append(len(tokens))

                            last_slice = 0
                            for current_slice in slices:
                                sliced_tokens = tokens[last_slice:current_slice]
                                start_timestamp_pos = (
                                    sliced_tokens[0].item() - tokenizer.timestamp_begin
                                )
                                end_timestamp_pos = (
                                    sliced_tokens[-1].item() - tokenizer.timestamp_begin
                                )
                                current_segments.append(
                                    new_segment(
                                        start=time_offset + start_timestamp_pos * time_precision,
                                        end=time_offset + end_timestamp_pos * time_precision,
                                        tokens=mx.array(sliced_tokens),
                                        result=result,
                                    )
                                )
                                last_slice = current_slice
                        else:
                            duration = segment_duration
                            timestamps = tokens[timestamp_tokens.nonzero()[0]]
                            if len(timestamps) > 0 and timestamps[-1].item() != tokenizer.timestamp_begin:
                                last_timestamp_pos = timestamps[-1].item() - tokenizer.timestamp_begin
                                duration = last_timestamp_pos * time_precision

                            current_segments.append(
                                new_segment(
                                    start=time_offset,
                                    end=time_offset + duration,
                                    tokens=mx.array(tokens),
                                    result=result,
                                )
                            )

                        if verbose:
                            for segment in current_segments:
                                start, end, text = segment["start"], segment["end"], segment["text"]
                                line = f"[{format_timestamp(start)} --> {format_timestamp(end)}] {text}"
                                print(make_safe(line))

                        # Clear empty segments
                        for segment in current_segments:
                            if segment["start"] == segment["end"] or segment["text"].strip() == "":
                                segment["text"] = ""
                                segment["tokens"] = []
                                segment["words"] = []

                        all_segments.extend(
                            [{"id": i, **segment} for i, segment in enumerate(current_segments, start=len(all_segments))]
                        )
                        all_tokens.extend([token for segment in current_segments for token in segment["tokens"]])

                        if not condition_on_previous_text or result.temperature > 0.5:
                            prompt_reset_since = len(all_tokens)

                    # Update seek and progress bar
                    previous_seek = seek
                    seek = batch_seek
                    pbar.update(min(content_frames, seek) - previous_seek)

                else:
                    # Original single-segment processing
                    time_offset = float(seek * HOP_LENGTH / SAMPLE_RATE)
                    window_end_time = float((seek + N_FRAMES) * HOP_LENGTH / SAMPLE_RATE)
                    segment_size = min(
                        N_FRAMES, content_frames - seek, seek_clip_end - seek
                    )
                    mel_segment = mel[seek : seek + segment_size]
                    segment_duration = segment_size * HOP_LENGTH / SAMPLE_RATE
                    mel_segment = pad_or_trim(mel_segment, N_FRAMES, axis=-2).astype(dtype)

                    decode_options["prompt"] = all_tokens[prompt_reset_since:]
                    result: DecodingResult = decode_with_fallback(mel_segment)

                    tokens = np.array(result.tokens)

                    if no_speech_threshold is not None:
                        # no voice activity check
                        should_skip = result.no_speech_prob > no_speech_threshold
                        if (
                            logprob_threshold is not None
                            and result.avg_logprob > logprob_threshold
                        ):
                            # don't skip if the logprob is high enough, despite the no_speech_prob
                            should_skip = False

                        if should_skip:
                            seek += (
                                segment_size  # fast-forward to the next segment boundary
                            )
                            continue

                    previous_seek = seek
                    current_segments = []

                    # anomalous words are very long/short/improbable
                    def word_anomaly_score(word: dict) -> float:
                        probability = word.get("probability", 0.0)
                        duration = word["end"] - word["start"]
                        score = 0.0
                        if probability < 0.15:
                            score += 1.0
                        if duration < 0.133:
                            score += (0.133 - duration) * 15
                        if duration > 2.0:
                            score += duration - 2.0
                        return score

                    def is_segment_anomaly(segment: Optional[dict]) -> bool:
                        if segment is None or not segment["words"]:
                            return False
                        words = [
                            w for w in segment["words"] if w["word"] not in punctuation
                        ]
                        words = words[:8]
                        score = sum(word_anomaly_score(w) for w in words)
                        return score >= 3 or score + 0.01 >= len(words)

                    def next_words_segment(segments: List[dict]) -> Optional[dict]:
                        return next((s for s in segments if s["words"]), None)

                    timestamp_tokens = tokens >= tokenizer.timestamp_begin
                    single_timestamp_ending = timestamp_tokens[-2:].tolist() == [
                        False,
                        True,
                    ]

                    consecutive = np.where(
                        np.logical_and(timestamp_tokens[:-1], timestamp_tokens[1:])
                    )[0]
                    consecutive += 1
                    if len(consecutive) > 0:
                        # if the output contains two consecutive timestamp tokens
                        slices = consecutive.tolist()
                        if single_timestamp_ending:
                            slices.append(len(tokens))

                        last_slice = 0
                        for current_slice in slices:
                            sliced_tokens = tokens[last_slice:current_slice]
                            start_timestamp_pos = (
                                sliced_tokens[0].item() - tokenizer.timestamp_begin
                            )
                            end_timestamp_pos = (
                                sliced_tokens[-1].item() - tokenizer.timestamp_begin
                            )
                            current_segments.append(
                                new_segment(
                                    start=time_offset
                                    + start_timestamp_pos * time_precision,
                                    end=time_offset + end_timestamp_pos * time_precision,
                                    tokens=mx.array(sliced_tokens),
                                    result=result,
                                )
                            )
                            last_slice = current_slice

                        if single_timestamp_ending:
                            # single timestamp at the end means no speech after the last timestamp.
                            seek += segment_size
                        else:
                            # otherwise, ignore the unfinished segment and seek to the last timestamp
                            last_timestamp_pos = (
                                tokens[last_slice - 1].item() - tokenizer.timestamp_begin
                            )
                            seek += last_timestamp_pos * input_stride
                    else:
                        duration = segment_duration
                        timestamps = tokens[timestamp_tokens.nonzero()[0]]
                        if (
                            len(timestamps) > 0
                            and timestamps[-1].item() != tokenizer.timestamp_begin
                        ):
                            # no consecutive timestamps but it has a timestamp; use the last one.
                            last_timestamp_pos = (
                                timestamps[-1].item() - tokenizer.timestamp_begin
                            )
                            duration = last_timestamp_pos * time_precision

                        current_segments.append(
                            new_segment(
                                start=time_offset,
                                end=time_offset + duration,
                                tokens=mx.array(tokens),
                                result=result,
                            )
                        )
                        seek += segment_size

                    if word_timestamps:
                        add_word_timestamps(
                            segments=current_segments,
                            model=model,
                            tokenizer=tokenizer,
                            mel=mel_segment,
                            num_frames=segment_size,
                            prepend_punctuations=prepend_punctuations,
                            append_punctuations=append_punctuations,
                            last_speech_timestamp=last_speech_timestamp,
                        )

                        if not single_timestamp_ending:
                            last_word_end = _get_end(current_segments)
                            if last_word_end is not None and last_word_end > time_offset:
                                seek = round(last_word_end * FRAMES_PER_SECOND)

                        # skip silence before possible hallucinations
                        if hallucination_silence_threshold is not None:
                            threshold = hallucination_silence_threshold
                            if not single_timestamp_ending:
                                last_word_end = _get_end(current_segments)
                                if (
                                    last_word_end is not None
                                    and last_word_end > time_offset
                                ):
                                    remaining_duration = window_end_time - last_word_end
                                    if remaining_duration > threshold:
                                        seek = round(last_word_end * FRAMES_PER_SECOND)
                                    else:
                                        seek = previous_seek + segment_size

                            # if first segment might be a hallucination, skip leading silence
                            first_segment = next_words_segment(current_segments)
                            if first_segment is not None and is_segment_anomaly(
                                first_segment
                            ):
                                gap = first_segment["start"] - time_offset
                                if gap > threshold:
                                    seek = previous_seek + round(gap * FRAMES_PER_SECOND)
                                    continue

                            # skip silence before any possible hallucination that is surrounded
                            # by silence or more hallucinations
                            hal_last_end = last_speech_timestamp
                            for si in range(len(current_segments)):
                                segment = current_segments[si]
                                if not segment["words"]:
                                    continue
                                if is_segment_anomaly(segment):
                                    next_segment = next_words_segment(
                                        current_segments[si + 1 :]
                                    )
                                    if next_segment is not None:
                                        hal_next_start = next_segment["words"][0]["start"]
                                    else:
                                        hal_next_start = time_offset + segment_duration
                                    silence_before = (
                                        segment["start"] - hal_last_end > threshold
                                        or segment["start"] < threshold
                                        or segment["start"] - time_offset < 2.0
                                    )
                                    silence_after = (
                                        hal_next_start - segment["end"] > threshold
                                        or is_segment_anomaly(next_segment)
                                        or window_end_time - segment["end"] < 2.0
                                    )
                                    if silence_before and silence_after:
                                        seek = round(
                                            max(time_offset + 1, segment["start"])
                                            * FRAMES_PER_SECOND
                                        )
                                        if content_duration - segment["end"] < threshold:
                                            seek = content_frames
                                        current_segments[si:] = []
                                        break
                                hal_last_end = segment["end"]

                        last_word_end = _get_end(current_segments)
                        if last_word_end is not None:
                            last_speech_timestamp = last_word_end

                    if verbose:
                        for segment in current_segments:
                            start, end, text = (
                                segment["start"],
                                segment["end"],
                                segment["text"],
                            )
                            line = f"[{format_timestamp(start)} --> {format_timestamp(end)}] {text}"
                            print(make_safe(line))

                    # if a segment is instantaneous or does not contain text, clear it
                    for i, segment in enumerate(current_segments):
                        if (
                            segment["start"] == segment["end"]
                            or segment["text"].strip() == ""
                        ):
                            segment["text"] = ""
                            segment["tokens"] = []
                            segment["words"] = []

                    all_segments.extend(
                        [
                            {"id": i, **segment}
                            for i, segment in enumerate(
                                current_segments, start=len(all_segments)
                            )
                        ]
                    )
                    all_tokens.extend(
                        [
                            token
                            for segment in current_segments
                            for token in segment["tokens"]
                        ]
                    )

                    if not condition_on_previous_text or result.temperature > 0.5:
                        # do not feed the prompt tokens if a high temperature was used
                        prompt_reset_since = len(all_tokens)

                    # update progress bar
                    pbar.update(min(content_frames, seek) - previous_seek)

    return dict(
        text=tokenizer.decode(all_tokens[len(initial_prompt_tokens) :]),
        segments=all_segments,
        language=language,
    )
