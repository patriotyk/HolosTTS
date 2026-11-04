"""Chunking and boundary-silence trimming shared by the streaming decoders.

The text encoder and prosody run on the whole utterance, so every frame is
known before decoding starts. Streaming cuts the frames into chunks and decodes
each one inside a window with ``context_frames`` of real context on both
sides; the chunks then concatenate to the whole-utterance audio.

This module has no torch dependency. The holos-onnx runtime keeps a copy of
these functions (it cannot import holos without torch); the tests in
holos_onnx/python/tests/test_streaming_parity.py check that both behave the same.
"""

from typing import Iterator, Tuple

import numpy as np

# The boundary token at the start lasts about 50 ms. Streaming drops it except
# this margin and fades in over FADE_IN_SAMPLES, so the onset does not click.
SILENCE_MARGIN_SECONDS = 0.010
FADE_IN_SAMPLES = 120


def stream_windows(
    frame_count: int,
    chunk_frames: int,
    context_frames: int,
    fixed_width: bool = False,
) -> Iterator[Tuple[int, int, int, int]]:
    """Yield ``(start, end, window_start, window_end)`` for each chunk.

    Windows stop at the utterance edges, so the first and last ones are
    shorter: the first chunk, which sets the time to first audio, decodes
    ``chunk_frames + context_frames`` frames.

    With ``fixed_width`` every window is ``chunk_frames + 2 * context_frames``
    wide (or the whole utterance), shifted inside it at the edges. The CUDA
    execution provider re-plans its convolutions whenever the input shape
    changes, about 30 ms per change for this decoder, far more than decoding
    the extra frames.
    """
    if chunk_frames < 1:
        raise ValueError("chunk_frames must be positive")
    width = chunk_frames + 2 * context_frames
    for start in range(0, frame_count, chunk_frames):
        end = min(start + chunk_frames, frame_count)
        window_start = max(0, start - context_frames)
        window_end = min(frame_count, end + context_frames)
        if fixed_width:
            window_start = max(0, min(window_start, frame_count - width))
            window_end = min(frame_count, window_start + width)
        yield start, end, window_start, window_end


def leading_silence_samples(
    first_token_frames: int,
    samples_per_frame: int,
    sample_rate: int,
) -> int:
    """Samples to drop before the speech onset: the boundary token minus the margin."""
    margin = round(SILENCE_MARGIN_SECONDS * sample_rate)
    return max(0, first_token_frames * samples_per_frame - margin)


def trim_leading(chunks: Iterator, skip: int, fade: int = FADE_IN_SAMPLES) -> Iterator:
    """Drop ``skip`` leading samples and ramp in the next ``fade``, across
    chunk boundaries. Chunks are NumPy arrays or torch tensors, with samples
    on the last axis."""
    if not skip:
        fade = 0
    faded = 0
    for chunk in chunks:
        if skip:
            dropped = min(skip, chunk.shape[-1])
            chunk = chunk[..., dropped:]
            skip -= dropped
        if faded < fade and chunk.shape[-1]:
            count = min(fade - faded, chunk.shape[-1])
            ramp = np.arange(faded + 1, faded + count + 1, dtype=np.float32) / fade
            if isinstance(chunk, np.ndarray):
                chunk = chunk.copy()
            else:
                chunk = chunk.clone()
                ramp = chunk.new_tensor(ramp)
            chunk[..., :count] *= ramp
            faded += count
        if chunk.shape[-1]:
            yield chunk
