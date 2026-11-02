from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .decoder import Decoder
from .layers import DurationPredictor, ProsodyPredictor, TextEncoder
from .tokenizer import Tokenizer

VoiceInput = Union[str, np.ndarray, Sequence[float]]

DEFAULT_MODEL = "patriotyk/HolosTTS/mlx"
_FILES = ("config.json", "model.safetensors", "voices.npz")
# The boundary token before the text decodes to silence. Streaming drops all
# of it except this margin and fades the rest in to avoid a click.
_SILENCE_MARGIN_SECONDS = 0.010
_FADE_IN_SAMPLES = 120  # 5 ms at 24 kHz


@dataclass(frozen=True)
class InferenceResult:
    audio: np.ndarray
    sample_rate: int
    audio_length: int
    frame_count: int
    durations: np.ndarray


@dataclass(frozen=True)
class _Encoded:
    voice: mx.array
    acoustic: mx.array
    f0: mx.array
    energy: mx.array
    durations: np.ndarray


class _Model(nn.Module):
    def __init__(self, n_symbols: int):
        super().__init__()
        self.text_encoder = TextEncoder(n_symbols, n_layers=4)
        self.prosodic_text_encoder = TextEncoder(n_symbols, n_layers=6, conv_pe=True)
        self.duration_predictor = DurationPredictor()
        self.prosodic_predictor = ProsodyPredictor()
        self.decoder = Decoder()


class HolosTTS:
    """HolosTTS on Apple silicon with MLX.

    The model runs on the GPU with dynamic shapes: ``generate`` decodes the
    whole utterance at once, and ``stream`` decodes chunks with enough context
    on both sides that they concatenate to the same audio. The input text is
    expected to use the checkpoint vocabulary; in the demo application this is
    the phoneme string produced by the Ukrainian frontend.
    """

    def __init__(
        self,
        model: Union[str, Path] = DEFAULT_MODEL,
        revision: Optional[str] = None,
        cache_dir: Optional[Union[str, Path]] = None,
        token: Optional[Union[str, bool]] = None,
    ):
        """Load a model directory, or a Hugging Face model: ``owner/repo``
        optionally followed by its folder in the repository, such as
        ``patriotyk/HolosTTS/mlx``. Only the model files are downloaded."""
        directory = self._resolve_model(model, revision, cache_dir, token)
        config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
        if config.get("format") != "holos-mlx" or config.get("format_version") != 1:
            raise ValueError(f"Unsupported HolosTTS MLX model in {directory}")
        self.config = config
        self.sample_rate = int(config["sample_rate"])
        self.samples_per_frame = int(config["samples_per_frame"])
        self.context_frames = int(config["context_frames"])
        self.n_symbols = int(config["n_symbols"])
        self.tokenizer = Tokenizer(config["vocab"])
        self.model = _Model(self.n_symbols)
        self.model.load_weights(str(directory / "model.safetensors"))
        self.model.eval()
        mx.eval(self.model.parameters())
        with np.load(directory / "voices.npz", allow_pickle=False) as archive:
            names = archive["names"].astype(str).tolist()
            values = np.asarray(archive["values"], dtype=np.float32)
        self._voices = dict(zip(names, values))

    @staticmethod
    def _resolve_model(model, revision, cache_dir, token) -> Path:
        path = Path(model).expanduser()
        if path.exists():
            return path.resolve()
        parts = str(model).strip("/").split("/")
        if isinstance(model, Path) or len(parts) < 2 or not all(parts):
            raise FileNotFoundError(f"HolosTTS MLX model not found: {model}")

        from huggingface_hub import snapshot_download

        prefix = "".join(f"{part}/" for part in parts[2:])
        snapshot = snapshot_download(
            "/".join(parts[:2]),
            revision=revision,
            cache_dir=cache_dir,
            token=token,
            allow_patterns=[prefix + name for name in _FILES],
        )
        directory = Path(snapshot) / prefix
        if not (directory / "config.json").is_file():
            raise FileNotFoundError(f"No HolosTTS MLX model in {prefix.rstrip('/') or 'the root'} of {model}")
        return directory

    def list_voices(self) -> List[str]:
        return list(self._voices)

    def _voice_array(self, voice: Optional[VoiceInput]) -> mx.array:
        if voice is None:
            value = next(iter(self._voices.values()))
        elif isinstance(voice, str):
            if voice not in self._voices:
                raise ValueError(f"Unknown voice {voice!r}. Available voices: {', '.join(self._voices)}")
            value = self._voices[voice]
        else:
            value = np.asarray(voice, dtype=np.float32)
        value = np.asarray(value, dtype=np.float32).reshape(-1)
        if value.shape != (256,):
            raise ValueError(f"Expected a voice of 256 values, got {value.size}")
        return mx.array(value[None])

    def _encode(self, text: str, voice: Optional[VoiceInput], speed: float) -> _Encoded:
        speed = float(speed)
        if not math.isfinite(speed) or speed <= 0:
            raise ValueError("speed must be a positive finite number")
        tokens = self.tokenizer.encode_with_boundaries(text)
        if tokens.max() >= self.n_symbols:
            unknown = sorted({self.tokenizer.vocab[t] for t in tokens if t >= self.n_symbols})
            raise ValueError(f"The model has no embedding for: {''.join(unknown)}")
        voice = self._voice_array(voice)
        style = voice[:, 128:]
        model = self.model
        tokens = mx.array(tokens[None])

        prosodic = model.prosodic_text_encoder(tokens)
        expected = model.duration_predictor(prosodic, style) / speed
        # Rounded on the host, half to even like torch.round.
        durations = np.round(np.array(expected[0])).astype(np.int64)
        if durations.sum() <= 0:
            raise ValueError("The text produced no frames")
        index = mx.array(np.repeat(np.arange(durations.size), durations))
        f0, energy = model.prosodic_predictor(prosodic[:, index], style)
        acoustic = model.text_encoder(tokens)[:, index]
        return _Encoded(voice[:, :128], acoustic, f0, energy, durations)

    def generate_with_details(
        self,
        text: str,
        voice: Optional[VoiceInput] = None,
        speed: float = 1.0,
    ) -> InferenceResult:
        encoded = self._encode(text, voice, speed)
        decoder = self.model.decoder
        source = decoder.generator.source(encoded.f0)
        audio = decoder(encoded.acoustic, encoded.f0, encoded.energy, encoded.voice, source)
        audio = np.array(audio[0])
        if not np.isfinite(audio).all():
            raise RuntimeError("HolosTTS MLX decoder produced NaN or Inf")
        return InferenceResult(
            audio=audio,
            sample_rate=self.sample_rate,
            audio_length=audio.size,
            frame_count=int(encoded.durations.sum()),
            durations=encoded.durations,
        )

    def generate(self, text: str, voice: Optional[VoiceInput] = None, speed: float = 1.0) -> np.ndarray:
        """Audio of the whole utterance, float32 at ``sample_rate``. Speed
        above one speaks faster, below one slower."""
        return self.generate_with_details(text, voice, speed).audio

    __call__ = generate

    @staticmethod
    def _stream_windows(frame_count: int, chunk_frames: int, context_frames: int) -> Iterator[Tuple[int, int, int, int]]:
        """Yield ``(start, end, window_start, window_end)``: each chunk with
        ``context_frames`` of context on both sides, cut at the edges."""
        for start in range(0, frame_count, chunk_frames):
            end = min(start + chunk_frames, frame_count)
            yield start, end, max(0, start - context_frames), min(frame_count, end + context_frames)

    def stream(
        self,
        text: str,
        voice: Optional[VoiceInput] = None,
        speed: float = 1.0,
        chunk_frames: int = 40,
        trim_silence: bool = True,
    ) -> Iterator[np.ndarray]:
        """Yield audio chunk by chunk; each covers ``chunk_frames`` frames
        (40 = 1 s). Text encoding and prosody run once for the whole text,
        and the chunks concatenate to the ``generate`` audio.

        ``trim_silence`` drops the leading boundary-token silence (about
        50 ms) except a 10 ms margin and fades in the next 5 ms.
        """
        if chunk_frames < 1:
            raise ValueError("chunk_frames must be positive")
        encoded = self._encode(text, voice, speed)
        chunks = self._decode_windows(encoded, int(chunk_frames))
        if not trim_silence:
            return chunks
        silence = int(encoded.durations[0]) * self.samples_per_frame
        skip = max(0, silence - round(_SILENCE_MARGIN_SECONDS * self.sample_rate))
        return self._trim_leading(chunks, skip, _FADE_IN_SAMPLES if skip else 0)

    def _decode_windows(self, encoded: _Encoded, chunk_frames: int) -> Iterator[np.ndarray]:
        decoder = self.model.decoder
        step = self.samples_per_frame
        source = decoder.generator.source(encoded.f0)
        for start, end, window_start, window_end in self._stream_windows(
            int(encoded.durations.sum()), chunk_frames, self.context_frames,
        ):
            audio = decoder(
                encoded.acoustic[:, window_start:window_end],
                encoded.f0[:, 2 * window_start:2 * window_end],
                encoded.energy[:, 2 * window_start:2 * window_end],
                encoded.voice,
                source[:, step * window_start:step * window_end],
            )
            offset = start - window_start
            yield np.array(audio[0, step * offset:step * (offset + end - start)])

    @staticmethod
    def _trim_leading(chunks: Iterator[np.ndarray], skip: int, fade: int) -> Iterator[np.ndarray]:
        """Drop ``skip`` leading samples and ramp in the next ``fade``,
        across chunk boundaries."""
        faded = 0
        for chunk in chunks:
            if skip:
                dropped = min(skip, chunk.size)
                chunk = chunk[dropped:]
                skip -= dropped
            if faded < fade and chunk.size:
                count = min(fade - faded, chunk.size)
                ramp = np.arange(faded + 1, faded + count + 1, dtype=np.float32) / fade
                chunk = np.concatenate([chunk[:count] * ramp, chunk[count:]])
                faded += count
            if chunk.size:
                yield chunk
