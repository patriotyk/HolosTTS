"""HolosTTS with LiteRT, for the bundles in the litert/ folder of the Hugging
Face repository.

It needs only NumPy and ai-edge-litert, and does on the host what the Kotlin
library does: the embedding lookups, bucket selection and padding, rounding
the durations, the frame alignment, the harmonic excitation and the reflect
padding for its STFT, the decoder windows and the trimming of the leading
silence. Its decoder windows do not depend on timing, so it is the
reproducible reference for the ports of the bundle.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Union

import numpy as np

VoiceInput = Union[str, np.ndarray, Sequence[float]]

DEFAULT_MODEL = "patriotyk/HolosTTS/litert"
_MANIFEST = "holos_litert.json"
_FILES = (_MANIFEST, "encoder.tflite", "prosody.tflite", "decoder.tflite", "embeddings.bin", "voices.bin")
# The boundary token before the text decodes to silence. Streaming drops all
# of it except this margin and fades the rest in to avoid a click.
_SILENCE_MARGIN_SECONDS = 0.010
_FADE_IN_SAMPLES = 120  # 5 ms at 24 kHz
_ACOUSTIC_VOICE_DIM = 128


@dataclass(frozen=True)
class _Encoded:
    acoustic: np.ndarray  # [channels, frames]
    f0: np.ndarray        # [2 * frames]
    energy: np.ndarray    # [2 * frames]
    voice: np.ndarray     # [1, 128]
    durations: np.ndarray


class _Stage:
    """One .tflite with signatures, compiled once for the chosen accelerator."""

    def __init__(self, path: Path, gpu: bool, fp16: bool, threads: int):
        import ai_edge_litert
        from ai_edge_litert.compiled_model import CompiledModel
        from ai_edge_litert.environment import Environment, EnvironmentOptions
        from ai_edge_litert.hardware_accelerator import HardwareAccelerator
        from ai_edge_litert.options import CpuOptions, GpuOptions, Options

        if gpu:
            environment = Environment.create(options=EnvironmentOptions(
                runtime_path=str(Path(ai_edge_litert.__file__).parent)))
            options = Options(hardware_accelerators=HardwareAccelerator.GPU | HardwareAccelerator.CPU,
                              gpu_options=GpuOptions(enforce_f32=not fp16, constant_tensor_sharing=True))
        else:
            environment = None
            options = Options(hardware_accelerators=HardwareAccelerator.CPU, cpu_options=CpuOptions(num_threads=threads))
        self.model = CompiledModel.from_file(str(path), options=options, environment=environment)
        self._buffers = {}

    def __call__(self, signature: str, *inputs: np.ndarray) -> List[np.ndarray]:
        if signature not in self._buffers:
            details_in = self.model.get_input_tensor_details(signature)
            details_out = self.model.get_output_tensor_details(signature)
            names_in = sorted(details_in, key=lambda n: int(n.split("_")[-1]))
            names_out = sorted(details_out, key=lambda n: int(n.split("_")[-1]))
            self._buffers[signature] = (
                [(n, details_in[n], self.model.create_input_buffer_by_name(signature, n)) for n in names_in],
                [(n, details_out[n], self.model.create_output_buffer_by_name(signature, n)) for n in names_out],
            )
        buffers_in, buffers_out = self._buffers[signature]
        for (name, detail, buffer), value in zip(buffers_in, inputs):
            buffer.write(np.ascontiguousarray(np.asarray(value, dtype=detail["dtype"]).reshape(detail["shape"])))
        self.model.run_by_name(signature, {n: b for n, _, b in buffers_in}, {n: b for n, _, b in buffers_out})
        return [buffer.read(int(np.prod(detail["shape"])), detail["dtype"]).reshape(detail["shape"])
                for _, detail, buffer in buffers_out]


class HolosTTS:
    """HolosTTS with LiteRT.

    The encoder, prosody and decoder graphs have static shapes: the text goes
    into one of the token and frame buckets, and the decoder works in fixed
    windows. ``generate`` decodes 200-frame windows; ``stream`` starts with
    an 80-frame window, to give its first chunk sooner. The input text is
    expected to use the model vocabulary; in the demo application this is
    the phoneme string produced by the Ukrainian frontend.
    """

    def __init__(
        self,
        model: Union[str, Path] = DEFAULT_MODEL,
        gpu: bool = False,
        fp16: bool = True,
        threads: int = 4,
        seed: Optional[int] = None,
        revision: Optional[str] = None,
        cache_dir: Optional[Union[str, Path]] = None,
        token: Optional[Union[str, bool]] = None,
    ):
        """Load a model directory, or a Hugging Face model: ``owner/repo``
        optionally followed by its folder in the repository, such as
        ``patriotyk/HolosTTS/litert``. Only the model files are downloaded.

        ``gpu`` runs the graphs on the GPU (``fp16`` in FP16), otherwise on
        ``threads`` CPU threads in FP32. ``seed`` fixes the source noise."""
        directory = self._resolve_model(model, revision, cache_dir, token)
        manifest = json.loads((directory / _MANIFEST).read_text(encoding="utf-8"))
        if manifest.get("format") != "holos-litert-gpu" or manifest.get("format_version") != 2:
            raise ValueError(f"Unsupported HolosTTS LiteRT model in {directory}")
        self.manifest = manifest
        self.sample_rate = int(manifest["sample_rate"])
        self.samples_per_frame = int(manifest["samples_per_frame"])
        self.vocab = {symbol: index for index, symbol in enumerate(manifest["vocab"])}
        self.n_symbols, channels = int(manifest["n_symbols"]), int(manifest["channels"])
        tables = np.fromfile(directory / manifest["embeddings"]["file"], dtype="<f4").reshape(2, self.n_symbols, channels)
        self._prosodic_table, self._acoustic_table = tables
        voices = manifest["voices"]
        values = np.fromfile(directory / voices["file"], dtype="<f4").reshape(len(voices["names"]), voices["dim"])
        self._voices = dict(zip(voices["names"], values))
        self._encoder = _Stage(directory / manifest["encoder"]["file"], gpu, fp16, threads)
        self._prosody = _Stage(directory / manifest["prosody"]["file"], gpu, fp16, threads)
        self._decoder = _Stage(directory / manifest["decoder"]["file"], gpu, fp16, threads)
        self._rng = np.random.default_rng(seed)
        self._n_fft = int(manifest["stft"]["n_fft"])

    @staticmethod
    def _resolve_model(model, revision, cache_dir, token) -> Path:
        path = Path(model).expanduser()
        if path.exists():
            return path.resolve()
        parts = str(model).strip("/").split("/")
        if isinstance(model, Path) or len(parts) < 2 or not all(parts):
            raise FileNotFoundError(f"HolosTTS LiteRT model not found: {model}")

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
        if not (directory / _MANIFEST).is_file():
            raise FileNotFoundError(f"No HolosTTS LiteRT model in {prefix.rstrip('/') or 'the root'} of {model}")
        return directory

    def list_voices(self) -> List[str]:
        return list(self._voices)

    def _voice_array(self, voice: Optional[VoiceInput]) -> np.ndarray:
        if voice is None:
            value = next(iter(self._voices.values()))
        elif isinstance(voice, str):
            if voice not in self._voices:
                raise ValueError(f"Unknown voice {voice!r}. Available voices: {', '.join(self._voices)}")
            value = self._voices[voice]
        else:
            value = voice
        value = np.asarray(value, dtype=np.float32).reshape(-1)
        dim = int(self.manifest["voices"]["dim"])
        if value.shape != (dim,):
            raise ValueError(f"Expected a voice of {dim} values, got {value.size}")
        return value[None]

    @staticmethod
    def _bucket(buckets, length, what):
        for capacity in buckets:
            if length <= capacity:
                return capacity
        raise ValueError(f"{length} {what} exceed the largest bucket ({buckets[-1]}); split the text into sentences")

    def _encode(self, text: str, voice: Optional[VoiceInput], speed: float) -> _Encoded:
        speed = float(speed)
        if not math.isfinite(speed) or speed <= 0:
            raise ValueError("speed must be a positive finite number")
        tokens = np.asarray([0, *[self.vocab[c] for c in text if c in self.vocab], 0])
        if tokens.max() >= self.n_symbols:
            raise ValueError("The text has symbols the model has no embedding for")
        voice = self._voice_array(voice)
        length = tokens.size
        capacity = self._bucket(self.manifest["encoder"]["buckets"], length, "tokens")
        padded = np.zeros(capacity, dtype=np.int64)
        padded[:length] = tokens
        acoustic, prosodic, expected = self._encoder(
            f"encoder_{capacity}", self._prosodic_table[padded][None], self._acoustic_table[padded][None],
            np.array([length]), voice, np.array([speed], dtype=np.float32))
        # Rounded on the host, half to even like torch.round.
        durations = np.round(expected[0, :length]).astype(np.int64)
        frames = int(durations.sum())
        if frames <= 0:
            raise ValueError("The text produced no frames")
        index = np.repeat(np.arange(length), durations)
        frame_capacity = self._bucket(self.manifest["prosody"]["buckets"], frames, "frames")
        aligned = np.zeros((1, prosodic.shape[2], frame_capacity), dtype=np.float32)
        aligned[0, :, :frames] = prosodic[0, index].T
        f0, energy = self._prosody(f"prosody_{frame_capacity}", aligned, np.array([frames]), voice)
        return _Encoded(acoustic[0][:, index], f0[0, :2 * frames], energy[0, :2 * frames],
                        voice[:, :_ACOUSTIC_VOICE_DIM], durations)

    def _source(self, f0: np.ndarray, noise: bool) -> np.ndarray:
        """Harmonic-plus-noise excitation, as ``SineGen`` with the phase start
        of every block of constant F0 accumulated in float64."""
        params = self.manifest["source"]
        upsample, harmonics = int(params["upsample"]), int(params["harmonics"])
        factors = np.arange(1, harmonics + 1, dtype=np.float32)
        rad = (f0[:, None].astype(np.float32) * factors / np.float32(self.sample_rate)) % np.float32(1)
        block = (rad.astype(np.float64) * upsample) % 1
        start = (np.cumsum(block, axis=0) - block) % 1
        steps = np.arange(1, upsample + 1, dtype=np.float64)[None, :, None]
        cycles = ((start[:, None] + steps * rad.astype(np.float64)[:, None]) % 1).astype(np.float32)
        phase = cycles.reshape(-1, harmonics) * np.float32(2 * np.pi)
        voiced = np.repeat((f0 > params["voiced_threshold"]).astype(np.float32), upsample)[:, None]
        weight = np.asarray(params["merge_weight"], dtype=np.float32)
        merged = (np.sin(phase) * np.float32(params["sine_amp"]) * voiced) @ weight + np.float32(params["merge_bias"])
        if noise:
            amplitude = voiced[:, 0] * params["noise_std"] + (1 - voiced[:, 0]) * params["sine_amp"] / 3
            merged = merged + amplitude * np.linalg.norm(weight) * self._rng.standard_normal(merged.shape).astype(np.float32)
        return np.tanh(merged).astype(np.float32)

    def _padded(self, source: np.ndarray) -> np.ndarray:
        """A window's excitation, reflect-padded for the decoder's STFT."""
        pad = self._n_fft // 2
        return np.concatenate([source[1:pad + 1][::-1], source, source[-pad - 1:-1][::-1]])

    def _windows(self, frames: int, stream: bool, chunk_frames: Optional[int] = None):
        """Decoder windows: (signature, width, start, end, window_start).

        Every window has context_frames of context before and after its chunk,
        or reaches the edge of the utterance, so the chunks join seamlessly
        whatever window computed them. ``generate`` uses the large windows;
        ``stream`` uses a small one for the first chunk, to start sooner, and
        the large ones after it, with less context recomputed per second of
        audio. The frames that fit in one small window go into it in one
        piece.

        A window that runs past the end of a short utterance is padded, which
        only approximates its end, so ``stream`` takes large windows only when
        the utterance fills them. The Kotlin library also waits until the
        audio given out and not yet played covers a large window, so a slow
        GPU stays with small windows.
        """
        decoder = self.manifest["decoder"]
        small, large = decoder["stream"], decoder["generate"]
        context = int(decoder["context_frames"])
        start = 0
        while start < frames:
            if frames - max(0, start - context) <= int(small["window_frames"]) and chunk_frames is None:
                config, end = small, frames
            elif chunk_frames is not None or (stream and (start == 0 or frames < int(large["window_frames"]))):
                config = small
                end = min(start + (chunk_frames or int(small["chunk_frames"])), frames)
            else:
                config, end = large, min(start + int(large["chunk_frames"]), frames)
            width = int(config["window_frames"])
            window_start = max(0, min(start - context, frames - width))
            yield config["signature"], width, start, end, window_start
            start = end

    def _decode(self, encoded: _Encoded, stream: bool, noise: bool,
                chunk_frames: Optional[int] = None) -> Iterator[np.ndarray]:
        frames = int(encoded.durations.sum())
        step = self.samples_per_frame
        source = self._source(encoded.f0, noise)
        for signature, width, start, end, window_start in self._windows(frames, stream, chunk_frames):
            window_end = window_start + width
            padding = max(0, window_end - frames)
            used = slice(window_start, min(window_end, frames))
            # Past the end: the last token and F0 repeated, no energy or excitation.
            aligned = np.pad(encoded.acoustic[:, used], ((0, 0), (0, padding)), mode="edge")
            window_f0 = np.pad(encoded.f0[2 * used.start:2 * used.stop], (0, 2 * padding), mode="edge")
            window_energy = np.pad(encoded.energy[2 * used.start:2 * used.stop], (0, 2 * padding))
            window_source = np.pad(source[step * used.start:step * used.stop], (0, step * padding))
            (audio,) = self._decoder(signature, aligned[None], window_f0[None], window_energy[None],
                                     self._padded(window_source)[None], encoded.voice)
            offset = start - window_start
            yield audio[0, step * offset:step * (offset + end - start)]

    def generate(self, text: str, voice: Optional[VoiceInput] = None, speed: float = 1.0,
                 noise: bool = True) -> np.ndarray:
        """The whole utterance as float32 samples at ``sample_rate``;
        ``noise=False`` turns off the source noise, for reproducible output."""
        encoded = self._encode(text, voice, speed)
        return np.concatenate(list(self._decode(encoded, False, noise)))

    def stream(
        self,
        text: str,
        voice: Optional[VoiceInput] = None,
        speed: float = 1.0,
        chunk_frames: Optional[int] = None,
        trim_silence: bool = True,
        noise: bool = True,
    ) -> Iterator[np.ndarray]:
        """Yield audio chunk by chunk: 1 s first, then 4 s. ``chunk_frames``
        (1-40, 40 = 1 s) decodes every chunk in an 80-frame window instead.
        Text encoding and prosody run once for the whole text, and the chunks
        concatenate to the ``generate`` audio.

        ``trim_silence`` drops the leading boundary-token silence (about
        50 ms) except a 10 ms margin and fades in the next 5 ms.
        """
        if chunk_frames is not None:
            chunk_frames = int(chunk_frames)
            limit = int(self.manifest["decoder"]["stream"]["chunk_frames"])
            if not 0 < chunk_frames <= limit:
                raise ValueError(f"chunk_frames must be between 1 and {limit}")
        encoded = self._encode(text, voice, speed)
        chunks = self._decode(encoded, True, noise, chunk_frames)
        if not trim_silence:
            return chunks
        silence = int(encoded.durations[0]) * self.samples_per_frame
        skip = max(0, silence - round(_SILENCE_MARGIN_SECONDS * self.sample_rate))
        return self._trim_leading(chunks, skip, _FADE_IN_SAMPLES if skip else 0)

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
