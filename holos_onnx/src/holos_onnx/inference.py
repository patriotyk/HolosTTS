from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
import onnxruntime as ort

from .tokenizer import Tokenizer


VoiceInput = Union[str, np.ndarray, Sequence[float]]

DEFAULT_MODEL = "patriotyk/HolosTTS/onnx"
_MANIFEST = "holos_onnx.json"
# Bundle files downloaded from a model repository, shared by both modes and
# per mode: the whole-utterance decoder or the streaming window decoder.
_SHARED_FILES = (_MANIFEST, "voices.npz", "encoder.onnx", "prosody.onnx")
_EXACT_FILES = ("decoder.onnx",)
_STREAMING_FILES = ("source.onnx", "decoder_window.onnx")

_TENSORRT_PROVIDER = "TensorrtExecutionProvider"
_CUDA_PROVIDER = "CUDAExecutionProvider"
_CPU_PROVIDER = "CPUExecutionProvider"
_GPU_PROVIDERS = {_TENSORRT_PROVIDER, _CUDA_PROVIDER}
# The boundary token before the text decodes to silence. Streaming drops all
# of it except this margin and fades the rest in to avoid a click.
_SILENCE_MARGIN_SECONDS = 0.010
_FADE_IN_SAMPLES = 120  # 5 ms at 24 kHz
_TENSORRT_PROFILE_OPTIONS = (
    "trt_profile_min_shapes",
    "trt_profile_opt_shapes",
    "trt_profile_max_shapes",
)


@dataclass(frozen=True)
class InferenceResult:
    audio: np.ndarray
    sample_rate: int
    audio_length: int
    frame_count: int
    durations: np.ndarray


@dataclass(frozen=True)
class _Encoded:
    voice: np.ndarray
    acoustic_tokens: np.ndarray
    frame_to_token: np.ndarray
    f0: np.ndarray
    energy: np.ndarray
    durations: np.ndarray


class HolosTTS:
    """Three-stage HolosTTS inference implemented without a PyTorch dependency.

    The encoder predicts token features and durations, the prosody stage turns
    them into frame-level F0 and energy, and the decoder synthesizes audio.
    With ``streaming=True`` a source stage computes the excitation and a window
    decoder decodes fixed-size windows, so ``stream`` returns audio chunk by
    chunk; otherwise one decoder call produces the whole utterance.
    The input text is expected to use the checkpoint vocabulary. In the demo
    application this is the phoneme string produced by the Ukrainian frontend.
    """

    def __init__(
        self,
        model: Union[str, Path] = DEFAULT_MODEL,
        streaming: bool = False,
        providers: Optional[Sequence[Any]] = None,
        session_options: Optional[ort.SessionOptions] = None,
        revision: Optional[str] = None,
        cache_dir: Optional[Union[str, Path]] = None,
        token: Optional[Union[str, bool]] = None,
    ):
        """Load a bundle.

        ``model`` is a local bundle directory (or its ``holos_onnx.json``).
        When no such path exists, it is a Hugging Face model: ``owner/repo``
        optionally followed by the bundle's folder in the repository, such as
        ``patriotyk/HolosTTS/onnx``. Only the bundle files are downloaded, into
        the Hugging Face cache (``cache_dir``), so later loads work offline.
        ``revision`` and ``token`` apply to the download.

        ``streaming`` loads the window decoder used by ``stream`` instead of
        the whole-utterance decoder; only the chosen decoder is downloaded.
        """
        model_path = self._resolve_model(model, streaming, revision, cache_dir, token)
        manifest_path = model_path if model_path.is_file() else model_path / _MANIFEST
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Holos ONNX manifest not found: {manifest_path}")

        self.model_dir = manifest_path.parent
        self.manifest = self._load_manifest(manifest_path)
        self.sample_rate = int(self.manifest["sample_rate"])
        self.samples_per_frame = int(self.manifest["samples_per_frame"])
        self.max_frames = int(self.manifest["max_frames"])
        self.tokenizer = Tokenizer(self.manifest["vocab"])
        self._voices = self._load_voices(self.manifest.get("voices"))

        selected_providers = list(providers or [_CPU_PROVIDER])
        selected_providers = self._add_tensorrt_fallbacks(selected_providers)
        self._preload_gpu_libraries(selected_providers)
        self._tensorrt_enabled = any(
            self._provider_name(value) == _TENSORRT_PROVIDER
            for value in selected_providers
        )
        self.tensorrt_profile = self._load_tensorrt_profile()
        files = {"encoder": self.manifest["encoder"], "prosody": self.manifest["prosody"]}
        if streaming:
            if "streaming" not in self.manifest:
                raise ValueError("This bundle has no streaming decoder; re-export it to use streaming=True")
            self.streaming = self.manifest["streaming"]
            files.update(source=self.streaming["source"], decoder=self.streaming["decoder"])
        else:
            self.streaming = None
            files["decoder"] = self.manifest["decoder"]
        if session_options is None:
            session_options = self._default_session_options()
        loaded = {
            stage: ort.InferenceSession(
                str(self.model_dir / filename),
                sess_options=session_options,
                providers=self._configure_stage_providers(selected_providers, stage=stage),
            )
            for stage, filename in files.items()
        }
        self.encoder_session = loaded["encoder"]
        self.prosody_session = loaded["prosody"]
        self.source_session = loaded.get("source")
        self.decoder_session = loaded["decoder"]
        sessions = list(loaded.values())
        self.providers = list(dict.fromkeys(
            provider
            for session in sessions
            for provider in session.get_providers()
        ))
        self.alignment_format = self._resolve_alignment_format()

    @staticmethod
    def _resolve_model(
        model: Union[str, Path],
        streaming: bool,
        revision: Optional[str],
        cache_dir: Optional[Union[str, Path]],
        token: Optional[Union[str, bool]],
    ) -> Path:
        path = Path(model).expanduser()
        if path.exists():
            return path.resolve()
        parts = str(model).strip("/").split("/")
        if isinstance(model, Path) or len(parts) < 2 or not all(parts):
            raise FileNotFoundError(f"Holos ONNX bundle not found: {model}")

        from huggingface_hub import snapshot_download
        from huggingface_hub.utils import RepositoryNotFoundError

        repo_id = "/".join(parts[:2])
        prefix = "".join(f"{part}/" for part in parts[2:])
        try:
            snapshot = snapshot_download(
                repo_id,
                revision=revision,
                cache_dir=cache_dir,
                token=token,
                allow_patterns=[
                    prefix + name
                    for name in _SHARED_FILES + (_STREAMING_FILES if streaming else _EXACT_FILES)
                ],
            )
        except RepositoryNotFoundError as error:
            raise FileNotFoundError(
                f"{model!r} is neither a local path nor a Hugging Face model"
            ) from error
        bundle = Path(snapshot) / prefix
        if not (bundle / _MANIFEST).is_file():
            raise FileNotFoundError(
                f"No Holos ONNX bundle in {prefix.rstrip('/') or 'the root'} of {repo_id}"
            )
        return bundle

    @staticmethod
    def _provider_name(provider: Any) -> str:
        if isinstance(provider, (tuple, list)):
            return str(provider[0])
        return str(provider)

    @classmethod
    def _add_tensorrt_fallbacks(cls, providers: List[Any]) -> List[Any]:
        """Keep unsupported TensorRT nodes on CUDA instead of moving to CPU."""
        names = [cls._provider_name(value) for value in providers]
        if _TENSORRT_PROVIDER not in names:
            return providers

        available = set(ort.get_available_providers())
        for fallback in (_CUDA_PROVIDER, _CPU_PROVIDER):
            if fallback in available and fallback not in names:
                providers.append(fallback)
                names.append(fallback)
        return providers

    @staticmethod
    def _default_session_options() -> ort.SessionOptions:
        options = ort.SessionOptions()
        # The stages run one after another. Spinning intra-op threads of a
        # finished stage would compete with the next one for the CPU, which
        # made the CPU bundle about 2.5x slower than a single graph.
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        return options

    @classmethod
    def _preload_gpu_libraries(cls, providers: Sequence[Any]) -> None:
        names = {cls._provider_name(value) for value in providers}
        if not names.intersection(_GPU_PROVIDERS):
            return

        preload_dlls = getattr(ort, "preload_dlls", None)
        if callable(preload_dlls):
            # An empty directory makes ONNX Runtime prefer NVIDIA pip package
            # libraries (including nvidia/cu13/lib) over unrelated PyTorch DLLs.
            preload_dlls(directory="")

    def _load_tensorrt_profile(self) -> Optional[Dict[str, int]]:
        profile = self.manifest.get("tensorrt_profile")
        if profile is None:
            return None
        if not isinstance(profile, dict):
            raise ValueError("Manifest tensorrt_profile must be an object")

        required = {
            "min_text_length",
            "opt_text_length",
            "max_text_length",
            "min_frames",
            "opt_frames",
            "max_frames",
        }
        missing = sorted(required.difference(profile))
        if missing:
            raise ValueError(
                "Manifest tensorrt_profile is missing fields: "
                + ", ".join(missing)
            )
        values = {name: int(profile[name]) for name in required}
        for dimension in ("text_length", "frames"):
            minimum = values[f"min_{dimension}"]
            optimum = values[f"opt_{dimension}"]
            maximum = values[f"max_{dimension}"]
            if not 0 < minimum <= optimum <= maximum:
                raise ValueError(
                    f"Invalid TensorRT {dimension} profile: "
                    f"{minimum} <= {optimum} <= {maximum}"
                )
        return values

    def _stage_profile_options(self, stage: str) -> Dict[str, str]:
        if self.tensorrt_profile is None:
            return {}

        profile = self.tensorrt_profile
        shapes = {}
        for point in ("min", "opt", "max"):
            # TensorRT needs a profile for every dynamic input of a stage.
            text_length = profile[f"{point}_text_length"]
            frames = profile[f"{point}_frames"]
            if stage == "encoder":
                shapes[point] = f"tokens:1x{text_length}"
            elif stage == "prosody":
                shapes[point] = (
                    f"prosodic_tokens:1x{text_length}x512,"
                    f"frame_to_token:{frames}"
                )
            elif stage == "source":
                shapes[point] = f"f0:1x{2 * frames}"
            elif stage == "decoder" and self.streaming:
                # Optimized for full windows; shorter utterances use one
                # narrower window.
                window = 2 if point == "min" else int(self.streaming["window_frames"])
                shapes[point] = (
                    f"acoustic_tokens:1x512x{text_length},"
                    f"frame_to_token:{window},"
                    f"f0:1x{2 * window},"
                    f"energy:1x{2 * window},"
                    f"source:1x{self.samples_per_frame * window}"
                )
            elif stage == "decoder":
                shapes[point] = (
                    f"acoustic_tokens:1x512x{text_length},"
                    f"frame_to_token:{frames},"
                    f"f0:1x{2 * frames},"
                    f"energy:1x{2 * frames}"
                )
            else:
                raise ValueError(f"Unknown ONNX stage: {stage!r}")
        return {
            option: shapes[point]
            for option, point in zip(
                _TENSORRT_PROFILE_OPTIONS,
                ("min", "opt", "max"),
            )
        }

    def _configure_stage_providers(
        self,
        providers: Sequence[Any],
        stage: str,
    ) -> List[Any]:
        profile_options = self._stage_profile_options(stage)
        configured = []
        for provider in providers:
            if self._provider_name(provider) != _TENSORRT_PROVIDER:
                configured.append(provider)
                continue

            options = {}
            if isinstance(provider, (tuple, list)) and len(provider) > 1:
                options.update(dict(provider[1]))
            # A bundle has different dynamic inputs in each stage. Always use
            # its matching profile instead of applying one shared profile to
            # every TensorRT session.
            options.update(profile_options)
            configured.append((_TENSORRT_PROVIDER, options))
        return configured

    def _resolve_alignment_format(self) -> str:
        configured = self.manifest.get("alignment_format")
        get_inputs = getattr(self.decoder_session, "get_inputs", None)
        detected = None
        if callable(get_inputs):
            input_names = {value.name for value in get_inputs()}
            if "frame_to_token" in input_names:
                detected = "gather"

        alignment_format = configured or detected or "gather"
        if alignment_format != "gather":
            raise ValueError(
                "Exact-length decoder requires gather alignment; got "
                f"{alignment_format!r}"
            )
        if configured and detected and configured != detected:
            raise ValueError(
                "Manifest alignment_format does not match decoder inputs: "
                f"{configured!r} != {detected!r}"
            )
        return alignment_format

    @staticmethod
    def _load_manifest(path: Path) -> Dict[str, Any]:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("format") != "holos-onnx":
            raise ValueError(f"Unsupported Holos ONNX manifest format in {path}")
        if manifest.get("format_version") != 2:
            raise ValueError(
                f"Unsupported Holos ONNX format version: "
                f"{manifest.get('format_version')!r}"
            )
        required = {
            "encoder",
            "prosody",
            "decoder",
            "vocab",
            "sample_rate",
            "samples_per_frame",
            "max_frames",
            "decoder_mode",
            "alignment_format",
        }
        missing = sorted(required.difference(manifest))
        if missing:
            raise ValueError(
                f"Manifest is missing fields: {', '.join(missing)}; "
                "re-export the bundle with export_onnx.py --bundle-dir"
            )
        if manifest["decoder_mode"] != "exact":
            raise ValueError(
                f"Unsupported decoder mode: {manifest['decoder_mode']!r}"
            )
        streaming = manifest.get("streaming")
        if streaming is not None:
            fields = {"decoder", "source", "chunk_frames", "context_frames", "window_frames"}
            if not isinstance(streaming, dict) or not fields <= set(streaming):
                raise ValueError(
                    f"Manifest streaming must have {', '.join(sorted(fields))}"
                )
            if streaming["window_frames"] != streaming["chunk_frames"] + 2 * streaming["context_frames"]:
                raise ValueError("Manifest window_frames must be chunk_frames + 2 * context_frames")
        return manifest

    def _load_voices(self, filename: Optional[str]) -> Dict[str, np.ndarray]:
        if filename is None:
            return {}
        path = self.model_dir / filename
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != {"names", "values"}:
                raise ValueError(
                    f"{path} must contain exactly the 'names' and 'values' arrays"
                )
            names = archive["names"].astype(str).tolist()
            values = np.asarray(archive["values"], dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != 256:
            raise ValueError(
                f"Expected voices with shape [N, 256], got {values.shape}"
            )
        if len(names) != values.shape[0]:
            raise ValueError("Voice names and embeddings have different lengths")
        if len(set(names)) != len(names):
            raise ValueError("Voice names must be unique")
        if not np.isfinite(values).all():
            raise ValueError("Voice embeddings contain NaN or Inf")
        return {
            name: np.ascontiguousarray(value)
            for name, value in zip(names, values)
        }

    def list_voices(self) -> List[str]:
        return list(self._voices)

    def _voice_array(self, voice: Optional[VoiceInput]) -> np.ndarray:
        if voice is None:
            if not self._voices:
                raise ValueError("No preset voices are available; pass a voice array")
            value = next(iter(self._voices.values()))
        elif isinstance(voice, str):
            try:
                value = self._voices[voice]
            except KeyError as error:
                available = ", ".join(self.list_voices())
                raise ValueError(
                    f"Unknown voice {voice!r}. Available voices: {available}"
                ) from error
        else:
            value = np.asarray(voice, dtype=np.float32)

        value = np.asarray(value, dtype=np.float32)
        if value.shape == (256,):
            value = value[None, :]
        if value.shape != (1, 256):
            raise ValueError(
                f"Expected voice shape (256,) or (1, 256), got {value.shape}"
            )
        return np.ascontiguousarray(value)

    def _validate_frame_count(self, frame_count: int) -> None:
        if frame_count <= 0:
            raise ValueError(f"Predicted invalid frame count: {frame_count}")
        if frame_count > self.max_frames:
            raise ValueError(
                f"Predicted {frame_count} frames, exceeding "
                f"max_frames={self.max_frames}"
            )
        if self._tensorrt_enabled and self.tensorrt_profile is not None:
            minimum = self.tensorrt_profile["min_frames"]
            maximum = self.tensorrt_profile["max_frames"]
            if not minimum <= frame_count <= maximum:
                raise ValueError(
                    f"Predicted {frame_count} frames outside the TensorRT "
                    f"profile range [{minimum}, {maximum}]"
                )

    @staticmethod
    def _build_frame_to_token(
        durations: np.ndarray,
    ) -> np.ndarray:
        durations = np.asarray(durations, dtype=np.int64)
        if durations.ndim != 2 or durations.shape[0] != 1:
            raise ValueError(
                f"Expected durations with shape (1, T), got {durations.shape}"
            )
        if np.any(durations < 0):
            raise ValueError("Durations must be non-negative")

        frame_to_token = np.repeat(
            np.arange(durations.shape[1], dtype=np.int64),
            durations[0],
        )
        return np.ascontiguousarray(frame_to_token, dtype=np.int64)

    def _encode(self, text: str, voice: Optional[VoiceInput], speed: float) -> _Encoded:
        speed = float(speed)
        if not np.isfinite(speed) or speed <= 0:
            raise ValueError("speed must be a positive finite number")
        token_values = self.tokenizer.encode_with_boundaries(text)
        tokens = np.ascontiguousarray(token_values[None, :])
        if self._tensorrt_enabled and self.tensorrt_profile is not None:
            text_length = tokens.shape[1]
            minimum = self.tensorrt_profile["min_text_length"]
            maximum = self.tensorrt_profile["max_text_length"]
            if not minimum <= text_length <= maximum:
                raise ValueError(
                    f"Token length {text_length} is outside the TensorRT "
                    f"profile range [{minimum}, {maximum}]"
                )
        voice_array = self._voice_array(voice)

        acoustic_tokens, prosodic_tokens, durations = self.encoder_session.run(
            ["acoustic_tokens", "prosodic_tokens", "durations"],
            {
                "tokens": tokens,
                "voice": voice_array,
                "speed": np.asarray([speed], dtype=np.float32),
            },
        )
        durations = np.asarray(durations, dtype=np.int64)
        self._validate_frame_count(int(durations.sum()))
        frame_to_token = self._build_frame_to_token(durations)
        f0, energy = self.prosody_session.run(
            ["f0", "energy"],
            {
                "prosodic_tokens": np.asarray(prosodic_tokens, dtype=np.float32),
                "frame_to_token": frame_to_token,
                "voice": voice_array,
            },
        )
        return _Encoded(
            voice=voice_array,
            acoustic_tokens=np.asarray(acoustic_tokens, dtype=np.float32),
            frame_to_token=frame_to_token,
            f0=np.asarray(f0, dtype=np.float32),
            energy=np.asarray(energy, dtype=np.float32),
            durations=np.ascontiguousarray(durations[0]),
        )

    def _decode_exact(self, encoded: _Encoded) -> np.ndarray:
        frame_count = encoded.frame_to_token.size
        audio, audio_lengths = self.decoder_session.run(
            ["audio", "audio_lengths"],
            {
                "acoustic_tokens": encoded.acoustic_tokens,
                "frame_to_token": encoded.frame_to_token,
                "f0": encoded.f0,
                "energy": encoded.energy,
                "voice": encoded.voice,
            },
        )
        audio_length = int(np.asarray(audio_lengths).reshape(-1)[0])
        expected_length = frame_count * self.samples_per_frame
        if audio_length != expected_length:
            raise RuntimeError(
                f"Decoder returned {audio_length} valid samples; "
                f"expected {expected_length}"
            )
        waveform = np.asarray(audio, dtype=np.float32).reshape(-1)
        if not 0 < audio_length <= waveform.size:
            raise RuntimeError(
                f"Invalid audio length {audio_length} for output size {waveform.size}"
            )
        return waveform[:audio_length]

    @staticmethod
    def _stream_windows(
        frame_count: int,
        chunk_frames: int,
        context_frames: int,
    ) -> Iterator[Tuple[int, int, int, int]]:
        """Yield ``(start, end, window_start, window_end)`` for each chunk.

        Every chunk gets ``context_frames`` of real context on both sides, or
        reaches the utterance edge, so the chunks concatenate to the
        whole-utterance audio. Windows are shifted inside the utterance
        rather than padded.
        """
        window = chunk_frames + 2 * context_frames
        for start in range(0, frame_count, chunk_frames):
            end = min(start + chunk_frames, frame_count)
            window_start = max(0, min(start - context_frames, frame_count - window))
            yield start, end, window_start, min(window_start + window, frame_count)

    def _decode_windows(self, encoded: _Encoded, chunk_frames: Optional[int]) -> Iterator[np.ndarray]:
        maximum = int(self.streaming["chunk_frames"])
        chunk_frames = maximum if chunk_frames is None else int(chunk_frames)
        if not 0 < chunk_frames <= maximum:
            # The decoder's TensorRT profile ends at the exported window width.
            raise ValueError(f"chunk_frames must be between 1 and {maximum}")
        context_frames = int(self.streaming["context_frames"])
        step = self.samples_per_frame
        (source,) = self.source_session.run(["source"], {"f0": encoded.f0})
        for start, end, window_start, window_end in self._stream_windows(
            encoded.frame_to_token.size, chunk_frames, context_frames,
        ):
            (audio,) = self.decoder_session.run(
                ["audio"],
                {
                    "acoustic_tokens": encoded.acoustic_tokens,
                    "frame_to_token": encoded.frame_to_token[window_start:window_end],
                    "f0": encoded.f0[:, 2 * window_start:2 * window_end],
                    "energy": encoded.energy[:, 2 * window_start:2 * window_end],
                    "source": source[:, step * window_start:step * window_end],
                    "voice": encoded.voice,
                },
            )
            offset = start - window_start
            chunk = np.asarray(audio, dtype=np.float32)[0, step * offset:step * (offset + end - start)]
            if not np.isfinite(chunk).all():
                raise RuntimeError("Holos ONNX decoder produced NaN or Inf")
            yield np.ascontiguousarray(chunk)

    def stream(
        self,
        text: str,
        voice: Optional[VoiceInput] = None,
        speed: float = 1.0,
        chunk_frames: Optional[int] = None,
        trim_silence: bool = True,
    ) -> Iterator[np.ndarray]:
        """Yield audio chunk by chunk as the decoder windows finish.

        Needs ``streaming=True``. Each chunk covers ``chunk_frames`` frames,
        at most the exported chunk size.

        ``trim_silence`` drops the leading boundary-token silence (about
        50 ms) except a 10 ms margin and fades in the next 5 ms, so the first
        chunk starts at the speech onset. Those frames are still decoded as
        context, so the onset itself is unchanged.
        """
        if not self.streaming:
            raise RuntimeError("stream() needs a model loaded with HolosTTS(..., streaming=True)")
        return self._stream(text, voice, speed, chunk_frames, trim_silence)

    def _stream(self, text, voice, speed, chunk_frames, trim_silence) -> Iterator[np.ndarray]:
        encoded = self._encode(text, voice, speed)
        chunks = self._decode_windows(encoded, chunk_frames)
        if not trim_silence:
            yield from chunks
            return
        silence = int(encoded.durations[0]) * self.samples_per_frame
        skip = max(0, silence - round(_SILENCE_MARGIN_SECONDS * self.sample_rate))
        yield from self._trim_leading(chunks, skip, _FADE_IN_SAMPLES if skip else 0)

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

    def generate_with_details(
        self,
        text: str,
        voice: Optional[VoiceInput] = None,
        speed: float = 1.0,
    ) -> InferenceResult:
        encoded = self._encode(text, voice, speed)
        if self.streaming:
            waveform = np.concatenate(list(self._decode_windows(encoded, None)))
        else:
            waveform = self._decode_exact(encoded)
        if not np.isfinite(waveform).all():
            raise RuntimeError("Holos ONNX decoder produced NaN or Inf")

        frame_count = encoded.frame_to_token.size
        return InferenceResult(
            audio=np.ascontiguousarray(waveform),
            sample_rate=self.sample_rate,
            audio_length=waveform.size,
            frame_count=frame_count,
            durations=encoded.durations,
        )

    def generate(
        self,
        text: str,
        voice: Optional[VoiceInput] = None,
        speed: float = 1.0,
    ) -> np.ndarray:
        return self.generate_with_details(text, voice, speed=speed).audio

    __call__ = generate
