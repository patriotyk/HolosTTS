from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import onnxruntime as ort

from .tokenizer import Tokenizer


VoiceInput = Union[str, np.ndarray, Sequence[float]]

_TENSORRT_PROVIDER = "TensorrtExecutionProvider"
_CUDA_PROVIDER = "CUDAExecutionProvider"
_CPU_PROVIDER = "CPUExecutionProvider"
_GPU_PROVIDERS = {_TENSORRT_PROVIDER, _CUDA_PROVIDER}
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


class HolosTTS:
    """Two-stage HolosTTS inference implemented without a PyTorch dependency.

    The input text is expected to use the checkpoint vocabulary. In the demo
    application this is the phoneme string produced by the Ukrainian frontend.
    """

    def __init__(
        self,
        model_dir: Union[str, Path],
        providers: Optional[Sequence[Any]] = None,
        session_options: Optional[ort.SessionOptions] = None,
    ):
        model_path = Path(model_dir).expanduser().resolve()
        manifest_path = (
            model_path
            if model_path.is_file()
            else model_path / "holos_onnx.json"
        )
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
        encoder_providers = self._configure_stage_providers(
            selected_providers,
            stage="encoder",
        )
        prosody_filename = self.manifest.get("prosody")
        if prosody_filename is None:
            prosody_providers = None
            decoder_providers = self._configure_stage_providers(
                selected_providers,
                stage="decoder",
            )
        else:
            prosody_providers = self._configure_stage_providers(
                selected_providers,
                stage="prosody",
            )
            decoder_providers = self._without_tensorrt(selected_providers)
        self.providers = [self._provider_name(value) for value in selected_providers]
        self.encoder_session = ort.InferenceSession(
            str(self.model_dir / self.manifest["encoder"]),
            sess_options=session_options,
            providers=encoder_providers,
        )
        self.encoder_input_names = {
            value.name
            for value in self.encoder_session.get_inputs()
        }
        self.prosody_session = None
        if prosody_filename is not None:
            self.prosody_session = ort.InferenceSession(
                str(self.model_dir / prosody_filename),
                sess_options=session_options,
                providers=prosody_providers,
            )
        self.decoder_session = ort.InferenceSession(
            str(self.model_dir / self.manifest["decoder"]),
            sess_options=session_options,
            providers=decoder_providers,
        )
        sessions = [self.encoder_session, self.prosody_session, self.decoder_session]
        self.providers = list(dict.fromkeys(
            provider
            for session in sessions
            if session is not None
            for provider in session.get_providers()
        ))
        self.alignment_format = self._resolve_alignment_format()

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

    @classmethod
    def _without_tensorrt(cls, providers: Sequence[Any]) -> List[Any]:
        fallbacks = [
            provider
            for provider in providers
            if cls._provider_name(provider) != _TENSORRT_PROVIDER
        ]
        return fallbacks or [_CPU_PROVIDER]

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
            text_length = profile[f"{point}_text_length"]
            if stage == "encoder":
                shapes[point] = f"tokens:1x{text_length}"
            elif stage == "prosody":
                frames = profile[f"{point}_frames"]
                shapes[point] = (
                    f"prosodic_tokens:1x{text_length}x512,"
                    f"frame_to_token:{frames}"
                )
            elif stage == "decoder":
                frames = profile[f"{point}_frames"]
                shapes[point] = (
                    f"acoustic_tokens:1x512x{text_length},"
                    f"prosodic_tokens:1x{text_length}x512,"
                    f"frame_to_token:{frames}"
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
            # both TensorRT sessions.
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
            "decoder",
            "vocab",
            "sample_rate",
            "samples_per_frame",
            "max_frames",
            "decoder_mode",
            "alignment_format",
            "normalization_format",
        }
        missing = sorted(required.difference(manifest))
        if missing:
            raise ValueError(f"Manifest is missing fields: {', '.join(missing)}")
        if manifest["decoder_mode"] != "exact":
            raise ValueError(
                f"Unsupported decoder mode: {manifest['decoder_mode']!r}"
            )
        if manifest["normalization_format"] != "instance_norm":
            raise ValueError(
                "Unsupported decoder normalization format: "
                f"{manifest['normalization_format']!r}"
            )
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

    def generate_with_details(
        self,
        text: str,
        voice: Optional[VoiceInput] = None,
    ) -> InferenceResult:
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
        input_lengths = np.asarray([tokens.shape[1]], dtype=np.int64)
        voice_array = self._voice_array(voice)

        encoder_inputs = {
            "tokens": tokens,
            "voice": voice_array,
        }
        if "input_lengths" in self.encoder_input_names:
            encoder_inputs["input_lengths"] = input_lengths
        acoustic_tokens, prosodic_tokens, durations = self.encoder_session.run(
            ["acoustic_tokens", "prosodic_tokens", "durations"],
            encoder_inputs,
        )
        durations = np.asarray(durations, dtype=np.int64)
        frame_count = int(durations.sum())
        self._validate_frame_count(frame_count)
        frame_to_token = self._build_frame_to_token(durations)
        decoder_inputs = {
            "acoustic_tokens": np.asarray(acoustic_tokens, dtype=np.float32),
            "voice": voice_array,
            "frame_to_token": frame_to_token,
        }
        if self.prosody_session is None:
            decoder_inputs["prosodic_tokens"] = np.asarray(
                prosodic_tokens,
                dtype=np.float32,
            )
        else:
            f0, energy = self.prosody_session.run(
                ["f0", "energy"],
                {
                    "prosodic_tokens": np.asarray(
                        prosodic_tokens,
                        dtype=np.float32,
                    ),
                    "frame_to_token": frame_to_token,
                    "voice": voice_array,
                },
            )
            decoder_inputs["f0"] = np.asarray(f0, dtype=np.float32)
            decoder_inputs["energy"] = np.asarray(energy, dtype=np.float32)

        audio, audio_lengths = self.decoder_session.run(
            ["audio", "audio_lengths"],
            decoder_inputs,
        )
        if not np.isfinite(audio).all():
            raise RuntimeError("Holos ONNX decoder produced NaN or Inf")

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

        return InferenceResult(
            audio=np.ascontiguousarray(waveform[:audio_length]),
            sample_rate=self.sample_rate,
            audio_length=audio_length,
            frame_count=frame_count,
            durations=np.ascontiguousarray(durations[0]),
        )

    def generate(
        self,
        text: str,
        voice: Optional[VoiceInput] = None,
    ) -> np.ndarray:
        return self.generate_with_details(text, voice).audio

    __call__ = generate
