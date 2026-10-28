import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


RUNTIME_SRC = Path(__file__).parents[1] / "src"
sys.path.insert(0, str(RUNTIME_SRC))

from holos_onnx.inference import HolosTTS  # noqa: E402
from holos_onnx.tokenizer import Tokenizer  # noqa: E402


class TokenizerTest(unittest.TestCase):
    def test_boundaries_and_unknown_symbols(self):
        tokenizer = Tokenizer(["$", "a", "b"])
        encoded = tokenizer.encode_with_boundaries("ab?ba")
        np.testing.assert_array_equal(encoded, np.asarray([0, 1, 2, 2, 1, 0]))


class ExactAlignmentTest(unittest.TestCase):
    def test_frame_to_token_has_no_padding_sentinel(self):
        durations = np.asarray([[2, 0, 3]], dtype=np.int64)
        frame_to_token = HolosTTS._build_frame_to_token(durations)
        np.testing.assert_array_equal(
            frame_to_token,
            np.asarray([0, 0, 2, 2, 2], dtype=np.int64),
        )

    def test_frame_count_validation_enforces_dynamic_limit(self):
        runtime = HolosTTS.__new__(HolosTTS)
        runtime.max_frames = 64
        runtime._tensorrt_enabled = False
        runtime.tensorrt_profile = None
        runtime._validate_frame_count(64)
        with self.assertRaises(ValueError):
            runtime._validate_frame_count(0)
        with self.assertRaises(ValueError):
            runtime._validate_frame_count(65)


class DependencyBoundaryTest(unittest.TestCase):
    def test_runtime_sources_do_not_import_torch(self):
        package_dir = RUNTIME_SRC / "holos_onnx"
        for source in package_dir.glob("*.py"):
            self.assertNotIn("import torch", source.read_text(encoding="utf-8"))


class _FakeInput:
    def __init__(self, name):
        self.name = name


class _FakeEncoderSession:
    def get_inputs(self):
        return [_FakeInput(name) for name in ("tokens", "voice", "speed")]

    def get_providers(self):
        return ["CPUExecutionProvider"]

    def run(self, output_names, inputs):
        token_count = inputs["tokens"].shape[1]
        acoustic = np.zeros((1, 512, token_count), dtype=np.float32)
        prosodic = np.zeros((1, token_count, 512), dtype=np.float32)
        speed = float(np.asarray(inputs["speed"]).reshape(-1)[0])
        duration = max(1, int(round(3.0 / speed)))
        durations = np.full((1, token_count), duration, dtype=np.int64)
        return acoustic, prosodic, durations


class _FakeProsodySession:
    def get_providers(self):
        return ["CPUExecutionProvider"]

    def run(self, output_names, inputs):
        frames = inputs["frame_to_token"].shape[0]
        return (
            np.zeros((1, 2 * frames), dtype=np.float32),
            np.zeros((1, 2 * frames), dtype=np.float32),
        )


class _FakeGatherDecoderSession:
    def get_inputs(self):
        names = ["acoustic_tokens", "frame_to_token", "f0", "energy", "voice"]
        return [_FakeInput(name) for name in names]

    def get_providers(self):
        return ["CPUExecutionProvider"]

    def run(self, output_names, inputs):
        decoder_frames = inputs["frame_to_token"].shape[0]
        if set(inputs) != {"acoustic_tokens", "frame_to_token", "f0", "energy", "voice"}:
            raise AssertionError(f"Unexpected decoder inputs: {sorted(inputs)}")
        if inputs["f0"].shape != (1, 2 * decoder_frames):
            raise AssertionError("Decoder must receive the prosody stage outputs")
        return (
            np.zeros((1, decoder_frames * 600), dtype=np.float32),
            np.asarray([decoder_frames * 600], dtype=np.int64),
        )


def _manifest(**overrides):
    manifest = {
        "format": "holos-onnx",
        "format_version": 2,
        "encoder": "encoder.onnx",
        "prosody": "prosody.onnx",
        "decoder": "decoder.onnx",
        "voices": "voices.npz",
        "vocab": ["$", "a"],
        "sample_rate": 24000,
        "samples_per_frame": 600,
        "max_frames": 64,
        "decoder_mode": "exact",
        "alignment_format": "gather",
        "tensorrt_profile": {
            "min_text_length": 2,
            "opt_text_length": 128,
            "max_text_length": 512,
            "min_frames": 2,
            "opt_frames": 256,
            "max_frames": 1000,
        },
    }
    manifest.update(overrides)
    return manifest


class RuntimeBundleTest(unittest.TestCase):
    def test_generate_loads_exact_length_bundle(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            bundle = Path(temp_dir)
            (bundle / "holos_onnx.json").write_text(json.dumps(_manifest()))
            np.savez_compressed(
                bundle / "voices.npz",
                names=np.asarray(["test"]),
                values=np.zeros((1, 256), dtype=np.float32),
            )

            sessions = [_FakeEncoderSession(), _FakeProsodySession(), _FakeGatherDecoderSession()]
            with mock.patch(
                "holos_onnx.inference.ort.InferenceSession",
                side_effect=sessions,
            ) as session_factory:
                runtime = HolosTTS(bundle)
                result = runtime.generate_with_details("aaaa", voice="test")

        loaded = [Path(call.args[0]).name for call in session_factory.call_args_list]
        self.assertEqual(loaded, ["encoder.onnx", "prosody.onnx", "decoder.onnx"])
        options = session_factory.call_args_list[0].kwargs["sess_options"]
        self.assertEqual(options.get_session_config_entry("session.intra_op.allow_spinning"), "0")

        # Four symbols plus two boundary tokens, each with duration 3.
        self.assertEqual(result.frame_count, 18)
        self.assertEqual(result.audio_length, 18 * 600)
        self.assertEqual(result.audio.shape, (18 * 600,))

    def test_generate_passes_speed_to_encoder(self):
        runtime = HolosTTS.__new__(HolosTTS)
        runtime.tokenizer = Tokenizer(["$", "a"])
        runtime._tensorrt_enabled = False
        runtime.tensorrt_profile = None
        runtime.streaming = None
        runtime.encoder_session = _FakeEncoderSession()
        runtime.prosody_session = _FakeProsodySession()
        runtime.decoder_session = _FakeGatherDecoderSession()
        runtime.max_frames = 64
        runtime.samples_per_frame = 600
        runtime.sample_rate = 24000
        runtime._voices = {"test": np.zeros((256,), dtype=np.float32)}

        result = runtime.generate_with_details(
            "aaaa",
            voice="test",
            speed=1.5,
        )

        # Six tokens, round(3 / 1.5) == 2 frames per token.
        self.assertEqual(result.frame_count, 12)

    def test_generate_rejects_non_positive_speed(self):
        runtime = HolosTTS.__new__(HolosTTS)
        with self.assertRaisesRegex(ValueError, "speed"):
            runtime.generate_with_details("a", speed=0)

    def test_legacy_manifest_version_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "holos_onnx.json"
            path.write_text(json.dumps(_manifest(format_version=1)))
            with self.assertRaisesRegex(ValueError, "format version"):
                HolosTTS._load_manifest(path)

    def test_two_stage_bundle_without_prosody_is_rejected(self):
        manifest = _manifest()
        del manifest["prosody"]
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "holos_onnx.json"
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "prosody"):
                HolosTTS._load_manifest(path)

    def test_tensorrt_profiles_cover_every_dynamic_stage_input(self):
        runtime = HolosTTS.__new__(HolosTTS)
        runtime.manifest = _manifest()
        runtime.streaming = None
        runtime.samples_per_frame = 600
        runtime.tensorrt_profile = runtime._load_tensorrt_profile()
        providers = [("TensorrtExecutionProvider", {"trt_fp16_enable": True}), "CUDAExecutionProvider"]

        decoder = runtime._configure_stage_providers(providers, stage="decoder")
        options = decoder[0][1]
        self.assertTrue(options["trt_fp16_enable"])
        self.assertEqual(
            options["trt_profile_max_shapes"],
            "acoustic_tokens:1x512x512,frame_to_token:1000,f0:1x2000,energy:1x2000",
        )
        self.assertEqual(decoder[1], "CUDAExecutionProvider")
        prosody = runtime._configure_stage_providers(providers, stage="prosody")[0][1]
        self.assertEqual(prosody["trt_profile_min_shapes"], "prosodic_tokens:1x2x512,frame_to_token:2")



_STREAMING = {
    "decoder": "decoder_window.onnx",
    "source": "source.onnx",
    "chunk_frames": 4,
    "context_frames": 2,
    "window_frames": 8,
}


class _FakeSourceSession:
    def get_providers(self):
        return ["CPUExecutionProvider"]

    def run(self, output_names, inputs):
        samples = inputs["f0"].shape[1] * 300
        return (np.arange(samples, dtype=np.float32)[None, :],)


class _FakeWindowDecoderSession:
    """Returns each window's excitation so chunks can be traced back."""

    def __init__(self):
        self.widths = []

    def get_inputs(self):
        names = ["acoustic_tokens", "frame_to_token", "f0", "energy", "source", "voice"]
        return [_FakeInput(name) for name in names]

    def get_providers(self):
        return ["CPUExecutionProvider"]

    def run(self, output_names, inputs):
        width = inputs["frame_to_token"].shape[0]
        self.widths.append(width)
        if inputs["source"].shape != (1, width * 600) or inputs["f0"].shape != (1, 2 * width):
            raise AssertionError("Window inputs must cover the same frames")
        return (inputs["source"],)


class StreamingBundleTest(unittest.TestCase):
    def test_windows_cover_each_frame_once_with_full_context(self):
        for frame_count in (1, 5, 8, 9, 23, 64):
            covered = []
            for start, end, window_start, window_end in HolosTTS._stream_windows(frame_count, 4, 2):
                covered.extend(range(start, end))
                self.assertLessEqual(window_end - window_start, 8)
                self.assertTrue(window_start == 0 or start - window_start >= 2)
                self.assertTrue(window_end == frame_count or window_end - end >= 2)
            self.assertEqual(covered, list(range(frame_count)))

    def _runtime(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            bundle = Path(temp_dir)
            (bundle / "holos_onnx.json").write_text(json.dumps(_manifest(streaming=_STREAMING)))
            np.savez_compressed(
                bundle / "voices.npz",
                names=np.asarray(["test"]),
                values=np.zeros((1, 256), dtype=np.float32),
            )
            self.decoder = _FakeWindowDecoderSession()
            sessions = [_FakeEncoderSession(), _FakeProsodySession(), _FakeSourceSession(), self.decoder]
            with mock.patch("holos_onnx.inference.ort.InferenceSession", side_effect=sessions) as factory:
                runtime = HolosTTS(bundle, streaming=True)
        loaded = [Path(call.args[0]).name for call in factory.call_args_list]
        self.assertEqual(loaded, ["encoder.onnx", "prosody.onnx", "source.onnx", "decoder_window.onnx"])
        return runtime

    def test_stream_chunks_concatenate_to_the_whole_excitation(self):
        runtime = self._runtime()
        chunks = list(runtime.stream("aaaa", voice="test", trim_silence=False))

        # Six tokens x 3 frames = 18 frames in chunks of 4.
        self.assertEqual([chunk.size // 600 for chunk in chunks], [4, 4, 4, 4, 2])
        np.testing.assert_array_equal(np.concatenate(chunks), np.arange(18 * 600, dtype=np.float32))
        self.assertEqual(set(self.decoder.widths), {8})
        result = runtime.generate_with_details("aaaa", voice="test")
        self.assertEqual(result.audio_length, 18 * 600)

    def test_stream_trims_boundary_silence_with_fade_in(self):
        runtime = self._runtime()
        audio = np.concatenate(list(runtime.stream("aaaa", voice="test")))

        # The boundary token lasts 3 frames (1800 samples); 10 ms (240) stay.
        expected = np.arange(18 * 600, dtype=np.float32)[1560:]
        expected[:120] *= np.arange(1, 121, dtype=np.float32) / 120
        np.testing.assert_allclose(audio, expected)

    def test_trim_spans_chunk_boundaries(self):
        chunks = [np.ones(100, dtype=np.float32), np.ones(100, dtype=np.float32)]
        trimmed = list(HolosTTS._trim_leading(iter(chunks), skip=150, fade=10))
        self.assertEqual([chunk.size for chunk in trimmed], [50])
        np.testing.assert_allclose(trimmed[0][:10], np.arange(1, 11) / 10)
        self.assertEqual(len(list(HolosTTS._trim_leading(iter(chunks), skip=250, fade=10))), 0)

    def test_smaller_chunks_are_allowed_but_not_larger(self):
        runtime = self._runtime()
        self.assertEqual(len(list(runtime.stream("aaaa", voice="test", chunk_frames=2, trim_silence=False))), 9)
        with self.assertRaisesRegex(ValueError, "chunk_frames"):
            list(runtime.stream("aaaa", voice="test", chunk_frames=5))

    def test_window_decoder_profile_is_optimized_for_full_windows(self):
        runtime = HolosTTS.__new__(HolosTTS)
        runtime.manifest = _manifest(streaming=_STREAMING)
        runtime.streaming = _STREAMING
        runtime.samples_per_frame = 600
        runtime.tensorrt_profile = runtime._load_tensorrt_profile()
        options = runtime._configure_stage_providers(["TensorrtExecutionProvider"], stage="decoder")[0][1]
        self.assertEqual(
            options["trt_profile_opt_shapes"],
            "acoustic_tokens:1x512x128,frame_to_token:8,f0:1x16,energy:1x16,source:1x4800",
        )
        source = runtime._configure_stage_providers(["TensorrtExecutionProvider"], stage="source")[0][1]
        self.assertEqual(source["trt_profile_max_shapes"], "f0:1x2000")

    def test_streaming_manifest_requires_all_streaming_fields(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "holos_onnx.json"
            path.write_text(json.dumps(_manifest(streaming={"chunk_frames": 4})))
            with self.assertRaisesRegex(ValueError, "streaming"):
                HolosTTS._load_manifest(path)


class ModeTest(unittest.TestCase):
    def _bundle(self, temp_dir, **overrides):
        bundle = Path(temp_dir)
        (bundle / "holos_onnx.json").write_text(json.dumps(_manifest(**overrides)))
        np.savez_compressed(bundle / "voices.npz", names=np.asarray(["test"]),
                            values=np.zeros((1, 256), dtype=np.float32))
        return bundle

    def test_default_mode_loads_only_the_whole_utterance_decoder(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            bundle = self._bundle(temp_dir, streaming=_STREAMING)
            sessions = [_FakeEncoderSession(), _FakeProsodySession(), _FakeGatherDecoderSession()]
            with mock.patch("holos_onnx.inference.ort.InferenceSession", side_effect=sessions) as factory:
                runtime = HolosTTS(bundle)
                audio = runtime.generate("aaaa", voice="test")
        loaded = [Path(call.args[0]).name for call in factory.call_args_list]
        self.assertEqual(loaded, ["encoder.onnx", "prosody.onnx", "decoder.onnx"])
        self.assertEqual(audio.size, 18 * 600)
        # Raised at the call, not on the first iteration.
        with self.assertRaisesRegex(RuntimeError, "streaming=True"):
            runtime.stream("aaaa", voice="test")

    def test_streaming_needs_a_bundle_with_a_window_decoder(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            bundle = self._bundle(temp_dir)
            with mock.patch("holos_onnx.inference.ort.InferenceSession") as factory:
                with self.assertRaisesRegex(ValueError, "no streaming decoder"):
                    HolosTTS(bundle, streaming=True)
        factory.assert_not_called()


class ModelResolutionTest(unittest.TestCase):
    def test_existing_local_path_is_used_without_download(self):
        with tempfile.TemporaryDirectory() as temp_dir, \
                mock.patch("huggingface_hub.snapshot_download") as download:
            self.assertEqual(HolosTTS._resolve_model(temp_dir, False, None, None, None), Path(temp_dir).resolve())
        download.assert_not_called()

    def test_hub_model_downloads_only_bundle_files_of_its_folder(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            bundle = Path(temp_dir) / "onnx" / "tensorrt"
            bundle.mkdir(parents=True)
            (bundle / "holos_onnx.json").write_text("{}")
            with mock.patch("huggingface_hub.snapshot_download", return_value=temp_dir) as download:
                resolved = HolosTTS._resolve_model("user/repo/onnx/tensorrt", False, "v1", None, "token")
                HolosTTS._resolve_model("user/repo/onnx/tensorrt", True, "v1", None, "token")
        self.assertEqual(resolved, bundle)
        (exact_args, exact), (_, streaming) = [(call.args, call.kwargs) for call in download.call_args_list]
        self.assertEqual(exact_args, ("user/repo",))
        self.assertEqual((exact["revision"], exact["token"]), ("v1", "token"))
        prefix = "onnx/tensorrt/"
        shared = {prefix + name for name in ("holos_onnx.json", "voices.npz", "encoder.onnx", "prosody.onnx")}
        self.assertEqual(set(exact["allow_patterns"]), shared | {prefix + "decoder.onnx"})
        self.assertEqual(set(streaming["allow_patterns"]),
                         shared | {prefix + "source.onnx", prefix + "decoder_window.onnx"})

    def test_repository_root_bundle_needs_no_folder(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            (Path(temp_dir) / "holos_onnx.json").write_text("{}")
            with mock.patch("huggingface_hub.snapshot_download", return_value=temp_dir) as download:
                self.assertEqual(HolosTTS._resolve_model("user/repo", False, None, None, None), Path(temp_dir))
        self.assertIn("encoder.onnx", download.call_args.kwargs["allow_patterns"])

    def test_missing_model_is_a_clear_error(self):
        from huggingface_hub.utils import RepositoryNotFoundError

        missing = RepositoryNotFoundError.__new__(RepositoryNotFoundError)
        with mock.patch("huggingface_hub.snapshot_download", side_effect=missing):
            with self.assertRaisesRegex(FileNotFoundError, "neither a local path"):
                HolosTTS._resolve_model("user/missing", False, None, None, None)
        with tempfile.TemporaryDirectory() as temp_dir, \
                mock.patch("huggingface_hub.snapshot_download", return_value=temp_dir):
            with self.assertRaisesRegex(FileNotFoundError, "No Holos ONNX bundle"):
                HolosTTS._resolve_model("user/repo/onnx", False, None, None, None)
        with mock.patch("huggingface_hub.snapshot_download") as download:
            with self.assertRaises(FileNotFoundError):
                HolosTTS._resolve_model("missing_bundle", False, None, None, None)
        download.assert_not_called()


if __name__ == "__main__":
    unittest.main()
