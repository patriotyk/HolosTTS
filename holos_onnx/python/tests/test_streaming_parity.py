"""The runtime's streaming helpers must match holos/streaming.py.

holos-onnx cannot import holos (it needs torch), so it keeps a copy of the
window and trimming code. This test loads the repository's module by path
and fails when the two diverge.
"""
import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from holos_onnx import inference  # noqa: E402
from holos_onnx.inference import HolosTTS  # noqa: E402

SHARED = Path(__file__).resolve().parents[3] / "holos" / "streaming.py"


def _load_shared():
    spec = importlib.util.spec_from_file_location("holos_streaming", SHARED)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@unittest.skipUnless(SHARED.is_file(), "needs the HolosTTS repository checkout")
class StreamingParityTest(unittest.TestCase):
    def setUp(self):
        self.shared = _load_shared()

    def test_windows_match(self):
        for frame_count in range(1, 200):
            for chunk_frames in (1, 7, 20, 40):
                for context_frames in (0, 3, 20):
                    for fixed_width in (False, True):
                        args = (frame_count, chunk_frames, context_frames, fixed_width)
                        self.assertEqual(
                            list(HolosTTS._stream_windows(*args)),
                            list(self.shared.stream_windows(*args)),
                        )

    def test_trimming_constants_match(self):
        self.assertEqual(inference._SILENCE_MARGIN_SECONDS, self.shared.SILENCE_MARGIN_SECONDS)
        self.assertEqual(inference._FADE_IN_SAMPLES, self.shared.FADE_IN_SAMPLES)

    def test_trimming_matches(self):
        rng = np.random.default_rng(0)
        sizes = [37, 600, 5, 1200, 90]
        chunks = [rng.standard_normal(size).astype(np.float32) for size in sizes]
        for skip in (0, 1, 30, 37, 640, 960, 1900, 5000):
            fade = inference._FADE_IN_SAMPLES if skip else 0
            runtime = list(HolosTTS._trim_leading(iter(chunks), skip, fade))
            shared = list(self.shared.trim_leading(iter(chunks), skip))
            self.assertEqual([chunk.size for chunk in runtime], [chunk.size for chunk in shared])
            for actual, expected in zip(runtime, shared):
                np.testing.assert_array_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()
