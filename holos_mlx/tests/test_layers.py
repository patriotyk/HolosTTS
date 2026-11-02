"""Tests of the MLX building blocks that need neither PyTorch nor a checkpoint."""

import sys
import unittest
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from holos_mlx.decoder import STFT, atan2_approx  # noqa: E402
from holos_mlx.inference import HolosTTS  # noqa: E402
from holos_mlx.layers import pixel_shuffle  # noqa: E402


class LayerTest(unittest.TestCase):
    def test_stft_round_trip(self):
        signal = np.random.default_rng(0).normal(size=(1, 6000)).astype(np.float32)
        stft = STFT(40, 10)
        magnitude, phase = stft.transform(mx.array(signal))
        self.assertEqual(magnitude.shape, (1, 601, 21))
        restored = np.array(stft.inverse(magnitude, phase))
        np.testing.assert_allclose(restored, signal, atol=2e-3)

    def test_atan2_approx(self):
        imag, real = np.random.default_rng(1).normal(size=(2, 1000)).astype(np.float32)
        angle = np.array(atan2_approx(mx.array(imag), mx.array(real)))
        np.testing.assert_allclose(angle, np.arctan2(imag, real), atol=2e-5)

    def test_pixel_shuffle_moves_channel_groups_to_time(self):
        # Channel c * 2 + j at time t lands on time t * 2 + j, channel c.
        x = np.arange(2 * 4).reshape(1, 2, 4).astype(np.float32)
        out = np.array(pixel_shuffle(mx.array(x), 2))
        np.testing.assert_array_equal(out[0, :, 0], [0, 1, 4, 5])
        np.testing.assert_array_equal(out[0, :, 1], [2, 3, 6, 7])


class StreamingTest(unittest.TestCase):
    def test_windows_cover_each_frame_once_with_context(self):
        for frame_count in (1, 5, 40, 41, 123):
            covered = []
            for start, end, window_start, window_end in HolosTTS._stream_windows(frame_count, 40, 20):
                covered.extend(range(start, end))
                self.assertEqual((window_start, window_end), (max(0, start - 20), min(frame_count, end + 20)))
            self.assertEqual(covered, list(range(frame_count)))

    def test_trim_spans_chunk_boundaries(self):
        chunks = [np.ones(100, dtype=np.float32), np.ones(100, dtype=np.float32)]
        trimmed = list(HolosTTS._trim_leading(iter(chunks), skip=150, fade=10))
        self.assertEqual([chunk.size for chunk in trimmed], [50])
        np.testing.assert_allclose(trimmed[0][:10], np.arange(1, 11) / 10)


if __name__ == "__main__":
    unittest.main()
