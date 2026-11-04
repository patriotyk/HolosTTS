"""The LiteRT runtime on the CPU, with the source noise off.

Loads the model from HOLOS_LITERT_MODEL, a directory or a Hugging Face
model, by default patriotyk/HolosTTS/litert (downloaded once, 270 MB):

    python holos_litert/python/tests/test_inference.py
"""

import os
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from holos_litert import HolosTTS  # noqa: E402
from holos_litert.inference import DEFAULT_MODEL  # noqa: E402

SHORT = "ˈdʲakʊju."
LONG = (
    "nɐʊˈkɔʋɐ kɔnfeˈrɛnʲt͡sʲijɐ ʋʲidˈbudet͡sʲːɐ nɐˈstupnɔɦɔ ˈtɪʒnʲɐ ˈu ʋeˈɫɪkʲii̯ ˈzalʲi ʊnʲiʋersɪˈtɛtʊ, "
    "ˈi ˈkɔʒen ɔˈxɔt͡ʃɪi̯ ˈzmɔʒe posˈɫuxɐtɪ ˈdɔpɔʋʲidʲi prɔʋʲidˈnɪx dosʲˈlʲidnɪkʲiu̯ z ˈrʲiznɪx krɐˈjin ˈsʲʋʲitʊ."
)


class InferenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = HolosTTS(os.environ.get("HOLOS_LITERT_MODEL", DEFAULT_MODEL))

    def assert_joins(self, text, **options):
        """The streamed chunks are the generate() audio without the trimmed
        silence, after the fade-in."""
        whole = self.model.generate(text, noise=False)
        chunks = list(self.model.stream(text, noise=False, **options))
        streamed = np.concatenate(chunks)
        offset = whole.size - streamed.size
        self.assertGreater(offset, 0)
        np.testing.assert_array_equal(streamed[120:], whole[offset + 120:])
        return chunks

    def test_short_text_is_one_chunk(self):
        self.assertEqual(len(self.assert_joins(SHORT)), 1)

    def test_stream_switches_to_large_windows(self):
        chunks = self.assert_joins(LONG)
        seconds = [chunk.size / self.model.sample_rate for chunk in chunks]
        self.assertLess(seconds[0], 1.0)
        self.assertEqual(seconds[1], 4.0)

    def test_chunk_frames(self):
        chunks = self.assert_joins(LONG, chunk_frames=20)
        self.assertEqual(chunks[1].size, 20 * self.model.samples_per_frame)

    def test_noise_changes_the_audio(self):
        self.assertFalse(np.array_equal(self.model.generate(SHORT), self.model.generate(SHORT, noise=False)))

    def test_voices(self):
        voices = self.model.list_voices()
        self.assertTrue(voices)
        vector = self.model._voices[voices[0]]
        np.testing.assert_array_equal(self.model.generate(SHORT, voices[0], noise=False),
                                      self.model.generate(SHORT, vector, noise=False))
        with self.assertRaises(ValueError):
            self.model.generate(SHORT, "no such voice")

    def test_limits(self):
        with self.assertRaises(ValueError):
            self.model.generate(SHORT, speed=0)
        with self.assertRaisesRegex(ValueError, "split the text"):
            self.model.generate(SHORT * 60)
        with self.assertRaises(ValueError):
            list(self.model.stream(SHORT, chunk_frames=41))


if __name__ == "__main__":
    unittest.main()
