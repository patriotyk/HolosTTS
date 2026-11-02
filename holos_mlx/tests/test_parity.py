"""MLX against the PyTorch model, stage by stage, with the source noise off.

Needs PyTorch, the repository's ``holos`` package and its checkpoint:

    python holos_mlx/tests/test_parity.py --weights holos.pt --voices voices.pt

The MLX model is converted into a temporary directory first.
"""

import argparse
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "holos_mlx" / "src"))
sys.path.insert(0, str(ROOT))

SENTENCES = (
    "ˈdʲakʊju.",
    "ˈbudʲ ˈɫaskɐ, zɐteɫefoˈnui̯te meˈnʲi ˈzau̯trɐ ˈpʲisʲlʲɐ oˈbʲidʊ, kɔˈɫɪ ˈmatɪmete ˈʋʲilʲnʊ xʋɪˈɫɪnkʊ.",
    "nɐʊˈkɔʋɐ kɔnfeˈrɛnʲt͡sʲijɐ ʋʲidˈbudet͡sʲːɐ nɐˈstupnɔɦɔ ˈtɪʒnʲɐ ˈu ʋeˈɫɪkʲii̯ ˈzalʲi ʊnʲiʋersɪˈtɛtʊ, "
    "ˈi ˈkɔʒen ɔˈxɔt͡ʃɪi̯ ˈzmɔʒe posˈɫuxɐtɪ ˈdɔpɔʋʲidʲi prɔʋʲidˈnɪx dosʲˈlʲidnɪkʲiu̯ z ˈrʲiznɪx krɐˈjin ˈsʲʋʲitʊ.",
)
ARGS = argparse.Namespace(weights="holos.pt", voices="voices.pt")


def _relative(actual, expected):
    return float(np.abs(actual - expected).max() / np.abs(expected).max())


def _snr(actual, expected):
    return float(10 * np.log10(np.sum(expected ** 2) / max(np.sum((actual - expected) ** 2), 1e-20)))


class ParityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch

        from holos.models import HolosTTS as TorchHolosTTS
        from holos_mlx import HolosTTS
        from holos_mlx.convert import convert

        cls.torch = torch
        cls.reference = TorchHolosTTS(ARGS.weights, device="cpu", voices_path=ARGS.voices).eval()
        cls.temp = tempfile.TemporaryDirectory()
        convert(ARGS.weights, ARGS.voices, Path(cls.temp.name))
        cls.model = HolosTTS(cls.temp.name)
        cls.voice = cls.model.list_voices()[0]

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def _torch(self, text, speed=1.0):
        torch = self.torch
        with torch.inference_mode(), mock.patch("holos.decoder.torch.randn_like", torch.zeros_like):
            inputs = self.reference._prepare_inputs(text, self.voice)
            aligned, f0, energy, _, _, durations = self.reference._encode(*inputs, speed=speed)
            audio = self.reference.forward(*inputs, speed=speed)
        return durations.numpy(), f0.numpy(), energy.numpy(), audio.reshape(-1).numpy()

    def _mlx(self, text, speed=1.0):
        encoded = self.model._encode(text, self.voice, speed)
        decoder = self.model.model.decoder
        source = decoder.generator.source(encoded.f0, noise=False)
        audio = decoder(encoded.acoustic, encoded.f0, encoded.energy, encoded.voice, source)
        return encoded.durations, np.array(encoded.f0), np.array(encoded.energy), np.array(audio[0])

    def test_stages_and_audio_match_pytorch(self):
        for text in SENTENCES:
            for speed in (1.0, 1.3):
                with self.subTest(tokens=len(text), speed=speed):
                    durations, f0, energy, audio = self._mlx(text, speed)
                    expected = self._torch(text, speed)
                    np.testing.assert_array_equal(durations, expected[0])
                    self.assertLess(_relative(f0, expected[1]), 1e-3)
                    self.assertLess(_relative(energy, expected[2]), 1e-3)
                    self.assertEqual(audio.shape, expected[3].shape)
                    self.assertGreater(_snr(audio, expected[3]), 40)

    def test_stream_concatenates_to_generate(self):
        import mlx.core as mx

        text = SENTENCES[2]
        mx.random.seed(0)
        whole = self.model.generate(text, self.voice)
        mx.random.seed(0)
        streamed = np.concatenate(list(self.model.stream(text, self.voice, trim_silence=False)))
        self.assertEqual(streamed.shape, whole.shape)
        self.assertGreater(_snr(streamed, whole), 40)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="holos.pt")
    parser.add_argument("--voices", default="voices.pt")
    ARGS, rest = parser.parse_known_args()
    unittest.main(argv=[sys.argv[0], *rest])
