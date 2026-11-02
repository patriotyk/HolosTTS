# holos-tts-mlx

HolosTTS on Apple silicon with [MLX](https://github.com/ml-explore/mlx): the
whole model runs on the GPU, without PyTorch, with optional low-latency
streaming.

## Install

```bash
python -m pip install holos-tts-mlx
```

From a checkout of the repository, install `./holos_mlx` instead. The import
name is `holos_mlx`. It needs a Mac with Apple silicon and Python 3.10+.

## Quick start

```python
from holos_mlx import HolosTTS

model = HolosTTS()  # patriotyk/HolosTTS/mlx
audio = model.generate(phonemes, voice=model.list_voices()[0], speed=1.0)
# float32 samples at model.sample_rate (24 kHz)
```

`HolosTTS(model)` loads a local model directory when that path exists.
Otherwise `model` names a Hugging Face model: `owner/repo`, optionally
followed by the model's folder in the repository. Only the model files are
downloaded, into the Hugging Face cache, so later runs work offline;
`revision`, `cache_dir` and `token` control the download.

The input is a phoneme string in the model's vocabulary; the Ukrainian text
frontend (stress marks and IPA conversion) is not part of this package, see
the [repository README](https://github.com/patriotyk/HolosTTS#readme).
`speed` above one speaks faster, below one slower. `voice` is a preset name
or a 256-value voice vector, such as one from the PyTorch `VoiceEncoder`.

The model computes in FP32 and matches the PyTorch model: the same durations,
F0 and energy within 1e-5, and the audio 43-50 dB SNR from it with the source
noise turned off. MLX handles every text length as it is, so there are no
length buckets, and loading takes a fraction of a second.

## Streaming

```python
for chunk in model.stream(phonemes, voice="my_voice"):
    play(chunk)
```

Text encoding and prosody run once for the whole text; then each chunk of
`chunk_frames` frames (40 by default, 1 s of audio) is decoded with 20 frames
of context on both sides, so the chunks concatenate to the `generate` audio.
On an Apple M2 the first chunk is ready after about 65 ms; see the
[benchmarks](https://github.com/patriotyk/HolosTTS#benchmarks). `stream`
drops the boundary-token silence at the start (about 50 ms) except 10 ms and
fades in over 5 ms; pass `trim_silence=False` to keep it. `generate` decodes
the whole utterance in one pass and is faster when the phrase can be
generated before playback.

## Converting a checkpoint

The model directory has `config.json` (vocabulary, sample rate, streaming
context), `model.safetensors` and `voices.npz`. To make one from a PyTorch
checkpoint, with PyTorch and the repository's `holos` package installed:

```bash
python -m holos_mlx.convert --weights holos.pt --voices voices.pt --output holos_mlx_model
```

Weight norm is folded into the weights and the convolutions are stored in
MLX's channels-last layout.

## Tests

```bash
python holos_mlx/tests/test_layers.py
python holos_mlx/tests/test_parity.py --weights holos.pt --voices voices.pt  # needs PyTorch
```

The parity test converts the checkpoint and compares every stage with the
PyTorch model on three sentences at two speeds.

## License

MIT, see [LICENSE](https://github.com/patriotyk/HolosTTS/blob/main/holos_mlx/LICENSE).
