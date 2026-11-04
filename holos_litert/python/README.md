# holos-tts-litert

HolosTTS with [LiteRT](https://ai.google.dev/edge/litert) in Python, for the
model in the [`litert/`](https://huggingface.co/patriotyk/HolosTTS/tree/main/litert)
folder on Hugging Face, the same one the [Kotlin library](https://github.com/patriotyk/HolosTTS/blob/main/holos_litert/kotlin/README.md)
runs on Android GPUs.

I use it to check that model on a computer, and as the reference for the
ports: the code between the graphs (embedding lookups, buckets, rounding the
durations, frame alignment, harmonic excitation, decoder windows, trimming
the leading silence) is easier to follow and to compare here. To synthesize
on a computer, my other packages are faster: [MLX](https://github.com/patriotyk/HolosTTS/blob/main/holos_mlx/README.md)
on Apple silicon, ONNX Runtime or TensorRT elsewhere.

## Install

```bash
python -m pip install holos-tts-litert
```

From a checkout of the repository, install `./holos_litert/python` instead.
The import name is `holos_litert`. It needs Python 3.10+ and
[ai-edge-litert](https://pypi.org/project/ai-edge-litert/), which has
packages for Linux, macOS and Windows.

## Quick start

```python
from holos_litert import HolosTTS

model = HolosTTS()  # patriotyk/HolosTTS/litert, on the CPU
audio = model.generate(phonemes, voice=model.list_voices()[0], speed=1.0)
# float32 samples at model.sample_rate (24 kHz)
```

`HolosTTS(model)` loads a local model directory when that path exists.
Otherwise `model` names a Hugging Face model: `owner/repo`, optionally
followed by the model's folder in the repository. Only the model files are
downloaded, into the Hugging Face cache; `revision`, `cache_dir` and `token`
control the download.

The input is a phoneme string in the model's vocabulary, as for the other
packages. `voice` is a preset name or a 256-value voice vector. One utterance
can have up to 510 phonemes and 1000 frames (25 s of speech).

By default the graphs run on 4 CPU threads in FP32. `HolosTTS(gpu=True)`
runs them on the GPU in FP16, as on Android (`fp16=False` for FP32). On an
Apple M2, with Metal, the 10 s sentence from the
[benchmarks](https://github.com/patriotyk/HolosTTS#benchmarks) takes:

| | RTF | First chunk of `stream()` |
| --- | --- | --- |
| CPU, FP32 | 0.26 | 0.79 s |
| GPU, FP16 | 0.029 | 0.09 s |

## Streaming

```python
for chunk in model.stream(phonemes, voice="my_voice"):
    play(chunk)
```

The first chunk of 1 s comes from an 80-frame decoder window, the later ones
of 4 s from 200-frame windows, and every window has 20 frames of context on
both sides, so the chunks concatenate to the `generate()` audio. Unlike the
Kotlin library, which stays with the small windows on a GPU too slow for the
large ones, the windows here do not depend on timing, so the output is
reproducible. `chunk_frames` (1-40, 40 = 1 s) decodes every chunk in a small
window instead. `stream()` drops the boundary-token silence at the start
except 10 ms and fades in over 5 ms; pass `trim_silence=False` to keep it.

`noise=False` in `generate()` and `stream()` turns off the source noise, so
the same text gives the same audio.

## Tests

```bash
python holos_litert/python/tests/test_inference.py
```

The tests download the model, or load it from `HOLOS_LITERT_MODEL`, and check
on the CPU that the streamed chunks join into the `generate()` audio sample
for sample.
