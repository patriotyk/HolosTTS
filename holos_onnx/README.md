# holos-tts-onnx

PyTorch-free inference for HolosTTS on ONNX Runtime: CPU, CUDA and TensorRT,
with optional low-latency streaming.

## Install

```bash
python -m pip install holos-tts-onnx              # CPU and CUDA
python -m pip install "holos-tts-onnx[tensorrt]"  # plus TensorRT
```

From a checkout of the repository, install `./holos_onnx` instead. The import
name is `holos_onnx`.

On Linux and Windows this installs `onnxruntime-gpu` with its CUDA and cuDNN
libraries; on macOS it installs the CPU `onnxruntime`. The libraries come from
NVIDIA's pip packages, so no system CUDA or `LD_LIBRARY_PATH` setup is needed.
Python 3.9+ and ONNX Runtime 1.28+ are required.

## Quick start

```python
from holos_onnx import HolosTTS

model = HolosTTS()  # the default bundle, patriotyk/HolosTTS/onnx
audio = model.generate(phonemes, voice=model.list_voices()[0], speed=1.0)
# float32 samples at model.sample_rate (24 kHz)
```

`HolosTTS(model)` loads a local bundle directory when that path exists.
Otherwise `model` names a Hugging Face model: `owner/repo`, optionally
followed by the bundle's folder in the repository, such as
`patriotyk/HolosTTS/onnx`. Only the bundle files are downloaded, into the
Hugging Face cache, so later runs work offline; `revision`, `cache_dir` and
`token` control the download.

The input is a phoneme string in the model's vocabulary; the Ukrainian text
frontend (stress marks and IPA conversion) is not part of this package, see
the [repository README](https://github.com/patriotyk/HolosTTS#readme).
`speed` above one speaks faster, below one slower.

## Models

| Model | Use |
| --- | --- |
| `patriotyk/HolosTTS/onnx` | FP32 for CPU, CUDA and TensorRT (the default). |
| `patriotyk/HolosTTS/onnx-int8` | INT8 for CPU: about 1.5x faster than FP32 and half the download, sounds the same. |

## GPU

```python
model = HolosTTS(providers=["CUDAExecutionProvider"])
```

TensorRT uses the same default model. FP16 with a persistent engine cache is
the fastest option:

```python
model = HolosTTS(
    providers=[(
        "TensorrtExecutionProvider",
        {
            "trt_fp16_enable": True,
            "trt_engine_cache_enable": True,
            "trt_engine_cache_path": "trt_cache",
            "trt_timing_cache_enable": True,
            "trt_timing_cache_path": "trt_cache",
        },
    )],
)
```

The first load builds the engines, which takes a few minutes; later loads reuse
the cache. CUDA and CPU are added as fallbacks automatically. Each stage gets
its own TensorRT optimization profile from the bundle's manifest (2–512 tokens
and up to `max_frames` frames by default), and longer inputs are rejected
before inference.

## Streaming

Load the model with `streaming=True` to get audio chunk by chunk:

```python
model = HolosTTS("patriotyk/HolosTTS/onnx-int8", streaming=True)
for chunk in model.stream(phonemes, voice="my_voice"):
    play(chunk)
```

Streaming uses a window decoder: each chunk of `chunk_frames` frames (40 by
default, 1 s of audio) is decoded with 20 frames of context on both sides, so
the chunks concatenate to the same audio as decoding the whole utterance. On
CPU the windows at the utterance edges are shorter, so the first chunk decodes
fewer frames; on GPU every window has the full width, since a new input shape
costs the CUDA provider more than the extra frames. The first chunk is ready
after about 3 ms with TensorRT FP16 on an RTX 5090, or about 200 ms in INT8 on
an Apple M2 CPU; see the
[benchmarks](https://github.com/patriotyk/HolosTTS#benchmarks). When the whole
phrase can be generated before playback, `generate` without streaming is
faster, since streaming decodes the context of every chunk again. `stream`
drops the boundary-token silence at the start (about 50 ms) except 10 ms and
fades in over 5 ms; pass `trim_silence=False` to keep it.

`stream` raises an error on a model loaded without `streaming=True`.
`generate` works in both modes and always returns the full utterance.

## Bundle format

A bundle is a directory with `holos_onnx.json` (vocabulary, sample rate and
stage files), `voices.npz` (preset voice embeddings) and the stage graphs:

- `encoder.onnx`: tokens → token features and integer durations;
- `prosody.onnx`: token features gathered per frame → F0 and energy;
- `decoder.onnx`: acoustic features, F0 and energy → audio of the whole
  utterance (default mode);
- `source.onnx` and `decoder_window.onnx` (streaming mode): the harmonic
  excitation for the whole utterance, and the decoder for one window of it.

Only the files of the selected mode are downloaded and loaded. The runtime
builds the frame-to-token index between stages, and every stage runs on the
exact predicted length. Bundles from exporters older than the three-stage
format are rejected.

Without `session_options`, the runtime disables ONNX Runtime's intra-op thread
spinning: the stages run one after another, and spinning threads of a finished
stage would slow the next one down on CPU.

## Tests

```bash
python holos_onnx/tests/test_inference.py
python holos_onnx/tests/test_streaming_parity.py  # needs the repository checkout
```

## License

MIT, see [LICENSE](https://github.com/patriotyk/HolosTTS/blob/main/holos_onnx/LICENSE).
