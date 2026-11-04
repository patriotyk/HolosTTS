# HolosTTS

HolosTTS is an open-source Ukrainian text-to-speech model, with the best balance of quality and speed (the first audio chunk in 3 ms on an RTX 5090 or in 5 ms on an RTX 3090). It has preset voices and clones a voice from an audio sample. It produces mono audio at **24 kHz** and supports streaming. It runs on PyTorch, or without PyTorch on ONNX Runtime (CPU including INT8, CUDA and TensorRT), on MLX on Apple silicon, and on Android: on the GPU with LiteRT or on the CPU with ONNX Runtime.

[Demo](https://huggingface.co/spaces/patriotyk/HolosTTS) · [Models on Hugging Face](https://huggingface.co/patriotyk/HolosTTS) · [Code on GitHub](https://github.com/patriotyk/HolosTTS)

| Python package | Use |
| --- | --- |
| [`holos-tts`](https://pypi.org/project/holos-tts/) | PyTorch inference and voice cloning. |
| [`holos-tts-onnx`](https://pypi.org/project/holos-tts-onnx/) | ONNX Runtime inference without PyTorch: CPU, CUDA, TensorRT. |
| [`holos-tts-mlx`](https://pypi.org/project/holos-tts-mlx/) | MLX inference without PyTorch on Apple silicon. |
| [`holos-tts-litert`](https://pypi.org/project/holos-tts-litert/) | LiteRT inference on CPU or GPU, the reference for the Android library. |

## Phonemes

Both packages take **phonemes**, not text. Stress marks and phonetic transcription for Ukrainian come from separate packages on GitHub (Git is required):

```bash
python -m pip install git+https://github.com/patriotyk/ukrainian-word-stress.git \
                      git+https://github.com/patriotyk/ipa-uk.git
```

```python
from ipa_uk import ipa
from ukrainian_word_stress import Stressifier

stressify = Stressifier()
phonemes = ipa(stressify("Привіт! Це український синтез мовлення."))
```

Synthesize long texts sentence by sentence.

## PyTorch

```bash
python -m pip install holos-tts
```

NVIDIA GPUs need a CUDA build of PyTorch. The model downloads `holos.pt` and `voices.pt` from Hugging Face and caches them locally.

```python
import torch
import soundfile as sf
from holos.models import HolosTTS

device = "cuda" if torch.cuda.is_available() else "cpu"
model = HolosTTS("patriotyk/HolosTTS", device=device)

print(model.list_voices())
voice = model.list_voices()[0]
audio = model.generate(phonemes, voice=voice, speed=1.0)
sf.write("speech.wav", audio.squeeze().cpu().numpy(), 24000)
```

`speed` above one speaks faster, below one slower. For local files, use `HolosTTS("holos.pt", device=device, voices_path="voices.pt")`.

### Streaming

`stream()` yields audio in chunks of `chunk_frames` frames (40 frames = 1 s by default) as soon as each one is ready. The chunks concatenate to the same audio as `generate()`:

```python
for chunk in model.stream(phonemes, voice=voice):
    play(chunk.squeeze().cpu().numpy())  # tensor [1, samples]
```

Every utterance starts with the silence of a boundary token (about 50 ms). `stream()` drops it except 10 ms and fades in over 5 ms to avoid a click; `trim_silence=False` keeps it. A smaller `chunk_frames` gets the first audio out sooner but takes longer overall, since every chunk is decoded with context on both sides.

### Voice cloning

`reference.wav` is a recording of the target voice without music or other speakers:

```python
from holos.models import VoiceEncoder

encoder = VoiceEncoder("patriotyk/HolosTTS", device=device)
with torch.inference_mode():
    custom_voice = encoder("reference.wav")
audio = model.generate(phonemes, voice=custom_voice)
sf.write("custom_voice.wav", audio.squeeze().cpu().numpy(), 24000)
```

`voice_encoder.pt` is downloaded automatically; preset voices do not need it. The voice is a vector of 256 numbers that you can save and pass to the ONNX runtime as `custom_voice.squeeze().cpu().numpy()`.

## ONNX Runtime

`holos-tts-onnx` runs the model on ONNX Runtime without PyTorch: on CPU (FP32 or INT8), CUDA and TensorRT, with the same streaming as above.

```bash
python -m pip install holos-tts-onnx              # CPU and CUDA
python -m pip install "holos-tts-onnx[tensorrt]"  # plus TensorRT
```

See the [package README](https://github.com/patriotyk/HolosTTS/blob/main/holos_onnx/python/README.md) for usage, the models and GPU setup.

For Android, [`holos_onnx/kotlin/`](https://github.com/patriotyk/HolosTTS/tree/main/holos_onnx/kotlin) has a Kotlin port of this package with a demo app. It runs the same bundles on the phone's CPU, INT8 being the faster one there. See its [README](https://github.com/patriotyk/HolosTTS/blob/main/holos_onnx/kotlin/README.md) for usage.

## MLX (Apple silicon)

`holos-tts-mlx` runs the whole model on the GPU of a Mac with [MLX](https://github.com/ml-explore/mlx), in FP32 and without PyTorch, with the same streaming. It is the fastest option on a Mac and loads in a fraction of a second.

```bash
python -m pip install holos-tts-mlx
```

See the [package README](https://github.com/patriotyk/HolosTTS/blob/main/holos_mlx/README.md) for usage.

## Android (LiteRT GPU)

The [`holos_litert/kotlin/`](https://github.com/patriotyk/HolosTTS/tree/main/holos_litert/kotlin) folder has a library that runs the model on the phone's GPU with [LiteRT](https://ai.google.dev/edge/litert), and a demo app. On a Pixel 7 it synthesizes with an RTF of 0.13 and gives the first streaming chunk in about 0.4 s. See its [README](https://github.com/patriotyk/HolosTTS/blob/main/holos_litert/kotlin/README.md) for usage. On phones without a suitable GPU, the [ONNX library](https://github.com/patriotyk/HolosTTS/tree/main/holos_onnx/kotlin) runs on the CPU.

The same model also runs in Python, in the `holos-tts-litert` package from [`holos_litert/python/`](https://github.com/patriotyk/HolosTTS/tree/main/holos_litert/python) (`pip install holos-tts-litert`). I use it to check the model on a computer and as the reference for the ports.

## Benchmarks

- **RTF** (real-time factor) is for synthesizing a whole sentence (`generate()` in HolosTTS): the time it takes divided by the duration of the audio. 0.01 means one second of speech in 10 ms; lower is faster.
- **First audio** is the delay a listener notices: for HolosTTS, the first chunk of streaming with `stream()`. The models marked * do not stream within a sentence, so their first audio is the whole sentence.

Three sentences of 5–12 s of speech, the median of 10 runs (5 on CPU) after a warm-up; the first audio is averaged over the sentences. Every sample says "Будь ласка, зателефонуйте мені завтра після обіду, коли матимете вільну хвилинку."

### NVIDIA RTX 5090

Only HolosTTS so far; the samples are in the RTX 3090 table.

| Model | Runtime | RTF | First audio |
| --- | --- | --- | --- |
| **HolosTTS** | **ONNX, TensorRT FP16** | **0.0005** | **3 ms** |
| **HolosTTS** | **ONNX, TensorRT FP32** | **0.0012** | **6 ms** |
| **HolosTTS** | **ONNX, CUDA** | **0.0019** | **11 ms** |
| **HolosTTS** | **PyTorch** | **0.0094** | **58 ms** |

### NVIDIA RTX 3090

| Model | Runtime | RTF | First audio | Sample |
| --- | --- | --- | --- | --- |
| **HolosTTS** | **ONNX, TensorRT FP16** | **0.0013** | **5 ms** | <audio controls preload="none" src="https://huggingface.co/patriotyk/HolosTTS/resolve/main/samples/holos-onnx-tensorrt-fp16.wav"><a href="https://huggingface.co/patriotyk/HolosTTS/blob/main/samples/holos-onnx-tensorrt-fp16.wav">Listen</a></audio> |
| **HolosTTS** | **ONNX, TensorRT FP32** | **0.0031** | **12 ms** | <audio controls preload="none" src="https://huggingface.co/patriotyk/HolosTTS/resolve/main/samples/holos-onnx-tensorrt-fp32.wav"><a href="https://huggingface.co/patriotyk/HolosTTS/blob/main/samples/holos-onnx-tensorrt-fp32.wav">Listen</a></audio> |
| **HolosTTS** | **ONNX, CUDA** | **0.0039** | **14 ms** | <audio controls preload="none" src="https://huggingface.co/patriotyk/HolosTTS/resolve/main/samples/holos-onnx-cuda.wav"><a href="https://huggingface.co/patriotyk/HolosTTS/blob/main/samples/holos-onnx-cuda.wav">Listen</a></audio> |
| Supertonic 3 | ONNX, CUDA | 0.0052 | 45 ms* | <audio controls preload="none" src="https://huggingface.co/patriotyk/HolosTTS/resolve/main/samples/supertonic3-cuda.wav"><a href="https://huggingface.co/patriotyk/HolosTTS/blob/main/samples/supertonic3-cuda.wav">Listen</a></audio> |
| StyleTTS2 single-speaker | PyTorch | 0.0065 | 55 ms* | <audio controls preload="none" src="https://huggingface.co/patriotyk/HolosTTS/resolve/main/samples/styletts2-single-pytorch-cuda.wav"><a href="https://huggingface.co/patriotyk/HolosTTS/blob/main/samples/styletts2-single-pytorch-cuda.wav">Listen</a></audio> |
| Kokoro-82M (English) | PyTorch | 0.0067 | 45 ms* | <audio controls preload="none" src="https://huggingface.co/patriotyk/HolosTTS/resolve/main/samples/kokoro-cuda.wav"><a href="https://huggingface.co/patriotyk/HolosTTS/blob/main/samples/kokoro-cuda.wav">Listen</a></audio> |
| **HolosTTS** | **PyTorch** | **0.011** | **65 ms** | <audio controls preload="none" src="https://huggingface.co/patriotyk/HolosTTS/resolve/main/samples/holos-pytorch-cuda.wav"><a href="https://huggingface.co/patriotyk/HolosTTS/blob/main/samples/holos-pytorch-cuda.wav">Listen</a></audio> |
| StyleTTS2 multi-speaker | PyTorch | 0.013 | 105 ms* | <audio controls preload="none" src="https://huggingface.co/patriotyk/HolosTTS/resolve/main/samples/styletts2-multi-pytorch-cuda.wav"><a href="https://huggingface.co/patriotyk/HolosTTS/blob/main/samples/styletts2-multi-pytorch-cuda.wav">Listen</a></audio> |
| StyleTTS2 single-speaker | ONNX, CUDA | 0.013 | 112 ms* | <audio controls preload="none" src="https://huggingface.co/patriotyk/HolosTTS/resolve/main/samples/styletts2-single-onnx-cuda.wav"><a href="https://huggingface.co/patriotyk/HolosTTS/blob/main/samples/styletts2-single-onnx-cuda.wav">Listen</a></audio> |

### Intel Core i7-13700K

| Model | Runtime | RTF | First audio | Sample |
| --- | --- | --- | --- | --- |
| **HolosTTS** | **ONNX INT8** | **0.044** | **110 ms** | <audio controls preload="none" src="https://huggingface.co/patriotyk/HolosTTS/resolve/main/samples/holos-onnx-int8-cpu.wav"><a href="https://huggingface.co/patriotyk/HolosTTS/blob/main/samples/holos-onnx-int8-cpu.wav">Listen</a></audio> |
| **HolosTTS** | **ONNX FP32** | **0.066** | **147 ms** | <audio controls preload="none" src="https://huggingface.co/patriotyk/HolosTTS/resolve/main/samples/holos-onnx-cpu.wav"><a href="https://huggingface.co/patriotyk/HolosTTS/blob/main/samples/holos-onnx-cpu.wav">Listen</a></audio> |
| **HolosTTS** | **PyTorch** | **0.074** | **196 ms** | <audio controls preload="none" src="https://huggingface.co/patriotyk/HolosTTS/resolve/main/samples/holos-pytorch-cpu.wav"><a href="https://huggingface.co/patriotyk/HolosTTS/blob/main/samples/holos-pytorch-cpu.wav">Listen</a></audio> |
| StyleTTS2 single-speaker | ONNX | 0.132 | 1104 ms* | <audio controls preload="none" src="https://huggingface.co/patriotyk/HolosTTS/resolve/main/samples/styletts2-single-onnx-cpu.wav"><a href="https://huggingface.co/patriotyk/HolosTTS/blob/main/samples/styletts2-single-onnx-cpu.wav">Listen</a></audio> |
| Kokoro-82M (English) | PyTorch | 0.137 | 923 ms* | <audio controls preload="none" src="https://huggingface.co/patriotyk/HolosTTS/resolve/main/samples/kokoro-cpu.wav"><a href="https://huggingface.co/patriotyk/HolosTTS/blob/main/samples/kokoro-cpu.wav">Listen</a></audio> |
| StyleTTS2 single-speaker | PyTorch | 0.138 | 1154 ms* | <audio controls preload="none" src="https://huggingface.co/patriotyk/HolosTTS/resolve/main/samples/styletts2-single-pytorch-cpu.wav"><a href="https://huggingface.co/patriotyk/HolosTTS/blob/main/samples/styletts2-single-pytorch-cpu.wav">Listen</a></audio> |
| Supertonic 3 | ONNX | 0.186 | 1582 ms* | <audio controls preload="none" src="https://huggingface.co/patriotyk/HolosTTS/resolve/main/samples/supertonic3-cpu.wav"><a href="https://huggingface.co/patriotyk/HolosTTS/blob/main/samples/supertonic3-cpu.wav">Listen</a></audio> |
| StyleTTS2 multi-speaker | PyTorch | 0.335 | 2756 ms* | <audio controls preload="none" src="https://huggingface.co/patriotyk/HolosTTS/resolve/main/samples/styletts2-multi-pytorch-cpu.wav"><a href="https://huggingface.co/patriotyk/HolosTTS/blob/main/samples/styletts2-multi-pytorch-cpu.wav">Listen</a></audio> |

### Apple M2

| Model | Runtime | RTF | First audio | Sample |
| --- | --- | --- | --- | --- |
| **HolosTTS** | **MLX (GPU)** | **0.025** | **66 ms** | <audio controls preload="none" src="https://huggingface.co/patriotyk/HolosTTS/resolve/main/samples/holos-mlx-m2.wav"><a href="https://huggingface.co/patriotyk/HolosTTS/blob/main/samples/holos-mlx-m2.wav">Listen</a></audio> |
| **HolosTTS** | **ONNX INT8** | **0.086** | **204 ms** | <audio controls preload="none" src="https://huggingface.co/patriotyk/HolosTTS/resolve/main/samples/holos-onnx-int8-m2.wav"><a href="https://huggingface.co/patriotyk/HolosTTS/blob/main/samples/holos-onnx-int8-m2.wav">Listen</a></audio> |
| **HolosTTS** | **PyTorch (CPU)** | **0.107** | **351 ms** | <audio controls preload="none" src="https://huggingface.co/patriotyk/HolosTTS/resolve/main/samples/holos-pytorch-m2.wav"><a href="https://huggingface.co/patriotyk/HolosTTS/blob/main/samples/holos-pytorch-m2.wav">Listen</a></audio> |
| **HolosTTS** | **ONNX FP32** | **0.129** | **289 ms** | <audio controls preload="none" src="https://huggingface.co/patriotyk/HolosTTS/resolve/main/samples/holos-onnx-m2.wav"><a href="https://huggingface.co/patriotyk/HolosTTS/blob/main/samples/holos-onnx-m2.wav">Listen</a></audio> |

The other models run the way their packages do by default, and like HolosTTS they get the text already converted to phonemes, except Supertonic, which reads the text itself:

- [Supertonic 3](https://huggingface.co/Supertone/supertonic-3): the `supertonic` package with 8 steps and the F1 voice (44.1 kHz); on the GPU with the CUDA provider enabled. On the i7-13700K it is about 1.7x faster when limited to 4 threads; the table uses the default.
- [StyleTTS2 single-speaker](https://huggingface.co/patriotyk/styletts2_ukrainian_single) and [multi-speaker](https://huggingface.co/patriotyk/styletts2_ukrainian_multispeaker): my earlier Ukrainian StyleTTS2 models, with the same phonemes. The single-speaker model has an iSTFTNet decoder, the multi-speaker one HiFi-GAN. The single-speaker ONNX model runs on ONNX Runtime directly, after normalizing one negative `Transpose` permutation that ONNX Runtime 1.28 rejects.
- [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) does not speak Ukrainian; it is measured on English translations of the same sentences, with the af_heart voice.

Run the HolosTTS measurement on your machine from a checkout of this repository:

```bash
python benchmark.py onnx                                        # ONNX FP32 on CPU
python benchmark.py onnx --model patriotyk/HolosTTS/onnx-int8   # ONNX INT8 on CPU
python benchmark.py onnx --provider tensorrt-fp16               # or cuda, tensorrt
python benchmark.py torch --device cuda                         # PyTorch
python benchmark.py mlx                                         # MLX on Apple silicon
```

## Files on Hugging Face

| File | Contents |
| --- | --- |
| `holos.pt` | PyTorch model weights and the phoneme vocabulary. |
| `voices.pt` | Preset voices: names and 256-dimensional vectors. |
| `voice_encoder.pt` | The model that computes a voice vector from an audio sample. |
| `samples/` | The audio samples from [Benchmarks](#benchmarks). |
| `onnx/` | FP32 ONNX model for `holos-tts-onnx`: CPU, CUDA and TensorRT. |
| `onnx-int8/` | INT8 ONNX model for CPU. |
| `mlx/` | MLX model for `holos-tts-mlx`: `config.json`, `model.safetensors` and `voices.npz`. |
| `litert/` | LiteRT model with FP16 weights for the [Android library](https://github.com/patriotyk/HolosTTS/blob/main/holos_litert/kotlin/README.md). |

Each ONNX folder holds `holos_onnx.json` (vocabulary and parameters), `voices.npz` (preset voices) and the model stages: `encoder.onnx`, `prosody.onnx`, `decoder.onnx`, plus `source.onnx` and `decoder_window.onnx` for streaming.

## License

The code and models are released under the [MIT license](https://github.com/patriotyk/HolosTTS/blob/main/LICENSE).

I took the preset voices without the consent of their owners. For commercial use, either agree on it with the voice owners, clone your own voices from audio prompts whose license allows commercial use, or mix several voices so that none of them is recognizable.
