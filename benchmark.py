"""Measure HolosTTS speed on this machine.

    python benchmark.py onnx                              # ONNX FP32 on CPU
    python benchmark.py onnx --model patriotyk/HolosTTS/onnx-int8
    python benchmark.py onnx --provider cuda              # or tensorrt, tensorrt-fp16
    python benchmark.py onnx --provider coreml            # Apple silicon
    python benchmark.py torch --device cuda               # PyTorch
    python benchmark.py mlx                               # MLX on Apple silicon

For each sentence it reports:
- RTF (real-time factor): the time generate() takes for the whole sentence
  divided by the duration of the audio. 0.01 means 1 s of speech in 10 ms.
- First chunk: the time until stream() yields the first audio, which is the
  latency a listener notices in streaming mode.

The onnx backend needs holos-tts-onnx, the torch backend holos-tts and the mlx
backend holos-tts-mlx. They download the models from Hugging Face on the
first run.
"""

import argparse
import statistics
import time
from pathlib import Path

# Three Ukrainian sentences, 5.0-10.3 s of speech, already in phonemes.
SENTENCES = [
    "sʲɔˈɦɔdʲnʲi ˈwranʲt͡sʲi ˈnad ˈmʲistɔm prɔi̯ˈʃɔu̯ ˈsɪlʲnɪi̯ ˈdɔʃt͡ʃ, "
    "ˈi ˈwuɫɪt͡sʲi ˈʃt͡ʃɛ ˈdɔu̯ɦɔ zɐɫɪˈʃaɫɪsʲɐ ˈmɔkrɪmɪ.",
    "ˈbudʲ ˈɫaskɐ, zɐteɫefoˈnui̯te meˈnʲi ˈzau̯trɐ ˈpʲisʲlʲɐ oˈbʲidʊ, "
    "kɔˈɫɪ ˈmatɪmete ˈʋʲilʲnʊ xʋɪˈɫɪnkʊ.",
    "nɐʊˈkɔʋɐ kɔnfeˈrɛnʲt͡sʲijɐ ʋʲidˈbudet͡sʲːɐ nɐˈstupnɔɦɔ ˈtɪʒnʲɐ ˈu "
    "ʋeˈɫɪkʲii̯ ˈzalʲi ʊnʲiʋersɪˈtɛtʊ, ˈi ˈkɔʒen ɔˈxɔt͡ʃɪi̯ ˈzmɔʒe "
    "posˈɫuxɐtɪ ˈdɔpɔʋʲidʲi prɔʋʲidˈnɪx dosʲˈlʲidnɪkʲiu̯ z ˈrʲiznɪx krɐˈjin ˈsʲʋʲitʊ.",
]


def onnx_backend(args):
    from holos_onnx import HolosTTS

    model = args.model or "patriotyk/HolosTTS/onnx"
    tensorrt = (
        "TensorrtExecutionProvider",
        {
            "trt_fp16_enable": args.provider == "tensorrt-fp16",
            "trt_engine_cache_enable": True,
            "trt_engine_cache_path": args.trt_cache,
            "trt_timing_cache_enable": True,
            "trt_timing_cache_path": args.trt_cache,
        },
    )
    if args.provider.startswith("tensorrt"):
        Path(args.trt_cache).mkdir(parents=True, exist_ok=True)
    providers = {
        "cpu": ["CPUExecutionProvider"],
        "cuda": ["CUDAExecutionProvider"],
        "tensorrt": [tensorrt],
        "tensorrt-fp16": [tensorrt],
        "coreml": ["CoreMLExecutionProvider"],
    }[args.provider]
    whole = HolosTTS(model, providers=providers)
    streaming = HolosTTS(model, providers=providers, streaming=True)
    voice = whole.list_voices()[0]
    print(f"ONNX Runtime, {model}, providers {whole.providers}")

    def generate(text):
        return whole.generate(text, voice=voice).size / whole.sample_rate

    def stream(text):
        return streaming.stream(text, voice=voice)

    return generate, stream, lambda: None


def torch_backend(args):
    import torch
    from holos.models import HolosTTS

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if args.model:
        model = HolosTTS(args.model, device=device, voices_path=args.voices)
    else:
        model = HolosTTS(device=device)
    voice = model.list_voices()[0]
    synchronize = torch.cuda.synchronize if device.startswith("cuda") else (lambda: None)
    print(f"PyTorch {torch.__version__}, device {device}")

    def generate(text):
        audio = model.generate(text, voice)
        synchronize()
        return audio.shape[-1] / 24000

    def stream(text):
        return model.stream(text, voice)

    return generate, stream, synchronize


def mlx_backend(args):
    import mlx.core as mx
    from holos_mlx import HolosTTS

    model = HolosTTS(args.model or "patriotyk/HolosTTS/mlx")
    voice = model.list_voices()[0]
    print(f"MLX {mx.__version__}, device {mx.default_device()}")

    def generate(text):
        return model.generate(text, voice=voice).size / model.sample_rate

    def stream(text):
        return model.stream(text, voice=voice)

    return generate, stream, lambda: None


def measure(generate, stream, synchronize, runs):
    rows = []
    for text in SENTENCES:
        # Warm-up: the first runs pick kernels and, with TensorRT, build engines.
        generate(text)
        for _chunk in stream(text):
            pass
        generate_times, first_chunk_times = [], []
        for _ in range(runs):
            start = time.perf_counter()
            seconds = generate(text)
            generate_times.append(time.perf_counter() - start)
        for _ in range(runs):
            start = time.perf_counter()
            chunks = stream(text)
            next(chunks)
            synchronize()
            first_chunk_times.append(time.perf_counter() - start)
            for _chunk in chunks:
                pass
        rows.append((seconds, statistics.median(generate_times), statistics.median(first_chunk_times)))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("backend", choices=["onnx", "torch", "mlx"])
    parser.add_argument("--model", help="Hugging Face model or local path (default: the published model)")
    parser.add_argument("--provider", choices=["cpu", "cuda", "tensorrt", "tensorrt-fp16", "coreml"], default="cpu",
                        help="ONNX Runtime execution provider")
    parser.add_argument("--trt-cache", default="trt_cache", help="TensorRT engine cache directory")
    parser.add_argument("--device", help="PyTorch device (default: cuda when available)")
    parser.add_argument("--voices", default="voices.pt", help="voices file for a local PyTorch --model")
    parser.add_argument("--runs", type=int, default=10, help="timed runs per sentence (the median is reported)")
    args = parser.parse_args()

    backend = {"onnx": onnx_backend, "torch": torch_backend, "mlx": mlx_backend}[args.backend]
    rows = measure(*backend(args), args.runs)

    print(f"\n{'audio, s':>9} {'generate, ms':>13} {'RTF':>7} {'first chunk, ms':>16}")
    for seconds, generate_time, first_chunk in rows:
        print(f"{seconds:9.2f} {generate_time * 1000:13.1f} {generate_time / seconds:7.4f} {first_chunk * 1000:16.1f}")
    total_audio = sum(row[0] for row in rows)
    total_time = sum(row[1] for row in rows)
    mean_first = statistics.mean(row[2] for row in rows)
    print(f"{'total':>9} {total_time * 1000:13.1f} {total_time / total_audio:7.4f} {mean_first * 1000:16.1f}")


if __name__ == "__main__":
    main()
