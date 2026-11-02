"""Convert a HolosTTS PyTorch checkpoint into an MLX model directory.

Needs PyTorch and the ``holos`` package of this repository; the converted
model does not. Weight norm is folded into the weights, convolutions move to
MLX's ``[out, kernel, in]`` layout, and the directory gets the vocabulary,
the streaming context and the preset voices.

    python -m holos_mlx.convert --weights holos.pt --voices voices.pt --output holos_mlx_model
"""

import argparse
import json
from pathlib import Path

import numpy as np

# PyTorch module paths that the MLX modules name differently.
_RENAMES = (
    (".cnn.{}.0.", ".cnn.{}.conv."),
    (".cnn.{}.1.", ".cnn.{}.norm."),
)


def _rename(name: str) -> str:
    for index in range(16):
        for old, new in _RENAMES:
            name = name.replace(old.format(index), new.format(index))
    return name.replace(".pool.0.", ".pool.").replace("decoder.asr_res.0.", "decoder.asr_res.")


def convert(weights: str, voices: str, output: Path):
    import mlx.core as mx
    import torch
    from torch import nn
    from torch.nn.utils import parametrize

    from holos.models import HolosTTS

    model = HolosTTS(weights, device="cpu", voices_path=voices).eval()
    for module in model.modules():
        if parametrize.is_parametrized(module, "weight"):
            parametrize.remove_parametrizations(module, "weight", leave_parametrized=True)
    modules = dict(model.named_modules())
    arrays = {}
    for name, tensor in model.state_dict().items():
        if name.endswith(("cos_cached", "sin_cached", "mask_token")):
            continue  # The RoPE tables, which MLX computes, and a training-only token.
        owner, attribute = name.rsplit(".", 1)
        module = modules.get(owner)
        value = tensor.detach().float().numpy()
        if isinstance(module, nn.ConvTranspose1d) and attribute == "weight":
            value = value.transpose(1, 2, 0)  # [in, out, k] -> [out, k, in]
        elif isinstance(module, nn.Conv1d) and attribute == "weight":
            value = value.transpose(0, 2, 1)  # [out, in, k] -> [out, k, in]
        elif isinstance(module, nn.ParameterList):  # Snake alphas [1, C, 1]
            value = value.reshape(1, 1, -1)
        elif name.endswith("conv_pe.gate"):  # [scales, 1, C, 1]
            value = value.reshape(value.shape[0], 1, 1, -1)
        elif name.endswith("style_weight"):
            value = value.reshape(1)  # safetensors stores scalars as [1]
        arrays[_rename(name)] = mx.array(np.ascontiguousarray(value))

    output.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(output / "model.safetensors"), arrays)
    config = {
        "format": "holos-mlx",
        "format_version": 1,
        "vocab": list(model.tokenizer.vocab),
        "n_symbols": int(model.text_encoder.embedding.num_embeddings),
        "sample_rate": 24000,
        "samples_per_frame": int(2 * model.decoder.generator.total_upsample),
        "context_frames": int(model.decoder.streaming_context_frames()),
    }
    (output / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    names = list(model.voices)
    values = np.stack([model.voices[name].detach().float().reshape(-1).numpy() for name in names])
    np.savez(output / "voices.npz", names=np.asarray(names), values=values.astype(np.float32))
    print(f"Converted {len(arrays)} tensors and {len(names)} voices to {output}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", default="patriotyk/HolosTTS")
    parser.add_argument("--voices", default="voices.pt")
    parser.add_argument("--output", type=Path, default=Path("holos_mlx_model"))
    args = parser.parse_args(argv)
    convert(args.weights, args.voices, args.output)


if __name__ == "__main__":
    main()
