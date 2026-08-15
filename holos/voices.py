"""Framework-neutral voice embedding storage and legacy conversion."""

import argparse
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch


VOICE_DIMENSION = 256


def _as_numpy_voice(name, value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.shape != (VOICE_DIMENSION,):
        raise ValueError(
            f"Voice {name!r} has shape {array.shape}; "
            f"expected ({VOICE_DIMENSION},)"
        )
    if not np.isfinite(array).all():
        raise ValueError(f"Voice {name!r} contains NaN or Inf")
    return array


def save_voices_npz(voices, output, force=False):
    """Save a mapping of voice names to embeddings without pickle."""
    output = Path(os.path.abspath(Path(output).expanduser()))
    if output.suffix.lower() != ".npz":
        raise ValueError("Canonical voice files must use the .npz extension")
    if output.exists() and not force:
        raise FileExistsError(f"{output} already exists; pass force=True to overwrite it")
    output.parent.mkdir(parents=True, exist_ok=True)

    if not isinstance(voices, Mapping) or not voices:
        raise ValueError("Voices must be a non-empty mapping")

    names = [str(name) for name in voices]
    if len(set(names)) != len(names):
        raise ValueError("Voice names must be unique after string conversion")
    values = np.stack(
        [_as_numpy_voice(name, value) for name, value in voices.items()]
    )
    np.savez_compressed(
        output,
        names=np.asarray(names, dtype=np.str_),
        values=values,
    )
    return output


def load_voices_npz(path):
    """Load canonical voices as NumPy arrays with pickle disabled."""
    path = Path(os.path.abspath(Path(path).expanduser()))
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {"names", "values"}:
            raise ValueError(
                f"{path} must contain exactly the 'names' and 'values' arrays"
            )
        names = archive["names"].astype(str).tolist()
        values = np.asarray(archive["values"], dtype=np.float32)

    if values.ndim != 2 or values.shape[1] != VOICE_DIMENSION:
        raise ValueError(
            f"Expected voice values with shape [N, {VOICE_DIMENSION}], "
            f"got {values.shape}"
        )
    if len(names) != values.shape[0]:
        raise ValueError("Voice names and embeddings have different lengths")
    if len(set(names)) != len(names):
        raise ValueError("Voice names must be unique")
    if not np.isfinite(values).all():
        raise ValueError("Voice embeddings contain NaN or Inf")

    return {
        name: np.ascontiguousarray(value)
        for name, value in zip(names, values)
    }


def load_voice_embeddings(path, device="cpu"):
    """Load canonical .npz or legacy .pt voices as float32 tensors."""
    path = Path(os.path.abspath(Path(path).expanduser()))
    suffix = path.suffix.lower()
    if suffix == ".npz":
        numpy_voices = load_voices_npz(path)
        return {
            name: torch.from_numpy(value.copy()).to(device=device)
            for name, value in numpy_voices.items()
        }
    if suffix != ".pt":
        raise ValueError(f"Unsupported voices format {suffix!r}; use .npz or .pt")

    try:
        voices = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        voices = torch.load(path, map_location=device)
    if not isinstance(voices, Mapping) or not voices:
        raise ValueError("Legacy voices.pt must contain a non-empty mapping")
    result = {}
    for name, value in voices.items():
        array = _as_numpy_voice(name, value)
        result[str(name)] = torch.from_numpy(array.copy()).to(device=device)
    return result


def convert_voices(source, output, force=False):
    """Convert legacy voices.pt (or normalize an .npz) to canonical .npz."""
    source = Path(os.path.abspath(Path(source).expanduser()))
    output = Path(os.path.abspath(Path(output).expanduser()))
    if source.resolve() == output.resolve():
        raise ValueError("Source and output paths must be different")
    voices = load_voice_embeddings(source, device="cpu")
    return save_voices_npz(voices, output, force=force)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Convert HolosTTS voice embeddings to canonical NPZ"
    )
    parser.add_argument("source", type=Path, help="Existing voices.pt or voices.npz")
    parser.add_argument("output", type=Path, help="Output voices.npz")
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None):
    args = build_parser().parse_args(argv)
    output = convert_voices(args.source, args.output, force=args.force)
    print(f"Converted voices: {output}")


if __name__ == "__main__":
    main()
