"""Static INT8 quantization for the single-file HolosTTS ONNX graph."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np


# ORT's CPU EP fuses QDQ Conv/MatMul to integer kernels. QDQ Gemm remains a
# floating-point Gemm surrounded by conversions in ORT 1.28, so quantizing it
# adds overhead without providing INT8 compute.
DEFAULT_QUANTIZED_OPS = ("Conv", "MatMul")

_WAVEFORM_SENSITIVE_PREFIXES = (
    "/acoustic_decoder/m_source/",
    "/acoustic_decoder/noise_convs.",
    "/acoustic_decoder/noise_res.",
    "/acoustic_decoder/conv_post/",
)

_WAVEFORM_OUTPUT_PREFIXES = (
    "/acoustic_decoder/conv_post/",
)

_WAVEFORM_RESBLOCK_PREFIXES = (
    "/acoustic_decoder/resblocks.",
    "/acoustic_decoder/conv_post/",
)

_WAVEFORM_EARLY_PREFIXES = tuple(
    f"/acoustic_decoder/resblocks.{index}/"
    for index in range(3)
) + _WAVEFORM_OUTPUT_PREFIXES

_WAVEFORM_LATE_PREFIXES = tuple(
    f"/acoustic_decoder/resblocks.{index}/"
    for index in range(3, 6)
) + _WAVEFORM_OUTPUT_PREFIXES

_WAVEFORM_ALL_PREFIXES = _WAVEFORM_SENSITIVE_PREFIXES + (
    "/acoustic_decoder/resblocks.",
)

WAVEFORM_PRECISIONS = (
    "fp32",
    "mixed",
    "resblocks",
    "early",
    "late",
    "output",
    "int8",
)
QUANTIZATION_TYPES = ("s8s8", "u8s8", "u8u8")


def _load_voices(voices_path: Path) -> np.ndarray:
    with np.load(voices_path, allow_pickle=False) as archive:
        if "values" not in archive.files:
            raise ValueError(f"{voices_path} does not contain a 'values' array")
        voices = np.asarray(archive["values"], dtype=np.float32)

    if voices.ndim != 2 or voices.shape[1] != 256 or voices.shape[0] == 0:
        raise ValueError(
            f"Expected voices with shape [N, 256], got {voices.shape}"
        )
    if not np.isfinite(voices).all():
        raise ValueError("Voice embeddings contain NaN or Inf")
    return np.ascontiguousarray(voices)


def _calibration_lengths(samples: int, max_text_length: int) -> List[int]:
    if samples <= 0:
        raise ValueError("samples must be positive")
    if max_text_length < 2:
        raise ValueError("max_text_length must be at least 2")

    # Log-spaced lengths cover short UI prompts and long sentences without
    # spending most calibration time on the largest waveform shapes.
    values = np.geomspace(2, max_text_length, num=samples)
    lengths = np.rint(values).astype(np.int64).clip(2, max_text_length)
    lengths[0] = 2
    lengths[-1] = max_text_length
    return lengths.tolist()


def _make_calibration_inputs(
    vocab_size: int,
    voices: np.ndarray,
    samples: int,
    max_text_length: int,
    seed: int,
) -> List[Dict[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    inputs = []
    for sample_index, text_length in enumerate(
        _calibration_lengths(samples, max_text_length)
    ):
        tokens = np.zeros((1, text_length), dtype=np.int64)
        if text_length > 2:
            tokens[:, 1:-1] = rng.integers(
                1,
                vocab_size,
                size=(1, text_length - 2),
                dtype=np.int64,
            )
        voice = voices[sample_index % voices.shape[0]][None, :]
        inputs.append(
            {
                "tokens": np.ascontiguousarray(tokens),
                "voice": np.ascontiguousarray(voice, dtype=np.float32),
            }
        )
    return inputs


def _validate_single_graph(model_path: Path) -> int:
    import onnx

    model = onnx.load(str(model_path), load_external_data=False)
    inputs = {value.name for value in model.graph.input}
    if inputs != {"tokens", "voice"}:
        raise ValueError(
            "Static CPU quantization expects a dynamic single-file graph with "
            f"inputs ['tokens', 'voice']; got {sorted(inputs)}. Re-export it "
            "with --dynamic-frames."
        )
    embedding_sizes = {
        int(initializer.dims[0])
        for initializer in model.graph.initializer
        if initializer.name.endswith("embedding.weight")
        and len(initializer.dims) == 2
    }
    if len(embedding_sizes) != 1:
        raise ValueError(
            "Could not determine one vocabulary size from ONNX embedding "
            f"weights; found {sorted(embedding_sizes)}"
        )
    vocab_size = embedding_sizes.pop()
    if vocab_size < 2:
        raise ValueError(f"Invalid ONNX vocabulary size: {vocab_size}")
    return vocab_size


def _duration_sensitive_nodes(
    model_path: Path,
    op_types: Iterable[str],
) -> List[str]:
    """Keep every producer of duration rounding in floating point.

    Duration values are rounded to integer frame counts, so a small INT8 error
    can discontinuously change the waveform length and speaking rhythm.  The
    exporter gives the duration and acoustic text encoders separate graph
    branches; walking backwards from Round preserves only the sensitive branch
    while allowing the acoustic branch to be quantized.
    """
    import onnx

    model = onnx.load(str(model_path), load_external_data=False)
    nodes = list(model.graph.node)
    producer = {
        output: node
        for node in nodes
        for output in node.output
        if output
    }
    roots = [
        node
        for node in nodes
        if node.op_type == "Round" and "duration_encoder" in node.name
    ]
    if len(roots) != 1:
        raise ValueError(
            "Expected one duration Round node in the single graph; found "
            f"{len(roots)}"
        )

    ancestors = set()
    pending = list(roots)
    while pending:
        node = pending.pop()
        identity = id(node)
        if identity in ancestors:
            continue
        ancestors.add(identity)
        pending.extend(
            parent
            for value in node.input
            if (parent := producer.get(value)) is not None
        )

    selected_types = set(op_types)
    return [
        node.name
        for node in nodes
        if id(node) in ancestors
        and node.op_type in selected_types
        and node.name
    ]


def _waveform_decoder_nodes(
    model_path: Path,
    op_types: Iterable[str],
    waveform_precision: str,
) -> List[str]:
    """Return PTQ-sensitive waveform generator nodes.

    Its residual Conv stack predicts log magnitude and phase immediately before
    the inverse STFT. Small post-training INT8 errors are amplified by Exp and
    accumulate through the residual blocks, producing metallic or bubbling
    artifacts. Keeping this relatively local subgraph in FP32 preserves the
    useful INT8 work in the text/prosody encoders and the feature decoder.
    """
    import onnx

    if waveform_precision not in WAVEFORM_PRECISIONS:
        raise ValueError(
            "waveform_precision must be one of "
            + ", ".join(WAVEFORM_PRECISIONS)
        )
    if waveform_precision == "int8":
        return []

    selected_types = set(op_types)
    if waveform_precision == "fp32":
        prefixes = _WAVEFORM_ALL_PREFIXES
    elif waveform_precision == "mixed":
        prefixes = _WAVEFORM_SENSITIVE_PREFIXES
    elif waveform_precision == "resblocks":
        prefixes = _WAVEFORM_RESBLOCK_PREFIXES
    elif waveform_precision == "early":
        prefixes = _WAVEFORM_EARLY_PREFIXES
    elif waveform_precision == "late":
        prefixes = _WAVEFORM_LATE_PREFIXES
    else:
        prefixes = _WAVEFORM_OUTPUT_PREFIXES
    model = onnx.load(str(model_path), load_external_data=False)
    return [
        node.name
        for node in model.graph.node
        if node.name
        and node.op_type in selected_types
        and (
            (
                waveform_precision == "fp32"
                and node.name == "/acoustic_decoder/Conv"
            )
            or node.name.startswith(prefixes)
        )
    ]


def quantize_onnx_cpu(
    model_input: Path,
    model_output: Path,
    voices_path: Path,
    samples: int = 64,
    max_text_length: int = 128,
    seed: int = 0,
    op_types: Iterable[str] = DEFAULT_QUANTIZED_OPS,
    waveform_precision: str = "late",
    quantization_type: str = "s8s8",
    reduce_range: bool = False,
    symmetric_activations: bool = False,
    force: bool = False,
) -> Path:
    try:
        import onnx
        import onnxruntime as ort
        from onnxruntime.quantization import (
            CalibrationDataReader,
            CalibrationMethod,
            QuantFormat,
            QuantType,
            quantize_static,
        )
    except ImportError as error:
        raise RuntimeError(
            "Install ONNX dependencies with: pip install -e '.[onnx]'"
        ) from error

    model_input = model_input.expanduser().resolve()
    model_output = model_output.expanduser().resolve()
    voices_path = voices_path.expanduser().resolve()
    if not model_input.is_file():
        raise FileNotFoundError(model_input)
    if model_input == model_output:
        raise ValueError("The INT8 output must not overwrite the FP32 input")
    if model_output.exists() and not force:
        raise FileExistsError(
            f"{model_output} already exists; pass --force to overwrite it"
        )
    model_output.parent.mkdir(parents=True, exist_ok=True)
    vocab_size = _validate_single_graph(model_input)

    op_types = list(dict.fromkeys(str(value) for value in op_types))
    if not op_types:
        raise ValueError("At least one op type must be selected")
    if quantization_type not in QUANTIZATION_TYPES:
        raise ValueError(
            "quantization_type must be one of "
            + ", ".join(QUANTIZATION_TYPES)
        )

    activation_type = (
        QuantType.QInt8
        if quantization_type.startswith("s8")
        else QuantType.QUInt8
    )
    weight_type = (
        QuantType.QInt8
        if quantization_type.endswith("s8")
        else QuantType.QUInt8
    )

    calibration_inputs = _make_calibration_inputs(
        vocab_size=vocab_size,
        voices=_load_voices(voices_path),
        samples=samples,
        max_text_length=max_text_length,
        seed=seed,
    )
    duration_nodes = _duration_sensitive_nodes(model_input, op_types)
    waveform_nodes = _waveform_decoder_nodes(
        model_input,
        op_types,
        waveform_precision,
    )
    nodes_to_exclude = list(dict.fromkeys(duration_nodes + waveform_nodes))

    class HolosCalibrationDataReader(CalibrationDataReader):
        def __init__(self, values: Sequence[Dict[str, np.ndarray]]):
            self.values = list(values)
            self.rewind()

        def get_next(self):
            return next(self.iterator, None)

        def rewind(self):
            self.iterator = iter(self.values)

    print(
        f"Calibrating {model_input} with {samples} samples "
        f"(token lengths 2..{max_text_length})..."
    )
    print(f"Quantized operator types: {', '.join(op_types)}")
    print(
        f"Quantization type: {quantization_type.upper()}"
        + (" with reduced weight range" if reduce_range else "")
        + (" and symmetric activations" if symmetric_activations else "")
    )
    print(
        "Keeping the duration branch in FP32: "
        f"{len(duration_nodes)} compute nodes"
    )
    if waveform_nodes:
        print(
            f"Keeping waveform={waveform_precision} nodes in FP32: "
            f"{len(waveform_nodes)} compute nodes"
        )
    quantize_static(
        model_input=str(model_input),
        model_output=str(model_output),
        calibration_data_reader=HolosCalibrationDataReader(calibration_inputs),
        quant_format=QuantFormat.QDQ,
        op_types_to_quantize=op_types,
        nodes_to_exclude=nodes_to_exclude,
        per_channel=True,
        reduce_range=reduce_range,
        activation_type=activation_type,
        weight_type=weight_type,
        use_external_data_format=False,
        calibrate_method=CalibrationMethod.MinMax,
        calibration_providers=["CPUExecutionProvider"],
        extra_options={
            "ActivationSymmetric": (
                symmetric_activations
                and activation_type == QuantType.QInt8
            ),
            # A tensor shared by the duration and acoustic branches must not
            # reuse one Q/DQ pair: doing so would silently quantize the FP32
            # duration consumer as well and could change the frame count.
            "DedicatedQDQPair": True,
            "WeightSymmetric": weight_type == QuantType.QInt8,
        },
    )

    onnx.checker.check_model(str(model_output))
    session = ort.InferenceSession(
        str(model_output),
        providers=["CPUExecutionProvider"],
    )
    audio, audio_lengths = session.run(None, calibration_inputs[0])
    if not np.isfinite(audio).all():
        raise RuntimeError("Quantized ONNX model produced NaN or Inf")
    valid_samples = int(np.asarray(audio_lengths).reshape(-1)[0])
    if not 0 < valid_samples <= audio.shape[-1]:
        raise RuntimeError(
            f"Invalid quantized output length {valid_samples} for {audio.shape}"
        )

    model = onnx.load(str(model_output), load_external_data=False)
    quantize_nodes = sum(
        node.op_type == "QuantizeLinear" for node in model.graph.node
    )
    dequantize_nodes = sum(
        node.op_type == "DequantizeLinear" for node in model.graph.node
    )
    size_mib = model_output.stat().st_size / (1024 * 1024)
    print(
        f"INT8 export complete: {model_output} ({size_mib:.1f} MiB, "
        f"Q={quantize_nodes}, DQ={dequantize_nodes}, "
        f"smoke-test samples={valid_samples})"
    )
    return model_output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Statically quantize a dynamic single-file HolosTTS ONNX model"
    )
    parser.add_argument("--input", type=Path, default=Path("holos.onnx"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("holos_cpu_int8.onnx"),
    )
    parser.add_argument("--voices", type=Path, default=Path("voices.npz"))
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--max-text-length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--op-types",
        nargs="+",
        default=list(DEFAULT_QUANTIZED_OPS),
    )
    parser.add_argument(
        "--waveform-precision",
        choices=WAVEFORM_PRECISIONS,
        default="late",
        help=(
            "fp32 is safest; mixed keeps harmonic/noise/output layers in "
            "FP32; resblocks keeps the main residual stack and output in "
            "FP32; early/late keep half of that stack in FP32; output keeps "
            "only the final spectral projection in FP32; "
            "int8 is fastest but commonly causes severe artifacts"
        ),
    )
    parser.add_argument(
        "--quantization-type",
        choices=QUANTIZATION_TYPES,
        default="s8s8",
    )
    parser.add_argument(
        "--reduce-range",
        action="store_true",
        help="quantize weights to 7 bits to reduce accumulator saturation",
    )
    parser.add_argument(
        "--symmetric-activations",
        action="store_true",
        help="force symmetric signed activation ranges instead of ORT defaults",
    )
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    quantize_onnx_cpu(
        model_input=args.input,
        model_output=args.output,
        voices_path=args.voices,
        samples=args.samples,
        max_text_length=args.max_text_length,
        seed=args.seed,
        op_types=args.op_types,
        waveform_precision=args.waveform_precision,
        quantization_type=args.quantization_type,
        reduce_range=args.reduce_range,
        symmetric_activations=args.symmetric_activations,
        force=args.force,
    )


if __name__ == "__main__":
    main()
