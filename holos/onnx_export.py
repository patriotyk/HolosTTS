"""Command-line ONNX export for HolosTTS."""

import argparse
import inspect
import json
import shutil
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
from torch import nn

from holos.models import HolosTTS
from holos.modules import EncoderLayerONNX, MaskedInstanceNorm1d
from holos.voices import save_voices_npz


class OnnxInferenceWrapper(nn.Module):
    """Expose tensor-only inputs and the valid output-audio length."""

    def __init__(self, model: HolosTTS):
        super().__init__()
        self.model = model

    def forward(self, tokens, voice, input_lengths):
        encoded = self.model.encode(tokens, voice, input_lengths)
        audio = self.model.decoder(*encoded).squeeze(0)

        decoder_lengths = encoded[-1]
        audio_lengths = decoder_lengths * self.model.decoder.generator.total_upsample
        audio_lengths = torch.clamp(audio_lengths, max=audio.size(-1)).to(torch.int64)
        return audio, audio_lengths


class OnnxDurationEncoderWrapper(nn.Module):
    """Stage 1: text and voice -> token features and integer durations."""

    def __init__(self, model: HolosTTS):
        super().__init__()
        self.prosodic_text_encoder = model.prosodic_text_encoder
        self.text_encoder = model.text_encoder
        self.duration_predictor = model.duration_predictor

    def forward(self, tokens, voice):
        voice_prosodic = voice[:, 128:]

        prosodic_encoded = self.prosodic_text_encoder.inference(tokens)
        prosodic_tokens = prosodic_encoded.transpose(1, 2)
        duration_logits = self.duration_predictor.inference(
            prosodic_tokens,
            voice_prosodic,
        )
        durations = torch.round(torch.sigmoid(duration_logits).sum(dim=-1))
        durations = durations.clamp(min=1).to(torch.int64)

        acoustic_tokens = self.text_encoder.inference(tokens)
        return acoustic_tokens, prosodic_tokens, durations


class OnnxAcousticDecoderWrapper(nn.Module):
    """Stage 2: gather token features -> waveform for a dynamic frame width."""

    def __init__(self, model: HolosTTS):
        super().__init__()
        self.prosodic_predictor = model.prosodic_predictor
        self.decoder = model.decoder
        self.samples_per_frame = int(
            model.decoder.generator.total_upsample * 2
        )

    def forward(
        self,
        acoustic_tokens,
        prosodic_tokens,
        frame_to_token,
        voice,
    ):
        voice_acoustic = voice[:, :128]
        voice_prosodic = voice[:, 128:]

        aligned_prosody = self._gather_frames(
            prosodic_tokens.transpose(1, 2),
            frame_to_token,
        )
        f0, energy = self.prosodic_predictor.inference(
            aligned_prosody,
            voice_prosodic,
        )

        aligned_acoustic = self._gather_frames(
            acoustic_tokens,
            frame_to_token,
        )
        audio = self.decoder.inference(
            aligned_acoustic,
            f0,
            energy,
            voice_acoustic,
        ).squeeze(1)
        audio_lengths = torch._shape_as_tensor(audio)[-1:].to(torch.int64)
        return audio, audio_lengths

    @staticmethod
    def _gather_frames(token_features, frame_to_token):
        return torch.index_select(token_features, 2, frame_to_token)


class OnnxProsodyDecoderWrapper(nn.Module):
    """Stage 2: aligned prosody tokens -> frame-level F0 and energy."""

    def __init__(self, model: HolosTTS):
        super().__init__()
        self.prosodic_predictor = model.prosodic_predictor

    def forward(self, prosodic_tokens, frame_to_token, voice):
        aligned_prosody = torch.index_select(
            prosodic_tokens.transpose(1, 2),
            2,
            frame_to_token,
        )
        return self.prosodic_predictor.inference(
            aligned_prosody,
            voice[:, 128:],
        )


class OnnxWaveformDecoderWrapper(nn.Module):
    """Stage 3: acoustic features and prosody -> waveform."""

    def __init__(self, model: HolosTTS):
        super().__init__()
        self.decoder = model.decoder
        self.samples_per_frame = int(
            model.decoder.generator.total_upsample * 2
        )

    def forward(
        self,
        acoustic_tokens,
        frame_to_token,
        f0,
        energy,
        voice,
    ):
        aligned_acoustic = torch.index_select(
            acoustic_tokens,
            2,
            frame_to_token,
        )
        audio = self.decoder.inference(
            aligned_acoustic,
            f0,
            energy,
            voice[:, :128],
        ).squeeze(1)
        audio_lengths = torch._shape_as_tensor(audio)[-1:].to(torch.int64)
        return audio, audio_lengths


class OnnxDynamicInferenceWrapper(nn.Module):
    """Single graph with data-dependent alignment and waveform length.

    Dynamic inference operates on one exact-length, unpadded token sequence, so
    it deliberately has no ``input_lengths`` input.  Keeping an unused third
    argument would make the Python export signature disagree with the ONNX
    graph after dead-input elimination.
    """

    def __init__(self, model: HolosTTS, max_frames: int):
        super().__init__()
        self.duration_encoder = OnnxDurationEncoderWrapper(model)
        self.acoustic_decoder = OnnxAcousticDecoderWrapper(model)
        self.max_frames = int(max_frames)

    def forward(self, tokens, voice):
        acoustic_tokens, prosodic_tokens, durations = self.duration_encoder(
            tokens,
            voice,
        )
        frame_lengths = durations.sum(dim=1).clamp(max=self.max_frames)

        frame_positions = torch.arange(
            frame_lengths[0],
            device=durations.device,
            dtype=torch.int64,
        )
        # ``repeat_interleave`` exports as SplitToSequence + Loop, which is
        # expensive on CPU and poorly supported by mobile execution providers.
        # A frame belongs to the number of tokens whose end position is not
        # greater than that frame.  This vectorized form exports as Range,
        # CumSum, comparison and ReduceSum only.
        token_end_positions = torch.cumsum(durations[0], dim=0)[:-1]
        frame_to_token = torch.sum(
            frame_positions[:, None] >= token_end_positions[None, :],
            dim=1,
        ).to(torch.int64)

        return self.acoustic_decoder(
            acoustic_tokens,
            prosodic_tokens,
            frame_to_token,
            voice,
        )


def _dummy_tokens(model: HolosTTS, sequence_length: int, device: torch.device):
    if sequence_length < 2:
        raise ValueError("sequence_length must be at least 2")

    vocab_size = model.tokenizer.symbols_count
    if vocab_size < 2:
        raise ValueError("The checkpoint vocabulary must contain at least two symbols")

    tokens = torch.zeros((1, sequence_length), dtype=torch.int64, device=device)
    if sequence_length > 2:
        values = torch.arange(1, sequence_length - 1, dtype=torch.int64, device=device)
        tokens[:, 1:-1] = values.remainder(vocab_size - 1).add(1)
    return tokens


def _load_voice_tensor(
    model: HolosTTS,
    voice_name: Optional[str],
    voice_tensor_path: Optional[Path],
    device: torch.device,
):
    if voice_tensor_path is not None:
        value = torch.load(voice_tensor_path, map_location=device)
        if isinstance(value, dict):
            if "voice" not in value:
                raise ValueError("Voice tensor dictionary must contain a 'voice' key")
            value = value["voice"]
        if not isinstance(value, torch.Tensor):
            raise TypeError("The voice file must contain a torch.Tensor")
        voice = value
    else:
        if not model.voices:
            raise ValueError("No preset voices are available; pass --voice-tensor")
        selected_name = voice_name or next(iter(model.voices))
        if selected_name not in model.voices:
            available = ", ".join(model.list_voices())
            raise ValueError(f"Unknown voice '{selected_name}'. Available voices: {available}")
        voice = model.voices[selected_name]

    if voice.dim() == 1:
        voice = voice.unsqueeze(0)
    if voice.dim() != 2 or voice.size(0) != 1 or voice.size(1) != 256:
        raise ValueError(f"Expected voice shape [256] or [1, 256], got {tuple(voice.shape)}")
    return voice.to(device=device, dtype=torch.float32)


def _frame_to_token_tensor(durations: torch.Tensor):
    """Build one token index for every exact output frame."""
    durations = durations.to(dtype=torch.int64)
    token_count = durations.size(1)
    token_ids = torch.arange(
        token_count,
        device=durations.device,
        dtype=torch.int64,
    )
    frame_to_token = torch.repeat_interleave(token_ids, durations[0])
    return frame_to_token


def _export_voices(model: HolosTTS, output: Path):
    if not model.voices:
        return None

    source = getattr(model, "voices_path", None)
    if source is not None and Path(source).suffix.lower() == ".npz":
        source = Path(source).expanduser().resolve()
        if source != output.resolve():
            shutil.copyfile(source, output)
    else:
        save_voices_npz(model.voices, output, force=True)
    return output.name


def _check_output_targets(paths: Sequence[Path], force: bool):
    existing = [str(path) for path in paths if path.exists()]
    if existing and not force:
        raise FileExistsError(
            "Output files already exist; pass --force to overwrite them: "
            + ", ".join(existing)
        )


def _expected_attention_nodes(module: nn.Module) -> int:
    return sum(
        isinstance(child, EncoderLayerONNX)
        for child in module.modules()
    )


def _set_attention_format(module: nn.Module, attention_format: str) -> int:
    layers = [
        child
        for child in module.modules()
        if isinstance(child, EncoderLayerONNX)
    ]
    for layer in layers:
        layer.onnx_attention_format = attention_format
    return len(layers)


def _check_fused_attention(onnx_model, expected: int, path: Path) -> int:
    actual = sum(
        node.domain == "com.microsoft"
        and node.op_type == "MultiHeadAttention"
        for node in onnx_model.graph.node
    )
    if actual != expected:
        raise RuntimeError(
            f"Fused attention validation failed for {path}: "
            f"expected {expected} ONNX Runtime MultiHeadAttention nodes, "
            f"found {actual}"
        )
    return actual


def _check_tensorrt_attention(onnx_model, expected: int, path: Path) -> int:
    """Validate TensorRT's recognizable BMM -> mask -> Softmax -> BMM pattern."""
    nodes = list(onnx_model.graph.node)
    producer = {
        output: node
        for node in nodes
        for output in node.output
        if output
    }
    consumers = {}
    for node in nodes:
        for value in node.input:
            consumers.setdefault(value, []).append(node)

    def has_upstream_matmul(node, depth: int = 6) -> bool:
        if depth < 0:
            return False
        parents = [producer.get(value) for value in node.input]
        return any(
            parent is not None
            and (
                parent.op_type == "MatMul"
                or has_upstream_matmul(parent, depth - 1)
            )
            for parent in parents
        )

    forbidden = [
        node.name or node.op_type
        for node in nodes
        if node.op_type == "Attention"
        or (
            node.domain == "com.microsoft"
            and node.op_type == "MultiHeadAttention"
        )
    ]
    softmax_nodes = [node for node in nodes if node.op_type == "Softmax"]
    invalid = [
        node.name or node.output[0]
        for node in softmax_nodes
        if not has_upstream_matmul(node)
        or not any(
            consumer.op_type == "MatMul"
            for consumer in consumers.get(node.output[0], ())
        )
    ]
    actual = len(softmax_nodes)
    if actual != expected or invalid or forbidden:
        raise RuntimeError(
            f"TensorRT attention validation failed for {path}: "
            f"expected {expected} BMM/mask/Softmax/BMM patterns, found {actual}; "
            f"invalid={invalid}, unsupported attention nodes={forbidden}"
        )
    return actual


def _expected_fused_projections(module: nn.Module):
    layers = [
        child
        for child in module.modules()
        if isinstance(child, EncoderLayerONNX)
    ]
    qkv = sum(layer.qkv_proj is not None for layer in layers)
    gate_up = sum(
        layer.activation_type == "swiglu"
        and layer.gate_up_proj is not None
        for layer in layers
    )
    return qkv, gate_up


def _check_fused_projections(onnx_model, expected, path: Path):
    projection_nodes = [
        node
        for node in onnx_model.graph.node
        if node.op_type in {"Gemm", "MatMul"}
    ]
    qkv = sum("/qkv_proj" in node.name for node in projection_nodes)
    gate_up = sum("/gate_up_proj" in node.name for node in projection_nodes)
    old_projection_names = (
        "/q_proj",
        "/k_proj",
        "/v_proj",
        "/gate_proj",
        "/up_proj",
    )
    old_nodes = [
        node.name
        for node in projection_nodes
        if any(name in node.name for name in old_projection_names)
    ]
    actual = (qkv, gate_up)
    if actual != expected or old_nodes:
        raise RuntimeError(
            f"Fused projection validation failed for {path}: "
            f"expected qkv/gate_up={expected}, found {actual}, "
            f"unfused nodes={old_nodes}"
        )
    return actual


def _expected_instance_norm_nodes(module: nn.Module) -> int:
    return sum(
        isinstance(child, MaskedInstanceNorm1d)
        for child in module.modules()
    )


def _check_instance_norm(onnx_model, expected: int, path: Path) -> int:
    actual = sum(
        node.domain in {"", "ai.onnx"}
        and node.op_type == "InstanceNormalization"
        for node in onnx_model.graph.node
    )
    decomposed = [
        node.name
        for node in onnx_model.graph.node
        if "/norm/" in node.name
        and node.op_type in {"ReduceMean", "ReduceSum", "Sqrt"}
    ]
    # ONNX common-subexpression elimination can share normalization results
    # when parallel residual branches receive the same tensor. Therefore the
    # exported count may be lower than the module count, but it must never be
    # higher or fall back to decomposed masked reductions.
    if actual == 0 or actual > expected or decomposed:
        raise RuntimeError(
            f"InstanceNormalization validation failed for {path}: "
            f"found {actual} standard nodes for {expected} modules, "
            f"decomposed nodes={decomposed}"
        )
    return actual


def _infer_symbolic_shapes(onnx_model, path: Path):
    """Add the intermediate tensor shapes required by TensorRT partitioning."""
    try:
        from onnxruntime.tools.symbolic_shape_infer import (
            SymbolicShapeInference,
        )
    except ImportError as error:
        raise RuntimeError(
            "TensorRT-compatible bundle export requires ONNX Runtime; "
            "install ONNX dependencies with: pip install -e '.[onnx]'"
        ) from error

    inferred_model = SymbolicShapeInference.infer_shapes(
        onnx_model,
        auto_merge=True,
        guess_output_rank=True,
        verbose=0,
    )
    if inferred_model is None:
        raise RuntimeError(f"Symbolic shape inference failed for {path}")

    _make_intermediate_symbolic_dims_unique(inferred_model)
    return inferred_model


def _make_intermediate_symbolic_dims_unique(onnx_model) -> None:
    """Discard unsafe equality guesses made while auto-merging dynamic dims.

    Symbolic shape inference must auto-merge some expressions to finish the
    decoder graph. Those inferred equalities are only hints, and convolution
    rounding can make two of them differ by one at runtime. Unique symbolic
    names retain rank metadata for TensorRT without misleading ORT's memory
    planner.
    """
    for value_index, value in enumerate(onnx_model.graph.value_info):
        tensor_type = value.type.tensor_type
        if not tensor_type.HasField("shape"):
            continue
        for axis, dimension in enumerate(tensor_type.shape.dim):
            if dimension.HasField("dim_param"):
                dimension.dim_param = f"holos_dim_{value_index}_{axis}"


def _check_complete_tensor_shapes(onnx_model, path: Path) -> int:
    """Ensure every produced tensor has type/shape metadata for TensorRT."""
    shaped_values = {
        value.name
        for value in (
            *onnx_model.graph.input,
            *onnx_model.graph.value_info,
            *onnx_model.graph.output,
        )
        if value.type.HasField("tensor_type")
        and value.type.tensor_type.HasField("shape")
    }
    missing = sorted(
        output
        for node in onnx_model.graph.node
        for output in node.output
        if output and output not in shaped_values
    )
    if missing:
        preview = ", ".join(missing[:5])
        suffix = "..." if len(missing) > 5 else ""
        raise RuntimeError(
            f"TensorRT shape validation failed for {path}: "
            f"{len(missing)} node outputs have no inferred shape "
            f"({preview}{suffix})"
        )
    return len(shaped_values)


def export_onnx(
    output: Path,
    weights: str = "patriotyk/HolosTTS",
    voices: str = "voices.npz",
    voice_name: Optional[str] = None,
    voice_tensor_path: Optional[Path] = None,
    device_name: str = "cpu",
    opset: int = 18,
    sequence_length: int = 16,
    max_frames: int = 1000,
    harmonic_representation: str = "polar",
    dynamic_text: bool = True,
    verify_runtime: bool = False,
    force: bool = False,
    dynamic_frames: bool = False,
):
    try:
        import onnx
    except ImportError as error:
        raise RuntimeError("Install ONNX dependencies with: pip install -e '.[onnx]'") from error

    output = output.expanduser().resolve()
    if output.exists() and not force:
        raise FileExistsError(f"{output} already exists; pass --force to overwrite it")
    output.parent.mkdir(parents=True, exist_ok=True)

    if max_frames <= 0:
        raise ValueError("max_frames must be positive")
    if opset < 18:
        raise ValueError("opset must be at least 18")

    device = torch.device(device_name)
    print(f"Loading HolosTTS from {weights!r} on {device}...")
    model = HolosTTS(
        weights_path=weights,
        voices_path=voices,
        device=str(device),
        harmonic_representation=harmonic_representation,
        onnx_max_frames=max_frames,
    ).eval()



    tokens = _dummy_tokens(model, sequence_length, device)
    voice = _load_voice_tensor(model, voice_name, voice_tensor_path, device)
    input_lengths = torch.tensor([sequence_length], dtype=torch.int64, device=device)
    if dynamic_frames:
        wrapper = OnnxDynamicInferenceWrapper(
            model,
            max_frames=max_frames,
        ).eval()
        export_args = (tokens, voice)
        input_names = ["tokens", "voice"]
    else:
        wrapper = OnnxInferenceWrapper(model).eval()
        export_args = (tokens, voice, input_lengths)
        input_names = ["tokens", "voice", "input_lengths"]

    dynamic_axes = {}
    if dynamic_text:
        dynamic_axes["tokens"] = {1: "text_length"}
    if dynamic_frames:
        dynamic_axes["audio"] = {1: "audio_samples"}

    export_kwargs = {
        "export_params": True,
        "input_names": input_names,
        "output_names": ["audio", "audio_lengths"],
        "opset_version": opset,
        "do_constant_folding": True,
        "dynamic_axes": dynamic_axes or None,
        "training": torch.onnx.TrainingMode.EVAL,
        "custom_opsets": {"com.microsoft": 1},
    }
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        export_kwargs["dynamo"] = False

    frame_mode = "dynamic" if dynamic_frames else "fixed-capacity"
    print(
        f"Exporting {output} (opset {opset}, {frame_mode} frames, "
        f"max_frames {max_frames})..."
    )
    with torch.inference_mode():
        torch.onnx.export(
            wrapper,
            export_args,
            str(output),
            **export_kwargs,
        )

    print("Checking the ONNX graph...")
    onnx.checker.check_model(str(output))
    onnx_model = onnx.load(str(output), load_external_data=False)
    attention_nodes = _check_fused_attention(
        onnx_model,
        expected=_expected_attention_nodes(wrapper),
        path=output,
    )
    projection_nodes = _check_fused_projections(
        onnx_model,
        expected=_expected_fused_projections(wrapper),
        path=output,
    )
    input_summary = ", ".join(value.name for value in onnx_model.graph.input)
    output_summary = ", ".join(value.name for value in onnx_model.graph.output)
    print(f"Inputs: {input_summary}")
    print(f"Outputs: {output_summary}")
    print(f"Fused ONNX Runtime MultiHeadAttention nodes: {attention_nodes}")
    print(
        "Fused projection nodes: "
        f"qkv={projection_nodes[0]}, gate_up={projection_nodes[1]}"
    )

    if verify_runtime:
        try:
            import onnxruntime as ort
        except ImportError as error:
            raise RuntimeError("--verify-runtime requires onnxruntime") from error

        with torch.inference_mode():
            expected_encoded = model.encode(tokens, voice, input_lengths)
            expected_length = int(
                torch.clamp(
                    expected_encoded[-1] * model.decoder.generator.total_upsample,
                    max=max_frames * model.decoder.generator.total_upsample * 2,
                ).item()
            )

        print("Running one ONNX Runtime smoke test...")
        session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
        runtime_inputs = {
            "tokens": tokens.cpu().numpy(),
            "voice": voice.cpu().numpy(),
        }
        if not dynamic_frames:
            runtime_inputs["input_lengths"] = input_lengths.cpu().numpy()
        audio, audio_lengths = session.run(None, runtime_inputs)
        if not np.isfinite(audio).all():
            raise RuntimeError("ONNX Runtime produced non-finite audio")
        valid_length = int(audio_lengths[0])
        if valid_length < 0 or valid_length > audio.shape[-1]:
            raise RuntimeError(
                f"Invalid audio length {valid_length} for output shape {audio.shape}"
            )
        if valid_length != expected_length:
            raise RuntimeError(
                "ONNX duration mismatch: "
                f"PyTorch produced {expected_length} valid samples, "
                f"ONNX Runtime produced {valid_length}"
            )
        if dynamic_frames and audio.shape[-1] != expected_length:
            raise RuntimeError(
                "Dynamic ONNX output was not sized to its valid duration: "
                f"shape={audio.shape}, valid_samples={expected_length}"
            )
        print(f"Runtime output: audio={audio.shape}, valid_samples={valid_length}")

    size_mib = output.stat().st_size / (1024 * 1024)
    print(f"Export complete: {output} ({size_mib:.1f} MiB)")
    return output


def export_onnx_bundle(
    output_dir: Path,
    weights: str = "patriotyk/HolosTTS",
    voices: str = "voices.npz",
    voice_name: Optional[str] = None,
    voice_tensor_path: Optional[Path] = None,
    device_name: str = "cpu",
    opset: int = 18,
    sequence_length: int = 16,
    max_frames: int = 1000,
    harmonic_representation: str = "polar",
    attention_backend: str = "tensorrt",
    verify_runtime: bool = False,
    force: bool = False,
):
    """Export a two-stage bundle with an exact-length dynamic decoder."""
    try:
        import onnx
    except ImportError as error:
        raise RuntimeError("Install ONNX dependencies with: pip install -e '.[onnx]'") from error

    if max_frames <= 0:
        raise ValueError("max_frames must be positive")
    if opset < 18:
        raise ValueError("opset must be at least 18")
    attention_formats = {
        "tensorrt": "tensorrt_fused_attention",
        "onnxruntime": "ort_multi_head_attention",
    }
    try:
        attention_format = attention_formats[attention_backend]
    except KeyError as error:
        raise ValueError(
            "attention_backend must be 'tensorrt' or 'onnxruntime'"
        ) from error

    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    encoder_path = output_dir / "encoder.onnx"
    prosody_path = output_dir / "prosody.onnx"
    decoder_path = output_dir / "decoder.onnx"
    voices_path = output_dir / "voices.npz"
    manifest_path = output_dir / "holos_onnx.json"
    _check_output_targets(
        [encoder_path, prosody_path, decoder_path, voices_path, manifest_path],
        force=force,
    )

    device = torch.device(device_name)
    print(f"Loading HolosTTS from {weights!r} on {device}...")
    model = HolosTTS(
        weights_path=weights,
        voices_path=voices,
        device=str(device),
        harmonic_representation=harmonic_representation,
        onnx_max_frames=max_frames,
    ).eval()

    tokens = _dummy_tokens(model, sequence_length, device)
    voice = _load_voice_tensor(model, voice_name, voice_tensor_path, device)
    encoder = OnnxDurationEncoderWrapper(model).eval()
    prosody = OnnxProsodyDecoderWrapper(model).eval()
    decoder = OnnxWaveformDecoderWrapper(model).eval()
    encoder_attention_layers = _set_attention_format(
        encoder,
        attention_format,
    )
    prosody_attention_layers = _set_attention_format(
        prosody,
        attention_format,
    )

    export_common = {
        "export_params": True,
        "opset_version": opset,
        "do_constant_folding": True,
        "training": torch.onnx.TrainingMode.EVAL,
    }
    if attention_backend == "onnxruntime":
        export_common["custom_opsets"] = {"com.microsoft": 1}
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        export_common["dynamo"] = False

    print(f"Exporting duration/text stage to {encoder_path}...")
    with torch.inference_mode():
        torch.onnx.export(
            encoder,
            (tokens, voice),
            str(encoder_path),
            input_names=["tokens", "voice"],
            output_names=[
                "acoustic_tokens",
                "prosodic_tokens",
                "durations",
            ],
            dynamic_axes={
                "tokens": {1: "text_length"},
                "acoustic_tokens": {2: "text_length"},
                "prosodic_tokens": {1: "text_length"},
                "durations": {1: "text_length"},
            },
            **export_common,
        )

        acoustic_tokens, prosodic_tokens, durations = encoder(
            tokens,
            voice,
        )
        frame_count = int(durations.sum().item())
        frame_to_token = _frame_to_token_tensor(durations)

    print(
        f"Exporting dynamic prosody stage to {prosody_path} "
        f"(example width {frame_count})..."
    )
    with torch.inference_mode():
        torch.onnx.export(
            prosody,
            (
                prosodic_tokens,
                frame_to_token,
                voice,
            ),
            str(prosody_path),
            input_names=[
                "prosodic_tokens",
                "frame_to_token",
                "voice",
            ],
            output_names=["f0", "energy"],
            dynamic_axes={
                "prosodic_tokens": {1: "text_length"},
                "frame_to_token": {0: "frames"},
                "f0": {1: "prosody_frames"},
                "energy": {1: "prosody_frames"},
            },
            **export_common,
        )
        f0, energy = prosody(prosodic_tokens, frame_to_token, voice)

    print(
        f"Exporting dynamic waveform stage to {decoder_path} "
        f"(example width {frame_count})..."
    )
    with torch.inference_mode():
        torch.onnx.export(
            decoder,
            (
                acoustic_tokens,
                frame_to_token,
                f0,
                energy,
                voice,
            ),
            str(decoder_path),
            input_names=[
                "acoustic_tokens",
                "frame_to_token",
                "f0",
                "energy",
                "voice",
            ],
            output_names=["audio", "audio_lengths"],
            dynamic_axes={
                "acoustic_tokens": {2: "text_length"},
                "frame_to_token": {0: "frames"},
                "f0": {1: "prosody_frames"},
                "energy": {1: "prosody_frames"},
                "audio": {1: "audio_samples"},
            },
            **export_common,
        )

    encoder_model = onnx.load(str(encoder_path), load_external_data=False)
    prosody_model = onnx.load(str(prosody_path), load_external_data=False)
    decoder_model = onnx.load(str(decoder_path), load_external_data=False)

    print("Inferring symbolic tensor shapes for TensorRT...")
    encoder_model = _infer_symbolic_shapes(encoder_model, encoder_path)
    prosody_model = _infer_symbolic_shapes(prosody_model, prosody_path)
    decoder_model = _infer_symbolic_shapes(decoder_model, decoder_path)
    onnx.save_model(encoder_model, str(encoder_path))
    onnx.save_model(prosody_model, str(prosody_path))
    onnx.save_model(decoder_model, str(decoder_path))

    print("Checking split ONNX graphs...")
    onnx.checker.check_model(encoder_model)
    onnx.checker.check_model(prosody_model)
    onnx.checker.check_model(decoder_model)
    encoder_shaped_values = _check_complete_tensor_shapes(
        encoder_model,
        encoder_path,
    )
    prosody_shaped_values = _check_complete_tensor_shapes(
        prosody_model,
        prosody_path,
    )
    decoder_shaped_values = _check_complete_tensor_shapes(
        decoder_model,
        decoder_path,
    )
    attention_check = (
        _check_tensorrt_attention
        if attention_backend == "tensorrt"
        else _check_fused_attention
    )
    encoder_attention_nodes = attention_check(
        encoder_model,
        expected=encoder_attention_layers,
        path=encoder_path,
    )
    prosody_attention_nodes = attention_check(
        prosody_model,
        expected=prosody_attention_layers,
        path=prosody_path,
    )
    decoder_attention_nodes = attention_check(
        decoder_model,
        expected=0,
        path=decoder_path,
    )
    encoder_projection_nodes = _check_fused_projections(
        encoder_model,
        expected=_expected_fused_projections(encoder),
        path=encoder_path,
    )
    prosody_projection_nodes = _check_fused_projections(
        prosody_model,
        expected=_expected_fused_projections(prosody),
        path=prosody_path,
    )
    decoder_projection_nodes = _check_fused_projections(
        decoder_model,
        expected=_expected_fused_projections(decoder),
        path=decoder_path,
    )
    prosody_instance_norm_nodes = _check_instance_norm(
        prosody_model,
        expected=_expected_instance_norm_nodes(prosody),
        path=prosody_path,
    )
    decoder_instance_norm_nodes = _check_instance_norm(
        decoder_model,
        expected=_expected_instance_norm_nodes(decoder),
        path=decoder_path,
    )
    print(
        f"{attention_backend} attention nodes/patterns: "
        f"encoder={encoder_attention_nodes}, prosody={prosody_attention_nodes}, "
        f"decoder={decoder_attention_nodes}"
    )
    print(
        "Fused projection nodes (qkv/gate_up): "
        f"encoder={encoder_projection_nodes}, "
        f"prosody={prosody_projection_nodes}, "
        f"decoder={decoder_projection_nodes}"
    )
    print(
        "Mask-free ONNX InstanceNormalization nodes: "
        f"prosody={prosody_instance_norm_nodes}, "
        f"decoder={decoder_instance_norm_nodes}"
    )
    print(
        "TensorRT shape metadata values: "
        f"encoder={encoder_shaped_values}, prosody={prosody_shaped_values}, "
        f"decoder={decoder_shaped_values}"
    )

    exported_voices = _export_voices(model, voices_path)
    if exported_voices is None and voices_path.exists():
        voices_path.unlink()

    manifest = {
        "format": "holos-onnx",
        "format_version": 2,
        "encoder": encoder_path.name,
        "prosody": prosody_path.name,
        "decoder": decoder_path.name,
        "voices": exported_voices,
        "vocab": list(model.tokenizer.vocab),
        "sample_rate": 24000,
        "samples_per_frame": decoder.samples_per_frame,
        "max_frames": max_frames,
        "decoder_mode": "exact",
        "alignment_format": "gather",
        "normalization_format": "instance_norm",
        "attention_format": (
            "tensorrt_fused_bmm"
            if attention_backend == "tensorrt"
            else "ort_multi_head_attention"
        ),
        "projection_format": "fused_qkv_swiglu",
        "shape_inference_format": "onnxruntime_symbolic",
        "tensorrt_profile": {
            "min_text_length": 2,
            "opt_text_length": max(128, sequence_length),
            "max_text_length": max(512, sequence_length),
            "min_frames": min(2, max_frames),
            "opt_frames": min(256, max_frames),
            "max_frames": max_frames,
        },
        "harmonic_representation": harmonic_representation,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    if verify_runtime:
        try:
            import onnxruntime as ort
        except ImportError as error:
            raise RuntimeError("--verify-runtime requires onnxruntime") from error

        print("Running split ONNX Runtime smoke tests...")
        encoder_session = ort.InferenceSession(
            str(encoder_path),
            providers=["CPUExecutionProvider"],
        )
        encoded = encoder_session.run(
            None,
            {
                "tokens": tokens.cpu().numpy(),
                "voice": voice.cpu().numpy(),
            },
        )
        np.testing.assert_allclose(
            encoded[0],
            acoustic_tokens.cpu().numpy(),
            atol=2e-5,
            rtol=2e-4,
        )
        np.testing.assert_allclose(
            encoded[1],
            prosodic_tokens.cpu().numpy(),
            atol=2e-5,
            rtol=2e-4,
        )
        np.testing.assert_array_equal(encoded[2], durations.cpu().numpy())

        prosody_session = ort.InferenceSession(
            str(prosody_path),
            providers=["CPUExecutionProvider"],
        )
        runtime_frame_to_token = _frame_to_token_tensor(durations).cpu().numpy()
        runtime_f0, runtime_energy = prosody_session.run(
            None,
            {
                "prosodic_tokens": encoded[1],
                "frame_to_token": runtime_frame_to_token,
                "voice": voice.cpu().numpy(),
            },
        )
        np.testing.assert_allclose(
            runtime_f0,
            f0.cpu().numpy(),
            atol=2e-5,
            rtol=2e-4,
        )
        np.testing.assert_allclose(
            runtime_energy,
            energy.cpu().numpy(),
            atol=2e-5,
            rtol=2e-4,
        )

        decoder_session = ort.InferenceSession(
            str(decoder_path),
            providers=["CPUExecutionProvider"],
        )
        audio, audio_lengths = decoder_session.run(
            None,
            {
                "acoustic_tokens": encoded[0],
                "frame_to_token": runtime_frame_to_token,
                "f0": runtime_f0,
                "energy": runtime_energy,
                "voice": voice.cpu().numpy(),
            },
        )
        expected_length = frame_count * decoder.samples_per_frame
        if audio.shape != (1, expected_length):
            raise RuntimeError(
                f"Unexpected decoder output shape {audio.shape}; "
                f"expected (1, {expected_length})"
            )
        if int(audio_lengths[0]) != expected_length:
            raise RuntimeError(
                f"Expected {expected_length} valid samples, got {audio_lengths[0]}"
            )
        if not np.isfinite(audio).all():
            raise RuntimeError("Split ONNX decoder produced non-finite audio")
        print(
            f"  exact_frames={frame_count}: audio={audio.shape}, "
            f"valid_samples={int(audio_lengths[0])}"
        )

    size_mib = sum(
        path.stat().st_size
        for path in (encoder_path, prosody_path, decoder_path)
    ) / (1024 * 1024)
    print(f"Bundle export complete: {output_dir} ({size_mib:.1f} MiB of ONNX)")
    return output_dir


def build_parser():
    parser = argparse.ArgumentParser(description="Export HolosTTS to ONNX")
    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument(
        "--output",
        type=Path,
        help=(
            "Export one self-contained graph; combine with --dynamic-frames "
            "for CPU and mobile runtimes"
        ),
    )
    output_group.add_argument(
        "--bundle-dir",
        type=Path,
        help="Export the multi-stage bundle used by the torch-free holos-onnx package",
    )
    parser.add_argument("--weights", default="patriotyk/HolosTTS")
    parser.add_argument("--voices", default="voices.npz")
    parser.add_argument("--voice", dest="voice_name")
    parser.add_argument("--voice-tensor", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--sequence-length", type=int, default=16)
    parser.add_argument("--max-frames", type=int, default=1000)
    parser.add_argument(
        "--harmonic-representation",
        choices=("polar", "cartesian"),
        default="polar",
        help="Use polar for existing checkpoints and cartesian for retrained decoders",
    )
    parser.add_argument(
        "--attention-backend",
        choices=("tensorrt", "onnxruntime"),
        default="tensorrt",
        help="Export TensorRT BMM attention or ONNX Runtime contrib MHA",
    )
    parser.add_argument(
        "--static-text",
        action="store_true",
        help="Keep the dummy token length static (useful for RKNN)",
    )
    parser.add_argument(
        "--dynamic-frames",
        action="store_true",
        help=(
            "Use an exact data-dependent waveform length in single-file "
            "--output mode (recommended for CPU/mobile; omit for fixed-capacity "
            "RKNN-style graphs)"
        ),
    )
    parser.add_argument("--verify-runtime", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None):
    parser = build_parser()
    args = parser.parse_args(argv)
    common = {
        "weights": args.weights,
        "voices": args.voices,
        "voice_name": args.voice_name,
        "voice_tensor_path": args.voice_tensor,
        "device_name": args.device,
        "opset": args.opset,
        "sequence_length": args.sequence_length,
        "max_frames": args.max_frames,
        "harmonic_representation": args.harmonic_representation,
        "verify_runtime": args.verify_runtime,
        "force": args.force,
    }
    if args.bundle_dir is not None:
        if args.dynamic_frames:
            parser.error(
                "--dynamic-frames applies only to --output; "
                "the split decoder is already dynamic"
            )
        export_onnx_bundle(
            output_dir=args.bundle_dir,
            attention_backend=args.attention_backend,
            **common,
        )
        return

    export_onnx(
        output=args.output or Path("holos.onnx"),
        dynamic_text=not args.static_text,
        dynamic_frames=args.dynamic_frames,
        **common,
    )


if __name__ == "__main__":
    main()
