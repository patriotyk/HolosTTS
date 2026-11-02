"""HolosTTS text, duration and prosody stages in MLX.

Tensors are channels-last, ``[batch, time, channels]``, the layout of MLX
convolutions. The modules keep the parameter names of the PyTorch model (see
convert.py) and implement its single-utterance inference path, so they have
no padding masks.
"""

import math

import mlx.core as mx
import mlx.nn as nn


def leaky_relu(x, slope):
    return mx.maximum(x, slope * x)


class Conv1d(nn.Module):
    """Convolution with weights ``[out, kernel, in / groups]`` (weight norm folded)."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super().__init__()
        self.weight = mx.zeros((out_channels, kernel_size, in_channels // groups))
        if bias:
            self.bias = mx.zeros((out_channels,))
        self.stride, self.padding, self.dilation, self.groups = stride, padding, dilation, groups

    def __call__(self, x):
        y = mx.conv1d(x, self.weight, self.stride, self.padding, self.dilation, self.groups)
        return y + self.bias if "bias" in self else y


class ConvTranspose1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.weight = mx.zeros((out_channels, kernel_size, in_channels))
        self.bias = mx.zeros((out_channels,))
        self.stride, self.padding = stride, padding

    def __call__(self, x):
        return mx.conv_transpose1d(x, self.weight, self.stride, self.padding) + self.bias


class ChannelNorm(nn.Module):
    """``LayerNormCausal``: layer norm over channels with gamma and beta."""

    def __init__(self, channels):
        super().__init__()
        self.gamma = mx.ones((channels,))
        self.beta = mx.zeros((channels,))

    def __call__(self, x):
        return mx.fast.layer_norm(x, self.gamma, self.beta, 1e-5)


class ConvNormBlock(nn.Module):
    def __init__(self, channels, kernel_size):
        super().__init__()
        self.conv = Conv1d(channels, channels, kernel_size, padding=(kernel_size - 1) // 2)
        self.norm = ChannelNorm(channels)

    def __call__(self, x):
        return leaky_relu(self.norm(self.conv(x)), 0.2)


class EncoderLayer(nn.Module):
    """Post-norm transformer layer with SwiGLU and optional interleaved RoPE."""

    def __init__(self, d_model, nhead, dim_feedforward):
        super().__init__()
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.gate_proj = nn.Linear(d_model, dim_feedforward)
        self.up_proj = nn.Linear(d_model, dim_feedforward)
        self.down_proj = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def __call__(self, x, rope_dims=0):
        batch, length, channels = x.shape

        def heads(y):
            return y.reshape(batch, length, self.nhead, self.head_dim).transpose(0, 2, 1, 3)

        q, k, v = heads(self.q_proj(x)), heads(self.k_proj(x)), heads(self.v_proj(x))
        if rope_dims:
            # Rotates the pairs (0, 1), (2, 3), ... of the first rope_dims
            # features, as RotaryPositionalEmbedding does.
            q = mx.fast.rope(q, rope_dims, traditional=True, base=10000.0, scale=1.0, offset=0)
            k = mx.fast.rope(k, rope_dims, traditional=True, base=10000.0, scale=1.0, offset=0)
        context = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.head_dim ** -0.5)
        context = context.transpose(0, 2, 1, 3).reshape(batch, length, channels)
        x = self.norm1(x + self.out_proj(context))
        return self.norm2(x + self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x)))


class MultiScaleConvPE(nn.Module):
    def __init__(self, channels, kernel_sizes=(3, 7, 15, 31)):
        super().__init__()
        self.convs = [Conv1d(channels, channels, k, padding=k // 2, groups=channels) for k in kernel_sizes]
        self.gate = mx.zeros((len(kernel_sizes), 1, 1, channels))

    def __call__(self, x):
        weights = mx.softmax(self.gate, axis=0)
        return sum(conv(x) * weights[i] for i, conv in enumerate(self.convs))


class TextEncoder(nn.Module):
    def __init__(self, n_symbols, channels=512, kernel_size=5, depth=3, n_heads=4, n_layers=4, conv_pe=False):
        super().__init__()
        self.embedding = nn.Embedding(n_symbols, channels)
        self.cnn = [ConvNormBlock(channels, kernel_size) for _ in range(depth)]
        self.layers = [EncoderLayer(channels, n_heads, int(channels * 8 / 3)) for _ in range(n_layers)]
        if conv_pe:
            self.conv_pe = MultiScaleConvPE(channels)
        # Partial RoPE: half of each head.
        self.rope_dims = channels // n_heads // 2

    def __call__(self, tokens):
        x = self.embedding(tokens)
        for block in self.cnn:
            x = block(x)
        if "conv_pe" in self:
            x = x + self.conv_pe(x)
        for layer in self.layers:
            x = layer(x, self.rope_dims)
        return x


class FiLM(nn.Module):
    """Style modulation blended with the input by a learned weight."""

    def __init__(self, channels, style_dim):
        super().__init__()
        self.gamma = nn.Linear(style_dim, channels)
        self.beta = nn.Linear(style_dim, channels)
        self.style_weight = mx.zeros((1,))

    def __call__(self, x, style):
        modulated = self.gamma(style)[:, None] * x + self.beta(style)[:, None]
        weight = mx.sigmoid(self.style_weight)
        return (1 - weight) * x + weight * modulated


class DurationPredictor(nn.Module):
    def __init__(self, channels=512, style_dim=128, n_layers=6, n_heads=8, max_duration=20):
        super().__init__()
        self.film = FiLM(channels, style_dim)
        self.conv_pe = MultiScaleConvPE(channels)
        self.layers = [EncoderLayer(channels, n_heads, channels * 4) for _ in range(n_layers)]
        self.films = [FiLM(channels, style_dim) for _ in range(n_layers)]
        self.ordinal_score = nn.Linear(channels, 1)
        self.ordinal_threshold_gaps = mx.zeros((max_duration - 1,))

    def __call__(self, x, style):
        """Expected frames per token: the sum of P(duration > k)."""
        x = self.film(x, style)
        x = x + self.conv_pe(x)
        for layer, film in zip(self.layers, self.films):
            x = layer(film(x, style))
        gaps = nn.softplus(self.ordinal_threshold_gaps) + 1e-4
        thresholds = mx.concatenate([mx.zeros((1,)), mx.cumsum(gaps)])
        return mx.sigmoid(self.ordinal_score(x) - thresholds).sum(axis=-1)


class AdaLN1d(nn.Module):
    def __init__(self, style_dim, channels):
        super().__init__()
        self.fc = nn.Linear(style_dim, channels * 2)

    def __call__(self, x, style):
        gamma, beta = mx.split(self.fc(style)[:, None], 2, axis=-1)
        return (1 + gamma) * mx.fast.layer_norm(x, None, None, 1e-5) + beta


def pixel_shuffle(x, factor):
    """``PixelShuffle1D`` for channels-last: channel c * factor + j moves to
    time step t * factor + j."""
    batch, length, channels = x.shape
    x = x.reshape(batch, length, channels // factor, factor).transpose(0, 1, 3, 2)
    return x.reshape(batch, length * factor, channels // factor)


class AdainResBlk1d(nn.Module):
    def __init__(self, dim_in, dim_out, style_dim, upsample=False):
        super().__init__()
        self.upsample = upsample
        self.conv1 = Conv1d(dim_in, dim_out, 3, padding=1)
        self.conv2 = Conv1d(dim_out, dim_out, 3, padding=1)
        self.norm1 = AdaLN1d(style_dim, dim_in)
        self.norm2 = AdaLN1d(style_dim, dim_out)
        if dim_in != dim_out:
            self.conv1x1 = Conv1d(dim_in, dim_out, 1, bias=False)
        if upsample:
            self.pool = Conv1d(dim_in, dim_in * 2, 3, padding=1, groups=dim_in)

    def __call__(self, x, style):
        residual = leaky_relu(self.norm1(x, style), 0.2)
        if self.upsample:
            residual = pixel_shuffle(self.pool(residual), 2)
        residual = self.conv2(leaky_relu(self.norm2(self.conv1(residual), style), 0.2))
        shortcut = mx.repeat(x, 2, axis=1) if self.upsample else x
        if "conv1x1" in self:
            shortcut = self.conv1x1(shortcut)
        return (residual + shortcut) / math.sqrt(2)


class ProsodyPredictor(nn.Module):
    def __init__(self, channels=512, style_dim=128, n_heads=4, n_layers=3):
        super().__init__()
        self.input_proj = nn.Linear(channels, channels)
        self.film = FiLM(channels, style_dim)
        self.conv_pe = MultiScaleConvPE(channels)
        self.layers = [EncoderLayer(channels, n_heads, int(channels * 8 / 3)) for _ in range(n_layers)]
        self.rope_dims = channels // n_heads
        half = channels // 2
        self.F0 = [
            AdainResBlk1d(channels, channels, style_dim),
            AdainResBlk1d(channels, half, style_dim, upsample=True),
            AdainResBlk1d(half, half, style_dim),
        ]
        self.N = [
            AdainResBlk1d(channels, channels, style_dim),
            AdainResBlk1d(channels, half, style_dim, upsample=True),
            AdainResBlk1d(half, half, style_dim),
        ]
        self.F0_proj = Conv1d(half, 1, 1)
        self.N_proj = Conv1d(half, 1, 1)

    def __call__(self, aligned, style):
        """Token features per frame ``[1, F, C]`` -> F0 and energy ``[1, 2F]``."""
        x = self.film(self.input_proj(aligned), style)
        x = x + self.conv_pe(x)
        for layer in self.layers:
            x = layer(x, self.rope_dims)
        f0, energy = x, x
        for block in self.F0:
            f0 = block(f0, style)
        for block in self.N:
            energy = block(energy, style)
        return self.F0_proj(f0)[..., 0], self.N_proj(energy)[..., 0]
