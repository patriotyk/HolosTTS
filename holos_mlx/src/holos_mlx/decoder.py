"""HolosTTS waveform decoder in MLX (channels-last)."""

import math

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .layers import AdainResBlk1d, Conv1d, ConvTranspose1d, leaky_relu


def atan2_approx(imag, real):
    """The PyTorch decoder's polynomial atan2 (max error about 1.2e-5 rad)."""
    abs_real, abs_imag = mx.abs(real), mx.abs(imag)
    high = mx.maximum(abs_real, abs_imag)
    low = mx.minimum(abs_real, abs_imag)
    ratio = low / mx.where(high > 0, high, 1.0)
    squared = ratio * ratio
    angle = ratio * (0.9998660 + squared * (-0.3302995 + squared * (0.1801410 + squared * (-0.0851330 + squared * 0.0208351))))
    angle = mx.where(abs_imag > abs_real, math.pi / 2 - angle, angle)
    angle = mx.where(real < 0, math.pi - angle, angle)
    angle = mx.where(imag < 0, -angle, angle)
    return mx.where(high == 0, 0.0, angle)


class STFT:
    """``CustomSTFT``: periodic Hann window, reflect padding, overlap-add with
    exact window normalization. Not an nn.Module: it holds no weights."""

    def __init__(self, n_fft=40, hop_length=10):
        self.n_fft, self.hop = n_fft, hop_length
        self.bins = n_fft // 2 + 1
        window = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n_fft) / n_fft)).astype(np.float32)
        t = np.arange(n_fft, dtype=np.float32)
        k = np.arange(self.bins, dtype=np.float32)
        angle = 2 * np.pi * k[:, None] * t[None, :] / n_fft
        cos_basis, sin_basis = np.cos(angle).astype(np.float32), np.sin(angle).astype(np.float32)
        # Analysis: frames [n, n_fft] @ [n_fft, 2 * bins] -> real, imaginary.
        self.dft = mx.array(np.concatenate([cos_basis, -sin_basis]).T * window[:, None])
        scale = np.full((self.bins, 1), 2.0 / n_fft, dtype=np.float32)
        scale[0] = scale[-1] = 1.0 / n_fft
        # Synthesis: [n, 2 * bins] @ [2 * bins, n_fft] -> windowed frames.
        self.idft = mx.array(np.concatenate([cos_basis * scale * window, -sin_basis * scale * window]))
        self.window_squared = mx.array(window ** 2)

    def transform(self, signal):
        """``[1, L]`` -> magnitude and phase, ``[1, L / hop + 1, bins]`` each."""
        pad = self.n_fft // 2
        x = mx.concatenate([signal[:, 1:pad + 1][:, ::-1], signal, signal[:, -pad - 1:-1][:, ::-1]], axis=1)
        frames = (x.shape[1] - self.n_fft) // self.hop + 1
        windows = mx.as_strided(x, (x.shape[0], frames, self.n_fft), (x.shape[1], self.hop, 1))
        spectrum = windows @ self.dft
        real, imag = spectrum[..., :self.bins], spectrum[..., self.bins:]
        magnitude = mx.sqrt(real * real + imag * imag + 1e-8)
        phase = atan2_approx(imag, real)
        # The DC bin's imaginary part is zero: its phase is 0 or pi, as in PyTorch.
        dc = mx.where(real[..., :1] < 0, math.pi, 0.0)
        return magnitude, mx.concatenate([dc, phase[..., 1:]], axis=-1)

    def _overlap_add(self, frames):
        """``[1, n, n_fft]`` frames with hop n_fft / q -> ``[1, (n + q - 1) * hop]``."""
        batch, count, _ = frames.shape
        parts = self.n_fft // self.hop
        frames = frames.reshape(batch, count, parts, self.hop)
        out = mx.zeros((batch, count + parts - 1, self.hop))
        for part in range(parts):
            out = out + mx.pad(frames[:, :, part], [(0, 0), (part, parts - 1 - part), (0, 0)])
        return out.reshape(batch, -1)

    def inverse(self, magnitude, phase):
        x = mx.concatenate([magnitude * mx.cos(phase), magnitude * mx.sin(phase)], axis=-1)
        output = self._overlap_add(x @ self.idft)
        window_sum = self._overlap_add(mx.broadcast_to(self.window_squared, (1, x.shape[1], self.n_fft)))
        output = output / mx.maximum(window_sum, 1e-8)
        cut = self.n_fft // 2
        return output[:, cut:-cut]


class FiLM1d(nn.Module):
    def __init__(self, style_dim, channels):
        super().__init__()
        self.fc = nn.Linear(style_dim, channels * 2)

    def __call__(self, style):
        """The style's gamma and beta, ``[1, 1, C]`` each."""
        return mx.split(self.fc(style)[:, None], 2, axis=-1)


_RESIDUAL_UNITS = {}


def _residual_unit(padding1, dilation1, padding2):
    """One FiLM-Snake-conv residual unit, compiled for any length. The
    generator's activations are long, and the element-wise passes between
    the convolutions took about a third of its time before fusing."""
    key = (padding1, dilation1, padding2)
    if key not in _RESIDUAL_UNITS:
        def unit(x, weight1, bias1, weight2, bias2, gamma1, beta1, alpha1, gamma2, beta2, alpha2):
            y = (1 + gamma1) * x + beta1
            y = y + (1 / alpha1) * mx.sin(alpha1 * y) ** 2
            y = mx.conv1d(y, weight1, 1, padding1, dilation1) + bias1
            y = (1 + gamma2) * y + beta2
            y = y + (1 / alpha2) * mx.sin(alpha2 * y) ** 2
            return x + mx.conv1d(y, weight2, 1, padding2) + bias2

        _RESIDUAL_UNITS[key] = mx.compile(unit, shapeless=True)
    return _RESIDUAL_UNITS[key]


class FiLMResBlock1(nn.Module):
    def __init__(self, channels, kernel_size, dilations, style_dim):
        super().__init__()
        self.convs1 = [
            Conv1d(channels, channels, kernel_size, padding=(kernel_size - 1) * d // 2, dilation=d) for d in dilations
        ]
        self.convs2 = [Conv1d(channels, channels, kernel_size, padding=(kernel_size - 1) // 2) for _ in dilations]
        self.film1 = [FiLM1d(style_dim, channels) for _ in dilations]
        self.film2 = [FiLM1d(style_dim, channels) for _ in dilations]
        self.alpha1 = [mx.ones((1, 1, channels)) for _ in dilations]
        self.alpha2 = [mx.ones((1, 1, channels)) for _ in dilations]

    def __call__(self, x, style):
        for conv1, conv2, film1, film2, alpha1, alpha2 in zip(
            self.convs1, self.convs2, self.film1, self.film2, self.alpha1, self.alpha2
        ):
            unit = _residual_unit(conv1.padding, conv1.dilation, conv2.padding)
            x = unit(x, conv1.weight, conv1.bias, conv2.weight, conv2.bias,
                     *film1(style), alpha1, *film2(style), alpha2)
        return x


class SourceMerge(nn.Module):
    """``SourceModuleHnNSF``'s linear merge of the harmonics."""

    def __init__(self, harmonics):
        super().__init__()
        self.l_linear = nn.Linear(harmonics, 1)


class Generator(nn.Module):
    """iSTFT generator with a harmonic-plus-noise source."""

    sample_rate = 24000
    harmonics = 9
    sine_amp = 0.1
    noise_std = 0.003
    voiced_threshold = 10.0

    def __init__(self, style_dim=128, channels=512, upsample_rates=(6, 5), upsample_kernels=(12, 10),
                 resblock_kernels=(3, 7, 11), dilations=(1, 3, 5), n_fft=40, hop=10):
        super().__init__()
        self.m_source = SourceMerge(self.harmonics)
        self.ups, self.noise_convs, self.noise_res, self.resblocks = [], [], [], []
        for i, (rate, kernel) in enumerate(zip(upsample_rates, upsample_kernels)):
            out_channels = channels // 2 ** (i + 1)
            self.ups.append(ConvTranspose1d(channels // 2 ** i, out_channels, kernel, rate, (kernel - rate) // 2))
            self.resblocks += [FiLMResBlock1(out_channels, k, dilations, style_dim) for k in resblock_kernels]
            if i + 1 < len(upsample_rates):
                stride = math.prod(upsample_rates[i + 1:])
                self.noise_convs.append(Conv1d(n_fft + 2, out_channels, stride * 2, stride, (stride + 1) // 2))
                self.noise_res.append(FiLMResBlock1(out_channels, 7, dilations, style_dim))
            else:
                self.noise_convs.append(Conv1d(n_fft + 2, out_channels, 1))
                self.noise_res.append(FiLMResBlock1(out_channels, 11, dilations, style_dim))
        self.conv_post = Conv1d(out_channels, n_fft + 2, 7, padding=3)
        self.num_kernels = len(resblock_kernels)
        self.upsample = math.prod(upsample_rates) * hop
        self.stft = STFT(n_fft, hop)

    def source(self, f0, noise=True):
        """F0 ``[1, 2F]`` -> harmonic excitation ``[1, 2F * upsample]``.

        The phase is accumulated as in ``SineGen``: its start in every block
        of constant F0 comes from a float64 cumulative sum on the host (the
        GPU has no float64), the steps inside the block from the GPU. The
        Gaussian noise of every harmonic merges into one Gaussian scaled by
        the merge weights' norm.
        """
        f0_host = np.asarray(f0, dtype=np.float32)[0]
        harmonics = np.arange(1, self.harmonics + 1, dtype=np.float32)
        rad = (f0_host[:, None] * harmonics / np.float32(self.sample_rate)) % np.float32(1)
        block = (rad.astype(np.float64) * self.upsample) % 1
        start = ((np.cumsum(block, axis=0) - block) % 1).astype(np.float32)
        steps = mx.arange(1, self.upsample + 1, dtype=mx.float32)[None, :, None]
        cycles = mx.array(start)[:, None] + steps * mx.array(rad)[:, None]
        phase = (cycles % 1) * (2 * math.pi)
        phase = phase.reshape(1, -1, self.harmonics)
        voiced = mx.repeat(mx.array(f0_host > self.voiced_threshold, dtype=mx.float32), self.upsample)[None, :, None]
        linear = self.m_source.l_linear
        merged = linear(mx.sin(phase) * self.sine_amp * voiced)
        if noise:
            amplitude = voiced * self.noise_std + (1 - voiced) * self.sine_amp / 3
            scale = mx.sqrt(mx.sum(linear.weight * linear.weight))
            merged = merged + amplitude * scale * mx.random.normal(merged.shape)
        return mx.tanh(merged)[..., 0]

    def spectrum(self, x, style, source):
        magnitude, phase = self.stft.transform(source)
        harmonic = mx.concatenate([magnitude, phase], axis=-1)
        for i, up in enumerate(self.ups):
            x = up(leaky_relu(x, 0.1))
            if i == len(self.ups) - 1:
                x = mx.concatenate([x[:, 1:2], x], axis=1)  # ReflectionPad1d((1, 0))
            excitation = self.noise_convs[i](harmonic)
            length = min(x.shape[1], excitation.shape[1])
            x = x[:, :length] + self.noise_res[i](excitation[:, :length], style)
            blocks = self.resblocks[i * self.num_kernels:(i + 1) * self.num_kernels]
            x = sum(block(x, style) for block in blocks) / self.num_kernels
        x = self.conv_post(leaky_relu(x, 0.01))
        bins = self.stft.bins
        return mx.exp(x[..., :bins]), mx.sin(x[..., bins:])

    def __call__(self, x, style, source):
        return self.stft.inverse(*self.spectrum(x, style, source))


class Decoder(nn.Module):
    def __init__(self, dim_in=512, style_dim=128):
        super().__init__()
        self.encode = AdainResBlk1d(dim_in + 2, 1024, style_dim)
        self.decode = [AdainResBlk1d(1024 + 2 + 64, 1024, style_dim) for _ in range(3)]
        self.decode.append(AdainResBlk1d(1024 + 2 + 64, 512, style_dim, upsample=True))
        self.F0_conv = Conv1d(1, 1, 3, stride=2, padding=1)
        self.N_conv = Conv1d(1, 1, 3, stride=2, padding=1)
        self.asr_res = Conv1d(dim_in, 64, 1)
        self.generator = Generator(style_dim)

    def features(self, asr, f0, energy, style):
        """Aligned acoustic features ``[1, F, C]``, F0 and energy ``[1, 2F]`` ->
        generator input ``[1, 2F, 512]``."""
        f0 = self.F0_conv(f0[..., None])
        energy = self.N_conv(energy[..., None])
        x = self.encode(mx.concatenate([asr, f0, energy], axis=-1), style)
        asr_res = self.asr_res(asr)
        residual = True
        for block in self.decode:
            # Every block up to and including the upsampling one gets the inputs again.
            if residual:
                x = mx.concatenate([x, asr_res, f0, energy], axis=-1)
            x = block(x, style)
            residual = residual and not block.upsample
        return x

    def __call__(self, asr, f0, energy, style, source):
        return self.generator(self.features(asr, f0, energy, style), style, source)
