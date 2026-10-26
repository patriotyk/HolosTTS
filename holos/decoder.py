import random
import math
from dataclasses import dataclass
import torch
import numpy as np
import torch.nn.functional as F

from torch import nn
from typing import Optional
from torch.nn.utils.parametrizations import weight_norm
from scipy.signal import get_window
from holos.modules import AdainResBlk1d, FiLMResBlock1
from torch.nn import Conv1d, ConvTranspose1d


def atan2_approx(imag, real):
    """Quadrant-correct polynomial atan2 without the ONNX Atan operator.

    The polynomial approximates atan(z) for z in [0, 1], after which the
    original quadrant is reconstructed. Keeping the ratio bounded avoids the
    instability of atan(imag / real) around a zero real component. The maximum
    FP32 angular error is approximately 1.2e-5 radians.
    """
    abs_real = real.abs()
    abs_imag = imag.abs()
    high = torch.maximum(abs_real, abs_imag)
    low = torch.minimum(abs_real, abs_imag)

    # Avoid division by zero without perturbing non-zero complex values. A
    # final mask gives atan2(0, 0) the same canonical value as PyTorch: zero.
    safe_high = torch.where(high > 0, high, torch.ones_like(high))
    ratio = low / safe_high
    ratio_squared = ratio * ratio
    atan_ratio = ratio * (
        0.9998660
        + ratio_squared * (
            -0.3302995
            + ratio_squared * (
                0.1801410
                + ratio_squared * (
                    -0.0851330
                    + ratio_squared * 0.0208351
                )
            )
        )
    )

    half_pi = real.new_tensor(np.pi / 2)
    pi = real.new_tensor(np.pi)
    angle = torch.where(abs_imag > abs_real, half_pi - atan_ratio, atan_ratio)
    angle = torch.where(real < 0, pi - angle, angle)
    angle = torch.where(imag < 0, -angle, angle)
    return torch.where(high == 0, torch.zeros_like(angle), angle)


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


@dataclass
class DecoderStreamState:
    """State for one utterance; pass it back to Decoder.forward_chunk unchanged."""
    asr: Optional[torch.Tensor] = None
    f0: Optional[torch.Tensor] = None
    energy: Optional[torch.Tensor] = None
    source: Optional[torch.Tensor] = None
    style: Optional[torch.Tensor] = None
    phase: Optional[torch.Tensor] = None
    start_frame: int = 0
    total_frames: int = 0
    emitted_samples: int = 0
    finished: bool = False


class CustomSTFT(torch.nn.Module):
    def __init__(self, n_fft=800, hop_length=200, win_length=800, window='hann'):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length

        w = torch.from_numpy(get_window(window, win_length, fftbins=True).astype(np.float32))
        self.register_buffer("window", w, persistent=False)

        t = torch.arange(n_fft).float()
        k = torch.arange(n_fft // 2 + 1).float()
        cos_basis = torch.cos(2 * np.pi * k[:, None] * t[None, :] / n_fft)
        sin_basis = torch.sin(2 * np.pi * k[:, None] * t[None, :] / n_fft)

        self.half_n_fft = n_fft // 2

        # Inverse DFT weights stored as non-persistent buffer (not saved in state_dict)
        scaler = torch.ones(n_fft // 2 + 1) * 2.0
        scaler[0] = 1.0; scaler[n_fft // 2] = 1.0
        scaler = (scaler / n_fft).view(-1, 1)
        re_inv = (cos_basis * scaler) * w
        im_inv = (-sin_basis * scaler) * w
        idft_weights_mat = torch.cat([re_inv, im_inv], dim=0) # [2F, n_fft]
        idft_mat = idft_weights_mat.transpose(0, 1).unsqueeze(-1) # [n_fft, 2F, 1]
        self.register_buffer("idft_weight", idft_mat, persistent=False)

        # Exact overlap-add denominator. It is expanded to the actual number of
        # frames in inverse(), avoiding incorrect tail normalization from a
        # fixed, precomputed maximum-frame buffer.
        self.register_buffer(
            "window_sq_weight",
            w.square().view(1, 1, n_fft),
            persistent=False,
        )

        # Pre-compute DFT basis for fast, accurate GPU transform
        dft_mat = torch.cat([cos_basis, -sin_basis], dim=0) # [2F, n_fft]
        self.register_buffer("dft_mat", dft_mat, persistent=False)


    def transform(self, input_data):
        x = input_data.unsqueeze(1)
        pad = self.n_fft // 2
        left = x[:, :, 1 : pad + 1].flip(dims=[-1])
        right = x.flip(dims=[-1])[:, :, 1 : pad + 1]
        x_padded = torch.cat([left, x, right], dim=-1) # [B, 1, T + 2*pad]

        weight = (self.dft_mat * self.window[None, :]).unsqueeze(1)
        out = F.conv1d(x_padded, weight, stride=self.hop_length)  # [B, 2F, frames]

        real = out[:, :self.n_fft // 2 + 1, :]
        imag = out[:, self.n_fft // 2 + 1:, :]

        magnitude = torch.sqrt(real**2 + imag**2 + 1e-8)
        phase = atan2_approx(imag, real)

        # The imaginary part of the DC bin is mathematically zero. For a
        # negative real value, PyTorch returns +pi while ONNX Runtime's
        # Atan2 decomposition returns -pi. Both angles describe the same
        # complex number, but the following Conv1d treats them as values
        # separated by 2*pi. Canonicalize DC to PyTorch's convention.
        dc_phase = torch.where(
            real[:, :1, :] < 0,
            torch.full_like(real[:, :1, :], torch.pi),
            torch.zeros_like(real[:, :1, :]),
        )
        phase = torch.cat([dc_phase, phase[:, 1:, :]], dim=1)

        return magnitude, phase

    def inverse(self, magnitude, phase):
        re = magnitude * torch.cos(phase)
        im = magnitude * torch.sin(phase)
        x = torch.cat([re, im], dim=1) # [B, 2*F, T]

        # 2. Frame Synthesis and Overlap-Add via ConvTranspose1d
        # x is [B, 2F, T]. idft_weight is [n_fft, 2F, 1]
        weight = self.idft_weight.squeeze(2).transpose(0, 1).unsqueeze(1) # [2F, 1, n_fft]
        output = F.conv_transpose1d(x, weight, stride=self.hop_length)

        # Exact overlap-add normalization for the actual frame count. Computing
        # this with ConvTranspose1d keeps dynamic lengths ONNX-exportable.
        window_sum = F.conv_transpose1d(
            torch.ones_like(x[:, :1, :]),
            self.window_sq_weight,
            stride=self.hop_length,
        )
        output = output / window_sum.clamp(min=1e-8)

        # center=True removes n_fft // 2 samples from both ends.
        cutoff = self.half_n_fft
        return output[:, :, cutoff : -cutoff]


class SineGen(torch.nn.Module):
    """ Definition of sine generator
    SineGen(samp_rate, harmonic_num = 0,
            sine_amp = 0.1, noise_std = 0.003,
            voiced_threshold = 0)
    samp_rate: sampling rate in Hz
    harmonic_num: number of harmonic overtones (default 0)
    sine_amp: amplitude of sine-wavefrom (default 0.1)
    noise_std: std of Gaussian noise (default 0.003)
    voiced_thoreshold: F0 threshold for U/V classification (default 0)
        segment is always sin(np.pi) or cos(0)
    """

    def __init__(self, samp_rate, upsample_scale, harmonic_num=0,
                 sine_amp=0.1, noise_std=0.003,
                 voiced_threshold=0):
        super(SineGen, self).__init__()
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.harmonic_num = harmonic_num
        self.dim = self.harmonic_num + 1
        self.sampling_rate = samp_rate
        self.voiced_threshold = voiced_threshold
        self.upsample_scale = upsample_scale

    def _f02uv(self, f0):
        # generate uv signal
        uv = (f0 > self.voiced_threshold).type(torch.float32)
        return uv

    def _f02sine(self, f0_values, initial_phase=None):
        """ f0_values: (batchsize, length, dim)
            where dim indicates fundamental tone and overtones

        Harmonics start phase-aligned with the fundamental (no random initial
        phase). The source STFT (n_fft=40, 600 Hz bins) merges several
        harmonics per bin, so random relative phases would make the
        generator's excitation features differ between draws. The previous
        downsampled-cumsum implementation also discarded its random offsets.
        initial_phase continues a previous chunk.
        """
        rad_values = (f0_values / self.sampling_rate) % 1
        phase = torch.cumsum(rad_values, dim=1) * (2 * np.pi)
        if initial_phase is not None:
            phase = initial_phase + phase

        final_phase = torch.remainder(phase[:, -1:, :], 2 * np.pi)
        sines = torch.sin(phase)
        return sines, final_phase

    def forward(self, f0, initial_phase=None):
        """ sine_tensor, uv, noise, final_phase = forward(f0, initial_phase)
        """
        harmonics = torch.arange(1, self.harmonic_num + 2, device=f0.device, dtype=f0.dtype).view(1, 1, -1)
        fn = f0 * harmonics

        # generate sine waveforms
        sine_waves, final_phase = self._f02sine(fn, initial_phase)
        sine_waves = sine_waves * self.sine_amp

        # generate uv signal
        uv = self._f02uv(f0)

        noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
        noise = noise_amp * torch.randn_like(sine_waves)

        # then: additive noise
        sine_waves = sine_waves * uv + noise
        return sine_waves, uv, noise, final_phase


class SourceModuleHnNSF(torch.nn.Module):
    """ SourceModule for hn-nsf
    SourceModule(sampling_rate, harmonic_num=0, sine_amp=0.1,
                 add_noise_std=0.003, voiced_threshod=0)
    sampling_rate: sampling_rate in Hz
    harmonic_num: number of harmonic above F0 (default: 0)
    sine_amp: amplitude of sine source signal (default: 0.1)
    add_noise_std: std of additive Gaussian noise (default: 0.003)
        note that amplitude of noise in unvoiced is decided
        by sine_amp
    voiced_threshold: threhold to set U/V given F0 (default: 0)
    Sine_source, noise_source = SourceModuleHnNSF(F0_sampled)
    F0_sampled (batchsize, length, 1)
    Sine_source (batchsize, length, 1)
    noise_source (batchsize, length 1)
    uv (batchsize, length, 1)
    """

    def __init__(self, sampling_rate, upsample_scale, harmonic_num=0, sine_amp=0.1,
                 add_noise_std=0.003, voiced_threshod=0):
        super(SourceModuleHnNSF, self).__init__()

        self.sine_amp = sine_amp
        self.noise_std = add_noise_std

        # to produce sine waveforms
        self.l_sin_gen = SineGen(sampling_rate, upsample_scale, harmonic_num,
                                 sine_amp, add_noise_std, voiced_threshod)

        # to merge source harmonics into a single excitation
        self.l_linear = torch.nn.Linear(harmonic_num + 1, 1)
        self.l_tanh = torch.nn.Tanh()

    def forward(self, x, initial_phase=None, *, return_phase=False):
        """
        Sine_source = SourceModuleHnNSF(F0_sampled, initial_phase)
        F0_sampled (batchsize, length, 1)
        """
        with torch.no_grad():
            sine_wavs, uv, _, final_phase = self.l_sin_gen(x, initial_phase)
        sine_merge = self.l_tanh(self.l_linear(sine_wavs))

        if return_phase or initial_phase is not None:
            return sine_merge, final_phase
        return sine_merge


class Generator(torch.nn.Module):
    def __init__(self, style_dim, resblock_kernel_sizes, upsample_rates, upsample_initial_channel, resblock_dilation_sizes, upsample_kernel_sizes, gen_istft_n_fft, gen_istft_hop_size):
        super(Generator, self).__init__()

        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        resblock = FiLMResBlock1

        self.m_source = SourceModuleHnNSF(
                    sampling_rate=24000,
                    upsample_scale=np.prod(upsample_rates) * gen_istft_hop_size,
                    harmonic_num=8, voiced_threshod=10)
        for param in self.m_source.parameters():
            param.requires_grad = False
        self.f0_upsamp = torch.nn.Upsample(scale_factor=np.prod(upsample_rates) * gen_istft_hop_size)
        self.noise_convs = nn.ModuleList()
        self.noise_res = nn.ModuleList()

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(weight_norm(
                ConvTranspose1d(upsample_initial_channel//(2**i), upsample_initial_channel//(2**(i+1)),
                                k, u, padding=(k-u)//2)))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel//(2**(i+1))
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes,resblock_dilation_sizes)):
                self.resblocks.append(resblock(ch, k, d, style_dim))

            c_cur = upsample_initial_channel // (2 ** (i + 1))

            if i + 1 < len(upsample_rates):  #
                stride_f0 = np.prod(upsample_rates[i + 1:])
                self.noise_convs.append(Conv1d(
                    gen_istft_n_fft + 2, c_cur, kernel_size=stride_f0 * 2, stride=stride_f0, padding=(stride_f0+1) // 2))
                self.noise_res.append(resblock(c_cur, 7, [1,3,5], style_dim))
            else:
                self.noise_convs.append(Conv1d(gen_istft_n_fft + 2, c_cur, kernel_size=1))
                self.noise_res.append(resblock(c_cur, 11, [1,3,5], style_dim))


        self.post_n_fft = gen_istft_n_fft
        self.conv_post = weight_norm(Conv1d(ch, self.post_n_fft + 2, 7, 1, padding=3))
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)
        self.reflection_pad = torch.nn.ReflectionPad1d((1, 0))
        self.total_upsample = np.prod(upsample_rates) * gen_istft_hop_size

        self.stft = CustomSTFT(
            n_fft=gen_istft_n_fft,
            hop_length=gen_istft_hop_size,
            win_length=gen_istft_n_fft,
        )

    def harmonic_features(self, source):
        magnitude, phase = self.stft.transform(source)
        return torch.cat([magnitude, phase], dim=1)


    def forward(self, x, s, f0, full_mask, lengths):
        with torch.no_grad():
            f0_upsampled = self.f0_upsamp(f0[:, None]* full_mask )

            upsample_factor = self.total_upsample
            audio_mask = torch.arange(f0_upsampled.size(-1), device=f0.device)[None, None, :] < (lengths[:, None, None] * upsample_factor)
            audio_mask = audio_mask.float()

            f0_upsampled = (f0_upsampled * audio_mask).transpose(1, 2)
            har_source = self.m_source(f0_upsampled)
            har_source = (har_source.transpose(1, 2) * audio_mask).squeeze(1)

            har = self.harmonic_features(har_source)

        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, 0.1)
            x = self.ups[i](x)
            if i == self.num_upsamples - 1:
                x = self.reflection_pad(x)


            x_source = self.noise_convs[i](har)
            min_len = min(x.size(-1), x_source.size(-1))
            x = x[..., :min_len]
            x_source = x_source[..., :min_len]
            current_mask = F.interpolate(full_mask, size=min_len, mode='nearest')

            x_source = self.noise_res[i](x_source, s, current_mask)

            x = (x + x_source) * current_mask
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i*self.num_kernels+j](x, s, current_mask)
                else:
                    xs += self.resblocks[i*self.num_kernels+j](x, s, current_mask)
            x = (xs / self.num_kernels) * current_mask
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        spec = torch.exp(x[:,:self.post_n_fft // 2 + 1, :]) * current_mask
        phase = torch.sin(x[:, self.post_n_fft // 2 + 1:, :]) * current_mask

        return self.stft.inverse(spec, phase)


    def source(self, f0, initial_phase=None):
        """Generate excitation once per new frame, including its continuation phase."""
        with torch.no_grad():
            upsampled = self.f0_upsamp(f0[:, None]).transpose(1, 2)
            source, phase = self.m_source(upsampled, initial_phase, return_phase=True)
        return source.squeeze(-1), phase

    def spectrum(self, x, s, source):
        """Shared exact-length path for offline and contextual streaming windows."""
        har = self.harmonic_features(source)
        for i in range(self.num_upsamples):
            x = self.ups[i](F.leaky_relu(x, 0.1))
            if i == self.num_upsamples - 1:
                x = self.reflection_pad(x)
            x_source = self.noise_convs[i](har)
            min_len = min(x.size(-1), x_source.size(-1))
            x = x[..., :min_len] + self.noise_res[i](x_source[..., :min_len], s)
            x = sum(self.resblocks[i * self.num_kernels + j](x, s)
                    for j in range(self.num_kernels)) / self.num_kernels
        x = self.conv_post(F.leaky_relu(x))
        return torch.exp(x[:, :self.post_n_fft // 2 + 1]), torch.sin(x[:, self.post_n_fft // 2 + 1:])

    def inference(self, x, s, f0, *, source=None):
        if source is None:
            source, _ = self.source(f0)
        spec, phase = self.spectrum(x, s, source)
        return self.stft.inverse(spec, phase)


class Decoder(nn.Module):
    def __init__(self, dim_in=512, style_dim=128,
                resblock_kernel_sizes = [3,7,11],
                upsample_rates = [6, 5],
                upsample_initial_channel=512,
                resblock_dilation_sizes=[[1,3,5], [1,3,5], [1,3,5]],
                upsample_kernel_sizes=[12, 10],
                gen_istft_n_fft=40, gen_istft_hop_size=10):
        super().__init__()

        self.decode = nn.ModuleList()

        self.encode = AdainResBlk1d(dim_in + 2, 1024, style_dim)

        self.decode.append(AdainResBlk1d(1024 + 2 + 64, 1024, style_dim))
        self.decode.append(AdainResBlk1d(1024 + 2 + 64, 1024, style_dim))
        self.decode.append(AdainResBlk1d(1024 + 2 + 64, 1024, style_dim))
        self.decode.append(AdainResBlk1d(1024 + 2 + 64, 512, style_dim, upsample=True))

        self.F0_conv = weight_norm(nn.Conv1d(1, 1, kernel_size=3, stride=2, groups=1, padding=1))

        self.N_conv = weight_norm(nn.Conv1d(1, 1, kernel_size=3, stride=2, groups=1, padding=1))

        self.asr_res = nn.Sequential(
            weight_norm(nn.Conv1d(512, 64, kernel_size=1)),
        )

        self.generator = Generator(style_dim, resblock_kernel_sizes, upsample_rates,
                                   upsample_initial_channel, resblock_dilation_sizes,
                                   upsample_kernel_sizes, gen_istft_n_fft, gen_istft_hop_size)

    def forward(self, asr, F0_curve, N, s, lengths=None):
        if self.training:
            downlist = [0, 3, 7]
            F0_down = downlist[random.randint(0, 2)]
            downlist = [0, 3, 7, 15]
            N_down = downlist[random.randint(0, 3)]
            if F0_down:
                F0_curve = F.avg_pool1d(F0_curve.unsqueeze(1), kernel_size=F0_down, stride=1, padding=F0_down//2)[:, 0, :]
            if N_down:
                N = F.avg_pool1d(N.unsqueeze(1), kernel_size=N_down, stride=1, padding=N_down//2)[:, 0, :]

        if lengths is not None:
            full_mask = torch.arange(F0_curve.size(-1), device=F0_curve.device)[None, None, :] < lengths[:, None, None]
            full_mask = full_mask.float()

            asr_lengths = lengths // 2
            asr_mask = torch.arange(asr.size(-1), device=asr.device)[None, None, :] < asr_lengths[:, None, None]
            asr_mask = asr_mask.float()
        else:
            full_mask = torch.ones_like(F0_curve[:, None, :])
            asr_mask = torch.ones_like(asr[:, :1, :])

        F0 = self.F0_conv(F0_curve.unsqueeze(1)) * asr_mask
        N_feat = self.N_conv(N.unsqueeze(1)) * asr_mask

        x = torch.cat([asr, F0, N_feat], axis=1)
        x = self.encode(x, s, mask_in=asr_mask, mask_out=asr_mask) * asr_mask

        asr_res = self.asr_res(asr) * asr_mask

        res = True
        for block in self.decode:
            if res:
                x = torch.cat([x, asr_res, F0, N_feat], axis=1)

            if block.upsample_type == "none":
                x = block(x, s, mask_in=asr_mask, mask_out=asr_mask)
                x = x * asr_mask
            else:
                x = block(x, s, mask_in=asr_mask, mask_out=full_mask)
                x = x * full_mask
                res = False
        if lengths is not None:
            x = self.generator(x, s, F0_curve, full_mask, lengths)
        else:
            x = self.generator.inference(x, s, F0_curve)
        return x

    def features(self, asr, F0_curve, N, s):
        """Shared unpadded feature path for inference and streaming windows."""
        F0 = self.F0_conv(F0_curve.unsqueeze(1))
        N_feat = self.N_conv(N.unsqueeze(1))
        x = self.encode(torch.cat([asr, F0, N_feat], dim=1), s)
        asr_res = self.asr_res(asr)
        res = True
        for block in self.decode:
            if res:
                x = torch.cat([x, asr_res, F0, N_feat], dim=1)
            x = block(x, s)
            if block.upsample_type != "none":
                res = False
        return x

    def inference(self, asr, F0_curve, N, s, *, source=None):
        """Decode an exact-length window; source may supply cached excitation."""
        x = self.features(asr, F0_curve, N, s)
        return self.generator.inference(x, s, F0_curve, source=source)

    def streaming_context_frames(self):
        """Conservative left/right receptive-field bound in ASR frames.

        AdaLN and FiLM act within each frame. Only convolutions, resampling and STFT
        expand temporal support. Include rounding slack for the sample grids.
        """
        def radius(conv):
            return (conv.kernel_size[0] - 1) * conv.dilation[0] / 2

        def res_radius(block):
            return sum(radius(c) for c in list(block.convs1) + list(block.convs2))

        jump = 1.0
        reach = max(radius(self.F0_conv), radius(self.N_conv)) / 2
        for block in [self.encode, *self.decode]:
            if block.upsample_type != "none":
                reach += radius(block.pool[0]) * jump
                jump /= 2
                reach += jump  # PixelShuffle / nearest-neighbor grid offset.
            reach += (radius(block.conv1) + radius(block.conv2)) * jump
        gen = self.generator
        samples_per_asr = 2 * int(gen.total_upsample)
        spectral_jump = gen.stft.hop_length / samples_per_asr
        for i, up in enumerate(gen.ups):
            # Full transpose-kernel span also bounds its grid offset.
            reach += 2 * radius(up) * jump / up.stride[0]
            jump /= up.stride[0]
            if i == gen.num_upsamples - 1:
                reach += jump  # ReflectionPad1d((1, 0)).
            source_reach = gen.stft.n_fft / samples_per_asr
            source_reach += radius(gen.noise_convs[i]) * spectral_jump
            source_reach += res_radius(gen.noise_res[i]) * jump
            reach = max(reach, source_reach)
            reach += max(res_radius(gen.resblocks[i * gen.num_kernels + j])
                         for j in range(gen.num_kernels)) * jump
        reach += radius(gen.conv_post) * jump + gen.stft.n_fft / samples_per_asr
        return math.ceil(reach) + 1

    @torch.inference_mode()
    def forward_chunk(self, asr, F0_curve, N, s, state=None, *, final=False):
        """Return (stable_audio, state), with audio shaped [batch, 1, samples].

        Pass the returned state on every call. Symmetric convolutions require
        lookahead, so early calls can return empty audio. Use final=True on the
        last input, or finish through StreamingDecoder.flush(). Phase alone is
        insufficient: state also retains bounded input and excitation context.
        """
        if self.training:
            raise ValueError("Streaming requires decoder.eval()")
        state = DecoderStreamState() if state is None else state
        if state.finished:
            raise ValueError("The decoder stream is finished; start a new state")
        if (asr.ndim != 3 or F0_curve.ndim != 2 or N.ndim != 2 or s.ndim != 2
                or F0_curve.shape != N.shape
                or F0_curve.shape != (asr.shape[0], asr.shape[-1] * 2)
                or s.shape[0] != asr.shape[0] or asr.shape[-1] == 0):
            raise ValueError("Expected nonempty ASR [B,C,T], F0/energy [B,2*T], style [B,S]")
        if any(t.device != asr.device or t.dtype != asr.dtype for t in (F0_curve, N, s)):
            raise ValueError("All streaming inputs must share device and dtype")
        if state.asr is not None:
            if (asr.shape[:2] != state.asr.shape[:2] or asr.device != state.asr.device
                    or asr.dtype != state.asr.dtype or not torch.equal(s, state.style)):
                raise ValueError("Batch, device, dtype and style must stay fixed; reset the stream")
        source, state.phase = self.generator.source(F0_curve, state.phase)
        for name, value in (("asr", asr), ("f0", F0_curve), ("energy", N), ("source", source)):
            previous = getattr(state, name)
            setattr(state, name, value.clone() if previous is None else torch.cat([previous, value], dim=-1))
        if state.style is None:
            state.style = s.clone()
        state.total_frames += asr.shape[-1]
        return self._drain_stream(state, final=final), state

    def _drain_stream(self, state, *, final):
        samples_per_asr = 2 * int(self.generator.total_upsample)
        context = self.streaming_context_frames()
        end_frame = state.total_frames if final else max(0, state.total_frames - context)
        end_sample = end_frame * samples_per_asr
        if end_sample > state.emitted_samples:
            audio = self.inference(state.asr, state.f0, state.energy, state.style, source=state.source)
            start = state.emitted_samples - state.start_frame * samples_per_asr
            end = end_sample - state.start_frame * samples_per_asr
            if end > audio.shape[-1]:
                raise RuntimeError("Decoder output length does not match its upsample factor")
            result = audio[..., start:end].clone()
            state.emitted_samples = end_sample
        else:
            result = state.asr.new_empty(state.asr.shape[0], 1, 0)
        if final:
            state.finished = True
            state.asr = state.f0 = state.energy = state.source = None
        else:
            # Clone slices so a small retained window cannot hold an entire
            # utterance's backing storage alive.
            drop = max(0, end_frame - context - state.start_frame)
            if drop:
                state.asr = state.asr[..., drop:].clone()
                state.f0 = state.f0[..., 2 * drop:].clone()
                state.energy = state.energy[..., 2 * drop:].clone()
                state.source = state.source[..., drop * samples_per_asr:].clone()
                state.start_frame += drop
        return result


class StreamingDecoder(nn.Module):
    """Bounded-context streaming for an eval-mode Decoder.

    step() accepts up to chunk_asr_frames ASR frames (including a short final
    chunk) and may return empty audio during lookahead. Concatenate all step()
    outputs and flush() exactly once to recover the complete aligned utterance.
    Lookahead is exposed as lookahead_samples; reset() starts another utterance.

    reset(skip_samples=...) drops leading output, typically the BOS silence:
    pred_dur[0] * samples_per_asr_frame minus a short margin. Those frames are
    still decoded as left context, so the speech onset is unchanged.
    """
    def __init__(self, decoder: Decoder, chunk_asr_frames: int = 40):
        super().__init__()
        if not isinstance(chunk_asr_frames, int) or chunk_asr_frames < 1:
            raise ValueError("chunk_asr_frames must be a positive integer")
        self.decoder = decoder
        self.chunk_asr_frames = chunk_asr_frames
        self.chunk_f0_frames = chunk_asr_frames * 2
        self.samples_per_asr_frame = 2 * int(decoder.generator.total_upsample)
        self.chunk_audio_samples = chunk_asr_frames * self.samples_per_asr_frame
        self.lookahead_samples = decoder.streaming_context_frames() * self.samples_per_asr_frame
        self.reset()

    def reset(self, skip_samples: int = 0, fade_samples: int = 120):
        """Start an utterance; skip_samples > 0 trims leading output and ramps in
        the next fade_samples (120 = 5 ms at 24 kHz) to avoid a click."""
        if skip_samples < 0 or fade_samples < 0:
            raise ValueError("skip_samples and fade_samples must be nonnegative")
        self.state = None
        self.skip_left = int(skip_samples)
        self.fade_samples = int(fade_samples) if skip_samples else 0
        self.fade_done = 0

    def _trim(self, audio):
        if self.skip_left:
            skip = min(self.skip_left, audio.shape[-1])
            audio = audio[..., skip:]
            self.skip_left -= skip
        if self.fade_done < self.fade_samples and audio.shape[-1]:
            count = min(self.fade_samples - self.fade_done, audio.shape[-1])
            ramp = torch.arange(self.fade_done + 1, self.fade_done + count + 1,
                                device=audio.device, dtype=audio.dtype) / self.fade_samples
            audio = torch.cat([audio[..., :count] * ramp, audio[..., count:]], dim=-1)
            self.fade_done += count
        return audio

    @torch.inference_mode()
    def step(self, asr_chunk, f0_chunk, energy_chunk, s):
        if asr_chunk.shape[-1] > self.chunk_asr_frames:
            raise ValueError("Input exceeds chunk_asr_frames")
        audio, self.state = self.decoder.forward_chunk(
            asr_chunk, f0_chunk, energy_chunk, s, self.state,
        )
        return self._trim(audio.squeeze(1))

    @torch.inference_mode()
    def flush(self):
        """Emit the remaining real samples, including the utterance's right edge."""
        if self.state is None:
            raise ValueError("Call step() before flush()")
        if self.state.finished:
            return self.state.style.new_empty(self.state.style.shape[0], 0)
        if self.decoder.training:
            raise ValueError("Streaming requires decoder.eval()")
        return self._trim(self.decoder._drain_stream(self.state, final=True).squeeze(1))
