import random
import torch
import numpy as np
import torch.nn.functional as F

from torch import nn
from typing import Optional
from torch.nn.utils.parametrizations import weight_norm
from torch.nn.utils import remove_weight_norm
from scipy.signal import get_window
from holos.modules import AdainResBlk1d, AdaIN1d, AdaINResBlock1, get_padding
from torch.nn import Conv1d, ConvTranspose1d


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


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
        
        # Forward DFT weights (kept as non-persistent buffer, not used in transform but preserved for reference)
        
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

        # Pre-compute frame offsets and DFT basis for fast, accurate GPU transform
        self.register_buffer("frame_offsets", torch.arange(n_fft), persistent=False)
        dft_mat = torch.cat([cos_basis, -sin_basis], dim=0) # [2F, n_fft]
        self.register_buffer("dft_mat", dft_mat, persistent=False)

    
    
    def transform_components(self, input_data):
        """Return continuous real/imaginary STFT components.

        This representation contains the same information as magnitude/phase but
        has no discontinuity at -pi/pi, so it is the preferred harmonic-source
        representation for newly trained models.
        """
        x = input_data.unsqueeze(1)
        pad = self.n_fft // 2
        left = x[:, :, 1 : pad + 1].flip(dims=[-1])
        right = x.flip(dims=[-1])[:, :, 1 : pad + 1]
        x_padded = torch.cat([left, x, right], dim=-1) # [B, 1, T + 2*pad]

        weight = (self.dft_mat * self.window[None, :]).unsqueeze(1)
        out = F.conv1d(x_padded, weight, stride=self.hop_length)  # [B, 2F, frames]

        real = out[:, :self.n_fft // 2 + 1, :]
        imag = out[:, self.n_fft // 2 + 1:, :]
        return real, imag

    def transform(self, input_data):
        """Return magnitude/phase features compatible with existing checkpoints."""
        real, imag = self.transform_components(input_data)

        magnitude = torch.sqrt(real**2 + imag**2 + 1e-8)
        phase = torch.atan2(imag, real)

        # The ONNX Atan2 decomposition evaluates 0 / 0 as NaN, whereas
        # torch.atan2(0, 0) returns 0. Masked harmonic-source tails contain
        # exactly these zero complex bins, so canonicalize them explicitly.
        zero_complex = (real == 0) & (imag == 0)
        phase = torch.where(zero_complex, torch.zeros_like(phase), phase)

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

    def forward(self, input_data):
        self.magnitude, self.phase = self.transform(input_data)
        reconstruction = self.inverse(self.magnitude, self.phase)
        return reconstruction



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

    def _f02sine(self, f0_values):
        """ f0_values: (batchsize, length, dim)
            where dim indicates fundamental tone and overtones
        """
        # convert to F0 in rad. The interger part n can be ignored
        # because 2 * np.pi * n doesn't affect phase
        rad_values = (f0_values / self.sampling_rate) % 1

        # Initial phase offset for overtones only.
        # Avoid in-place indexing here to prevent ScatterND in ONNX graph.
        batch_size, seq_len, dim = rad_values.shape
        rand_ini = torch.rand(batch_size, dim, device=f0_values.device)
        harmonic_mask = (torch.arange(dim, device=f0_values.device) > 0).to(rand_ini.dtype)
        rand_ini = rand_ini * harmonic_mask[None, :]
        first_frame_mask = (torch.arange(seq_len, device=f0_values.device) == 0).to(rand_ini.dtype)
        rad_values = rad_values + first_frame_mask[None, :, None] * rand_ini[:, None, :]

        
        # Downsample frequency to reduce the number of cumsum steps
        # This prevents float32 precision loss (hoarseness) without needing float64
        rad_values = torch.nn.functional.interpolate(rad_values.transpose(1, 2), 
                                                        scale_factor=1/self.upsample_scale, 
                                                        mode="linear").transpose(1, 2)

        # Accumulate continuous phase on low-res frequency
        phase = torch.cumsum(rad_values, dim=1) * 2 * np.pi
        
        # Upsample the continuous phase
        phase = torch.nn.functional.interpolate(phase.transpose(1, 2) * self.upsample_scale, 
                                                scale_factor=self.upsample_scale, mode="linear").transpose(1, 2)
        
        sines = torch.sin(phase)
            
        return sines

    def forward(self, f0):
        """ sine_tensor, uv = forward(f0)
        """
        f0_buf = torch.zeros(f0.shape[0], f0.shape[1], self.dim,
                             device=f0.device)
        # fundamental component
        fn = torch.multiply(f0, torch.FloatTensor([[range(1, self.harmonic_num + 2)]]).to(f0.device))

        # generate sine waveforms
        sine_waves = self._f02sine(fn) * self.sine_amp

        # generate uv signal
        uv = self._f02uv(f0)

        noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
        noise = noise_amp * torch.randn_like(sine_waves)

        # then: additive noise
        sine_waves = sine_waves * uv + noise
        return sine_waves, uv, noise


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

    def forward(self, x):
        """
        Sine_source, noise_source = SourceModuleHnNSF(F0_sampled)
        F0_sampled (batchsize, length, 1)
        Sine_source (batchsize, length, 1)
        noise_source (batchsize, length 1)
        """
        # source for harmonic branch
        with torch.no_grad():
            sine_wavs, uv, _ = self.l_sin_gen(x)
        sine_merge = self.l_tanh(self.l_linear(sine_wavs))

        # source for noise branch, in the same shape as uv
        #noise = torch.randn_like(uv) * self.sine_amp / 3
        return sine_merge#, noise, uv


class Generator(torch.nn.Module):
    def __init__(self, style_dim, resblock_kernel_sizes, upsample_rates, upsample_initial_channel, resblock_dilation_sizes, upsample_kernel_sizes, gen_istft_n_fft, gen_istft_hop_size, harmonic_representation="polar"):
        super(Generator, self).__init__()

        if harmonic_representation not in {"polar", "cartesian"}:
            raise ValueError(
                "harmonic_representation must be 'polar'"
                "or 'cartesian'"
            )
        self.harmonic_representation = harmonic_representation

        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        resblock = AdaINResBlock1

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
        if self.harmonic_representation == "cartesian":
            real, imag = self.stft.transform_components(source)
            return torch.cat([real, imag], dim=1)

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


    def inference(self, x, s, f0):
        with torch.no_grad():
            f0_upsampled = self.f0_upsamp(f0[:, None])
 
            f0_upsampled = f0_upsampled.transpose(1, 2)
            har_source = self.m_source(f0_upsampled)
            har_source = har_source.transpose(1, 2)[:, 0, :]

            har = self.harmonic_features(har_source)
        
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, 0.1)
            x = self.ups[i](x)
            if i == self.num_upsamples - 1:
                x = self.reflection_pad(x)

            x_source = self.noise_convs[i](har)
            x_source = self.noise_res[i](x_source, s)

            x = (x + x_source)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i*self.num_kernels+j](x, s)
                else:
                    xs += self.resblocks[i*self.num_kernels+j](x, s)
            x = (xs / self.num_kernels)
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        spec = torch.exp(x[:,:self.post_n_fft // 2 + 1, :])
        phase = torch.sin(x[:, self.post_n_fft // 2 + 1:, :])

        return self.stft.inverse(spec, phase)




    def remove_weight_norm(self):
        print('Removing weight norm...')
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_post)



class Decoder(nn.Module):
    def __init__(self, dim_in=512, style_dim=128,
                resblock_kernel_sizes = [3,7,11],
                upsample_rates = [6, 5],
                upsample_initial_channel=512,
                resblock_dilation_sizes=[[1,3,5], [1,3,5], [1,3,5]],
                upsample_kernel_sizes=[12, 10], 
                gen_istft_n_fft=40, gen_istft_hop_size=10,
                harmonic_representation="polar"):
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
                                   upsample_kernel_sizes, gen_istft_n_fft, gen_istft_hop_size,
                                   harmonic_representation=harmonic_representation)
        
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
        N = self.N_conv(N.unsqueeze(1)) * asr_mask
        
        x = torch.cat([asr, F0, N], axis=1)
        x = self.encode(x, s, mask_in=asr_mask, mask_out=asr_mask) * asr_mask
        
        asr_res = self.asr_res(asr) * asr_mask
        
        res = True
        for block in self.decode:
            if res:
                x = torch.cat([x, asr_res, F0, N], axis=1)
            
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
    


class DecoderNoN(nn.Module):
    def __init__(self, dim_in=512, style_dim=128,
                resblock_kernel_sizes = [3,7,11],
                upsample_rates = [6, 5],
                upsample_initial_channel=512,
                resblock_dilation_sizes=[[1,3,5], [1,3,5], [1,3,5]],
                upsample_kernel_sizes=[12, 10], 
                gen_istft_n_fft=40, gen_istft_hop_size=10,
                harmonic_representation="polar"):
        super().__init__()
        
        self.decode = nn.ModuleList()
        
        self.encode = AdainResBlk1d(dim_in + 1, 1024, style_dim)
        
        self.decode.append(AdainResBlk1d(1024 + 1 + 64, 1024, style_dim))
        self.decode.append(AdainResBlk1d(1024 + 1 + 64, 1024, style_dim))
        self.decode.append(AdainResBlk1d(1024 + 1 + 64, 1024, style_dim))
        self.decode.append(AdainResBlk1d(1024 + 1 + 64, 512, style_dim, upsample=True))

        self.F0_conv = weight_norm(nn.Conv1d(1, 1, kernel_size=3, stride=2, groups=1, padding=1))
        
        #elf.N_conv = weight_norm(nn.Conv1d(1, 1, kernel_size=3, stride=2, groups=1, padding=1))
        
        self.asr_res = nn.Sequential(
            weight_norm(nn.Conv1d(512, 64, kernel_size=1)),
        )
        
        
        self.generator = Generator(style_dim, resblock_kernel_sizes, upsample_rates, 
                                   upsample_initial_channel, resblock_dilation_sizes, 
                                   upsample_kernel_sizes, gen_istft_n_fft, gen_istft_hop_size,
                                   harmonic_representation=harmonic_representation)
        
    def forward(self, asr, F0_curve, s, lengths=None):
        
        if self.training:
            downlist = [0, 3, 7]
            F0_down = downlist[random.randint(0, 2)]
            # downlist = [0, 3, 7, 15]
            # N_down = downlist[random.randint(0, 3)]
            if F0_down:
                F0_curve = F.avg_pool1d(F0_curve.unsqueeze(1), kernel_size=F0_down, stride=1, padding=F0_down//2)[:, 0, :]
            # if N_down:
            #     N = F.avg_pool1d(N.unsqueeze(1), kernel_size=N_down, stride=1, padding=N_down//2)[:, 0, :]

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
        #N = self.N_conv(N.unsqueeze(1)) * asr_mask
        
        x = torch.cat([asr, F0], axis=1)
        x = self.encode(x, s, mask_in=asr_mask, mask_out=asr_mask) * asr_mask
        
        asr_res = self.asr_res(asr) * asr_mask
        
        res = True
        for block in self.decode:
            if res:
                x = torch.cat([x, asr_res, F0], axis=1)
            
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
