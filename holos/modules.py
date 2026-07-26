import math
import torch
import torch.nn as nn
from torch.nn import Conv1d
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm
from torchaudio.models import Conformer
from transformers import AlbertConfig, AlbertModel



def length_to_mask(lengths, max_len=None):
    max_len = max_len if max_len is not None else lengths.max()
    mask = torch.arange(max_len, device=lengths.device).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
    mask = torch.gt(mask+1, lengths.unsqueeze(1))
    return mask


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)


class Tokenizer:
    def __init__(self, vocab):
        self.dict = {}
        self.symbols_count = len(vocab)
        for i in range(self.symbols_count):
            self.dict[vocab[i]] = i

    def __call__(self, text):
        indexes = []
        for char in text:
            try:
                indexes.append(self.dict[char])
            except KeyError:
                pass
        return indexes


class MaskedInstanceNorm1d(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x, mask=None):
        if mask is None:
            if not torch.onnx.is_in_onnx_export():
                return F.instance_norm(x, eps=self.eps)
            mask = torch.ones(x.shape[0], 1, x.shape[2], device=x.device, dtype=x.dtype)
    
        if not torch.onnx.is_in_onnx_export() and mask.min() == 1.0:
            return F.instance_norm(x, eps=self.eps)

        valid_lengths = mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
        mean = (x * mask).sum(dim=-1, keepdim=True) / valid_lengths
        var = (((x - mean) * mask) ** 2).sum(dim=-1, keepdim=True) / valid_lengths
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return x_norm * mask


class AdaIN1d(nn.Module):
    def __init__(self, style_dim, num_features):
        super().__init__()
        self.norm = MaskedInstanceNorm1d(num_features)
        self.fc = nn.Linear(style_dim, num_features*2)

    def forward(self, x, s, mask=None):
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        return (1 + gamma) * self.norm(x, mask) + beta


class UpSample1d(nn.Module):
    def __init__(self, layer_type):
        super().__init__()
        self.layer_type = layer_type

    def forward(self, x):
        if self.layer_type == 'none':
            return x
        else:
            return F.interpolate(x, scale_factor=2, mode='nearest')


class Snake1d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x, eps=1e-9):
        return x + (1.0 / (self.alpha + eps)) * (torch.sin(self.alpha * x) ** 2)


class PixelShuffle1D(nn.Module):
    def __init__(self, upscale_factor):
        super().__init__()
        self.upscale_factor = upscale_factor

    def forward(self, x):
        b, c, t = x.size()
        r = self.upscale_factor
        if c % r != 0:
            raise ValueError(f"Channels ({c}) not divisible by upscale factor ({r})")
        x = x.view(b, c // r, r, t)
        x = x.permute(0, 1, 3, 2)
        x = x.contiguous().view(b, c // r, t * r)
        return x


class AdainResBlk1d(nn.Module):
    def __init__(self, dim_in, dim_out, style_dim=64, actv=nn.LeakyReLU(0.2),
                 upsample='none', dropout_p=0.0):
        super().__init__()
        self.actv = actv
        self.upsample_type = upsample
        self.upsample = UpSample1d(upsample)
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out, style_dim)
        self.dropout = nn.Dropout(dropout_p)
        
        if upsample == 'none':
            self.pool = nn.Identity()
        else:
            self.pool = nn.Sequential(
                weight_norm(nn.Conv1d(
                    in_channels=dim_in,
                    out_channels=dim_in * 2,
                    kernel_size=3,
                    padding=1,
                    groups=dim_in
                )),
                PixelShuffle1D(2)
            )
        
    def _build_weights(self, dim_in, dim_out, style_dim):
        self.conv1 = weight_norm(nn.Conv1d(dim_in, dim_out, 3, 1, 1))
        self.conv2 = weight_norm(nn.Conv1d(dim_out, dim_out, 3, 1, 1))
        if style_dim is not None:
            self.norm1 = AdaIN1d(style_dim, dim_in)
            self.norm2 = AdaIN1d(style_dim, dim_out)
        else:
            self.norm1 = MaskedInstanceNorm1d(dim_in)
            self.norm2 = MaskedInstanceNorm1d(dim_out)
        self.style_dim = style_dim
        if self.learned_sc:
            self.conv1x1 = weight_norm(nn.Conv1d(dim_in, dim_out, 1, 1, 0, bias=False))

    def _shortcut(self, x):
        x = self.upsample(x)
        if self.learned_sc:
            x = self.conv1x1(x)
        return x

    def _residual(self, x, s=None, mask_in=None, mask_out=None):
        if self.style_dim is not None:
            x = self.norm1(x, s, mask_in)
        else:
            x = self.norm1(x, mask_in)
        if mask_in is not None: x = x * mask_in
        x = self.actv(x)
        x = self.pool(x)
        x = self.conv1(self.dropout(x))
        if self.style_dim is not None:
            x = self.norm2(x, s, mask_out)
        else:
            x = self.norm2(x, mask_out)
        if mask_out is not None: x = x * mask_out
        x = self.actv(x)
        x = self.conv2(self.dropout(x))
        return x

    def forward(self, x, s=None, mask_in=None, mask_out=None):
        if mask_out is None:
            mask_out = mask_in
        out = self._residual(x, s, mask_in, mask_out)
        out = (out + self._shortcut(x)) / math.sqrt(2)
        if mask_out is not None:
            out = out * mask_out
        return out



class AdaINResBlock1(torch.nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5), style_dim=64):
        super(AdaINResBlock1, self).__init__()
        self.convs1 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                               padding=get_padding(kernel_size, dilation[1]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[2],
                               padding=get_padding(kernel_size, dilation[2])))
        ])
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1)))
        ])
        self.convs2.apply(init_weights)
        
        self.adain1 = nn.ModuleList([
            AdaIN1d(style_dim, channels),
            AdaIN1d(style_dim, channels),
            AdaIN1d(style_dim, channels),
        ])
        
        self.adain2 = nn.ModuleList([
            AdaIN1d(style_dim, channels),
            AdaIN1d(style_dim, channels),
            AdaIN1d(style_dim, channels),
        ])
        
        self.alpha1 = nn.ParameterList([nn.Parameter(torch.ones(1, channels, 1)) for i in range(len(self.convs1))])
        self.alpha2 = nn.ParameterList([nn.Parameter(torch.ones(1, channels, 1)) for i in range(len(self.convs2))])


    def forward(self, x, s, mask=None):
        for c1, c2, n1, n2, a1, a2 in zip(self.convs1, self.convs2, self.adain1, self.adain2, self.alpha1, self.alpha2):
            xt = n1(x, s, mask)
            xt = xt + (1 / a1) * (torch.sin(a1 * xt) ** 2)  # Snake1D
            if mask is not None: xt = xt * mask
            xt = c1(xt)
            xt = n2(xt, s, mask)
            xt = xt + (1 / a2) * (torch.sin(a2 * xt) ** 2)  # Snake1D
            if mask is not None: xt = xt * mask
            xt = c2(xt)
            x = xt + x
            if mask is not None: x = x * mask
        return x


class LayerNormCausal(nn.Module):
    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps

        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        x= x.transpose(1, 2)
        x = F.layer_norm(x, (self.channels,), self.gamma, self.beta, self.eps)
        return x.transpose(1, 2)




class RotaryPositionalEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE). Applies rotation to Q and K
    so attention scores capture relative distance between positions."""
    def __init__(self, dim, max_len=4096, partial_rope=False):
        super().__init__()
        self.partial_rope = partial_rope
        self.rope_dim = dim // 2 if partial_rope else dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.rope_dim, 2).float() / self.rope_dim))
        t = torch.arange(max_len).float()
        freqs = torch.outer(t, inv_freq)  # [max_len, rope_dim/2]
        self.register_buffer('cos_cached', freqs.cos().unsqueeze(0).unsqueeze(0))  # [1, 1, max_len, rope_dim/2]
        self.register_buffer('sin_cached', freqs.sin().unsqueeze(0).unsqueeze(0))  # [1, 1, max_len, rope_dim/2]

    def forward(self, q, k):
        # q, k: [B, nhead, T, head_dim]
        T = q.size(2)
        cos = self.cos_cached[:, :, :T, :]  # [1, 1, T, rope_dim/2]
        sin = self.sin_cached[:, :, :T, :]
        if self.partial_rope:
            q_rope, q_pass = q[..., :self.rope_dim], q[..., self.rope_dim:]
            k_rope, k_pass = k[..., :self.rope_dim], k[..., self.rope_dim:]
            q = torch.cat([self._apply_rope(q_rope, cos, sin), q_pass], dim=-1)
            k = torch.cat([self._apply_rope(k_rope, cos, sin), k_pass], dim=-1)
        else:
            q = self._apply_rope(q, cos, sin)
            k = self._apply_rope(k, cos, sin)
        return q, k
    
    @staticmethod
    def _apply_rope(x, cos, sin):
        # x: [B, nhead, T, head_dim] — split into even/odd pairs
        x1 = x[..., 0::2]  # [B, nhead, T, head_dim/2]
        x2 = x[..., 1::2]
        # Rotate: (x1*cos - x2*sin, x1*sin + x2*cos)
        out1 = x1 * cos - x2 * sin
        out2 = x1 * sin + x2 * cos
        return torch.stack([out1, out2], dim=-1).flatten(-2)  # [B, nhead, T, head_dim]


class EncoderLayerONNX(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1, activation="gelu"):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        assert self.head_dim * nhead == d_model, "d_model must be divisible by nhead"

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.activation_type = activation
        
        # FFN
        if activation == "swiglu":
            self.gate_proj = nn.Linear(d_model, dim_feedforward)
            self.up_proj = nn.Linear(d_model, dim_feedforward)
            self.down_proj = nn.Linear(dim_feedforward, d_model)
        else:
            self.fc1 = nn.Linear(d_model, dim_feedforward)
            self.fc2 = nn.Linear(dim_feedforward, d_model)
            if activation == "gelu":
                self.activation = F.gelu
            else:
                self.activation = F.relu

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x):
        # x: [B, T, D] -> [B, nhead, T, head_dim]
        B, T, D = x.size()
        x = x.view(B, T, self.nhead, self.head_dim)
        x = x.permute(0, 2, 1, 3)
        return x

    def _combine_heads(self, x):
        # x: [B, nhead, T, head_dim] -> [B, T, D]
        x = x.permute(0, 2, 1, 3).contiguous()
        B, T, nhead, head_dim = x.size()
        return x.view(B, T, nhead * head_dim)

    def forward(self, src, src_key_padding_mask=None, rope=None):
        # src: [B, T, D]
        B, T, D = src.size()

        # --- Self-attention (scaled dot-product) ---
        q = self.q_proj(src)  # [B, T, D]
        k = self.k_proj(src)
        v = self.v_proj(src)

        q = self._split_heads(q)  # [B, nhead, T, head_dim]
        k = self._split_heads(k)
        v = self._split_heads(v)

        if rope is not None:
            q, k = rope(q, k)

        if not torch.onnx.is_in_onnx_export():
            # Use SDPA for FlashAttention / memory-efficient attention
            attn_mask = None
            if src_key_padding_mask is not None:
                # src_key_padding_mask: [B, T] with True where padding
                # SDPA expects additive mask: 0 for valid, -inf for masked
                attn_mask = torch.zeros(B, 1, 1, T, device=q.device, dtype=q.dtype)
                attn_mask.masked_fill_(src_key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
            context = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=False,
            )
        else:
            # Manual attention for ONNX export compatibility
            k_t = k.permute(0, 1, 3, 2)
            scores = torch.matmul(q, k_t) / (self.head_dim ** 0.5)
            if src_key_padding_mask is not None:
                mask = src_key_padding_mask.unsqueeze(1).unsqueeze(2)
                scores = scores + (mask.to(dtype=scores.dtype) * -1e9)
            attn = torch.softmax(scores, dim=-1)
            attn = self.dropout(attn)
            context = torch.matmul(attn, v)

        context = self._combine_heads(context)  # [B,T,D]
        attn_out = self.out_proj(context)
        attn_out = self.dropout(attn_out)

        # Residual + Norm
        src2 = self.norm1(src + attn_out)

        # --- Feed-forward ---
        if self.activation_type == "swiglu":
            ff = self.down_proj(F.silu(self.gate_proj(src2)) * self.up_proj(src2))
        else:
            ff = self.fc2(self.activation(self.fc1(src2)))
            
        ff = self.dropout(ff)
        out = self.norm2(src2 + ff)

        return out



class TextEncoder(nn.Module):
    def __init__(self, channels=512, kernel_size=5, depth=3, n_symbols=181, n_heads=4, n_layers=4, actv=nn.LeakyReLU(0.2), use_pe=False, scale_embedding=False, partial_rope=False, use_conv_pe=False):
        super().__init__()
        self.channels = channels
        self.scale_embedding = scale_embedding
        self.embedding = nn.Embedding(n_symbols, channels)

        padding = (kernel_size - 1) // 2
        self.cnn = nn.ModuleList()
        for _ in range(depth):
            self.cnn.append(nn.Sequential(
                weight_norm(nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding)),
                LayerNormCausal(channels),
                actv,
                nn.Dropout(0.2),
            ))


        ffn_dim = int(channels * 8 / 3)
        self.layers = nn.ModuleList([
            EncoderLayerONNX(d_model=channels, nhead=n_heads, dim_feedforward=ffn_dim, dropout=0.1, activation="swiglu")
            for _ in range(n_layers)
        ])
        
        if use_pe:
            self.rope = RotaryPositionalEmbedding(channels // n_heads, partial_rope=partial_rope)
        else:
            self.rope = None
        
        if use_conv_pe:
            self.conv_pe = MultiScaleConvPE(channels)
        else:
            self.conv_pe = None

    def forward(self, x, input_lengths):
        x = self.embedding(x)
        if self.scale_embedding:
            x = x * math.sqrt(self.channels)
        m = length_to_mask(input_lengths, max_len=x.size(1))
        mask_for_cnn = m.unsqueeze(1).to(x.dtype)
        src_key_padding_mask = m.bool()
   
        x_conv = x.transpose(1, 2)
        x_residual = x_conv * (1 - mask_for_cnn)
        x_conv = x_residual
        for c in self.cnn:
            x_conv = c(x_conv)
            x_conv = x_conv * (1-mask_for_cnn)
            
        x = x_conv.transpose(1, 2)  # [B, T, C]

        if self.conv_pe is not None:
            x = x + self.conv_pe(x)

        for layer in self.layers:
            x = layer(x, src_key_padding_mask=src_key_padding_mask, rope=self.rope)

        x = x * (~m).unsqueeze(-1).to(x.dtype)
        x = x.transpose(2, 1)
        return x


class AttentivePooling(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.Tanh(),
            nn.Linear(dim // 4, 1),
        )

    def forward(self, x, mask):
        w = self.attn(x)
        w = w.masked_fill(~mask.bool(), float('-inf'))
        w = torch.softmax(w, dim=1)
        return (x * w).sum(dim=1)


class StyleEncoder(nn.Module):
    def __init__(self, mel_dim=80, proj_dim=512, style_dim=128, num_heads=8, num_layers=6):
        super().__init__()        
        self.mel_proj = nn.Conv1d(mel_dim, proj_dim, kernel_size=3, padding=1)
        self.proj_norm = nn.LayerNorm(proj_dim)
        self.conformer_pre = Conformer(
             input_dim=proj_dim,
             num_heads=num_heads,
             ffn_dim=proj_dim * 2,
             num_layers=1,
             depthwise_conv_kernel_size=31,
             use_group_norm=True,
        )
        self.conformer_body = Conformer(
             input_dim=proj_dim,
             num_heads=num_heads,
             ffn_dim=proj_dim * 2,
             num_layers=num_layers - 1,
             depthwise_conv_kernel_size=15,
             use_group_norm=True,
        )
        self.out = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(proj_dim, style_dim),
        )
        self.pooling = AttentivePooling(proj_dim)

        
    def forward(self, mel, mel_lengths=None):
        assert mel.dim() == 3, f"Expected 3D mel [B, n_mels, T], got {mel.dim()}D"
        
        x = self.mel_proj(mel)
        x = x.transpose(-1, -2)
        x = self.proj_norm(x)
        
        if mel_lengths is None:
            mel_lengths = torch.full(
                (mel.size(0),), mel.size(-1),
                dtype=torch.long, device=mel.device
            )

        T_max = x.size(1)
        mask = torch.arange(T_max, device=x.device)[None, :] < mel_lengths[:, None]
        mask = mask.unsqueeze(-1).float()
        
        x = x * mask

        x, _ = self.conformer_pre(x, mel_lengths)
        x, _ = self.conformer_body(x, mel_lengths)
        
        x = x * mask
        x_avg = self.pooling(x, mask)
        
        s = self.out(x_avg)        
        return s


class FiLM(nn.Module):
    def __init__(self, d_hid, style_dim, style_weight=1.0, learnable_weight=False):
        """
        Feature-wise Linear Modulation with controllable style influence.
        
        Args:
            d_hid: hidden dimension
            style_dim: style embedding dimension
            style_weight: blending factor between identity and FiLM output.
                          0.0 = no style influence (identity), 1.0 = full FiLM modulation.
            learnable_weight: if True, style_weight becomes a trainable nn.Parameter.
        """
        super().__init__()
        self.gamma = nn.Linear(style_dim, d_hid)
        self.beta = nn.Linear(style_dim, d_hid)

        nn.init.zeros_(self.gamma.weight)
        nn.init.zeros_(self.beta.weight)
        nn.init.zeros_(self.gamma.bias)
        nn.init.zeros_(self.beta.bias)
        with torch.no_grad():
            self.gamma.bias.data.fill_(1.0)
            self.beta.bias.data.fill_(0.0)

        if learnable_weight:
            self.style_weight = nn.Parameter(torch.tensor(float(style_weight)))
        else:
            self.register_buffer('style_weight', torch.tensor(float(style_weight)))

    def forward(self, h, style):
        """
        h: [B, T, D]
        style: [B, style_dim]
        
        Returns: (1 - w) * h + w * (gamma * h + beta)
            where w = sigmoid(style_weight) if learnable, else clamped style_weight
        """
        gamma = self.gamma(style).unsqueeze(1)  # [B, 1, D]
        beta = self.beta(style).unsqueeze(1)    # [B, 1, D]
        modulated = gamma * h + beta
        
        w = torch.sigmoid(self.style_weight) if self.style_weight.requires_grad else self.style_weight.clamp(0.0, 1.0)
        return (1.0 - w) * h + w * modulated


class MultiScaleConvPE(nn.Module):
    """Multi-Scale Convolutional Positional Encoding.
    
    Uses multiple depthwise convolutions with different kernel sizes to capture
    positional context at multiple scales. Outputs are combined via a learnable
    gated sum per channel.
    
    Operates on [B, D, T] (channels-first) internally.
    Input/output: [B, T, D] (time-first) for consistency with transformer convention.
    """
    def __init__(self, d_model, kernel_sizes=(3, 7, 15, 31)):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(d_model, d_model, kernel_size=k, padding=k // 2, groups=d_model)
            for k in kernel_sizes
        ])
        # Learnable gate per scale per channel: [num_scales, 1, D, 1]
        self.gate = nn.Parameter(torch.ones(len(kernel_sizes), 1, d_model, 1) / len(kernel_sizes))

    def forward(self, x):
        """x: [B, T, D] → returns positional encoding [B, T, D] to add to x."""
        xt = x.transpose(1, 2)  # [B, D, T]
        # Stack conv outputs: [num_scales, B, D, T]
        multi = torch.stack([conv(xt) for conv in self.convs], dim=0)
        # Gated sum: softmax over scales, then weighted sum
        weights = torch.softmax(self.gate, dim=0)  # [num_scales, 1, D, 1]
        pe = (multi * weights).sum(dim=0)  # [B, D, T]
        return pe.transpose(1, 2)  # [B, T, D]


class DurationPredictor(nn.Module):

    def __init__(self, d_hid=640, style_dim=128, nlayers=6, max_dur=20, dropout=0.1, n_heads=8, mask_prob=0.1, use_rope=False):
        super().__init__() 
        self.mask_prob = mask_prob
        if mask_prob > 0:
            self.mask_token = nn.Parameter(torch.zeros(1, 1, d_hid)) 
        self.dropout_dur = dropout
        self.film = FiLM(d_hid, style_dim, style_weight=0.35, learnable_weight=True)
        
        self.conv_pe = MultiScaleConvPE(d_hid)
        
        self.use_rope = use_rope
        if use_rope:
            self.rope = RotaryPositionalEmbedding(d_hid // n_heads)
        else:
            self.rope = None
        
        # Transformer backbone
        ffn_dim = d_hid * 4
        self.layers = nn.ModuleList([
            EncoderLayerONNX(
                d_model=d_hid, nhead=n_heads,
                dim_feedforward=ffn_dim, dropout=dropout,
                activation="swiglu"
            )
            for _ in range(nlayers)
        ])
        
        self.films = nn.ModuleList([
            FiLM(d_hid, style_dim, style_weight=0.5, learnable_weight=True)
            for _ in range(nlayers)
        ])
        
        self.proj = nn.Linear(d_hid, max_dur)
        nn.init.trunc_normal_(self.proj.weight, std=0.02)
        nn.init.constant_(self.proj.bias, -2.0)
    
    def forward(self, x, style, text_lengths):
        x = self.film(x, style)
        
        if self.training and getattr(self, "mask_prob", 0) > 0:
            mask = torch.rand(x.size(0), x.size(1), device=x.device) < self.mask_prob
            padding_mask = torch.arange(x.size(1), device=x.device)[None, :] >= text_lengths[:, None]
            mask = mask & (~padding_mask)
            x = torch.where(mask.unsqueeze(-1), self.mask_token, x)
            
        x = x + self.conv_pe(x)
        
        T = x.size(1)
        padding_mask = torch.arange(T, device=x.device)[None, :] >= text_lengths[:, None]  # [B, T]
        
        for layer, film_layer in zip(self.layers, self.films):
            x = film_layer(x, style)
            x = layer(x, src_key_padding_mask=padding_mask, rope=getattr(self, "rope", None))

        text_mask = torch.arange(x.size(1), device=x.device)[None, :] < text_lengths[:, None]
        x = x.masked_fill(~text_mask.unsqueeze(-1), 0.0)
        
        x = F.dropout(x, p=self.dropout_dur, training=self.training)
        dur = self.proj(x)                # [B, T, max_dur]
        return dur


class ProsodyPredictor(nn.Module):

    def __init__(self, style_dim=128, d_hid=640, dropout=0.1,
                 n_heads=4, n_transformer_layers=3, mask_prob=0):
        super().__init__() 
        
        self.mask_prob = mask_prob
        if mask_prob > 0:
            self.mask_token = nn.Parameter(torch.zeros(1, 1, d_hid))
        
        input_dim = d_hid
        self.input_proj = nn.Linear(input_dim, d_hid)
        
        self.film = FiLM(d_hid, style_dim, learnable_weight=True)
        
        self.conv_pe = MultiScaleConvPE(d_hid)
        
        self.rope = RotaryPositionalEmbedding(d_hid // n_heads)
        
        ffn_dim = int(d_hid * 8 / 3)
        self.layers = nn.ModuleList([
            EncoderLayerONNX(
                d_model=d_hid, nhead=n_heads,
                dim_feedforward=ffn_dim, dropout=dropout,
                activation="swiglu"
            )
            for _ in range(n_transformer_layers)
        ])
        
        self.F0 = nn.ModuleList()
        self.F0.append(AdainResBlk1d(d_hid, d_hid, style_dim=style_dim, dropout_p=dropout))
        self.F0.append(AdainResBlk1d(d_hid, d_hid // 2, style_dim=style_dim, upsample=True, dropout_p=dropout))
        self.F0.append(AdainResBlk1d(d_hid // 2, d_hid // 2, style_dim=style_dim, dropout_p=dropout))

        self.N = nn.ModuleList()
        self.N.append(AdainResBlk1d(d_hid, d_hid, style_dim=style_dim, dropout_p=dropout))
        self.N.append(AdainResBlk1d(d_hid, d_hid // 2, style_dim=style_dim, upsample=True, dropout_p=dropout))
        self.N.append(AdainResBlk1d(d_hid // 2, d_hid // 2, style_dim=style_dim, dropout_p=dropout))
        
        self.F0_proj = nn.Conv1d(d_hid // 2, 1, 1, 1, 0)
        self.N_proj = nn.Conv1d(d_hid // 2, 1, 1, 1, 0)



    def forward(self, en, style, text_lengths):
        en_t = en.transpose(-1, -2)
        
        # text_lengths is full mel resolution. The backbone operates at half resolution.
        half_lengths = text_lengths // 2
        
        # Project to transformer dim
        x = self.input_proj(en_t)
        
        x = self.film(x, style)
        
        if self.training and getattr(self, "mask_prob", 0) > 0:
            mask = torch.rand(x.size(0), x.size(1), device=x.device) < self.mask_prob
            padding_mask = torch.arange(x.size(1), device=x.device)[None, :] >= half_lengths[:, None]
            mask = mask & (~padding_mask)
            x = torch.where(mask.unsqueeze(-1), self.mask_token, x)
            
        x = x + self.conv_pe(x)

        
        T = x.size(1)
        padding_mask = torch.arange(T, device=x.device)[None, :] >= half_lengths[:, None]  # [B, T]
        
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=padding_mask, rope=self.rope)
        
        len1 = en.size(2)
        len2 = len1 * 2
        mask1x = (torch.arange(len1, device=x.device)[None, :] < half_lengths[:, None]).unsqueeze(1).float()
        mask2x = (torch.arange(len2, device=x.device)[None, :] < text_lengths[:, None]).unsqueeze(1).float()

        x = x.transpose(-1, -2)

        F0 = x
        F0 = self.F0[0](F0, style, mask_in=mask1x, mask_out=mask1x)
        F0 = self.F0[1](F0, style, mask_in=mask1x, mask_out=mask2x)
        F0 = self.F0[2](F0, style, mask_in=mask2x, mask_out=mask2x)
        F0 = self.F0_proj(F0) * mask2x

        N = x
        N = self.N[0](N, style, mask_in=mask1x, mask_out=mask1x)
        N = self.N[1](N, style, mask_in=mask1x, mask_out=mask2x)
        N = self.N[2](N, style, mask_in=mask2x, mask_out=mask2x)
        N = self.N_proj(N) * mask2x
        
        return F0[:, 0], N[:, 0]
