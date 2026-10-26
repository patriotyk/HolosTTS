import os
import torch
import numpy as np
import torch.nn as nn
import soundfile as sf
import librosa
import torchaudio
from typing import Union
from dataclasses import dataclass
from huggingface_hub import hf_hub_download
from holos.modules import StyleEncoder, TextEncoder, ProsodyPredictor, DurationPredictor, Tokenizer
from holos.decoder import Decoder


class VoiceEncoder(nn.Module):
    def __init__(self, weights_path="patriotyk/HolosTTS", device='cuda'):
        super().__init__()
        
        self.add_module('style_encoder', StyleEncoder())
        self.add_module('prosodic_style_encoder', StyleEncoder())
        
        self.to_mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_fft=2048,
            win_length=1200,
            hop_length=300,
            n_mels=80
        )
        
        self.to(device)
        self.device = device
        self.eval()

        if weights_path is not None:
            resolved_weights_path = None
            if os.path.isfile(weights_path):
                resolved_weights_path = weights_path
            else:
                try:
                    resolved_weights_path = hf_hub_download(repo_id=weights_path, filename="voice_encoder.pt")
                except Exception:
                    resolved_weights_path = None

            if resolved_weights_path and os.path.isfile(resolved_weights_path):
                params = torch.load(resolved_weights_path, map_location=device)
                if isinstance(params, dict) and 'state_dict' in params:
                    state_dict = params['state_dict']
                else:
                    state_dict = params
                
                state_dict = {k.removeprefix('module.'): v for k, v in state_dict.items()}
                self.load_state_dict(state_dict, strict=False)

    def preprocess_audio(self, audio_path: str) -> torch.Tensor:
        wave, sr = sf.read(audio_path)
        if wave.ndim > 1:
            wave = wave[:, 0].squeeze()
        if sr != 24000:
            wave = librosa.resample(wave, orig_sr=sr, target_sr=24000)


        mean, std = -4, 4
        wave_tensor = torch.from_numpy(wave).float().to(self.device)
        mel_tensor = self.to_mel(wave_tensor)
        mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - mean) / std

        mel_tensor = mel_tensor.squeeze(0)
        length_feature = mel_tensor.size(1)
        mel_tensor = mel_tensor[:, :(length_feature - length_feature % 2)]
        return mel_tensor.unsqueeze(0)  # [1, 80, T]

    @torch.inference_mode()
    def forward(self, audio_path_or_mel, max_sec: float = 30.0) -> torch.Tensor:
        if isinstance(audio_path_or_mel, (str, os.PathLike)):
            mel = self.preprocess_audio(str(audio_path_or_mel))
        elif isinstance(audio_path_or_mel, torch.Tensor):
            mel = audio_path_or_mel
            if mel.dim() == 2:
                mel = mel.unsqueeze(0)
            mel = mel.to(self.device)
        else:
            raise ValueError(f"Expected audio file path (str) or mel spectrogram (torch.Tensor), got {type(audio_path_or_mel)}")

        mel_input_length = torch.tensor([mel.shape[-1]], dtype=torch.long, device=self.device)

        max_crop_len = 3300 if max_sec is None else min(int(max_sec * 24000 // 300), 3300)
        crop_len = min(mel.shape[-1], max_crop_len)
        mel_cropped = mel[:, :, :crop_len]
        mel_crop_length = torch.tensor([crop_len], dtype=torch.long, device=self.device)

        acoustic_style = self.style_encoder(mel, mel_input_length)
        prosodic_style = self.prosodic_style_encoder(mel_cropped, mel_crop_length)

        voice = torch.cat([acoustic_style, prosodic_style], dim=-1)
        return voice

    encode = forward


class HolosTTS(nn.Module):
    def __init__(self, weights_path="patriotyk/HolosTTS", device='cuda', voices_path="voices.pt"):
        super().__init__()
        
        self.add_module('text_encoder', TextEncoder(use_pe=True, partial_rope=True, use_conv_pe=False))
        self.add_module('prosodic_text_encoder', TextEncoder(n_layers=6, use_pe=True, partial_rope=True, use_conv_pe=True))
        self.add_module('duration_predictor', DurationPredictor(d_hid=512))
        self.add_module('prosodic_predictor', ProsodyPredictor(d_hid=512))
        self.add_module('decoder', Decoder())
        self.to(device)
        self.device = device
        self.eval()
        self.voices = {}

        if weights_path is not None:
            resolved_weights_path = None
            if os.path.isfile(weights_path):
                resolved_weights_path = weights_path
            else:
                resolved_weights_path = hf_hub_download(repo_id=weights_path, filename="holos.pt")
            
            params = torch.load(resolved_weights_path, map_location=device)
            if isinstance(params, dict) and 'vocab' in params:
                self.tokenizer = Tokenizer(params['vocab'])
                state_dict = params['state_dict']
            else:
                state_dict = params
            self.load_state_dict(state_dict, strict=True)

            resolved_voices_path = None
            if voices_path and os.path.isfile(voices_path):
                resolved_voices_path = voices_path
            elif not os.path.isfile(weights_path):
                try:
                    resolved_voices_path = hf_hub_download(repo_id=weights_path, filename="voices.pt")
                except Exception:
                    resolved_voices_path = None

            if resolved_voices_path and os.path.isfile(resolved_voices_path):
                self.voices = torch.load(resolved_voices_path, map_location=device)
            else:
                raise FileNotFoundError("Voices file not found")

    def list_voices(self):
        return list(self.voices.keys())
    
    @torch.inference_mode()
    def generate(self, text: str, voice: Union[str, torch.Tensor]) -> torch.Tensor:
        if isinstance(voice, torch.Tensor):
            voice_tensor = voice
            if voice_tensor.dim() == 1:
                voice_tensor = voice_tensor.unsqueeze(0)
            voice_tensor = voice_tensor.to(self.device)
        else:
            voice_tensor = self.voices[voice].unsqueeze(0).to(self.device)

        tokens = self.tokenizer(text)
        tokens.insert(0, 0)
        tokens.append(0)
        
        tokens = torch.tensor(tokens).unsqueeze(0).to(self.device)
        input_lengths = torch.tensor([tokens.size(1)], dtype=torch.long, device=self.device)
        
        return self.forward(tokens, voice_tensor, input_lengths)

    
        
    def onnx_alignment(self, pred_dur, max_length):
        seq_length_dynamic = pred_dur.sum(dim=0, keepdim=True).squeeze(0).to(dtype=torch.int64)
        
        if torch.onnx.is_in_onnx_export():
            # RKNN requires fully static graphs. We must set a fixed maximum audio frame length.
            # 1000 frames is ~20 seconds of audio. Adjust if you need longer generation.
            seq_length = torch.tensor(1000, device=pred_dur.device, dtype=torch.long)
        else:
            seq_length = seq_length_dynamic

        device = pred_dur.device
                
        clipped_dur = torch.clamp(pred_dur, 0, seq_length)
        
        zero_tensor = torch.zeros(1, device=device)
        prefix_sum = torch.cat([zero_tensor, torch.cumsum(clipped_dur[:-1], dim=0)])
        
        positions = torch.arange(seq_length, device=device).unsqueeze(0).expand(max_length, -1)
        
        start_positions = prefix_sum.unsqueeze(1)
        
        end_positions = torch.min(
            start_positions + clipped_dur.unsqueeze(1),
            torch.ones_like(start_positions) * seq_length
        )
        
        mask = (positions >= start_positions) & (positions < end_positions)

        return mask.float()
    
    
    @torch.no_grad()
    def encode(self, tokens: torch.Tensor, voice: torch.Tensor, input_lengths: torch.Tensor):
        """Encoder part: text → aligned features, f0, energy. No decoder."""
        voice_acoustic = voice[:, :128]
        voice_prosodic = voice[:, 128:]
        
        t_en = self.prosodic_text_encoder(tokens, input_lengths)
        d_en = t_en.transpose(1, 2)
        
        d = self.duration_predictor(d_en, voice_prosodic, input_lengths)

        duration = torch.sigmoid(d).sum(axis=-1) / 1.0
        pred_dur = torch.round(duration).long()[0]
        print(pred_dur)
        #pred_dur[1] = 10

        pred_aln_trg = self.onnx_alignment(pred_dur, d_en.size(1))
        #print(pred_dur, duration)
        # indices = torch.repeat_interleave(torch.arange(tokens.shape[1], device=tokens.device), pred_dur)
        # pred_aln_trg = torch.zeros((tokens.shape[1], indices.shape[0]), device=tokens.device)
        # pred_aln_trg[indices, torch.arange(indices.shape[0], device=tokens.device)] = 1
        # pred_aln_trg = pred_aln_trg.unsqueeze(0)


        # attn_list = []
        # max_len = d_en.size(1)
        # max_dur = 0
        # pred_durs = []
        # pred_lens = []
        # for bib in range(len(d_en)):
        #     l = int(input_lengths[bib])
        #     duration = torch.sigmoid(d[bib]).sum(axis=-1)
        #     pred_dur = torch.round(duration.squeeze(0)[:l]).clamp(min=1)
        #     print(pred_dur)
        #     #pred_dur[-1] =1
        #     pred_durs.append(pred_dur)
        #     max_dur = max(max_dur, int(pred_dur.sum().item()))
        #     pred_lens.append(pred_dur.sum().item() * 2.0)

        # for bib in range(len(d_en)):
        #     l = int(input_lengths[bib])
        #     pred_dur = pred_durs[bib]
        #     pred_aln_trg = torch.zeros(max_len, max_dur)
        #     c_frame = 0
        #     for i in range(l):
        #         dur_val = int(pred_dur[i].item())
        #         pred_aln_trg[i, c_frame:c_frame + dur_val] = 1
        #         c_frame += dur_val
        #     attn_list.append(pred_aln_trg)
        # pred_aln_trg = torch.stack(attn_list).to(tokens.device)
        
        en = d_en.transpose(-1, -2) @ pred_aln_trg
        wav_lengths = pred_dur.sum(dim=0, keepdim=True) * 2
        
        f0, energy = self.prosodic_predictor(en, voice_prosodic, wav_lengths)
        t_en = self.text_encoder(tokens, input_lengths)
        aln = t_en @ pred_aln_trg
        return aln, f0, energy, voice_acoustic, wav_lengths


    @torch.no_grad()
    def forward(self, tokens: torch.Tensor, voice: torch.Tensor, input_lengths: torch.Tensor) -> torch.Tensor:
        encoded = self.encode(tokens, voice, input_lengths)
        wave = self.decoder(*encoded)
        return wave.squeeze(0)
