import torch
import numpy as np
import torch.nn as nn
from dataclasses import dataclass
from holos.modules import StyleEncoder, TextEncoder, ProsodyPredictor, DurationPredictor, Tokenizer
from holos.decoder import Decoder



class HolosTTS(nn.Module):
    def __init__(self, weights_path=None, device='cuda', onnx_phase_scale=0.0):
        super().__init__()
        
        self.add_module('text_encoder', TextEncoder())
        self.add_module('prosodic_text_encoder', TextEncoder(n_layers=6, use_pe=True, scale_embedding=False, partial_rope=True, use_conv_pe=True))
        self.add_module('duration_predictor', DurationPredictor(d_hid=512))
        self.add_module('prosodic_predictor', ProsodyPredictor(d_hid=512))
        self.add_module('decoder', Decoder(onnx_phase_scale=onnx_phase_scale))
        self.unk_style = nn.Parameter(torch.zeros(1, 128))
        self.to(device)
        self.device = device
        self.eval()
        self.voices = {}
        if weights_path is not None:
            params = torch.load(weights_path, map_location=device)
            self.tokenizer = Tokenizer(params['vocab'])
            self.load_state_dict(params['state_dict'], strict=False)
            self.voices = params['voices']

    def set_onnx_phase_scale(self, value: float):
        self.decoder.generator.stft.set_onnx_phase_scale(value)

    def list_voices(self):
        return list(self.voices.keys())
    
    @torch.inference_mode()
    def generate(self, text: str, voice_name: str, index, alpha: float = 0.3) -> torch.Tensor:
        #voice = torch.stack(self.voices[voice_name]).to(self.device).mean(dim=0, keepdim=True)
        #print(self.voices[voice_name][0].shape)
        voice = self.voices[voice_name][index].unsqueeze(0).to(self.device)
        #voice = torch.load('voice_1_1.pt').to(self.device)

        tokens = self.tokenizer(text)
        tokens.insert(0, 0)
        tokens.append(0)
        
        tokens = torch.tensor(tokens).unsqueeze(0).to(self.device)
        input_lengths = torch.tensor([tokens.size(1)], dtype=torch.long, device=self.device)
        
        return self.forward(tokens, voice, input_lengths, alpha)

    
        
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
    def encode(self, tokens: torch.Tensor, voice: torch.Tensor, input_lengths: torch.Tensor, alpha: float = 0.3):
        """Encoder part: text → aligned features, f0, energy. No decoder."""
        voice_acoustic = voice[:, :128]
        voice_prosodic = voice[:, 128:]
        
        t_en = self.prosodic_text_encoder(tokens, input_lengths)
        d_en = t_en.transpose(1, 2)
        
        if alpha > 0.0:
            unk = self.unk_style.expand(voice_prosodic.size(0), -1)
            d_uncond = self.duration_predictor(d_en, unk, input_lengths)
            d_cond = self.duration_predictor(d_en, voice_prosodic, input_lengths)
            d = d_cond * (1 - alpha) + d_uncond * alpha
        else:
            d = self.duration_predictor(d_en, voice_prosodic, input_lengths)
        
        duration = torch.sigmoid(d).sum(axis=-1) / 1.0
        pred_dur = torch.round(duration).clamp(min=1).long()[0]
        print(pred_dur)
        #pred_dur[0] = 8

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
        
        if alpha > 0.0:
            unk = self.unk_style.expand(voice_prosodic.size(0), -1)
            f0_uncond, energy_uncond = self.prosodic_predictor(en, unk, wav_lengths)
            f0_cond, energy_cond = self.prosodic_predictor(en, voice_prosodic, wav_lengths)
            f0 = f0_cond * (1 - alpha) + f0_uncond * alpha
            energy = energy_cond * (1 - alpha) + energy_uncond * alpha
        else:
            f0, energy = self.prosodic_predictor(en, voice_prosodic, wav_lengths)
        t_en = self.text_encoder(tokens, input_lengths)
        aln = t_en @ pred_aln_trg
        return aln, f0, energy, voice_acoustic, wav_lengths


    @torch.no_grad()
    def forward(self, tokens: torch.Tensor, voice: torch.Tensor, input_lengths: torch.Tensor, alpha: float = 0.3) -> torch.Tensor:
        aln, f0, energy, voice_acoustic, wav_lengths = self.encode(tokens, voice, input_lengths, alpha)
        wave = self.decoder(aln, f0, energy, voice_acoustic, wav_lengths)
        return wave.squeeze(0)