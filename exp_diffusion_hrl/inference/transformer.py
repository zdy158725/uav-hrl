import math
from typing import Optional, Union

import torch
import torch.nn as nn


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device) * -emb)
        emb = x[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class TransformerForDiffusion(nn.Module):
    def __init__(
        self, input_dim, output_dim, horizon, n_obs_steps=None, cond_dim=0,
        n_layer=8, n_head=8, n_emb=256, p_drop_emb=0.1, p_drop_attn=0.1,
        causal_attn=True, time_as_cond=True, obs_as_cond=True, n_cond_layers=4,
    ):
        super().__init__()
        self._dummy_variable = nn.Parameter()
        n_obs_steps = horizon if n_obs_steps is None else n_obs_steps
        token_count = horizon + (0 if time_as_cond else 1)
        cond_count = (1 if time_as_cond else 0) + (n_obs_steps if cond_dim > 0 else 0)
        self.input_emb = nn.Linear(input_dim, n_emb)
        self.pos_emb = nn.Parameter(torch.zeros(1, token_count, n_emb))
        self.drop = nn.Dropout(p_drop_emb)
        self.time_emb = SinusoidalPosEmb(n_emb)
        self.cond_obs_emb = nn.Linear(cond_dim, n_emb) if cond_dim > 0 else None
        self.cond_pos_emb = nn.Parameter(torch.zeros(1, cond_count, n_emb))
        encoder_layer = nn.TransformerEncoderLayer(
            n_emb, n_head, 4 * n_emb, p_drop_attn, "gelu", batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, n_cond_layers)
        decoder_layer = nn.TransformerDecoderLayer(
            n_emb, n_head, 4 * n_emb, p_drop_attn, "gelu", batch_first=True, norm_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, n_layer)
        self.ln_f = nn.LayerNorm(n_emb)
        self.head = nn.Linear(n_emb, output_dim)
        self.obs_as_cond = cond_dim > 0
        if causal_attn:
            mask = torch.triu(torch.ones(token_count, token_count), diagonal=1)
            self.register_buffer("mask", mask.masked_fill(mask == 1, float("-inf")))
            t, s = torch.meshgrid(torch.arange(token_count), torch.arange(cond_count), indexing="ij")
            memory_mask = (t >= (s - 1)).float()
            self.register_buffer(
                "memory_mask",
                memory_mask.masked_fill(memory_mask == 0, float("-inf")).masked_fill(memory_mask == 1, 0.0),
            )
        else:
            self.mask = None
            self.memory_mask = None

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def forward(
        self, sample: torch.Tensor, timestep: Union[torch.Tensor, float, int],
        cond: Optional[torch.Tensor] = None, token_offset=0, **kwargs,
    ):
        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], dtype=torch.long, device=sample.device)
        elif timestep.ndim == 0:
            timestep = timestep[None].to(sample.device)
        timestep = timestep.expand(sample.shape[0])
        time_emb = self.time_emb(timestep).unsqueeze(1)
        cond_emb = torch.cat([time_emb, self.cond_obs_emb(cond)], dim=1)
        tc = cond_emb.shape[1]
        memory = self.encoder(self.drop(cond_emb + self.cond_pos_emb[:, :tc]))
        t = sample.shape[1]
        start, end = int(token_offset), int(token_offset) + t
        x = self.drop(self.input_emb(sample) + self.pos_emb[:, start:end])
        tgt_mask = None if self.mask is None else self.mask[start:end, start:end]
        memory_mask = None if self.memory_mask is None else self.memory_mask[start:end, :tc]
        x = self.decoder(x, memory, tgt_mask=tgt_mask, memory_mask=memory_mask)
        return self.head(self.ln_f(x))
