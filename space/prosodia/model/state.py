# src/prosodia/model/state.py
from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from prosodia.model.pooling import AttentionPool, speaker_relative_norm, utterance_statistics


class SinusoidalPositionalEncoding(nn.Module):
    """Adds absolute-position information to a (B, T, D) sequence.

    C1 fix: without this, `nn.TransformerEncoder` self-attention is
    permutation-equivariant and `IsolatedBranches`' cross-attention is
    permutation-invariant over its keys, so the whole model was exactly
    invariant to reversing the audio's time axis (measured max|logit diff|
    ~1e-9 for a full reversal and for a rise-vs-fall ramp pair). Position is
    computed fresh at forward time from `x`'s own shape/device/dtype so it
    is always correct regardless of module placement, and works for any
    sequence length the pooled state happens to have (including the 2 extra
    utterance-statistics positions the C2 fix prepends).
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model

    def forward(self, x: Tensor) -> Tensor:
        b, t, d = x.shape
        position = torch.arange(t, device=x.device, dtype=x.dtype).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d, 2, device=x.device, dtype=x.dtype)
            * (-math.log(10000.0) / d)
        )
        pe = torch.zeros(t, d, device=x.device, dtype=x.dtype)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        return x + pe.unsqueeze(0)


class StateEncoder(nn.Module):
    """Frozen frame features -> the shared state, encoded ONCE per example."""

    def __init__(
        self, in_dim: int, d_model: int = 256, n_layers: int = 2,
        n_heads: int = 4, stride: int = 2,
    ) -> None:
        super().__init__()
        self.project = nn.Linear(in_dim, d_model)
        self.pool = AttentionPool(d_model, stride=stride)
        self.pos_enc = SinusoidalPositionalEncoding(d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model,
            batch_first=True, norm_first=True,
        )
        # enable_nested_tensor=False: the nested-tensor fast path is already
        # unavailable because norm_first=True (deliberate). Without this,
        # nn.TransformerEncoder emits a UserWarning on every construction.
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=n_layers, enable_nested_tensor=False,
        )

    def forward(self, audio: Tensor, audio_mask: Tensor) -> tuple[Tensor, Tensor]:
        # C2 fix: compute un-normalized level stats BEFORE the normalization
        # that strips them, and carry them forward as two extra state
        # positions (see `utterance_statistics` docstring).
        mean, std = utterance_statistics(audio, audio_mask)
        stat_tokens = self.project(torch.cat([mean, std], dim=1))  # (b, 2, d)
        has_audio = audio_mask.any(dim=1, keepdim=True)             # (b, 1)
        stat_mask = has_audio.expand(-1, 2)                         # (b, 2)

        x = speaker_relative_norm(audio, audio_mask)
        x = self.project(x)
        x, mask = self.pool(x, audio_mask)

        x = torch.cat([stat_tokens, x], dim=1)
        mask = torch.cat([stat_mask, mask], dim=1)

        # C1 fix: position must be injected before the (permutation-
        # equivariant) self-attention encoder for the encoder's output to
        # depend on time order at all.
        x = self.pos_enc(x)
        h = self.encoder(x, src_key_padding_mask=~mask)
        return h, mask
