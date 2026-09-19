# src/prosodia/model/state.py
from __future__ import annotations

import torch
from torch import Tensor, nn

from prosodia.model.pooling import AttentionPool, speaker_relative_norm


class StateEncoder(nn.Module):
    """Frozen frame features -> the shared state, encoded ONCE per example."""

    def __init__(
        self, in_dim: int, d_model: int = 256, n_layers: int = 2,
        n_heads: int = 4, stride: int = 2,
    ) -> None:
        super().__init__()
        self.project = nn.Linear(in_dim, d_model)
        self.pool = AttentionPool(d_model, stride=stride)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model,
            batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)

    def forward(self, audio: Tensor, audio_mask: Tensor) -> tuple[Tensor, Tensor]:
        x = speaker_relative_norm(audio, audio_mask)
        x = self.project(x)
        x, mask = self.pool(x, audio_mask)
        h = self.encoder(x, src_key_padding_mask=~mask)
        return h, mask
