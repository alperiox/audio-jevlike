# src/prosodia/model/branches.py
"""Isolated question branches — the Jev-shaped core.

Each question cross-attends to the shared state; no self-attention runs across
the question axis. Isolation is structural, not a mask on a shared attention:
questions are folded into the batch dimension so they cannot interact at all.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn


class _BranchLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm_ff = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, q: Tensor, kv: Tensor, kv_pad: Tensor) -> Tensor:
        attn_out, _ = self.attn(self.norm_q(q), self.norm_kv(kv), self.norm_kv(kv),
                                key_padding_mask=kv_pad, need_weights=False)
        q = q + attn_out
        return q + self.ff(self.norm_ff(q))


class IsolatedBranches(nn.Module):
    def __init__(self, d_model: int = 256, n_layers: int = 2, n_heads: int = 4) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [_BranchLayer(d_model, n_heads) for _ in range(n_layers)]
        )

    def forward(self, q: Tensor, h_state: Tensor, state_mask: Tensor) -> Tensor:
        """q: (B, Q, D); h_state: (B, T, D); state_mask: (B, T) True where valid."""
        b, n_q, d = q.shape
        t = h_state.shape[1]

        # Fold questions into batch: each becomes an independent sequence of
        # length 1. Cross-question attention is then impossible by construction.
        qf = q.reshape(b * n_q, 1, d)
        kv = h_state.unsqueeze(1).expand(b, n_q, t, d).reshape(b * n_q, t, d)
        kv_pad = (~state_mask).unsqueeze(1).expand(b, n_q, t).reshape(b * n_q, t)

        for layer in self.layers:
            qf = layer(qf, kv, kv_pad)
        return qf.reshape(b, n_q, d)
