# src/prosodia/model/pooling.py
"""Contour-preserving pooling.

Prosody is supra-segmental: it lives in contours over time, not in frames.
Mean pooling maps a rise and a fall to the same vector, which would delete the
thesis signal in the first layer with no visible symptom (spec §11 trap 3).
AttentionPool learns per-frame weights within each window instead.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def utterance_statistics(x: Tensor, mask: Tensor, eps: float = 1e-5) -> tuple[Tensor, Tensor]:
    """Un-normalized per-channel utterance mean and std, shape (b, 1, d) each.

    C2 fix: `speaker_relative_norm` below divides these back out, which is
    exactly the point of within-utterance normalization -- but it also means
    "loud" or "high-pitched" *for this speaker* becomes unrepresentable
    downstream: a loud utterance and a quiet one with the same contour SHAPE
    normalize to bit-identical tensors. `StateEncoder` appends these raw
    stats as extra state positions so the model can recover level while the
    pooled sequence still carries the normalized contour.
    """
    m = mask.unsqueeze(-1).float()
    n = m.sum(dim=1, keepdim=True).clamp(min=1.0)
    mean = (x * m).sum(dim=1, keepdim=True) / n
    var = (((x - mean) ** 2) * m).sum(dim=1, keepdim=True) / n
    std = (var + eps).sqrt()
    return mean, std


def speaker_relative_norm(x: Tensor, mask: Tensor, eps: float = 1e-5) -> Tensor:
    """Centre and scale within the utterance.

    'High pitch' is only meaningful relative to that speaker's own baseline,
    so normalization is within-utterance, not global (spec §5). This strips
    level by construction -- see `utterance_statistics`, which callers that
    need level (e.g. `StateEncoder`) should also use.
    """
    m = mask.unsqueeze(-1).float()
    mean, std = utterance_statistics(x, mask, eps)
    return (x - mean) / std * m


class AttentionPool(nn.Module):
    """Strided pooling with learned within-window attention weights."""

    def __init__(self, dim: int, stride: int = 2) -> None:
        super().__init__()
        self.stride = stride
        self.score = nn.Linear(dim, 1)

    def forward(self, x: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        b, t, d = x.shape
        s = self.stride
        pad = (-t) % s
        if pad:
            x = F.pad(x, (0, 0, 0, pad))
            mask = F.pad(mask, (0, pad), value=False)
        tw = (t + pad) // s

        xw = x.view(b, tw, s, d)
        mw = mask.view(b, tw, s)

        logits = self.score(xw).squeeze(-1)                      # (b, tw, s)
        logits = logits.masked_fill(~mw, torch.finfo(logits.dtype).min)
        weights = torch.softmax(logits, dim=-1)
        weights = weights * mw.float()
        weights = weights / weights.sum(-1, keepdim=True).clamp(min=1e-9)

        pooled = (xw * weights.unsqueeze(-1)).sum(dim=2)
        pooled_mask = mw.any(dim=-1)
        return pooled * pooled_mask.unsqueeze(-1), pooled_mask
