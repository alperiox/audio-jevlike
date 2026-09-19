"""Readout: z = Wh, strictly linear and bias-free, per spec §5.

Archer Hume's reverse-engineering of the model this project replicates
reported an affine readout, z = Wh + b. This implementation deliberately
drops the bias: a global bias is meaningless for a pointer head whose
option set changes between requests, and exact linearity from h to logits
is required by a later interpretability phase.

One pointer-style head serves all three primitives: logits are an inner
product between the projected branch vector and projected option embeddings,
so the head handles any number of candidates without retraining.

  Noul   -> two options ["false", "true"]; P(true) is probs[..., 1]
  Choice -> softmax over the supplied option set
  Score  -> softmax over ordered levels, read out as an expectation
"""
from __future__ import annotations

import torch
from torch import Tensor, nn


class ReadoutHead(nn.Module):
    def __init__(self, d_model: int = 256) -> None:
        super().__init__()
        # Both projections are bias-free: the map h -> logits must be exactly
        # linear (Phase 2 depends on it), and a global bias is meaningless for
        # a pointer head whose option set changes between requests.
        self.w_branch = nn.Linear(d_model, d_model, bias=False)
        self.w_option = nn.Linear(d_model, d_model, bias=False)
        self.scale = d_model ** -0.5

    def forward(self, branch_vec: Tensor, option_vecs: Tensor) -> Tensor:
        """branch_vec: (B, 1, D); option_vecs: (B, K, D) -> logits (B, K)."""
        h = self.w_branch(branch_vec)            # (B, 1, D)
        o = self.w_option(option_vecs)           # (B, K, D)
        return (h @ o.transpose(1, 2)).squeeze(1) * self.scale


def score_expectation(probs: Tensor) -> Tensor:
    """Probability-weighted level index: score = sum_i i * p(i)."""
    idx = torch.arange(probs.shape[-1], device=probs.device, dtype=probs.dtype)
    return (probs * idx).sum(-1)


def confidence(probs: Tensor) -> Tensor:
    """Normalized certainty in [0, 1]: 1 - H(p)/log(K)."""
    k = probs.shape[-1]
    if k < 2:
        return torch.ones(probs.shape[:-1], device=probs.device)
    entropy = -(probs.clamp_min(1e-12) * probs.clamp_min(1e-12).log()).sum(-1)
    return 1.0 - entropy / torch.log(torch.tensor(float(k), device=probs.device))
