"""Training objectives for the loss ablation (spec §6).

  Arm A: brier_weight = 0.0            (cross-entropy only)
  Arm B: brier_weight > 0.0            (RLCD stand-in — proper scoring composite)
  Arm C: Arm A + post-hoc temperature scaling (Task 13)

If B is indistinguishable from C, calibration is a scalar and that IS the
answer to "where does calibration live".
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def cross_entropy_loss(logits: Tensor, target: Tensor) -> Tensor:
    return F.cross_entropy(logits, target)


def brier_loss(logits: Tensor, target: Tensor) -> Tensor:
    """Multiclass Brier score: mean squared error against the one-hot target.

    Range [0, 2]. Strictly proper, so it is minimized only by the true
    probabilities — unlike CE it does not reward unbounded confidence.
    """
    probs = torch.softmax(logits, dim=-1)
    onehot = F.one_hot(target, num_classes=logits.shape[-1]).to(probs.dtype)
    return ((probs - onehot) ** 2).sum(-1).mean()


def composite_loss(logits: Tensor, target: Tensor, brier_weight: float = 0.0) -> Tensor:
    if brier_weight < 0.0:
        # Same defect family as the headline Arm C bug: a config field that
        # quietly stops meaning what its name says. A negative weight (a
        # config typo) used to silently fall through to Arm A (pure CE)
        # instead of raising, making a broken ablation arm look like a
        # deliberate one.
        raise ValueError(f"brier_weight must be >= 0.0, got {brier_weight!r}")
    if brier_weight == 0.0:
        return cross_entropy_loss(logits, target)
    ce = cross_entropy_loss(logits, target)
    br = brier_loss(logits, target)
    return (1.0 - brier_weight) * ce + brier_weight * br
