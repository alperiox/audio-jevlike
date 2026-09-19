"""Calibration metrics are first-class here, not an afterthought (spec §7).

All functions take PROBABILITIES and run in fp32. Accuracy answers "is it
right"; ECE answers "does it know when it is right", which is the quantity
this whole project is about.
"""
from __future__ import annotations

import torch
from torch import Tensor


def _fp32(x: Tensor) -> Tensor:
    return x.detach().to(torch.float32)


def accuracy(probs: Tensor, targets: Tensor) -> Tensor:
    return (_fp32(probs).argmax(-1) == targets).float().mean()


def macro_f1(probs: Tensor, targets: Tensor) -> Tensor:
    preds = _fp32(probs).argmax(-1)
    scores = []
    for c in range(probs.shape[-1]):
        tp = ((preds == c) & (targets == c)).sum().float()
        fp = ((preds == c) & (targets != c)).sum().float()
        fn = ((preds != c) & (targets == c)).sum().float()
        denom = 2 * tp + fp + fn
        scores.append(torch.tensor(0.0) if denom == 0 else 2 * tp / denom)
    return torch.stack(scores).mean()


def brier_score(probs: Tensor, targets: Tensor) -> Tensor:
    p = _fp32(probs)
    onehot = torch.zeros_like(p).scatter_(-1, targets.unsqueeze(-1), 1.0)
    return ((p - onehot) ** 2).sum(-1).mean()


def negative_log_likelihood(probs: Tensor, targets: Tensor) -> Tensor:
    p = _fp32(probs).clamp_min(1e-12)
    return -p.gather(-1, targets.unsqueeze(-1)).squeeze(-1).log().mean()


def expected_calibration_error(probs: Tensor, targets: Tensor, n_bins: int = 10) -> Tensor:
    """Top-label ECE: weighted mean |confidence - accuracy| across bins."""
    p = _fp32(probs)
    conf, pred = p.max(-1)
    correct = (pred == targets).float()

    edges = torch.linspace(0.0, 1.0, n_bins + 1, device=p.device)
    total = torch.tensor(0.0, device=p.device)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        in_bin = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        n = in_bin.sum()
        if n == 0:
            continue
        total = total + (n.float() / p.shape[0]) * \
            (conf[in_bin].mean() - correct[in_bin].mean()).abs()
    return total


def coverage_curve(
    probs: Tensor, targets: Tensor, n_points: int = 50
) -> tuple[Tensor, Tensor, Tensor]:
    """Accuracy-vs-coverage: at confidence threshold t, what fraction of traffic
    is automated and what error rate does it carry (spec §7)."""
    p = _fp32(probs)
    conf, pred = p.max(-1)
    correct = (pred == targets)

    thresholds = torch.linspace(0.0, conf.max().item(), n_points)
    coverage, error = [], []
    for t in thresholds:
        keep = conf >= t
        n = keep.sum()
        coverage.append(n.float() / p.shape[0])
        error.append(torch.tensor(0.0) if n == 0 else 1.0 - correct[keep].float().mean())
    return thresholds, torch.stack(coverage), torch.stack(error)
