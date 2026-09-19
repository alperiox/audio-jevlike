"""Post-hoc temperature scaling — ablation Arm C (spec §6).

This is the control that decides the Phase 2 interpretability question. If
CE+Brier is no better calibrated than CE plus a single learned scalar, then
calibration lives in a temperature and there is nothing distributed to find.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


class TemperatureScaler:
    def __init__(self) -> None:
        self.temperature = torch.ones(1)

    def fit(self, logits: Tensor, targets: Tensor, max_iter: int = 100) -> "TemperatureScaler":
        logits = logits.detach().to(torch.float32)
        log_t = torch.zeros(1, requires_grad=True)  # optimise log T to keep T > 0
        optimizer = torch.optim.LBFGS([log_t], lr=0.1, max_iter=max_iter)

        def closure():
            optimizer.zero_grad()
            loss = F.cross_entropy(logits / log_t.exp(), targets)
            loss.backward()
            return loss

        optimizer.step(closure)
        self.temperature = log_t.exp().detach()
        return self

    def transform(self, logits: Tensor) -> Tensor:
        return logits.detach().to(torch.float32) / self.temperature.to(logits.device)
