# src/prosodia/model/prosodia.py
from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from prosodia.model.branches import IsolatedBranches
from prosodia.model.heads import ReadoutHead
from prosodia.model.qencoder import QuestionEncoder
from prosodia.model.state import StateEncoder


class ProsodiaModel(nn.Module):
    """state (audio + context) -> per-question logits over supplied options."""

    def __init__(
        self, in_dim: int, d_model: int = 256, state_layers: int = 2,
        branch_layers: int = 2, n_heads: int = 4, stride: int = 2,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.state_encoder = StateEncoder(in_dim, d_model, state_layers, n_heads, stride)
        self.question_encoder = QuestionEncoder(d_model=d_model)
        self.branches = IsolatedBranches(d_model, branch_layers, n_heads)
        self.readout = ReadoutHead(d_model)
        self.audio_absent = nn.Parameter(torch.zeros(d_model))
        self.context_absent = nn.Parameter(torch.zeros(d_model))

    def _encode_state(self, batch: dict[str, Any]) -> tuple[Tensor, Tensor]:
        h, mask = self.state_encoder(batch["audio"], batch["audio_mask"])

        # Modality dropout: replace the whole audio state with a learned token.
        audio_present = batch["audio_present"].to(h.device).view(-1, 1, 1)
        h = torch.where(audio_present, h, self.audio_absent.view(1, 1, -1).expand_as(h))

        ctx_present = batch["context_present"].to(h.device)
        ctx_vecs = self.question_encoder.embed_texts(list(batch["context"]))
        ctx_vecs = torch.where(ctx_present.view(-1, 1), ctx_vecs,
                               self.context_absent.view(1, -1).expand_as(ctx_vecs))

        # Context joins the state as one extra position the branches attend to.
        h = torch.cat([ctx_vecs.unsqueeze(1), h], dim=1)
        mask = torch.cat([torch.ones(h.shape[0], 1, dtype=torch.bool, device=mask.device),
                          mask], dim=1)
        return h, mask

    def forward(self, batch: dict[str, Any]) -> list[dict[str, Tensor]]:
        h, mask = self._encode_state(batch)
        results: list[dict[str, Tensor]] = []

        for i, questions in enumerate(batch["questions"]):
            keys = list(questions)
            if not keys:
                results.append({})
                continue

            q_vecs = self.question_encoder.embed_texts(
                [questions[k]["instructions"] for k in keys]
            ).unsqueeze(0)                                        # (1, Q, D)
            branch_out = self.branches(q_vecs, h[i : i + 1], mask[i : i + 1])

            per_question: dict[str, Tensor] = {}
            for j, key in enumerate(keys):
                opts = questions[key]["options"]
                opt_vecs = self.question_encoder.embed_texts(opts).unsqueeze(0)
                logits = self.readout(branch_out[:, j : j + 1], opt_vecs)
                per_question[key] = logits.squeeze(0)
            results.append(per_question)

        return results
