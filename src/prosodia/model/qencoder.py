# src/prosodia/model/qencoder.py
"""Frozen sentence encoder over question instructions and candidate labels.

Frozen is the load-bearing choice: you cannot LEARN a question encoder from
~3 base question types, but you can learn to READ a pretrained semantic space.
Paraphrase augmentation is what teaches that reading (spec §5).
"""
from __future__ import annotations

import torch
from torch import Tensor, nn


class QuestionEncoder(nn.Module):
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        d_model: int = 256,
    ) -> None:
        super().__init__()
        from sentence_transformers import SentenceTransformer

        self._st = SentenceTransformer(model_name)
        for p in self._st.parameters():
            p.requires_grad_(False)
        self._st.eval()
        self.project = nn.Linear(self._st.get_embedding_dimension(), d_model)
        self._cache: dict[str, Tensor] = {}
        self.cache_misses = 0

    def train(self, mode: bool = True) -> "QuestionEncoder":
        # nn.Module.train() recurses into every submodule, including the
        # frozen sentence-transformer. If a caller trains a larger model
        # this encoder is embedded in, `.train()` would otherwise flip the
        # frozen encoder into train mode too -- enabling its dropout layers
        # and making "frozen" embeddings nondeterministic even though their
        # gradients stay off. Keep it pinned to eval regardless.
        super().train(mode)
        self._st.eval()
        return self

    def _raw(self, texts: list[str]) -> Tensor:
        missing = [t for t in texts if t not in self._cache]
        if missing:
            self.cache_misses += len(missing)
            with torch.no_grad():
                vecs = self._st.encode(missing, convert_to_tensor=True,
                                       show_progress_bar=False).cpu()
            for t, v in zip(missing, vecs):
                self._cache[t] = v
        return torch.stack([self._cache[t] for t in texts])

    def embed_texts(self, texts: list[str]) -> Tensor:
        raw = self._raw(texts).to(self.project.weight.device,
                                  self.project.weight.dtype)
        return self.project(raw)
