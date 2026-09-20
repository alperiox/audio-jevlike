# src/prosodia/model/qencoder.py
"""Frozen sentence encoder over question instructions and candidate labels.

Frozen is the load-bearing choice: you cannot LEARN a question encoder from
~3 base question types, but you can learn to READ a pretrained semantic space.
Paraphrase augmentation is what teaches that reading (spec §5).
"""
from __future__ import annotations

import torch
from torch import Tensor, nn


def _embedding_dim(st_model) -> int:
    """Output width of a SentenceTransformer, across library versions.

    sentence-transformers renamed this accessor: 6.x exposes BOTH
    `get_embedding_dimension` and `get_sentence_embedding_dimension`, while
    3.x has only the latter. Calling the 6.x-only name works on a dev box and
    dies at import on a pinned deployment -- which is exactly how it failed on
    a Hugging Face Space, after a clean build, with the model already
    downloaded.

    Tries the long-standing name first so the common path does not depend on
    a newer alias, and raises with both names on failure rather than letting
    an AttributeError surface from inside nn.Linear.
    """
    for name in ("get_sentence_embedding_dimension", "get_embedding_dimension"):
        fn = getattr(st_model, name, None)
        if fn is not None:
            dim = fn()
            if dim:
                return int(dim)
    raise AttributeError(
        f"{type(st_model).__name__} exposes neither "
        "get_sentence_embedding_dimension() nor get_embedding_dimension(); "
        "cannot size the projection layer"
    )


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
        self.project = nn.Linear(_embedding_dim(self._st), d_model)
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
