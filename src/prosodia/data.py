"""Dataset and collation.

Modality dropout (spec §5) yields three eval conditions — full / audio-only /
context-only — from one checkpoint, and simultaneously prevents the model from
learning to ignore audio whenever context is informative. It never drops both:
an example with no state at all carries no signal.
"""
from __future__ import annotations

import random
from typing import Any, Sequence

import torch
from torch.utils.data import Dataset

from prosodia.features import FeatureCache
from prosodia.questions import paraphrase, permute_candidates
from prosodia.schema import Example, QuestionSpec


class ProsodiaDataset(Dataset):
    def __init__(
        self,
        examples: Sequence[Example],
        specs: Sequence[QuestionSpec],
        cache: FeatureCache,
        rng_seed: int = 0,
        augment: bool = True,
        modality_dropout: float = 0.15,
    ) -> None:
        self.examples = [e for e in examples if e.uid in cache]
        self.specs = list(specs)
        self.cache = cache
        self.augment = augment
        self.modality_dropout = modality_dropout
        self._seed = rng_seed

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        ex = self.examples[idx]
        rng = random.Random((self._seed, idx, random.random()).__hash__())

        audio_present, context_present = True, True
        if self.modality_dropout > 0 and rng.random() < self.modality_dropout:
            # drop exactly one modality, never both
            if rng.random() < 0.5:
                audio_present = False
            else:
                context_present = False

        questions: dict[str, Any] = {}
        targets: dict[str, Any] = {}
        for spec in self.specs:
            label = ex.labels.get(spec.key)
            if label is None:
                continue
            active = spec
            if self.augment:
                active = paraphrase(active, rng)
                gold = label.value if spec.qtype == "choice" else None
                active, _ = permute_candidates(active, rng, keep=gold)

            options = active.options
            if spec.qtype == "choice":
                # permute_candidates guarantees the gold option survives
                target = options.index(label.value)
            elif spec.qtype == "score":
                target = int(label.value)
            else:
                target = int(label.value)

            questions[spec.key] = {
                "instructions": active.instructions,
                "options": options,
                "qtype": spec.qtype,
            }
            targets[spec.key] = target

        return {
            "uid": ex.uid,
            "audio": self.cache.read(ex.uid),
            "audio_present": audio_present,
            "context": ex.context,
            "context_present": context_present,
            "questions": questions,
            "targets": targets,
            "speaker": ex.speaker,
        }


def collate_batch(items: Sequence[dict[str, Any]]) -> dict[str, Any]:
    lengths = [it["audio"].shape[0] for it in items]
    tmax, dim = max(lengths), items[0]["audio"].shape[1]
    audio = torch.zeros(len(items), tmax, dim)
    mask = torch.zeros(len(items), tmax, dtype=torch.bool)
    for i, it in enumerate(items):
        n = it["audio"].shape[0]
        audio[i, :n] = it["audio"]
        mask[i, :n] = True
    return {
        "uid": [it["uid"] for it in items],
        "audio": audio,
        "audio_mask": mask,
        "audio_present": torch.tensor([it["audio_present"] for it in items]),
        "context": [it["context"] for it in items],
        "context_present": torch.tensor([it["context_present"] for it in items]),
        "questions": [it["questions"] for it in items],
        "targets": [it["targets"] for it in items],
        "speaker": [it["speaker"] for it in items],
    }
