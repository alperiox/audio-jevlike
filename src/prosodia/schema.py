"""Core types. Label provenance is a first-class field, not a comment.

Spec §4.0 / §11 trap 9: HarperValleyBank's valence labels turned out to be a
proprietary audio model's outputs. Training a prosody thesis on them would
have been circular. The tier travels with every label so that failure mode
is impossible to repeat by accident.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Sequence


class LabelTier(str, Enum):
    GOLD = "gold"                  # true by construction (e.g. assigned task)
    HUMAN = "human"                # human annotators
    MODEL_OUTPUT = "model_output"  # another model's predictions — breadth only


THESIS_SAFE_TIERS = frozenset({LabelTier.GOLD, LabelTier.HUMAN})

QType = str  # "noul" | "choice" | "score"


@dataclass(frozen=True)
class Label:
    value: Any
    tier: LabelTier


@dataclass(frozen=True)
class QuestionSpec:
    key: str
    qtype: QType
    instructions: str
    criteria: Any = None  # choice: dict[str, str|None]; score: ordered list; noul: dict|None

    def __post_init__(self) -> None:
        if self.qtype not in {"noul", "choice", "score"}:
            raise ValueError(f"unknown qtype {self.qtype!r}")
        if self.qtype == "choice":
            if not isinstance(self.criteria, dict) or len(self.criteria) < 2:
                raise ValueError("choice requires a dict of >= 2 options")
        if self.qtype == "score":
            if not isinstance(self.criteria, Sequence) or len(self.criteria) < 2:
                raise ValueError("score requires an ordered sequence of >= 2 levels")

    @property
    def options(self) -> list[str]:
        if self.qtype == "noul":
            return ["false", "true"]
        if self.qtype == "choice":
            return list(self.criteria.keys())
        return [str(c) for c in self.criteria]

    @property
    def n_options(self) -> int:
        return len(self.options)


@dataclass(frozen=True)
class Example:
    uid: str
    corpus: str
    audio_path: str
    context: str
    labels: dict[str, Label] = field(default_factory=dict)
    speaker: str | None = None


def assert_thesis_safe(examples: Iterable[Example], question_keys: Sequence[str]) -> None:
    """Raise if any label backing a thesis-testing question is a model output."""
    offenders: set[str] = set()
    for ex in examples:
        for key in question_keys:
            lab = ex.labels.get(key)
            if lab is not None and lab.tier not in THESIS_SAFE_TIERS:
                offenders.add(f"{key}:{lab.tier.value}")
    if offenders:
        raise ValueError(
            "thesis-testing split contains non-gold/human labels — "
            f"MODEL_OUTPUT present for {sorted(offenders)}. See spec §4.0."
        )
