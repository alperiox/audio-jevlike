"""Two baselines doing different jobs (spec §7).

  TextOnlyBaseline — CONTROLLED. Identical architecture and training, audio
                     permanently absent. Isolates modality and nothing else.
  Jev via API      — PRACTICAL. The actual text-state System One model.
"""
from __future__ import annotations

from typing import Any, Sequence

from prosodia.config import RunConfig
from prosodia.schema import QuestionSpec

JEV_ENDPOINT = "https://api.typesafe.ai/v1/systemone"


def build_jev_request(state: str, specs: Sequence[QuestionSpec]) -> dict[str, Any]:
    """Serialize our question bank into the documented TypeSafe schema."""
    questions: dict[str, Any] = {}
    for spec in specs:
        q: dict[str, Any] = {"type": spec.qtype, "instructions": spec.instructions}
        if spec.qtype == "choice":
            q["criteria"] = {k: v for k, v in spec.criteria.items()}
        elif spec.qtype == "score":
            q["criteria"] = list(spec.criteria)
        elif spec.criteria:
            q["criteria"] = dict(spec.criteria)
        questions[spec.key] = q
    return {"state": state, "model": "jev-latest", "questions": questions}


def _arm_name(encoder: str, brier: float, temp: bool) -> str:
    loss = "C-temp" if temp else ("B-brier" if brier > 0 else "A-ce")
    return f"{encoder}__{loss}"


ARMS: list[RunConfig] = [
    RunConfig(name=_arm_name(enc, brier, temp), encoder=enc,
              brier_weight=brier, temperature_scale=temp)
    for enc in ("wavlm", "whisper", "prosody")
    for brier, temp in ((0.0, False), (0.5, False), (0.0, True))
]


class TextOnlyBaseline:
    """Wraps a RunConfig so audio is absent for every example.

    Implemented as modality forcing rather than a separate model, so the
    controlled comparison holds architecture, parameter count, optimizer and
    data order fixed — only the modality changes.
    """

    def __init__(self, cfg: RunConfig) -> None:
        self.cfg = cfg

    @staticmethod
    def mute_audio(batch: dict[str, Any]) -> dict[str, Any]:
        out = dict(batch)
        out["audio_present"] = batch["audio_present"].clone().fill_(False)
        return out
