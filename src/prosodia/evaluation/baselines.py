"""Two baselines doing different jobs (spec §7).

  TextOnlyBaseline — CONTROLLED. Identical architecture and training, audio
                     permanently absent. Isolates modality and nothing else.
  Jev via API      — PRACTICAL. The actual text-state System One model.

`ARMS` is the 9-arm ablation grid (3 loss regimes x 3 encoders) plus 3
`TextOnlyBaseline` arms, one per loss regime -- 12 entries total. See the
`_TEXT_ONLY_ARMS` comment below for why the text-only baseline is folded
into `ARMS` at all three loss regimes rather than once or per-encoder.

`companion_arm_a_name` (I9) names the Arm A run each Arm C config must be
DERIVED from -- see `scripts/run_ablation.py`'s module docstring for why
Arm C is no longer an independent training run.
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


def _loss_tag(brier: float, temp: bool) -> str:
    return "C-temp" if temp else ("B-brier" if brier > 0 else "A-ce")


def _arm_name(encoder: str, brier: float, temp: bool) -> str:
    return f"{encoder}__{_loss_tag(brier, temp)}"


# TEXT_ONLY_CACHE_ENCODER: which encoder's feature cache the text-only arms
# read from. Audio is muted for every example in these arms (see
# TextOnlyBaseline.mute_audio below), so the specific cache chosen is
# arbitrary -- it exists only to give the (discarded) audio tensor a
# concrete shape/in_dim. "wavlm" is picked for no reason beyond it being the
# primary encoder, i.e. whichever cache is guaranteed to exist.
TEXT_ONLY_CACHE_ENCODER = "wavlm"
TEXT_ONLY_ARM_PREFIX = "text_only"

LOSS_REGIME_GRID: tuple[tuple[float, bool], ...] = ((0.0, False), (0.5, False), (0.0, True))

# The 9-arm ablation grid: 3 loss regimes x 3 encoders.
_ENCODER_ARMS: list[RunConfig] = [
    RunConfig(name=_arm_name(enc, brier, temp), encoder=enc,
              brier_weight=brier, temperature_scale=temp)
    for enc in ("wavlm", "whisper", "prosody")
    for brier, temp in LOSS_REGIME_GRID
]

# The controlled text-only baseline: one arm PER LOSS REGIME, not per
# encoder. The encoder axis is meaningless here -- audio_present is forced
# False for every example (TextOnlyBaseline.mute_audio), so WavLM vs
# Whisper vs the explicit-prosody channel produce bit-identical runs and
# three such arms would just be the same experiment logged three times.
# The loss axis is NOT meaningless: Arm B (Brier) and Arm C (post-hoc
# temperature scaling) are calibration treatments that can behave
# differently on a text-only model than on an audio-bearing one, and
# success criterion #1 ("does audio improve calibration over a controlled
# text-only baseline") needs a same-loss-regime baseline to diff each
# encoder arm against -- comparing wavlm__B-brier only to a CE-only
# text-only arm would conflate "audio helped" with "Brier training helped".
_TEXT_ONLY_ARMS: list[RunConfig] = [
    RunConfig(name=f"{TEXT_ONLY_ARM_PREFIX}__{_loss_tag(brier, temp)}",
              encoder=TEXT_ONLY_CACHE_ENCODER, brier_weight=brier,
              temperature_scale=temp, text_only=True)
    for brier, temp in LOSS_REGIME_GRID
]

ARMS: list[RunConfig] = _ENCODER_ARMS + _TEXT_ONLY_ARMS


def companion_arm_a_name(cfg: RunConfig) -> str:
    """The name of the Arm A run that `cfg` (an Arm C config) must be
    DERIVED from (I9): same encoder-or-text_only axis, loss regime reset to
    Arm A's (brier_weight=0.0, temperature_scale=False).

    Built structurally from `cfg.encoder`/`cfg.text_only` via the same
    `_arm_name`/`_loss_tag` machinery `ARMS` itself uses -- never by
    string-editing `cfg.name` -- so a text-only Arm C is structurally
    incapable of resolving to an encoder arm's name (or vice versa): the
    `text_only` branch always routes through `TEXT_ONLY_ARM_PREFIX`, which
    shares no name with any encoder.

    Raises ValueError for a config that is not actually an Arm C config
    (`temperature_scale=False`) -- calling this for anything else is a
    caller bug, not a data problem, and should fail at the call site
    rather than quietly resolving to a nonsense companion.
    """
    if not cfg.temperature_scale:
        raise ValueError(
            f"companion_arm_a_name({cfg.name!r}): only meaningful for an Arm C "
            "config (temperature_scale=True); this config has "
            "temperature_scale=False, so it has no Arm A to derive from -- "
            "it MAY BE Arm A itself."
        )
    prefix = TEXT_ONLY_ARM_PREFIX if cfg.text_only else cfg.encoder
    return f"{prefix}__{_loss_tag(0.0, False)}"


def assert_derived_arms_follow_their_source(arms: Sequence[RunConfig]) -> None:
    """F3 (owner-flagged in I9's own review): every derived Arm C's
    companion Arm A must appear BEFORE it in `arms`. `run_arm` trains arms
    in list order and `run_derived_arm_c` loads its companion's checkpoint
    from disk (`_find_latest_checkpoint`), so an Arm C listed before its
    Arm A would hit that checkpoint missing -- `_find_latest_checkpoint`
    already fails loudly for that case, but only at RUN time, and only if
    someone actually runs the misordered grid. This is a cheap STATIC
    check -- a single pass building a name -> index map -- that catches a
    misordered `ARMS` at IMPORT time, before any training happens at all.
    It guards a different failure mode than the runtime `FileNotFoundError`:
    that one guards a missing/deleted checkpoint; this one guards the
    ordering `ARMS` itself must maintain by construction.

    Also exercises `companion_arm_a_name`'s text_only branch identically to
    its encoder branch: a text-only Arm C's companion is looked up the same
    structural way, so this check would equally catch a text-only Arm C
    misordered relative to its text-only Arm A.
    """
    index = {cfg.name: i for i, cfg in enumerate(arms)}
    for i, cfg in enumerate(arms):
        if not cfg.temperature_scale:
            continue
        source_name = companion_arm_a_name(cfg)
        if source_name not in index:
            raise ValueError(
                f"Arm {cfg.name!r} (index {i}) derives from {source_name!r}, "
                "which does not appear in this arm list at all."
            )
        source_index = index[source_name]
        if source_index >= i:
            raise ValueError(
                f"Arm {cfg.name!r} (index {i}) derives from its companion "
                f"Arm A {source_name!r}, but that arm appears at index "
                f"{source_index} -- at or after the derived arm. ARMS must "
                "list every Arm A before its derived Arm C so a full grid "
                "run trains the source before the derivation ever needs "
                "its checkpoint."
            )


# I9/F3: verified once at import time, over the real production ARMS list,
# so a future reordering of `_ENCODER_ARMS`/`_TEXT_ONLY_ARMS` (or however
# ARMS is assembled) fails immediately and loudly rather than only showing
# up as a `FileNotFoundError` the next time someone actually runs the grid.
assert_derived_arms_follow_their_source(ARMS)


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
