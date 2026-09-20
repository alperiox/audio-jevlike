"""The demo question bank: seven typed questions over one audio state.

Extends MELD's three human-annotated affect questions with four more that a
transcript cannot answer (or, for speaking rate, cannot answer alone). The
acoustic three carry GOLD labels computed from the signal by a fixed rule, so
they need no annotation and cover every cached utterance.

Two properties that must survive any edit here:

  * Tertile boundaries are fitted on TRAIN ONLY and applied unchanged to dev
    and test. Refitting per split would rebalance the classes by construction
    and make the balance meaningless -- the same in-sample error that made a
    calibration anchor score a fake 0.0000 in Phase 1.
  * Utterances whose statistic cannot be computed get NO label for that
    question rather than a guessed one. `ProsodiaDataset` already skips
    questions with a missing label, so an absent label is safe and a
    fabricated one is not.
"""
from __future__ import annotations

import dataclasses
from typing import Iterable, Sequence

from prosodia.features import FeatureCache
from prosodia.labels.acoustic import (
    LOUDNESS_NAMES, PITCH_NAMES, RATE_NAMES, TertileBinner, mean_loudness_db,
    pitch_slope_semitones_per_second, speaking_rate_wps,
)
from prosodia.schema import Example, Label, LabelTier, QuestionSpec

# The six recurring leads cover 83.3% of MELD; #7 drops to 89 utterances, so
# a wider set would be a long tail of classes with no test support.
LEAD_SPEAKERS = ("Chandler", "Joey", "Monica", "Phoebe", "Rachel", "Ross")

# MELD's own StartTime/EndTime are wrong for a minority of utterances, and
# the errors are not small: test clip meld-test-173-2 declares a 177 ms span
# for "No, she doesn't", i.e. 17 words/second. 2.45% of the test split is
# physically impossible speech by its own metadata (max observed: 166 w/s).
#
# This matters most for `speaking_rate`, whose gold label IS words divided by
# duration -- a broken timestamp manufactures a "fast" label that a model can
# then learn to predict from clip length alone, and be scored correct for it.
# Pitch and loudness computed over a 177 ms window are equally meaningless.
#
# So a clip whose declared timing is impossible gets NO derived labels at all.
MAX_WORDS_PER_SECOND = 10.0     # sustained human speech tops out well below this
MIN_PLAUSIBLE_SECONDS = 0.30


def timing_is_plausible(declared_seconds: float | None, transcript: str) -> bool:
    """False when MELD's own span cannot contain the words it claims."""
    if declared_seconds is None or declared_seconds < MIN_PLAUSIBLE_SECONDS:
        return False
    words = len(transcript.split())
    return words == 0 or (words / declared_seconds) <= MAX_WORDS_PER_SECOND


def demo_question_specs() -> list[QuestionSpec]:
    return [
        QuestionSpec(
            key="speaker", qtype="choice",
            instructions="Which of these characters is speaking?",
            criteria={s: None for s in LEAD_SPEAKERS},
        ),
        QuestionSpec(
            key="pitch_direction", qtype="choice",
            instructions="Over this utterance, is the speaker's pitch rising, falling, or staying level?",
            criteria={n: None for n in PITCH_NAMES},
        ),
        QuestionSpec(
            key="speaking_rate", qtype="score",
            instructions="How quickly is the speaker talking?",
            criteria=list(RATE_NAMES),
        ),
        QuestionSpec(
            key="loudness", qtype="score",
            instructions="How loud is the speaker?",
            criteria=list(LOUDNESS_NAMES),
        ),
    ]


def _raw_stats(ex: Example, cache: FeatureCache, transcripts: dict[str, str]):
    """(pitch slope, speaking rate, loudness) — any element may be None."""
    if ex.uid not in cache:
        return None, None, None
    p = cache.read(ex.uid)
    return (
        pitch_slope_semitones_per_second(p),
        speaking_rate_wps(p, transcripts.get(ex.uid, "")),
        mean_loudness_db(p),
    )


def attach_demo_labels(
    splits: dict[str, list[Example]],
    cache: FeatureCache,
    transcripts: dict[str, str],
    declared_seconds: dict[str, float] | None = None,
) -> dict[str, list[Example]]:
    """Returns new Examples carrying the four extra labels.

    `cache` must be the EXPLICIT-PROSODY cache — F0/RMS/voicing is what the
    statistics read. Passing a WavLM or Whisper cache here would compute
    nonsense from the wrong feature space, silently.
    """
    if "train" not in splits:
        raise ValueError("need the train split to fit tertile boundaries on it")

    def stats_for(ex: Example):
        if declared_seconds is not None and not timing_is_plausible(
                declared_seconds.get(ex.uid), transcripts.get(ex.uid, "")):
            return None, None, None
        return _raw_stats(ex, cache, transcripts)

    stats = {
        name: [stats_for(ex) for ex in exs] for name, exs in splits.items()
    }

    def fit(index: int, names) -> TertileBinner:
        vals = [s[index] for s in stats["train"] if s[index] is not None]
        return TertileBinner.fit(vals, names)

    pitch_bin = fit(0, PITCH_NAMES)
    rate_bin = fit(1, RATE_NAMES)
    loud_bin = fit(2, LOUDNESS_NAMES)

    out: dict[str, list[Example]] = {}
    for name, exs in splits.items():
        enriched = []
        for ex, (slope, rate, loud) in zip(exs, stats[name]):
            extra: dict[str, Label] = {}
            if ex.speaker in LEAD_SPEAKERS:
                extra["speaker"] = Label(ex.speaker, LabelTier.GOLD)
            if slope is not None:
                extra["pitch_direction"] = Label(pitch_bin(slope), LabelTier.GOLD)
            if rate is not None:
                extra["speaking_rate"] = Label(
                    RATE_NAMES.index(rate_bin(rate)), LabelTier.GOLD)
            if loud is not None:
                extra["loudness"] = Label(
                    LOUDNESS_NAMES.index(loud_bin(loud)), LabelTier.GOLD)
            enriched.append(
                dataclasses.replace(ex, labels={**ex.labels, **extra}))
        out[name] = enriched
    return out
