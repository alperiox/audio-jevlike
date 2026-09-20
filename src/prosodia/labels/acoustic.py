"""Deterministic acoustic labels derived from the explicit-prosody cache.

These are GOLD labels: computed from the signal by a fixed rule, so they need
no annotation and are available for every cached utterance. They exist to give
the demo question bank questions a transcript fundamentally cannot answer.

Read the caveat before using them as evidence for anything: the prosody
encoder receives F0/RMS/voicing *as its input features*, so for that encoder
these questions are reading the answer off the input. They are only a real
task for encoders that do not explicitly carry F0 (WavLM, Whisper).

Cache layout is (T, 3) at 50 Hz: [F0 Hz, RMS, voiced_prob]. Unvoiced frames
carry F0 = 0.0 (`_explicit_prosody` nan_to_num's them), which is why every
pitch statistic here masks on voicing before touching F0 -- averaging zeros
into a pitch contour is the same class of error as Phase 1's C2, where a
normalisation silently erased the quantity it was meant to preserve.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

FRAMES_PER_SECOND = 50.0
MIN_VOICED_FRAMES = 10          # below this a slope is noise, not a contour
_EPS = 1e-9

# Voicing is decided by F0 > 0 alone -- that IS pyin's own decision, since
# `_explicit_prosody` nan_to_num's unpitched frames to 0.0.
#
# An earlier version ALSO gated on `voiced_prob > 0.5`, which looked like a
# sensible belt-and-braces check and silently destroyed the label. On this
# corpus pyin's voiced_prob averages 0.124 on frames where it assigned a
# pitch, so the extra gate cut mean voiced frames from 73.8 to 5.8 and
# question coverage from 88.1% to 21.0%. Nothing raised; the only symptom was
# a coverage percentage. Do not reintroduce a probability threshold here
# without measuring the distribution of voiced_prob on the actual data first.


def duration_seconds(prosody: Tensor) -> float:
    return prosody.shape[0] / FRAMES_PER_SECOND


def pitch_slope_semitones_per_second(prosody: Tensor) -> float | None:
    """Least-squares slope of log-pitch over VOICED frames.

    Semitones rather than Hz: a 20 Hz rise means something different for a
    80 Hz voice than a 250 Hz one, and the log scale makes the statistic
    comparable across speakers. Returns None when there is too little voiced
    material to fit a line, so the caller can omit the label rather than
    invent one.
    """
    f0 = prosody[:, 0]
    voiced = f0 > 0
    n = int(voiced.sum())
    if n < MIN_VOICED_FRAMES:
        return None

    idx = torch.nonzero(voiced, as_tuple=True)[0].to(torch.float64)
    t = idx / FRAMES_PER_SECOND
    semitones = 12.0 * torch.log2(f0[voiced].to(torch.float64) + _EPS)

    t_c = t - t.mean()
    denom = float((t_c * t_c).sum())
    if denom < _EPS:
        return None
    return float((t_c * (semitones - semitones.mean())).sum() / denom)


def mean_loudness_db(prosody: Tensor) -> float:
    rms = prosody[:, 1].to(torch.float64)
    return float(20.0 * math.log10(float(rms.mean()) + _EPS))


def speaking_rate_wps(prosody: Tensor, transcript: str) -> float | None:
    """Words per second. None when the clip has no words or no duration.

    NOTE: the word count comes from the transcript, so this label is partly
    lexically determined even though the model never sees the transcript.
    That makes it an interesting middle case between the affect questions
    (lexical) and pitch/loudness (not lexical) -- not a flaw, but it should
    be reported as what it is.
    """
    words = len(transcript.split())
    dur = duration_seconds(prosody)
    if words == 0 or dur <= 0.0:
        return None
    return words / dur


@dataclass(frozen=True)
class TertileBinner:
    """Maps a continuous statistic to one of three ordered class names.

    Boundaries MUST be fitted on the training split alone. Fitting them on
    the data they then label is the same in-sample error that made a
    base-rate calibration anchor score a fake 0.0000 in Phase 1: the bins
    come out perfectly balanced by construction and the balance means
    nothing.
    """
    low: float
    high: float
    names: tuple[str, str, str]

    @classmethod
    def fit(cls, values: list[float], names: tuple[str, str, str]) -> "TertileBinner":
        if len(values) < 3:
            raise ValueError(
                f"need at least 3 values to fit tertiles, got {len(values)}"
            )
        ordered = sorted(values)
        n = len(ordered)
        low, high = ordered[n // 3], ordered[2 * n // 3]
        if low == high:
            raise ValueError(
                f"tertile boundaries collapsed to a single value ({low}); the "
                "statistic is near-constant and cannot support a 3-way "
                "question. Refusing to emit a label that is always one class."
            )
        return cls(low=low, high=high, names=names)

    def __call__(self, value: float) -> str:
        if value < self.low:
            return self.names[0]
        if value < self.high:
            return self.names[1]
        return self.names[2]


PITCH_NAMES = ("falling", "level", "rising")
RATE_NAMES = ("slow", "medium", "fast")
LOUDNESS_NAMES = ("quiet", "moderate", "loud")
