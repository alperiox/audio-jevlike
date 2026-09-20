"""Acoustic gold labels — the ways they can be silently wrong.

The headline trap is unvoiced frames. `_explicit_prosody` stores F0 = 0.0
where pyin found no pitch, so any statistic that does not mask on voicing is
averaging silence into a pitch contour and will report a confident number
for a clip with no pitch in it at all.
"""
import math

import pytest
import torch

from prosodia.labels.acoustic import (
    LOUDNESS_NAMES, PITCH_NAMES, TertileBinner, duration_seconds,
    mean_loudness_db, pitch_slope_semitones_per_second, speaking_rate_wps,
)


def _clip(f0, rms=0.1, voiced=1.0):
    n = len(f0)
    return torch.stack([
        torch.tensor(f0, dtype=torch.float32),
        torch.full((n,), rms),
        torch.full((n,), voiced),
    ], dim=-1)


def test_rising_pitch_gives_positive_slope():
    # 100 Hz -> 200 Hz over 1s is +12 semitones/second
    n = 50
    f0 = [100.0 * (2 ** (i / n)) for i in range(n)]
    slope = pitch_slope_semitones_per_second(_clip(f0))
    assert slope == pytest.approx(12.0, abs=0.5)


def test_falling_pitch_gives_negative_slope():
    n = 50
    f0 = [200.0 / (2 ** (i / n)) for i in range(n)]
    assert pitch_slope_semitones_per_second(_clip(f0)) < -10.0


def test_unvoiced_frames_do_not_contribute_to_pitch():
    """The C2-family trap: F0=0 filler must be masked, not averaged in."""
    n = 50
    rising = [100.0 * (2 ** (i / n)) for i in range(n)]
    clean = _clip(rising)

    # same contour, but with unvoiced (F0=0, low voiced_prob) frames spliced in
    f0_mixed, voiced_mixed = [], []
    for i, v in enumerate(rising):
        f0_mixed += [v, 0.0]
        voiced_mixed += [1.0, 0.0]
    n2 = len(f0_mixed)
    mixed = torch.stack([
        torch.tensor(f0_mixed, dtype=torch.float32),
        torch.full((n2,), 0.1),
        torch.tensor(voiced_mixed, dtype=torch.float32),
    ], dim=-1)

    a = pitch_slope_semitones_per_second(clean)
    b = pitch_slope_semitones_per_second(mixed)
    # frames are half as dense in time in `mixed`, so the slope halves; what
    # must NOT happen is the zeros dragging it toward some arbitrary value
    assert b is not None and b > 0, "unvoiced zeros flipped or killed the slope"
    assert b == pytest.approx(a / 2, rel=0.15)


def test_voicing_mask_does_not_over_restrict_on_low_confidence_pitch():
    """Regression: gating on voiced_prob > 0.5 cut real coverage from 88% to
    21% on MELD, because pyin reports mean voiced_prob 0.124 on frames it did
    pitch. F0 > 0 is the voicing decision; confidence is not a second gate."""
    n = 50
    rising = [100.0 * (2 ** (i / n)) for i in range(n)]
    low_conf = _clip(rising, voiced=0.12)
    slope = pitch_slope_semitones_per_second(low_conf)
    assert slope is not None, "low voiced_prob suppressed an otherwise valid contour"
    assert slope == pytest.approx(12.0, abs=0.5)


def test_too_little_voiced_material_returns_none_rather_than_a_number():
    assert pitch_slope_semitones_per_second(_clip([120.0] * 4)) is None
    silent = _clip([0.0] * 50, voiced=0.0)
    assert pitch_slope_semitones_per_second(silent) is None


def test_flat_pitch_gives_near_zero_slope():
    assert abs(pitch_slope_semitones_per_second(_clip([150.0] * 50))) < 0.5


def test_loudness_is_monotone_in_rms():
    quiet = mean_loudness_db(_clip([150.0] * 20, rms=0.01))
    loud = mean_loudness_db(_clip([150.0] * 20, rms=0.5))
    assert loud > quiet


def test_duration_and_rate():
    clip = _clip([150.0] * 100)          # 100 frames @ 50Hz = 2s
    assert duration_seconds(clip) == pytest.approx(2.0)
    assert speaking_rate_wps(clip, "one two three four") == pytest.approx(2.0)
    assert speaking_rate_wps(clip, "   ") is None


def test_tertile_binner_splits_roughly_evenly():
    b = TertileBinner.fit([float(i) for i in range(300)], PITCH_NAMES)
    counts = {n: 0 for n in PITCH_NAMES}
    for i in range(300):
        counts[b(float(i))] += 1
    assert all(80 <= c <= 120 for c in counts.values()), counts


def test_tertile_binner_refuses_a_degenerate_statistic():
    with pytest.raises(ValueError, match="collapsed"):
        TertileBinner.fit([7.0] * 100, LOUDNESS_NAMES)


def test_tertile_binner_needs_enough_values():
    with pytest.raises(ValueError, match="at least 3"):
        TertileBinner.fit([1.0, 2.0], PITCH_NAMES)


def test_binner_is_a_pure_threshold_so_train_fitted_bounds_transfer():
    """Boundaries are data, not state: applying train bounds to unseen values
    must not rebalance them. Guards against anyone 'fixing' imbalance on
    dev/test by refitting there."""
    b = TertileBinner.fit([float(i) for i in range(300)], PITCH_NAMES)
    shifted = [b(float(i)) for i in range(400, 500)]
    assert set(shifted) == {PITCH_NAMES[2]}, "bounds were not applied as fixed thresholds"
