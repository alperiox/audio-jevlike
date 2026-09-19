import torch

from prosodia.device import assert_close_across_devices, get_device
from prosodia.evaluation.metrics import (
    coverage_curve, expected_calibration_error, brier_score, macro_f1,
)
from prosodia.train.calibrate import TemperatureScaler


def test_ece_is_zero_for_a_perfectly_calibrated_set():
    """80 predictions at p=0.8, exactly 80% correct -> ECE 0."""
    probs = torch.full((100, 2), 0.2)
    probs[:, 1] = 0.8
    targets = torch.zeros(100, dtype=torch.long)
    targets[:80] = 1  # the p=0.8 class is right exactly 80% of the time
    assert expected_calibration_error(probs, targets, n_bins=10).item() < 1e-6


def test_ece_matches_a_hand_computed_case():
    # all mass in one bin: stated confidence 0.9, observed accuracy 0.5
    probs = torch.tensor([[0.1, 0.9]] * 10)
    targets = torch.tensor([1] * 5 + [0] * 5)
    torch.testing.assert_close(
        expected_calibration_error(probs, targets, n_bins=10),
        torch.tensor(0.4), rtol=1e-5, atol=1e-5,
    )


def test_ece_does_not_let_over_and_under_confidence_cancel():
    """Two bins with equal-magnitude, opposite-sign (conf - accuracy) errors.

    Without abs() per bin, the signed errors would net to ~0; ECE must sum
    the *magnitudes*, so it should land near 0.1, not near 0.
    """
    # bin (0.8, 0.9]: confidence 0.9, all 10 correct -> conf - acc = -0.1 (underconfident)
    probs_under = torch.tensor([[0.1, 0.9]] * 10)
    targets_under = torch.tensor([1] * 10)
    # bin (0.5, 0.6]: confidence 0.6, 5/10 correct -> conf - acc = +0.1 (overconfident)
    probs_over = torch.tensor([[0.4, 0.6]] * 10)
    targets_over = torch.tensor([1] * 5 + [0] * 5)

    probs = torch.cat([probs_under, probs_over])
    targets = torch.cat([targets_under, targets_over])
    torch.testing.assert_close(
        expected_calibration_error(probs, targets, n_bins=10),
        torch.tensor(0.1), rtol=1e-5, atol=1e-5,
    )


def test_brier_score_bounds():
    probs = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    assert brier_score(probs, torch.tensor([0, 1])).item() < 1e-6
    assert brier_score(probs, torch.tensor([1, 0])).item() > 1.9


def test_macro_f1_is_zero_not_nan_for_an_absent_class():
    """Class 2 appears in neither predictions nor targets -> its F1 is 0/0.

    Constructed directly on get_device() (this box's default is MPS) rather
    than CPU: macro_f1's 0/0 guard used a CPU-only torch.tensor(0.0) that
    torch.stack could not combine with the accelerator-resident per-class
    scores, raising instead of returning a value. A CPU-only test cannot
    distinguish that fixed state from the broken one.
    """
    device = get_device()
    # 3-class problem; only classes 0 and 1 ever appear, both predicted perfectly.
    probs = torch.tensor(
        [[0.9, 0.1, 0.0], [0.1, 0.9, 0.0], [0.9, 0.1, 0.0], [0.1, 0.9, 0.0]],
        device=device,
    )
    targets = torch.tensor([0, 1, 0, 1], device=device)
    f1 = macro_f1(probs, targets)
    assert not torch.isnan(f1)
    torch.testing.assert_close(f1, torch.tensor(2.0 / 3.0, device=device), rtol=1e-4, atol=1e-4)


def test_coverage_curve_is_monotone_in_coverage():
    """Constructed on get_device() (this box's default is MPS), not CPU.

    coverage_curve's thresholds/coverage/error tensors were built without a
    shared device=, so on an accelerator it silently returned a mixed-device
    tuple: no crash on its own, only on the first attempt to combine the
    three (e.g. torch.stack, or plotting code that assumes one device). A
    CPU-only test can't see that, since every tensor defaults to CPU there.
    """
    device = get_device()
    torch.manual_seed(0)
    logits = torch.randn(500, 4, device=device)
    probs = torch.softmax(logits, -1)
    targets = torch.randint(0, 4, (500,), device=device)
    thresholds, coverage, error = coverage_curve(probs, targets)
    assert torch.all(coverage[1:] <= coverage[:-1] + 1e-6)  # higher t -> less coverage
    assert thresholds.shape == coverage.shape
    # the three returned tensors must live on one device -- a caller that
    # stacks or concatenates them, or a reliability-diagram plot that moves
    # one to numpy and not the others, would otherwise silently drop data.
    assert thresholds.device == coverage.device == error.device


def test_temperature_scaling_reduces_ece_on_overconfident_logits():
    """Constructed on get_device() (this box's default is MPS), not CPU.

    TemperatureScaler.fit's log_t was built without device=logits.device, so
    on an accelerator the LBFGS closure mixed a CPU log_t with accelerator
    logits/targets and crashed outright. A CPU-only test can't see that,
    since log_t's default CPU placement matches everything else there.
    """
    device = get_device()
    torch.manual_seed(0)
    targets = torch.randint(0, 3, (600,), device=device)
    logits = torch.randn(600, 3, device=device)
    logits[torch.arange(600, device=device), targets] += 1.0
    logits = logits * 4.0  # deliberately overconfident

    before = expected_calibration_error(torch.softmax(logits, -1), targets)
    scaler = TemperatureScaler().fit(logits, targets)
    after = expected_calibration_error(torch.softmax(scaler.transform(logits), -1), targets)
    assert after < before
    assert scaler.temperature.item() > 1.0  # softening, as expected


def test_metrics_agree_across_devices():
    """Spec §10: an MPS numerical artifact and a finding look identical in a plot."""
    torch.manual_seed(0)
    probs = torch.softmax(torch.randn(256, 5), -1)
    targets = torch.randint(0, 5, (256,))
    for fn in (brier_score, expected_calibration_error):
        assert_close_across_devices(fn, probs, targets)
