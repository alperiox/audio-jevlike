import pytest
import torch

from prosodia.train.losses import brier_loss, composite_loss, cross_entropy_loss


def test_brier_is_zero_for_a_perfect_confident_prediction():
    logits = torch.tensor([[50.0, -50.0, -50.0]])
    assert brier_loss(logits, torch.tensor([0])).item() < 1e-6


def test_brier_is_maximal_for_a_confident_wrong_prediction():
    logits = torch.tensor([[50.0, -50.0, -50.0]])
    assert brier_loss(logits, torch.tensor([1])).item() > 1.9  # -> 2.0


def test_brier_penalises_overconfidence_more_than_ce_at_the_margin():
    """A proper scoring rule is what makes probabilities honest (spec §6)."""
    target = torch.tensor([0])
    mild = torch.tensor([[1.0, 0.0, 0.0]])
    overconfident = torch.tensor([[20.0, 0.0, 0.0]])
    # both are correct; CE rewards the confident one far more than Brier does
    ce_gain = cross_entropy_loss(mild, target) - cross_entropy_loss(overconfident, target)
    br_gain = brier_loss(mild, target) - brier_loss(overconfident, target)
    assert ce_gain > br_gain


def test_composite_with_zero_weight_equals_cross_entropy():
    logits, target = torch.randn(4, 5), torch.randint(0, 5, (4,))
    torch.testing.assert_close(composite_loss(logits, target, brier_weight=0.0),
                               cross_entropy_loss(logits, target))


def test_composite_is_between_its_components():
    logits, target = torch.randn(8, 4), torch.randint(0, 4, (8,))
    ce = cross_entropy_loss(logits, target)
    br = brier_loss(logits, target)
    mix = composite_loss(logits, target, brier_weight=0.5)
    assert min(ce, br) <= mix <= max(ce, br)


def test_composite_raises_on_negative_brier_weight():
    """A negative brier_weight is a config typo, not a valid Arm A request.
    The old `<= 0.0` check silently routed it to pure cross-entropy -- same
    defect family as the headline Arm C bug: a config field quietly not
    meaning what its name says. Only exactly 0.0 should mean 'Arm A'."""
    logits, target = torch.randn(4, 3), torch.randint(0, 3, (4,))
    with pytest.raises(ValueError, match="brier_weight"):
        composite_loss(logits, target, brier_weight=-0.1)
