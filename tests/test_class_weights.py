"""Arm D (inverse-frequency class weighting) — the two ways it can go wrong silently.

1. Weighting by SLOT instead of gold class. `permute_candidates` subsamples
   and shuffles Choice options during training, so a slot-indexed weight
   attaches to whichever class happened to land there. The result still
   trains and still produces plausible numbers.
2. Normalising a weighted loss sum by an unweighted term count, which makes
   the effective step size depend on the batch's class mix -- so Arm D's
   `lr` silently stops meaning what every other arm's `lr` means.
"""
import pytest
import torch
from torch.utils.data import DataLoader

from prosodia.config import RunConfig
from prosodia.data import ProsodiaDataset, collate_batch
from prosodia.features import FeatureCache
from prosodia.model.prosodia import ProsodiaModel
from prosodia.schema import Example, Label, LabelTier, QuestionSpec
from prosodia.train.losses import class_weights_from_counts
from prosodia.train.loop import train_one_epoch

SPECS = [QuestionSpec("emotion", "choice", "Which emotion?",
                      {"anger": None, "joy": None, "neutral": None})]


def _loader(tmp_path, labels, augment=False):
    cache = FeatureCache(tmp_path / "c")
    exs = []
    for i, lab in enumerate(labels):
        cache.write(f"u{i}", torch.randn(16, 8))
        exs.append(Example(f"u{i}", "meld", "/x.wav", "ctx",
                           {"emotion": Label(lab, LabelTier.HUMAN)}, speaker="Joey"))
    ds = ProsodiaDataset(exs, SPECS, cache, augment=augment, modality_dropout=0.0)
    return DataLoader(ds, batch_size=4, collate_fn=collate_batch)


def test_weights_are_normalised_to_mean_one():
    w = class_weights_from_counts({"a": 900, "b": 90, "c": 10})
    assert sum(w.values()) / len(w) == pytest.approx(1.0)
    assert w["c"] > w["b"] > w["a"]


def test_a_class_with_no_training_support_raises_rather_than_being_weighted():
    with pytest.raises(ValueError, match="zero training support"):
        class_weights_from_counts({"a": 100, "b": 0})


def test_batch_carries_gold_class_identity_not_just_the_slot(tmp_path):
    batch = next(iter(_loader(tmp_path, ["joy"] * 4)))
    assert "gold" in batch, "collate dropped the gold-class field"
    assert batch["gold"][0]["emotion"] == "joy"


def test_gold_class_survives_option_permutation(tmp_path):
    """The whole point: with augmentation on, the slot moves but the class does not."""
    torch.manual_seed(0)
    loader = _loader(tmp_path, ["joy"] * 8, augment=True)
    seen_slots, seen_classes = set(), set()
    for epoch in range(6):
        loader.dataset.set_epoch(epoch)
        for batch in loader:
            for tgt, gold in zip(batch["targets"], batch["gold"]):
                seen_slots.add(tgt["emotion"])
                seen_classes.add(gold["emotion"])
    assert seen_classes == {"joy"}, f"gold class drifted: {seen_classes}"
    assert len(seen_slots) > 1, (
        "options never permuted, so this test cannot detect slot-vs-class "
        "confusion; strengthen the fixture rather than trusting it"
    )


def test_none_weights_reproduce_the_unweighted_loss_exactly(tmp_path):
    """Every pre-Arm-D arm must be bit-identical under the new code path."""
    cfg = RunConfig(name="t", brier_weight=0.0, encoder="wavlm")
    losses = []
    for weights in (None, {"emotion": {"joy": 1.0, "anger": 1.0, "neutral": 1.0}}):
        torch.manual_seed(0)
        model = ProsodiaModel(in_dim=8, d_model=32)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        losses.append(train_one_epoch(model, _loader(tmp_path, ["joy"] * 8), opt,
                                      cfg, epoch=0, class_weights=weights)["loss"])
    assert losses[0] == pytest.approx(losses[1], abs=1e-12)


def test_weighting_a_rare_class_up_changes_the_loss(tmp_path):
    """Guards against the weight being accepted and then ignored."""
    cfg = RunConfig(name="t", brier_weight=0.0, encoder="wavlm")
    out = []
    for w in (1.0, 5.0):
        torch.manual_seed(0)
        model = ProsodiaModel(in_dim=8, d_model=32)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        loader = _loader(tmp_path, ["joy", "anger"] * 4)
        out.append(train_one_epoch(
            model, loader, opt, cfg, epoch=0,
            class_weights={"emotion": {"joy": 1.0, "anger": w, "neutral": 1.0}})["loss"])
    assert out[0] != pytest.approx(out[1]), "class weight had no effect on the loss"
