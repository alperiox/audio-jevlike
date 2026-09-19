import torch

from prosodia.data import ProsodiaDataset, collate_batch
from prosodia.features import FeatureCache
from prosodia.schema import Example, Label, LabelTier, QuestionSpec

SPECS = [
    QuestionSpec("emotion", "choice", "Which emotion?",
                 {"anger": None, "joy": None, "neutral": None}),
    QuestionSpec("sentiment", "score", "Rate sentiment.",
                 ["negative", "neutral", "positive"]),
]


def _dataset(tmp_path, cache_dir="c", **kw):
    cache = FeatureCache(tmp_path / cache_dir)
    exs = []
    for i, T in enumerate((20, 35, 12)):
        uid = f"u{i}"
        cache.write(uid, torch.randn(T, 8))
        exs.append(Example(uid, "meld", "/x.wav", f"ctx {i}", {
            "emotion": Label("joy", LabelTier.HUMAN),
            "sentiment": Label(2, LabelTier.HUMAN),
        }, speaker="Joey"))
    return ProsodiaDataset(exs, SPECS, cache, rng_seed=0, **kw)


def test_collate_pads_to_longest_and_masks(tmp_path):
    ds = _dataset(tmp_path, augment=False, modality_dropout=0.0)
    batch = collate_batch([ds[i] for i in range(3)])
    assert batch["audio"].shape[:2] == (3, 35)
    assert batch["audio_mask"][2].sum().item() == 12
    assert batch["audio_mask"][1].sum().item() == 35


def test_modality_dropout_never_drops_both(tmp_path):
    ds = _dataset(tmp_path, augment=False, modality_dropout=0.9)
    for epoch in range(200):
        ds.set_epoch(epoch)
        item = ds[0]
        assert item["audio_present"] or item["context_present"]


def test_augmentation_preserves_the_target(tmp_path):
    ds = _dataset(tmp_path, augment=True, modality_dropout=0.0)
    for epoch in range(100):
        ds.set_epoch(epoch)
        item = ds[0]
        emo = item["questions"]["emotion"]
        # whichever subset/order the options were presented in, the target
        # index must still point at "joy"
        assert emo["options"][item["targets"]["emotion"]] == "joy"


def test_score_options_are_never_reordered(tmp_path):
    ds = _dataset(tmp_path, augment=True, modality_dropout=0.0)
    for _ in range(100):
        assert ds[0]["questions"]["sentiment"]["options"] == \
            ["negative", "neutral", "positive"]


def test_same_seed_epoch_idx_is_reproducible_across_instances(tmp_path):
    """The 9-arm ablation grid compares arms that must differ only in loss
    function or encoder. If two independently-constructed datasets with the
    same rng_seed, epoch, and index drew different augmentation/dropout
    choices, that would inject uncontrolled noise into the comparison."""
    ds1 = _dataset(tmp_path, cache_dir="c1", augment=True, modality_dropout=0.5)
    ds2 = _dataset(tmp_path, cache_dir="c2", augment=True, modality_dropout=0.5)
    ds1.set_epoch(3)
    ds2.set_epoch(3)
    item1, item2 = ds1[0], ds2[0]
    assert item1["questions"] == item2["questions"]
    assert item1["targets"] == item2["targets"]
    assert item1["audio_present"] == item2["audio_present"]
    assert item1["context_present"] == item2["context_present"]


def test_different_epoch_changes_the_draw(tmp_path):
    ds = _dataset(tmp_path, augment=True, modality_dropout=0.5)
    draws = set()
    for epoch in range(20):
        ds.set_epoch(epoch)
        item = ds[0]
        draws.add((
            item["questions"]["emotion"]["instructions"],
            tuple(item["questions"]["emotion"]["options"]),
            item["audio_present"],
            item["context_present"],
        ))
    assert len(draws) > 1
