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


def _dataset(tmp_path, **kw):
    cache = FeatureCache(tmp_path / "c")
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
    for _ in range(200):
        item = ds[0]
        assert item["audio_present"] or item["context_present"]


def test_augmentation_preserves_the_target(tmp_path):
    ds = _dataset(tmp_path, augment=True, modality_dropout=0.0)
    for _ in range(100):
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
