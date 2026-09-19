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


def _partially_cached_examples(n, cached_fraction, tmp_path, cache_dir="cov"):
    """`n` examples, only the first `round(n * cached_fraction)` of which
    actually have a feature tensor written to `cache` -- i.e. a cache that
    is missing coverage for the rest, exactly the "wrong or
    partially-extracted cache path" I2 is about. `FeatureCache.__init__`'s
    `mkdir(parents=True, exist_ok=True)` means such a path is CREATED
    successfully rather than rejected, so this fixture is realistic, not
    contrived."""
    cache = FeatureCache(tmp_path / cache_dir)
    n_cached = round(n * cached_fraction)
    exs = []
    for i in range(n):
        uid = f"u{i}"
        if i < n_cached:
            cache.write(uid, torch.randn(20, 8))
        exs.append(Example(uid, "meld", "/x.wav", f"ctx {i}", {
            "emotion": Label("joy", LabelTier.HUMAN),
            "sentiment": Label(2, LabelTier.HUMAN),
        }, speaker="Joey"))
    return exs, cache


def test_dataset_raises_when_cache_coverage_falls_below_the_floor(tmp_path):
    """I2: `FeatureCache.__init__` creates (rather than rejects) a wrong or
    partially-extracted cache path, so nothing upstream of `ProsodiaDataset`
    stops a badly incomplete cache from silently shrinking one arm's
    training set relative to the others in the grid -- the comparison goes
    void with no warning anywhere, per the final review.

    Fault this catches: the pre-fix line this replaces --
    `self.examples = [e for e in examples if e.uid in cache]`, no floor at
    all -- which would have returned a 50-example dataset here with zero
    complaint. Confirmed by fault injection (see the task report): reverting
    to that line makes this test fail with `Failed: DID NOT RAISE
    ValueError`.
    """
    exs, cache = _partially_cached_examples(100, cached_fraction=0.5, tmp_path=tmp_path)
    try:
        ProsodiaDataset(exs, SPECS, cache, rng_seed=0)
        assert False, "expected a ValueError for a 50%-covered cache"
    except ValueError as e:
        msg = str(e)
        assert "100" in msg, msg
        assert "50" in msg, msg


def test_dataset_warns_but_proceeds_for_a_small_drop(tmp_path, capsys):
    """A drop small enough to stay above the default coverage floor (99%
    covered, one example missing out of 100) must NOT raise -- MeldCorpus's
    own loader already filters its "handful of undecodable clips" before an
    example ever reaches `ProsodiaDataset` (see `corpora/meld.py`), so a
    small residual drop here is expected, not a bug. But it must never be
    SILENT: the pre-fix code dropped examples with zero trace anywhere,
    which is exactly what let a partial cache pass for a full one. Fault
    this catches: a warning that never fires (or fires unconditionally --
    see the companion silence test below) -- confirmed by fault injection
    in the task report.
    """
    exs, cache = _partially_cached_examples(100, cached_fraction=0.99, tmp_path=tmp_path)
    ds = ProsodiaDataset(exs, SPECS, cache, rng_seed=0)
    assert len(ds) == 99
    assert ds.n_dropped == 1
    assert ds.dropped_uids == ["u99"]
    captured = capsys.readouterr()
    assert "FEATURE-CACHE COVERAGE WARNING" in captured.err
    assert "u99" in captured.err


def test_dataset_is_silent_when_cache_coverage_is_complete(tmp_path, capsys):
    """Companion to the warning test above: the warning must be conditional
    on an actual drop, not print unconditionally -- which would bury the
    one case that matters (a real drop) in noise printed for every healthy
    run."""
    exs, cache = _partially_cached_examples(20, cached_fraction=1.0, tmp_path=tmp_path)
    ds = ProsodiaDataset(exs, SPECS, cache, rng_seed=0)
    assert len(ds) == 20
    assert ds.n_dropped == 0
    captured = capsys.readouterr()
    assert captured.err == ""


def test_min_cache_coverage_is_overridable(tmp_path):
    """I2 explicitly asks for the floor to be overridable, not hardcoded --
    a caller with a legitimately smaller or noisier corpus must be able to
    relax it rather than being permanently blocked by the library
    default."""
    exs, cache = _partially_cached_examples(100, cached_fraction=0.5, tmp_path=tmp_path)
    ds = ProsodiaDataset(exs, SPECS, cache, rng_seed=0, min_cache_coverage=0.0)
    assert len(ds) == 50
