"""Question bank assembly — the two properties that must not drift."""
import pytest
import torch

from prosodia.features import FeatureCache
from prosodia.labels.acoustic import RATE_NAMES
from prosodia.labels.bank import (
    LEAD_SPEAKERS, attach_demo_labels, demo_question_specs,
)
from prosodia.schema import Example, Label, LabelTier


def _mk(tmp_path, n_per_split=60):
    cache = FeatureCache(tmp_path / "prosody")
    splits, transcripts = {}, {}
    for si, split in enumerate(("train", "dev", "test")):
        exs = []
        for i in range(n_per_split):
            uid = f"{split}-{i}"
            # rising / falling contours alternate; loudness varies with i
            base = 120.0
            step = (1.0 if i % 2 else -1.0) * (0.3 + i * 0.01)
            f0 = torch.tensor([base + step * t for t in range(40)])
            p = torch.stack([f0, torch.full((40,), 0.02 + i * 0.002),
                             torch.ones(40)], dim=-1)
            cache.write(uid, p)
            transcripts[uid] = " ".join(["w"] * (1 + i % 9))
            exs.append(Example(uid, "meld", "/x.wav", "ctx",
                               {"emotion": Label("joy", LabelTier.HUMAN)},
                               speaker=LEAD_SPEAKERS[i % 6] if i % 7 else "Gunther"))
        splits[split] = exs
    return splits, cache, transcripts


def test_specs_are_well_formed():
    keys = {s.key for s in demo_question_specs()}
    assert keys == {"speaker", "pitch_direction", "speaking_rate", "loudness"}


def test_labels_are_gold_tier(tmp_path):
    splits, cache, tr = _mk(tmp_path)
    out = attach_demo_labels(splits, cache, tr)
    for ex in out["train"]:
        for key in ("pitch_direction", "speaking_rate", "loudness"):
            if key in ex.labels:
                assert ex.labels[key].tier is LabelTier.GOLD


def test_only_lead_speakers_get_a_speaker_label(tmp_path):
    splits, cache, tr = _mk(tmp_path)
    out = attach_demo_labels(splits, cache, tr)
    for ex in out["test"]:
        if "speaker" in ex.labels:
            assert ex.labels["speaker"].value in LEAD_SPEAKERS
        else:
            assert ex.speaker not in LEAD_SPEAKERS


def test_score_labels_are_integer_indices(tmp_path):
    """`score` targets are read as int(label.value); a string would crash or,
    worse, coerce to something arbitrary."""
    splits, cache, tr = _mk(tmp_path)
    out = attach_demo_labels(splits, cache, tr)
    for ex in out["train"]:
        if "speaking_rate" in ex.labels:
            v = ex.labels["speaking_rate"].value
            assert isinstance(v, int) and 0 <= v < len(RATE_NAMES)


def test_boundaries_come_from_train_only(tmp_path):
    """Dev/test must be labelled with TRAIN's thresholds. Shifting dev's
    underlying distribution must therefore shift its class balance -- if it
    doesn't, someone refitted per split."""
    splits, cache, tr = _mk(tmp_path)
    for ex in splits["dev"]:                       # make dev uniformly loud
        p = cache.read(ex.uid).clone()
        p[:, 1] = 5.0
        cache.write(ex.uid, p)
    out = attach_demo_labels(splits, cache, tr)
    got = {ex.labels["loudness"].value for ex in out["dev"] if "loudness" in ex.labels}
    assert got == {2}, f"dev was re-binned against itself instead of train: {got}"


def test_missing_train_split_is_refused(tmp_path):
    splits, cache, tr = _mk(tmp_path)
    with pytest.raises(ValueError, match="train"):
        attach_demo_labels({"test": splits["test"]}, cache, tr)


def test_implausible_timing_suppresses_derived_labels(tmp_path):
    """MELD declares 177ms for a three-word utterance in at least one case.
    Such a clip must get NO derived labels -- a words/duration gold of 17 w/s
    is a broken timestamp, and a model can learn to predict 'fast' from clip
    length alone and be scored correct for it."""
    from prosodia.labels.bank import timing_is_plausible

    assert timing_is_plausible(2.5, "four words right here")
    assert not timing_is_plausible(0.177, "No she doesn't")   # 17 w/s
    assert not timing_is_plausible(0.05, "hi")                 # under the floor
    assert not timing_is_plausible(None, "anything")
    assert timing_is_plausible(1.0, "")                        # no words, no claim

    splits, cache, tr = _mk(tmp_path)
    doomed = {ex.uid for ex in splits["test"][:10]}
    declared = {ex.uid: (0.1 if ex.uid in doomed else 3.0)
                for exs in splits.values() for ex in exs}
    out = attach_demo_labels(splits, cache, tr, declared_seconds=declared)

    derived = {"pitch_direction", "speaking_rate", "loudness"}
    for ex in out["test"]:
        if ex.uid in doomed:
            assert not (derived & set(ex.labels)), f"{ex.uid} kept a derived label"
            # the human/metadata labels are unaffected by a bad timestamp
            assert "emotion" in ex.labels


def test_plausibility_filter_is_opt_in(tmp_path):
    """Callers that pass no timings keep the old behaviour, so this cannot
    silently change results for a pipeline that never supplied them."""
    splits, cache, tr = _mk(tmp_path)
    out = attach_demo_labels(splits, cache, tr)
    assert any("loudness" in ex.labels for ex in out["test"])
