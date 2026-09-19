import pytest
from prosodia.schema import (
    Example, Label, LabelTier, QuestionSpec, assert_speaker_disjoint, assert_thesis_safe,
)


def _ex(uid, tier):
    return Example(
        uid=uid, corpus="meld", audio_path=f"/tmp/{uid}.wav",
        context="SPEAKER: hello", labels={"sentiment": Label(1, tier)},
    )


def _speaker_ex(uid, speaker):
    return Example(
        uid=uid, corpus="iemocap", audio_path=f"/tmp/{uid}.wav",
        context="SPEAKER: hello", labels={"sentiment": Label(1, LabelTier.GOLD)},
        speaker=speaker,
    )


def test_choice_spec_requires_at_least_two_options():
    with pytest.raises(ValueError):
        QuestionSpec("x", "choice", "Pick one", {"only": None})


def test_score_spec_requires_ordered_levels():
    spec = QuestionSpec("sent", "score", "Rate sentiment",
                        ["negative", "neutral", "positive"])
    assert spec.n_options == 3


def test_score_spec_rejects_a_bare_string_as_criteria():
    """A `str` IS a `Sequence` in Python, so `isinstance(criteria, Sequence)`
    alone lets a bare string slip through validation as if it were a list of
    single-character levels -- e.g. criteria="ab" would validate with levels
    ["a", "b"]. That is silent label corruption for any caller who passes a
    string by mistake instead of a list/tuple of level names."""
    with pytest.raises(ValueError):
        QuestionSpec("s", "score", "Rate it", "ab")


def test_noul_spec_has_two_implicit_options():
    assert QuestionSpec("q", "noul", "Is it urgent?", None).n_options == 2


def test_thesis_safe_accepts_gold_and_human():
    assert_thesis_safe([_ex("a", LabelTier.GOLD), _ex("b", LabelTier.HUMAN)],
                       ["sentiment"])


def test_thesis_safe_rejects_model_output():
    with pytest.raises(ValueError, match="MODEL_OUTPUT"):
        assert_thesis_safe([_ex("a", LabelTier.MODEL_OUTPUT)], ["sentiment"])


def test_label_coerces_string_tier():
    """Label should coerce raw string tier values to LabelTier."""
    lab = Label(value=1, tier="gold")
    assert lab.tier == LabelTier.GOLD
    assert isinstance(lab.tier, LabelTier)


def test_label_rejects_invalid_tier():
    """Label should raise ValueError on invalid tier value."""
    with pytest.raises(ValueError, match="tier must be"):
        Label(value=1, tier="invalid_tier")


def test_assert_thesis_safe_rejects_missing_key():
    """assert_thesis_safe should raise if a question_key never appears in examples."""
    with pytest.raises(ValueError, match="unrecognized keys"):
        assert_thesis_safe([_ex("a", LabelTier.GOLD)], ["sentiment", "typo_key"])


def test_assert_speaker_disjoint_passes_when_no_speaker_repeats():
    splits = {
        "train": [_speaker_ex("a", "Ses01"), _speaker_ex("b", "Ses02")],
        "dev": [_speaker_ex("c", "Ses03")],
        "test": [_speaker_ex("d", "Ses04")],
    }
    assert_speaker_disjoint(splits) is None  # must not raise


def test_assert_speaker_disjoint_raises_and_names_the_overlapping_speaker():
    """Fault it exists to catch: MELD-style splits where a speaker (e.g. one
    of the six recurring leads) shows up in more than one split -- the exact
    confound Decision 1 documents rather than silently re-splits away."""
    splits = {
        "train": [_speaker_ex("a", "Joey"), _speaker_ex("b", "Ross")],
        "dev": [_speaker_ex("c", "Joey")],
        "test": [_speaker_ex("d", "Chandler")],
    }
    with pytest.raises(ValueError, match="Joey"):
        assert_speaker_disjoint(splits)


def test_assert_speaker_disjoint_ignores_examples_with_no_speaker():
    """Corpora that never populate `Example.speaker` (speaker=None) must not
    false-positive just because multiple splits share the same None value."""
    splits = {
        "train": [_ex("a", LabelTier.GOLD)],
        "dev": [_ex("b", LabelTier.GOLD)],
    }
    assert_speaker_disjoint(splits) is None  # must not raise
