import pytest
from prosodia.schema import (
    Example, Label, LabelTier, QuestionSpec, assert_thesis_safe,
)


def _ex(uid, tier):
    return Example(
        uid=uid, corpus="meld", audio_path=f"/tmp/{uid}.wav",
        context="SPEAKER: hello", labels={"sentiment": Label(1, tier)},
    )


def test_choice_spec_requires_at_least_two_options():
    with pytest.raises(ValueError):
        QuestionSpec("x", "choice", "Pick one", {"only": None})


def test_score_spec_requires_ordered_levels():
    spec = QuestionSpec("sent", "score", "Rate sentiment",
                        ["negative", "neutral", "positive"])
    assert spec.n_options == 3


def test_noul_spec_has_two_implicit_options():
    assert QuestionSpec("q", "noul", "Is it urgent?", None).n_options == 2


def test_thesis_safe_accepts_gold_and_human():
    assert_thesis_safe([_ex("a", LabelTier.GOLD), _ex("b", LabelTier.HUMAN)],
                       ["sentiment"])


def test_thesis_safe_rejects_model_output():
    with pytest.raises(ValueError, match="MODEL_OUTPUT"):
        assert_thesis_safe([_ex("a", LabelTier.MODEL_OUTPUT)], ["sentiment"])
