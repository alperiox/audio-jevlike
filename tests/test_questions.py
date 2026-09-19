import random

import pytest

from prosodia.corpora.meld import MeldCorpus
from prosodia.questions import (
    holdout_split, paraphrase, permute_candidates,
)
from prosodia.schema import QuestionSpec

CHOICE = QuestionSpec("emotion", "choice", "Which emotion is the speaker expressing?",
                      {"anger": None, "joy": None, "neutral": None, "sadness": None})
SCORE = QuestionSpec("sentiment", "score", "Rate the sentiment the speaker conveys.",
                     ["negative", "neutral", "positive"])


def test_paraphrase_changes_wording_but_not_identity():
    rng = random.Random(0)
    p = paraphrase(CHOICE, rng)
    assert p.key == CHOICE.key
    assert p.qtype == CHOICE.qtype
    assert p.options == CHOICE.options
    assert p.instructions != CHOICE.instructions


def test_permute_candidates_always_keeps_the_gold_option():
    """Dropping the gold option would make the question unanswerable, so the
    example would be silently skipped — quietly biasing the training set
    toward whichever classes survive sampling most often."""
    rng = random.Random(1)
    for _ in range(100):
        spec, mapping = permute_candidates(CHOICE, rng, keep="joy")
        assert "joy" in spec.options
        assert 2 <= spec.n_options <= CHOICE.n_options
        assert set(mapping).issubset(set(CHOICE.options))
        assert len(set(mapping.values())) == len(mapping)


def test_permute_candidates_respects_min_options_with_no_keep():
    """The >= min_options invariant must hold unconditionally, not only when
    `keep` is supplied. The brief's original formula —
    `k = rng.randint(max(min_options - 1, 0), len(pool))` — sizes the lower
    bound assuming `keep` will be appended back in, so with keep=None it can
    sample down to a single option, and QuestionSpec.__post_init__ rejects a
    choice spec with < 2 options. Simulating this against the original
    formula over 2000 seeds fails 519 times (~26%); 500 seeds is plenty to
    catch it reliably."""
    for seed in range(500):
        rng = random.Random(seed)
        spec, mapping = permute_candidates(CHOICE, rng)
        assert spec.n_options >= 2
        assert len(mapping) >= 2


def test_permute_candidates_raises_when_min_options_exceeds_available_options():
    """The docstring promises the result always has at least `min_options`
    options. When that's impossible — `min_options` exceeds the spec's own
    option count — the only honest response is a loud, diagnosable error.
    Silently returning fewer options than promised (the old clamped
    behaviour) would be a contract violation in a module whose entire job is
    provable label-preserving transforms."""
    rng = random.Random(3)
    with pytest.raises(ValueError):
        permute_candidates(CHOICE, rng, keep="joy", min_options=10)


def test_permute_never_applies_to_score_questions():
    # Score levels are ORDERED; subsetting or shuffling them destroys the label.
    rng = random.Random(2)
    spec, mapping = permute_candidates(SCORE, rng, keep="neutral")
    assert spec.criteria == SCORE.criteria
    assert mapping == {o: o for o in SCORE.options}


def test_holdout_split_is_disjoint():
    specs = MeldCorpus(".").question_specs()
    train, held = holdout_split(specs, ["is_negative"])
    assert {s.key for s in train}.isdisjoint({s.key for s in held})
    assert {s.key for s in held} == {"is_negative"}


def test_no_waveform_augmentation_is_exported():
    import prosodia.questions as q
    banned = {"speed_perturb", "pitch_shift", "add_noise", "time_stretch"}
    assert banned.isdisjoint(set(dir(q)))
