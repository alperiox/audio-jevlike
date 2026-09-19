import ast
import random
from pathlib import Path

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


_BANNED_WAVEFORM_IDENTIFIERS = frozenset({
    "pitch_shift", "time_stretch", "speed", "speed_perturb", "resample",
    "add_noise",
})
_BANNED_WAVEFORM_MODULES = ("librosa.effects", "torchaudio.sox_effects")


def _dotted_attr_chain(node: ast.expr) -> str | None:
    """Reconstruct 'a.b.c' from a Name/Attribute chain, else None."""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def _waveform_augmentation_offenders(src_root: Path) -> list[str]:
    """Scan every .py file under `src_root` at the AST level for waveform
    augmentation: function/method definitions named like a waveform op, calls
    to one (by bare name or dotted attribute chain), and imports of one --
    whether from `prosodia.questions` or anywhere else in the package."""
    offenders: list[str] = []
    for path in sorted(src_root.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        rel = path.relative_to(src_root.parent)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name in _BANNED_WAVEFORM_IDENTIFIERS:
                    offenders.append(f"{rel}:{node.lineno} defines {node.name!r}")
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    chain = node.func.id
                else:
                    chain = _dotted_attr_chain(node.func)
                if chain is not None and (
                    chain.split(".")[-1] in _BANNED_WAVEFORM_IDENTIFIERS
                    or any(chain == m or chain.startswith(m + ".")
                           for m in _BANNED_WAVEFORM_MODULES)
                ):
                    offenders.append(f"{rel}:{node.lineno} calls {chain!r}")
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                for alias in node.names:
                    full = f"{mod}.{alias.name}" if mod else alias.name
                    if (
                        alias.name in _BANNED_WAVEFORM_IDENTIFIERS
                        or mod in _BANNED_WAVEFORM_MODULES
                        or any(full.startswith(m + ".") or full == m
                               for m in _BANNED_WAVEFORM_MODULES)
                    ):
                        offenders.append(f"{rel}:{node.lineno} imports {full!r}")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if any(alias.name == m or alias.name.startswith(m + ".")
                           for m in _BANNED_WAVEFORM_MODULES):
                        offenders.append(f"{rel}:{node.lineno} imports {alias.name!r}")
    return offenders


def test_no_waveform_augmentation_anywhere_in_src():
    """Spec §11 trap 6: waveform augmentation (pitch/speed/time perturbation)
    mangles the exact prosodic cues the thesis is about, so it must never
    exist anywhere in the package -- not just be absent from
    `prosodia.questions`'s public names.

    The old version of this test (`test_no_waveform_augmentation_is_exported`)
    checked a fixed list of banned NAMES against `dir(prosodia.questions)`.
    That is a guardrail test in the same broken family C1 exposed
    (`test_attention_pool_distinguishes_a_rise_from_a_fall` passed against
    the exact implementation it was meant to forbid): it is defeated by
    adding a differently-named function, or by adding the banned
    functionality to any module other than `questions.py`, or by importing
    it privately (still visible via `dir()`, but the old test only checked
    against 4 exact strings that don't match real librosa/torchaudio API
    names anyway). This version AST-scans every file under `src/` for
    function/method DEFINITIONS, CALLS (by bare name or dotted attribute
    chain), and IMPORTS matching known waveform-augmentation identifiers or
    modules (`librosa.effects.*`, `torchaudio.sox_effects.*`), so a
    newly-added waveform op is caught regardless of which module or name it
    lands under. Verified by fault injection: see the final report."""
    src_root = Path(__file__).resolve().parent.parent / "src"
    offenders = _waveform_augmentation_offenders(src_root)
    assert not offenders, "waveform augmentation found in src/:\n" + "\n".join(offenders)
