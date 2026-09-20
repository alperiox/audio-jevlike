"""Question-side augmentation.

Spec §4.4: labels are expensive, question phrasings are free. Every transform
here provably preserves the target — rephrasing a question does not change
which emotion was expressed, and permuting option order does not change which
option is correct.

There is deliberately NO waveform augmentation in this codebase. Speed
perturbation and pitch shifting are label-preserving only by assumption, and
they mangle the exact cues the thesis is about (spec §11 trap 6).
"""
from __future__ import annotations

import random
from typing import Sequence

from prosodia.schema import QuestionSpec

PARAPHRASES: dict[str, list[str]] = {
    "emotion": [
        "Which emotion is the speaker expressing?",
        "What emotion comes through in this utterance?",
        "Identify the speaker's emotional state.",
        "How does the speaker feel here?",
        "Which of these emotions best fits what you hear?",
    ],
    "sentiment": [
        "Rate the sentiment the speaker conveys.",
        "How positive or negative is this utterance?",
        "Where does this fall on a negative-to-positive scale?",
        "Judge the overall sentiment expressed.",
        "Rate how favourable the speaker sounds.",
    ],
    "is_negative": [
        "Is the speaker expressing something negative?",
        "Does this utterance carry negative sentiment?",
        "Would you describe the speaker as sounding negative?",
        "Is there negativity in what the speaker says?",
        "Does the speaker come across as unhappy or critical?",
    ],
}


def paraphrase(spec: QuestionSpec, rng: random.Random) -> QuestionSpec:
    """Return the same question with different wording."""
    options = PARAPHRASES.get(spec.key)
    if not options:
        return spec
    alternatives = [o for o in options if o != spec.instructions] or options
    return QuestionSpec(spec.key, spec.qtype, rng.choice(alternatives), spec.criteria)


def permute_candidates(
    spec: QuestionSpec, rng: random.Random, keep: str | None = None,
    min_options: int = 2,
) -> tuple[QuestionSpec, dict[str, str]]:
    """Subsample and shuffle a Choice option set.

    `keep` is the gold option and, when given, is always retained — a
    question whose correct answer is not on the menu is unanswerable, and
    silently skipping those examples would bias the training set toward
    frequently-sampled classes.

    The result always has at least `min_options` options, whether or not
    `keep` is supplied. This must hold unconditionally: computing the lower
    sample bound as if `keep` will always be appended back in (regardless of
    whether it actually is) lets the count fall below `min_options` whenever
    `keep` is None or not among the spec's options — silently producing a
    single-option QuestionSpec that QuestionSpec.__post_init__ rejects.

    Raises `ValueError` if `min_options` exceeds the number of options the
    spec itself has — the guarantee above would otherwise be impossible to
    keep, and returning fewer options than promised without saying so would
    be silent, undiagnosable label corruption in exactly the module whose
    job is to make transforms provably label-preserving.

    Score questions are returned untouched: their levels are ORDERED, so
    subsetting or shuffling would corrupt the target. `min_options` is not
    validated against them for this reason.
    """
    identity = {o: o for o in spec.options}
    if spec.qtype != "choice":
        return spec, identity

    options = list(spec.criteria.keys())
    if min_options > len(options):
        raise ValueError(
            f"min_options={min_options} exceeds the {len(options)} options "
            f"available on spec {spec.key!r}"
        )
    keep_present = keep is not None and keep in options
    pool = [o for o in options if o != keep] if keep_present else list(options)

    # `keep_present` options are added back after sampling, so the pool only
    # needs to supply `min_options - 1` of them; otherwise the pool alone
    # must supply the full `min_options`. The validation above guarantees
    # this lower bound never exceeds len(pool), so no clamp is needed here.
    low = max(min_options - 1, 0) if keep_present else min_options
    k = rng.randint(low, len(pool))
    kept = rng.sample(pool, k)
    if keep_present:
        kept.append(keep)
    rng.shuffle(kept)
    return (
        QuestionSpec(spec.key, "choice", spec.instructions,
                     {o: spec.criteria[o] for o in kept}),
        {o: o for o in kept},
    )


def holdout_split(
    specs: Sequence[QuestionSpec], held_out_keys: Sequence[str]
) -> tuple[list[QuestionSpec], list[QuestionSpec]]:
    """Split the bank into seen and never-seen questions (zero-shot eval)."""
    held = set(held_out_keys)
    return ([s for s in specs if s.key not in held],
            [s for s in specs if s.key in held])
