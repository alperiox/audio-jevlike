# Phase 2: IEMOCAP, soft targets, and closing the IIA gap

**Date:** 2026-09-23
**Supersedes:** the 2026-09-20 Phase 2 draft, which proposed a bidirectional
rebuild before we had read any reference implementation.
**Phase 1 results:** `docs/results/2026-09-20-phase1-meld-grid.md`

## 1. What changed since the last draft

The earlier draft said the option-interaction fix meant rebuilding around a
bidirectional encoder, and that RLCD was unimplementable because TypeSafe has
not published it. Reading two open implementations changed both.

**Kev** (`jaredpalmer/kev`, `kev/model.py`) puts options in the sequence as
spans and scores them with a pointer head:

    z = (self.k(h_opts) @ self.q(h_decide)) * self.scale

which is the same readout shape as ours. The difference is upstream: its
`<decide>` token **attends over the option spans**, so `h_decide` is a function
of the whole option set. Adding an option shifts `h_decide` and therefore every
pairwise logit difference. IIA is violated through the decide vector, not
through option-to-option attention.

Our branch vector cross-attends to the STATE ONLY. Options first appear at the
readout. That is the entire reason `p(a)/p(b)` is invariant here, and the fix is
one attention step rather than a new architecture.

Kev also gives every option span **the same position ids**, making the
per-option representations and the decide vector's attention over them
permutation-invariant by construction. Archer Hume's probe found option
reordering moved a real Jev classification from ~0.84 to ~0.93, which is a
deployment hazard; this design is immune to it.

**Laya** (`NandhaKishorM/laya`, Apache-2.0) publishes a concrete RLCD recipe in
`notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb`:

    eps  = zero-mean Gaussian noise over the option axis      (sigma 0.4 -> 0.1)
    z    = logits.detach() + eps
    r    = proper_reward(softmax(z), target, ...)             (log + spherical + RPS)
    adv  = (r - r.mean(0)) / r.std()                          (GRPO group-mean baseline)
    loss = -(adv * logp).mean()  +  1.0 * soft_CE(target)

Characterised honestly: it perturbs **logits**, not sampled answers, so it is an
evolution-strategies estimator rather than answer-level RL, and the supervised
soft-cross-entropy term sits at weight **1.0** beside it. Worth implementing;
not worth mythologising. `target` is a distribution throughout, not a one-hot.

`laya/common.py::proper_reward` also contributes the ordinal piece:

    rps = (((cdf_q - cdf_t) ** 2) * mask).sum(-1) / (k - 1)

## 2. Corpus

IEMOCAP replaces MELD as the evidence corpus. MELD stays only as the Phase 1
record.

| | MELD | IEMOCAP |
|---|---|---|
| utterances | 13,708 | 10,039 |
| median duration | 2.46s | 3.52s |
| emotion majority baseline | **48.1%** | **~25%** (4-class subset: 30.9%) |
| splits | dialogue-disjoint, speaker-SHARED | 5 sessions, speaker-disjoint |
| annotator labels | collapsed to majority | **retained per annotator** |

Three properties matter:

1. **Leave-one-session-out is genuinely speaker-disjoint.** Every Phase 1 number
   was pipeline validation because the six *Friends* leads appear in all splits.
2. **A 25% majority baseline instead of 48.1%.** Phase 1's emotion arms all sat
   on top of their baseline; there was barely any room above it.
3. **Per-annotator labels**, categorical (`C-*`) and VAD (`A-*`), 1-5 integers.

**The 25% of clips marked `xxx` are kept, not discarded.** `xxx` means the
annotators did not reach a majority — but every one still carries its individual
annotator labels, so the target distribution exists. Those are the
highest-entropy targets in the corpus and they are the population this project
exists to measure. Standard practice drops them; dropping them here would remove
the evidence.

## 3. Changes, in dependency order

### 3.1 Corpus adapter and splits
`IemocapCorpus` parsing `dialog/EmoEvaluation/*.txt`. Leave-one-session-out.
**Call `assert_speaker_disjoint` on it** — the guard has existed since Phase 1
and has never been allowed to execute because MELD fails it by design. If
IEMOCAP fails it, the split is wrong and nothing downstream is worth running.

### 3.2 Soft targets from the annotator distribution
`target` becomes a distribution over the option set, and the loss becomes a soft
cross-entropy against it (Laya's `loss_ce`). Our Brier term was always aimed at
recovering the true conditional distribution; with one-hot labels it could only
ever do so indirectly by averaging over similar inputs.

### 3.3 RPS for ordinal questions
Valence, arousal and dominance are ordered levels. Plain cross-entropy treats
them as unrelated categories, so predicting "1" when the truth is "5" costs the
same as predicting "4". RPS penalises by distance along the scale. Applies to
`Score` questions **only**; using it on `Choice` would impose an ordering that
does not exist.

### 3.4 Branch attends over options (the IIA fix)
Add one attention step so the branch vector sees the option embeddings before
the readout. Keep the pointer head and its bias-free exact linearity — Phase 3
interpretability depends on it. Optionally adopt Kev's shared option position
ids for permutation invariance.

### 3.5 RLCD (last, and optional)
Laya's recipe is implementable. It goes last because their own weighting says
the supervised term carries at least half the work, and because every earlier
step is a prerequisite for interpreting it.

## 4. Gates

1. **Speaker-disjointness asserted**, not assumed (§3.1).
2. **IIA probe, before and after §3.4.** Current model must measure ~0. Post-fix
   must be non-zero with the right sign and a magnitude comparable to Jev's
   −0.28, across randomised blocks with controls.
3. **Acoustic questions reproduce** at their Phase 1 level on the new corpus,
   as a pipeline sanity check before any affect number is trusted.
4. **Brier and NLL as headline**, each printed beside a train-fitted base-rate
   anchor. ECE only as a diagnostic. See Phase 1 §1.
5. **Three seeds minimum** for any reported comparison.

## 5. Known traps

Phase 1's every defect was silent. Specific to this phase:

1. **The IIA probe passes trivially.** Any non-linearity across options produces
   *some* shift. Require sign, magnitude, and randomised blocks — a small
   non-zero number is not evidence.
2. **`xxx` becomes a class.** It is the absence of consensus, not an emotion.
   It must appear as a high-entropy target over the real classes, never as a
   thirteenth option.
3. **Multi-label annotator lines.** 1,753 lines read `Neutral; Anger;`. A
   first-label-wins parse silently discards half the annotation. Split the mass.
4. **RPS applied to `Choice`.** Imposes a false ordering on unordered
   categories; would look like a small consistent improvement.
5. **Soft targets hide a collapsed model.** A model can match a high-entropy
   target by being uniformly uncertain. Report discrimination (per-class
   precision and recall) alongside distributional fit, never the fit alone.
6. **Mixing MELD and IEMOCAP.** Different recording conditions, label schemes
   and leakage properties. Any result would be unattributable.
7. **VAD binning fitted on the wrong split.** If continuous VAD is binned to
   ordered levels, fit the boundaries on TRAIN only — see the Phase 1
   in-sample-anchor error.
8. **Whisper's 30s window.** IEMOCAP dialogues are long; only per-utterance
   `sentences/wav/` files are in scope, never the dialogue-level `dialog/wav/`.

## 6. Out of scope

A pretrained LLM backbone. Phase 1's evidence for needing one was `sadness`
failing to recover under a 6.9x upweight on MELD — a corpus with a 48.1%
majority class, speaker leakage and a laugh track. That inference should be
re-tested on IEMOCAP before it justifies days of work.
