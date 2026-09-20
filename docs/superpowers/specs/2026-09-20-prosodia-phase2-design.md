# Prosodia Phase 2: option interaction and joint encoding

**Date:** 2026-09-20
**Phase 1 spec:** `docs/superpowers/specs/2026-09-19-prosodia-design.md`
**Phase 1 results:** `docs/results/2026-09-20-phase1-meld-grid.md`

## 1. What Phase 1 actually established

Not what it set out to. Recorded plainly because every Phase 2 decision
rests on it:

- **ECE cannot rank these arms.** A train-fitted constant predictor scores
  ECE 0.0097 / 0.0097 / 0.0243. Ranking the 12 arms by ECE vs by Brier gives
  Spearman +0.147 / -0.294 / -0.266 — near-independent. Report Brier/NLL.
- **The control encoder beat the primary.** Whisper > WavLM on accuracy,
  macro-F1 and Brier across all three questions.
- **The prosody dissociation did not survive its control.** Scored against
  each stratum's own base rate, prosody never beats a constant anywhere.
  Only the *direction* survives (prosody relatively better on ambiguous,
  +0.085 skill; whisper better on clear, -0.078).
- **The model is weak.** Emotion weighted-F1 0.437 vs ~0.57-0.60 for
  published text-only MELD baselines.
- **The tail collapse has two causes.** Arm D: joy (1743 train) was
  optimization-limited and recovered (F1 0.216 -> 0.310 on whisper).
  Sadness (683 train) was not — recall never exceeded 2.9% under a 6.9x
  upweight on any encoder.
- **The architecture was never tested.** Constant across all 12 arms.
- **RLCD was never implemented.** Arm B is a supervised proper-scoring
  stand-in; the recipe is unpublished and no one outside TypeSafe has it.

## 2. The finding this phase is built on

`ReadoutHead.forward` computes

    logits = (W_branch @ h) @ (W_option @ o)^T * scale

where `h` is ONE vector per question and `o` are per-option embeddings from
a **frozen MiniLM-L6-v2**, each encoded independently by
`QuestionEncoder.embed_texts`. `IsolatedBranches` folds each question to a
sequence of length 1, so there is no self-attention within a question
either.

Consequence: each option's logit is an independent inner product.
`p(a)/p(b) = exp(z_a - z_b)` **cannot change** when a third option is added
or removed. Our model satisfies independence of irrelevant alternatives
exactly, by construction.

Jev does not. The IIA probe (append an irrelevant `weather` option to four
causes of payout failure) shifted mean log-odds between two *untouched*
options by **-0.28**, 95% paired interval roughly [-0.36, -0.19], every one
of ten randomised blocks decreasing. Jev's options interact before the
choice; ours cannot.

This is the first concrete, falsifiable architectural gap the project has
identified between what we built and what we are replicating — and it is
independent of scale.

## 3. The question

Phase 1's deficit has two candidate causes:

- **H-arch** — option representations are frozen, isolated, and never
  jointly encoded with the state or with each other. The model cannot
  express "this option, given these alternatives, given this audio".
- **H-scale** — the representation lacks large-scale pretrained knowledge,
  and only a much larger model fixes it.

**Phase 2 tests H-arch first**, because it is cheaper, because it is
supported by §2, and because if H-arch is true then H-scale work would have
been wasted effort.

Note on what H-scale would even mean: whether Jev post-trains a public LLM
or was pretrained bespoke (the tokenizer matches none of 192 public
tokenizers across 415 probes, which is consistent with bespoke), its
breadth requires frontier-scale pretraining that we will not reproduce
under either hypothesis. So the useful question is not "can we rebuild Jev"
but "how far does a well-designed, task-shaped small model get".

## 4. Architecture

Keep: shared state encoded once, structural question isolation, pointer
readout, bias-free exact linearity (Phase 3 interpretability depends on it).

Change: options become first-class token sequences that attend to each
other and to the state.

    audio features (frozen WavLM/Whisper, cached)  ─┐
                                                    ├─> StateEncoder ──> H_state   (computed ONCE per example)
    context text ───────────────────────────────────┘

    per question q:
      [opt_1 tokens] [opt_2 tokens] ... [opt_K tokens] [question tokens]
              │
              ├── self-attention ACROSS options      <- new; this is what makes IIA violable
              ├── cross-attention to H_state         <- existing mechanism, now carrying real tokens
              │
              └── pointer readout: score option i from ITS OWN final representation

- **Option/question encoder:** a trainable mid-size bidirectional text
  encoder (~180-300M) replacing frozen MiniLM. Full finetune; fits in the
  ~11 GB free on the run host, so no LoRA is required. Options are pooled
  from their own token spans, not from a sentence vector.
- **State sharing:** `H_state` does not depend on the question, so it is
  computed once and reused across every question branch. This is the
  mechanism behind Jev's efficiency claim and the reason its request limit
  counts the state once. Phase 1's batch-dim folding was a stand-in that
  shared nothing.
- **Dynamic arity** falls out: options are token spans, so candidate count
  is a runtime property.

Deliberately NOT included: an LLM backbone. That is §9's escalation.

## 5. Training

Derived from NanoJev's released recipe (`hard_lr1e5`), the closest open
reproduction:

- Differential learning rates: encoder **1e-5**, heads/adapters **1e-4**.
  A single LR either leaves the heads unmoved or destroys the encoder.
- Warm up heads with the text encoder frozen, then unfreeze. Report the
  schedule; do not tune it silently.
- All three questions trained jointly in one forward pass (already true).
- Loss regimes carried forward unchanged: CE, CE+Brier. Post-hoc
  temperature derived from the CE arm, exactly as Phase 1 does.
- **Class weighting is NOT the default.** Arm D established it trades
  calibration for macro-F1 without fixing the tail; it stays a diagnostic.

## 6. Arms

Minimal, following NanoJev's one-main-config-plus-variants scope rather
than a grid:

| arm | encoder | loss | purpose |
|---|---|---|---|
| `joint__whisper__A-ce` | whisper | CE | main run |
| `joint__whisper__B-brier` | whisper | CE+Brier | calibration variant |
| `joint__whisper__C-temp` | whisper | derived | calibration variant |
| `joint__prosody__A-ce` | prosody | CE | does joint encoding rescue the prosody channel |
| `joint__text_only__A-ce` | — | CE | controlled baseline, audio muted |

Whisper only on the audio side: it won every Phase 1 comparison, and a
second audio encoder buys less than the text-only control does.

## 7. Evaluation

Reuse Phase 1's analysis scripts unchanged — they are validated
(`analysis/verify.py` reproduces all 24 published cells to 4 decimals).

- **Headline: Brier and NLL.** ECE only ever printed beside the
  train-fitted base-rate anchor.
- **Per-class precision AND recall**, never macro-F1 alone. Arm D showed
  F1 moving off zero at precision 0.02-0.05, which is recall bought at
  nothing.
- **Brier skill score against each stratum's own base rate** for any
  stratified claim. Raw deltas between strata with different label
  distributions are not interpretable.
- **EmotionLines ambiguity stratification** (`analysis/`), entropy split
  only. The MISLEADING stratum is partly circular and is corroboration at
  best.

## 8. Success criteria

Phase 2 succeeds if it produces a clear answer, including a negative one.

1. **IIA probe (gate, runs before any training).** Append an irrelevant
   option and measure the log-odds shift between two untouched options.
   Phase 1's model must score ~0 (confirming §2 empirically, not just
   algebraically). Phase 2's model must score non-zero. If Phase 2 also
   scores ~0, the rebuild did not deliver the property it exists to
   deliver and nothing downstream is worth reading.
2. **Sadness recall** (683 train examples) rises materially above Phase 1's
   2.9% ceiling at usable precision. This is the H-arch/H-scale
   discriminator.
3. **Brier beats the train-fitted base rate** on all three questions —
   which Phase 1 managed only with whisper, and only on some strata.

A clean negative on (2) with (1) passing is a *valid and informative*
outcome: it says joint encoding was not the binding constraint, and
escalates to §9 on evidence rather than assumption.

## 9. Escalation (not this phase)

If (1) passes and (2) fails, the constraint is representational capacity
or pretrained knowledge, and the next step is a causal LM backbone —
Qwen3-0.6B, NanoJev parity, differential LRs as above. Note this forces
causal attention, which changes the state/option layout: state as a shared
KV prefix, options in the suffix. Worth spelling out then, not now.

## 10. Known traps

Phase 1's every defect was silent — a plausible result from a mechanism
that was not there. Specific to this phase:

1. **The IIA gate can pass trivially.** Any non-linearity across options
   produces *some* shift. Require the shift to have the right sign and a
   magnitude comparable to Jev's -0.28, across randomised blocks with
   controls, not a single probe. A tiny non-zero number is not evidence.
2. **Option pooling can destroy option identity.** Pooling token spans to a
   vector is the same failure family as Phase 1's C2, where
   `speaker_relative_norm` erased level and made two very different inputs
   identical to 0.0000. Test that two options with overlapping wording
   produce distinguishable representations before trusting any result.
3. **State sharing can silently stop being shared.** If `H_state` is
   recomputed per question the model still trains and still scores; only
   the efficiency claim dies, invisibly. Assert the state encoder runs
   exactly once per example per forward pass.
4. **Audio can leak through the padding mask** into the text-only arm.
   Phase 1 shipped a duration leak (4.702e-03) that content-only checking
   missed. Re-run that exact check; `audio_present` must gate the mask, not
   just the content.
5. **Option permutation is train-only.** `permute_candidates` subsamples
   and shuffles; anything keyed to slot index at eval is keyed to a class,
   but anything keyed to slot at TRAIN is keyed to noise.
6. **Differential LRs make checkpoints heavier.** Exclude frozen modules
   from the saved state dict; Phase 1 went 135MB -> 44MB that way.
7. **Class collapse will recur.** Report per-class precision/recall every
   run. A rising macro-F1 can be entirely recall at 0.03 precision.
8. **Do not report ECE without the anchor.** See §1.

## 11. Limitations

- MELD splits remain speaker-shared. Phase 2 is still pipeline validation,
  not evidence for an audio-improves-calibration claim. IEMOCAP's
  leave-one-session-out protocol remains what settles that, and its
  per-annotator VAD labels remain what makes §7.1's arousal-vs-valence
  contrast measurable.
- No arm sees the current utterance's transcript. That is the design goal
  (avoid ASR on the current turn), but it means the grid still lacks a
  transcript-reading *comparison* floor. Adding one remains open.
- One seed throughout. Phase 1's stratified result was never seed-replicated
  and Phase 2 inherits that gap.
