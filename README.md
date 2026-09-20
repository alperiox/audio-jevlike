# Prosodia — an audio Jev

A [System One](https://typesafe.ai/blog/introducing-system-one-models-and-jev)-shaped
decision model that takes **speech** as its state. Audio is encoded once; several
typed questions branch over that single encoding and are answered in parallel as
calibrated probability distributions. No transcript, no speech recognition, no
generated text — the Whisper *decoder never runs*.

Live demo (private Space): [`alperiox/prosodia`](https://huggingface.co/spaces/alperiox/prosodia)

## Why

Across the ~60 projects in [`awesome-jev`](https://github.com/cobanov/awesome-jev),
vision is covered — `PlayJev` reads a game frame in one forward pass — but every
audio entry routes **ASR → text → Jev**. The state is a transcript; the audio
never reaches the model. This one feeds audio features directly, which is the
point: a voice agent that skips ASR on the current turn avoids both its latency
and its error propagation.

## What is reproduced

- **Prefill-only** — no autoregressive decode; answers come from a readout
- **State encoded once**, shared across every question
- **Structural question isolation** — questions are folded into the batch
  dimension, so cross-question attention is unrepresentable rather than masked
- **Typed outputs** — Choice / Score / Noul
- **Pointer readout** — logits are an inner product between the branch vector and
  each option's embedding, so any candidate count works untrained
- **Dynamic candidate sets** — option count is a runtime property

Training is cross-entropy + 0.5 × Brier, a strictly proper scoring rule.
TypeSafe's **RLCD is unpublished**, so no open reproduction uses it;
[NanoJev](https://github.com/TianyuCodings/NanoJev), the closest one, states that
its calibration work "follows proper-scoring theory, not an unpublished TypeSafe
algorithm." Same position here.

## The one measured way this differs from Jev

Stated as a finding, not a disclaimer.

The readout scores each option as an **independent inner product**, so
`p(a)/p(b)` cannot change when a third option is added — this model satisfies
**independence of irrelevant alternatives exactly**. Jev demonstrably violates
it: appending an irrelevant option shifts log-odds between two untouched options
by roughly **−0.28** (ten randomised blocks, every one decreasing).

That makes an IIA probe the sharpest cheap conformance test for whether *any*
Jev reimplementation is in the right functional class — and this one is not yet.
Closing it means options that attend to each other instead of independent frozen
embeddings. See `docs/results/` and the Phase 2 design.

## Findings

Phase 1 was a 12-arm ablation (3 loss regimes × 3 encoders + 3 text-only
controls) on MELD. Most of what it established is not what it set out to.

| | |
|---|---|
| **ECE cannot rank these arms** | A train-fitted constant predictor scores ECE 0.0097. Ranking 12 arms by ECE vs by Brier gives Spearman **+0.147 / −0.294 / −0.266** — near-independent orderings. |
| **The control encoder beat the primary** | Whisper > WavLM on accuracy, macro-F1 and Brier on all three questions. Read with the explicit-prosody arm failing to beat a constant, MELD affect looks lexical. |
| **The prosody effect did not survive its control** | Strata differ in base rate (59.4% vs 29.7% neutral). Scored against each stratum's own marginal, prosody never beats a constant anywhere. |
| **Class collapse had two causes** | `joy` (1743 examples) was optimization-limited and recovered under reweighting; `sadness` (683) did not — recall never exceeded 2.9% under a 6.9× upweight. |
| **Whisper retained pitch** | `pitch_direction` reaches ~0.51 against a 0.36 majority — an ASR-trained encoder carrying contour information transcription never needed. |
| **The architecture was never tested** | It is constant across all twelve arms. Nothing here bears on whether the Jev shape helps. |

**Treat single numbers as approximate.** Everything is one seed, and removing
1.75% of labels on two questions once swung an unrelated question by 13 points.

## Layout

```
src/prosodia/       the model, data pipeline, losses, metrics
  model/            state encoder, isolated branches, pointer readout
  labels/           deterministic acoustic labels + the demo question bank
scripts/            training, feature extraction, the ablation grid, demos
analysis/           EmotionLines stratification and the re-analysis
space/              the Hugging Face Space (Docker)
docs/results/       what Phase 1 measured, with its corrections
tests/              165 tests
```

## Running it

```bash
uv sync
uv run pytest                       # 165 tests

scripts/fetch_meld.sh               # corpus
uv run python scripts/extract_features.py --encoder whisper
uv run python scripts/run_ablation.py --corpus-root ... --cache-root ...

# seven-question demo model, then a local mic demo on :8777
uv run python scripts/train_demo.py  --corpus-root ... --cache-root ... --ckpt-root ...
uv run python scripts/export_bins.py --corpus-root ... --cache-root ... --out <ckpt>/bins.json
uv run python scripts/serve_demo.py  --corpus-root ... --cache-root ... \
    --ckpt <ckpt>/best.pt --page scripts/live_demo.html --port 8777
```

Training needs ~4–5 GB; feature extraction caps audio at 30s because WavLM's
relative-position bias is O(T²) and MELD ships clips up to 305s.

## Data

[MELD](https://github.com/declare-lab/MELD), GPL-3.0. Please cite:

> S. Poria, D. Hazarika, N. Majumder, G. Naik, E. Cambria, R. Mihalcea.
> *MELD: A Multimodal Multi-Party Dataset for Emotion Recognition in
> Conversation.* ACL 2019.

Two corpus caveats that travel with every number here. MELD's splits are
dialogue-disjoint but **not speaker-disjoint** — the six recurring leads appear
in train, dev and test, which inflates audio arms specifically, so these are
pipeline-validation results rather than evidence that audio improves
calibration. And ~2.45% of clips declare a span too short for the words they
claim (one test clip gives 177 ms for "No, she doesn't"); those are filtered
from derived labels.
