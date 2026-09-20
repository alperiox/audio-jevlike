---
title: Prosodia
emoji: 🎧
colorFrom: purple
colorTo: pink
sdk: docker
app_port: 7860
license: gpl-3.0
pinned: false
---

# Prosodia — an audio Jev

Record a clip. It is encoded **once** by Whisper's encoder, then every question
branches over that single encoding and is answered in parallel as a probability
distribution. No transcript, no speech recognition, no generated text —
Whisper's *decoder never runs*.

Questions and their option sets are editable at request time.

Across the ~60 projects in `awesome-jev`, vision is covered (`PlayJev`,
`Jev Visual`) but every audio entry routes **ASR → text → Jev**: the state is a
transcript and the audio never reaches the model. This one takes audio features
directly.

## What is reproduced

- **Prefill-only.** No autoregressive decode; answers come from a readout.
- **State encoded once**, shared across every question.
- **Structural question isolation** — questions are folded into the batch
  dimension, so cross-question attention is unrepresentable rather than masked.
- **Typed outputs**: Choice / Score / Noul.
- **Pointer readout** — logits are an inner product between the branch vector
  and each option's embedding, so any candidate count works untrained.
- **Dynamic candidate sets** — option count is a runtime property, which is why
  you can edit the questions above and get answers.

Training is cross-entropy + 0.5 × Brier, a strictly proper scoring rule.
TypeSafe's **RLCD is unpublished**, so no open reproduction uses it —
[NanoJev](https://github.com/TianyuCodings/NanoJev), the closest one, states
plainly that its calibration work "follows proper-scoring theory, not an
unpublished TypeSafe algorithm." Same position here.

## One measured way this differs from Jev

Worth stating precisely, because it is a finding rather than a disclaimer.

The readout scores each option as an **independent inner product**, so
`p(a)/p(b)` cannot change when a third option is added — this model satisfies
independence of irrelevant alternatives **exactly**. Jev demonstrably violates
it: appending an irrelevant option shifts log-odds between two untouched
options by roughly **−0.28** (ten randomised blocks, every one decreasing).

So option interaction is the sharpest known behavioural test for whether a
Jev reimplementation is in the right functional class, and this one is not yet.
Fixing it means options that attend to each other rather than independent
frozen embeddings.

## Reading the answers

- Trained on **MELD** — acted sitcom audio. Your voice on a laptop mic is out of
  domain; treat affect answers sceptically.
- **Speaker is out of scope for your voice.** The option set is six sitcom
  characters, so it must pick one regardless. MELD's splits are also
  speaker-shared, so that question is leaked even in-domain.
- **Acoustic questions carry a ◆** marking the true answer, computed from your
  clip by the same fixed rule used in training. Those are self-verifying.
- Every card reports the **nearest question the model was actually trained on**
  with a cosine similarity. Below 0.6 the answer is unsupported — the model
  answers anything you type, including questions nothing in training resembles.

## Measurement caveats

Single-seed results. Removing 1.75% of labels on two questions once swung an
unrelated question by 13 points, so treat individual numbers as approximate.
Emotion sits at its majority-class baseline; the acoustic questions clear their
baselines comfortably (pitch ≈ 0.51 vs 0.36 majority).

## Data and licence

Trained on [MELD](https://github.com/declare-lab/MELD), GPL-3.0. This Space
contains model weights and code only — no dataset audio. Please cite:

> S. Poria, D. Hazarika, N. Majumder, G. Naik, E. Cambria, R. Mihalcea.
> *MELD: A Multimodal Multi-Party Dataset for Emotion Recognition in
> Conversation.* ACL 2019.
