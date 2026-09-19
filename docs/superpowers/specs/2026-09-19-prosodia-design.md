# Prosodia — Design Spec

**Date:** 2026-09-19
**Status:** Approved design, pending implementation plan

---

## 0. TL;DR

**Build** an audio-native Jev-shaped decision model — speech + context in, typed `Choice`/`Score`/`Noul` out, one prefill-only forward pass — and use it to answer whether **prosody carries calibration information, and where that calibration physically lives**.

| Decision | Choice |
|---|---|
| Data | HarperValleyBank only (~26k utterances, 23h, 59 speakers, public domain) |
| Thesis tasks | Emotional valence (`Score`), dialog acts (`Choice`) |
| Architecture | Frozen speech encoder + our state encoder, isolated branch layers, 3 heads, **linear readout** |
| State | audio + structured context, with modality dropout |
| Questions | Frozen sentence encoder + paraphrase/candidate augmentation + held-out-question split |
| Ablation | 3 losses (CE / CE+Brier / CE+temp-scaling) x 3 encoders (WavLM / Whisper / explicit-prosody) = 9 runs |
| Compute | `ssh mac` (M4 Pro, 24GB), ~20GB footprint; Colab as fallback |
| Signature experiment | F0-flattening dose-response sweep with lexical content held fixed |
| Headline target | An entropy neuron with an identified *physical* cause |

**The through-line:** every stage of this pipeline can silently destroy the evidence it is meant to measure, and none of them announce it. §11 enumerates eight such traps with guardrails. That section is the most important one in this document.

**Out of scope:** CANDOR, Tier-2 teacher labels for any thesis claim, turn completion as a question, production-readiness.

---

## 1. Research question

> **Does prosody carry calibration information — and if so, where does it go?**

One question, two halves:

- **Behavioral:** does an audio-native decision model produce better-*calibrated* probabilities than a text-only equivalent, particularly on ambiguous inputs?
- **Mechanistic:** if it does, what in the network implements that? Is calibration a learned scalar temperature, or something distributed?

The model is the **instrument**, not the deliverable. The voice-agent framing is the motivation and the demo.

## 2. Background

TypeSafe AI's **Jev** ("System One model") replaces autoregressive decoding with a single prefill pass. Context is encoded once into a shared state; N typed questions branch off it, attending to the evidence but **not to each other**; each branch reads out through a linear head (`z = Wh + b`) into a schema-constrained answer set.

Three primitives:

| Primitive | Returns |
|---|---|
| `Choice` | distribution over a caller-supplied option set |
| `Score` | probability-weighted expectation over an ordered rubric |
| `Noul` | Bernoulli probability, 0–1 |

Training uses **RLCD** (Reinforcement Learning for Calibrated Decisions), optimizing proper scoring rules so probabilities are honest rather than fluent. Measured ECE on MMLU: **0.0313**.

**The gap.** Surveying ~60 projects in `awesome-jev` — games, robotics, SQL, log triage, routing, code review — there is **no audio anywhere**. Jev is text-only. Every open reproduction (`openjev-sglang`, `jevmlx`, `Verdict-open-jev`, `NanoJev`) is text.

**The observation that makes voice agents the right application.** A voice agent doesn't make one decision per turn; it makes twenty — turn complete? which agent owns this? frustrated? consent given? escalate? Jev's shared-state/isolated-branch design encodes the context once and answers all twenty nearly free. An LLM cascade pays for that context twenty times.

**Separation worth keeping in view:** in Jev-like systems, *speed is architecture, calibration is objective*. A prefill-only readout gets you the latency for free; it does not get you calibration. That comes from the training objective, which is why the research content of this project lives on the objective side.

## 3. Scope

**In scope:**
- Audio-native decision model over HarperValleyBank
- Loss ablation (CE / CE+Brier / CE+temperature) as both the calibration experiment and the interp setup
- Encoder ablation (WavLM / Whisper / explicit-prosody)
- Prosodic intervention experiments
- Entropy-neuron localization
- Live state-panel demo

**Explicitly out of scope:**
- CANDOR (see §12)
- Tier-2 LLM-teacher labels for any thesis-testing claim
- Turn-completion as a question (leak-prone; see §11)
- Dynamic per-request question *training* beyond paraphrase/candidate augmentation
- Any claim of production-readiness

## 4. Data

**HarperValleyBank** (arXiv:2010.13929) — public domain.

| Property | Value |
|---|---|
| Conversations | 1,446 |
| Utterances | ~26,000 (2–60 per conversation, mean 18) |
| Audio | ~23 hours |
| Speakers | 59 |
| Vocabulary | ~700 unique words |

**Annotations used:**

| Label | Primitive | Prosody-critical? |
|---|---|---|
| Emotional valence | `Score` | **Yes — primary thesis vehicle** |
| Dialog actions (16) | `Choice` | Partially |
| Caller intent (8) | `Choice` | No (mostly lexical) |
| Derived (problem stated? etc.) | `Noul` | Varies |

**The 700-word vocabulary cuts both ways.** It means the corpus is scripted/simulated — so it's a near-controlled experiment with lexical content roughly held constant and prosody varying freely, which *strengthens* the thesis vehicle. It also means acted valence and no transfer to real traffic. Must be stated as a limitation, not buried.

**Splits:** speaker-disjoint, ~45/7/7 of 59 speakers. Test ≈ 3k utterances — enough for ECE at ~10 bins, not enough for finely stratified analysis. This bounds what can be claimed.

**Effective size:**

| Level | Count |
|---|---|
| Distinct states | ~26k — **the binding constraint** |
| (state, question) pairs | ~650k |
| With paraphrase augmentation | several million |

The state encoder only ever sees 26k distinct examples. Keep it small; lean on the frozen encoder.

**Question-side augmentation (free — no new labels):**
- **Paraphrase:** ~20 rephrasings per question. Same audio, same label.
- **Candidate-set:** subsample the 8 intents into random 3-/5-way sets, permute order, rewrite descriptions. Defeats positional shortcuts.
- **Held-out questions:** train on ~20 question types, evaluate on ~5 unseen — a genuine zero-shot generalization claim.

**Hard rule:** augment the *question* side freely; **never** the waveform. Speed perturbation and pitch shifting are label-preserving only by assumption, and here they would mangle the exact cues under study.

## 5. Architecture — the instrument

```
audio ──[frozen encoder]──┐
                          ├──[state encoder]──► h_state ──┐
context ──[serialized]────┘                               │
                                    ┌─────────────────────┼──────────────────┐
                                (q1,opts1)            (q2,opts2)         (q3,opts3)
                                    │                     │                  │
                              [branch layers — isolated by attention mask]
                                    │                     │                  │
                                 Choice                 Noul              Score
```

| # | Component | Ownership | Notes |
|---|---|---|---|
| 1 | Speech encoder | **Frozen** | WavLM-large primary; ablation arm (§6) |
| 2 | State encoder | Ours | Small transformer; **attention pooling**, speaker-relative normalization |
| 3 | Question + candidate encoder | **Frozen** | Pretrained sentence encoder over instructions + candidate descriptions |
| 4 | Branch layers | Ours | Cross-attention; isolation enforced by mask |
| 5 | Heads | Ours | Noul (sigmoid) / Choice (set-attention pointer) / Score (ordinal → expectation) |

**Design constraints that are load-bearing:**

- **Readout stays strictly linear** (`z = Wh + b`). An MLP here would make "which directions in `h` produce confidence" unanswerable, killing the interp half.
- **Pooling must preserve contours.** Attention pooling or strided conv — never mean pooling. Prosody is supra-segmental; the mean of a rise and the mean of a fall are identical.
- **Speaker-relative normalization.** "High pitch" is only meaningful against that speaker's own baseline. Normalize within-conversation.
- **Frozen question encoder** is what makes dynamic questions learnable from only ~25 base question types — the model learns to *read* a pretrained semantic space, not to build one.

**State composition:** `audio + structured context`. Per the Jev API, `state` is `string | object | array` — arbitrary application state, serialized to text and consumed by the frozen sentence encoder. Flexible at the interface, consistent at the encoding path.

Context includes recent turns (from streaming ASR, **never gold transcripts** — see §11), turn index, elapsed time, repeat/rephrase count, active state node, prior dialog acts. **Excludes** the model's own prior decisions, which would create system-level exposure bias for no research benefit.

**Modality dropout** during training: randomly zero audio or context. Yields three eval conditions from one checkpoint (full / audio-only / context-only), and prevents the model from learning to ignore audio when context is informative — the mitigation and the measurement are the same mechanism.

## 6. Training and ablation grid

**Loss axis** — the calibration experiment and the interp setup, simultaneously:

| Arm | Objective | Tests |
|---|---|---|
| **A** | Cross-entropy only | baseline |
| **B** | CE + Brier (proper scoring composite) | RLCD stand-in |
| **C** | A + post-hoc temperature scaling (L-BFGS) | **the control that decides the interp question** |

If **B ≈ C**, calibration is a scalar and that is itself the answer to "where does it live." If **B > C**, calibration is distributed and the localization hunt is warranted.

**Encoder axis:**

| Arm | Role |
|---|---|
| WavLM-large | Primary — SSL objective retains paralinguistic information |
| Whisper encoder | Control — ASR objective may discard prosody at the feature boundary |
| Explicit F0/energy/voicing channel | Diagnostic — distinguishes encoder failure from task failure on a null result |

**9 runs**, minutes each over cached features. Fixed seeds, speaker-disjoint splits, **Weights & Biases for every run**, checkpoints local (~10GB total), Drive as backup.

## 7. Evaluation

Calibration metrics are first-class, not an afterthought.

- **Discrimination:** accuracy, macro-F1
- **Calibration:** ECE, Brier, NLL, reliability diagrams
- **Accuracy-vs-coverage curves** — at threshold *t*, what coverage and what error rate. The practical artifact.
- **Latency**
- **Stratified by ambiguity** — the thesis-critical view. Define ambiguity independently (text-model entropy, or annotator disagreement), then report calibration *within* strata. Aggregate numbers will bury the effect.

**Two baselines, doing different jobs:**

| Baseline | Purpose |
|---|---|
| Same architecture, text-only state | **Controlled** — isolates modality, holds everything else fixed |
| Whisper → **real Jev** (Vercel AI Gateway) | **Practical** — the actual text-state System One model, not a prompted stand-in |

## 8. Interpretability study

1. **Prosodic interventions.** F0 flattening, pause lengthening, disfluency splicing via `praat-parselmouth`/PSOLA. Swept continuously (0/25/50/100%) for dose–response curves, with lexical content held exactly fixed. This is the signature experiment: a causal intervention on a physically meaningful, continuously parameterized input axis — an affordance text interpretability does not have.
2. **Entropy-neuron scan.** Ablate each late-layer unit; plot Δentropy against Δargmax-rate. Entropy neurons are the high-Δentropy, near-zero-Δargmax cluster.
3. **The intersection.** Are units responsive to F0 flattening the same units that are entropy neurons? If yes: **an entropy neuron with an identified physical cause.** Existing entropy-neuron work is all on autoregressive LMs where the cause is unknown.
4. **Linear readout probes.** Project `h` onto confidence-relevant directions directly — available because the readout stayed linear.

Tooling detail to be settled at implementation time.

## 9. Demo

Live conversational-state panel: audio in, question bank lighting up with calibrated probabilities in real time, plus a free-text box to pose a new question and watch it answer. Dynamic questions are what make this more than a static panel of lights.

## 10. Compute and environment

**Primary: `ssh mac`** — Apple M4 Pro, 24GB unified (~14GB free; `llama-server` holds ~4GB), 12 cores (8P), macOS 26.6.2, 73GB disk free. `uv 0.12.16` at `~/.local/bin/uv`, Python 3.11, `ffmpeg`, `brew`, `git` present. Torch/transformers/MLX **not yet installed**.

**Fallback: Google Colab** (4TB Drive available). Write device-agnostic code (`mps` → `cuda` → `cpu`). Because features are precomputed, the training loop never touches the audio stack, so device differences are minimal.

The Mac is primary because the interp work is interactive and long-running — ablation sweeps and intervention curves want a kernel alive for hours, which Colab's session limit actively fights.

**Budget:**

| Item | Size |
|---|---|
| Feature cache (WavLM-large, 50Hz, fp16, 23h) | ~8.5 GB |
| Checkpoints (9 runs × ~3) | ~10 GB |
| **Total** | **~20 GB** of 73 GB free |

**MPS numerical-fidelity guardrail.** This project claims small effects — ECE differences ~0.01, entropy shifts from single-neuron ablations. Those are exactly the magnitudes an fp16 quirk or backend inconsistency can manufacture. Therefore: **all interp measurements run in fp32**, and a cross-device consistency check re-computes key numbers on CPU and asserts agreement within tolerance. A numerical artifact and a finding look identical in a plot; the check is what tells them apart.

## 11. Known traps and guardrails

Every stage of this pipeline can destroy the evidence it is meant to measure, and **none of them announce it** — the model trains fine, the loss descends, and the negative result looks clean. These are design parameters, not defaults to inherit.

| # | Trap | Where it bites | Guardrail |
|---|---|---|---|
| 1 | **Transcription** discards prosody | Text baseline defines ground truth | Tier 0/1 labels only; never claim prosody results on teacher labels |
| 2 | **ASR-objective encoder** discards prosody | Feature boundary | Whisper-vs-WavLM as an explicit ablation arm |
| 3 | **Mean pooling** destroys contours | State encoder, first layer | Attention/strided pooling; never mean |
| 4 | **Gold transcripts leak turn boundaries** | Context construction | Context from streaming ASR; punctuation and segmentation stripped. Turn completion dropped as a question entirely |
| 5 | **Teacher labels** cap audio at text performance | Supervision | Tier 2 excluded from scope |
| 6 | **Waveform augmentation** mangles the cues under study | Data pipeline | Augment question side only |
| 7 | **MPS numerical drift** masquerades as a finding | Interp measurements | fp32 + cross-device assertion |
| 8 | **Upstream pre-aggregated features** (e.g. CANDOR's 1s bins) are prosody-blind | Any third-party feature set | Compute prosody from raw audio; never trust packaged aggregates |

Traps 1–3, 5, 6 produce **false nulls**. Trap 4 produces a **false positive so strong it also suppresses the real effect** — a punctuation detector scoring 97% would show no response to F0 flattening, indistinguishable from a genuine null. Leaks must be closed by construction, not detected afterwards.

**Enforcement:** label provenance (tier) is a field in the dataset schema, so a Tier-2 row cannot physically enter a thesis-testing eval split.

## 12. Limitations

- **23 hours, 59 speakers.** Small. Speaker-disjoint test ≈ 3k utterances bounds calibration resolution to ~10 bins.
- **Scripted and acted.** 700-word vocabulary; valence labels are performances, not genuine affect. Same critique that dogs IEMOCAP. No claim of transfer to real traffic.
- **Zero-shot question generalization may simply fail** from ~25 base question types. Measured, not assumed; fixed-bank remains the demo fallback.
- **We do not know Jev's actual architecture.** This is a Jev-*shaped* experiment built on a public reverse-engineering account, not a reproduction.
- **Single domain**, single language, single corpus.

## 13. Future work

**CANDOR generalization extension.** 1,656 unscripted conversations, 850h, 1TB+. Rejected for this phase: wrong domain (open-domain small talk between strangers, no task structure), no utterance-level human affect labels (survey is conversation-level; facial emotion is a model output), shipped acoustic features aggregated to 1s and therefore prosody-blind, and 1TB exceeds sensible local storage.

As a *next step*, pseudo-labeling CANDOR would broaden the question distribution and teach the model to produce meaningful probabilities for more general questions over richer context states. This is the legitimate use of Tier-2 labels — **breadth, never thesis-testing** — and is consistent with the tier rule in §11.

Also deferred: **branch-isolation capacity.** Jev asserts questions should not attend to one another; nobody has checked whether that is free. Do isolated branches occupy orthogonal subspaces of the shared state, and how many questions can one state carry before they interfere? Directly relevant to the twenty-questions-per-turn use case, but a separate study.

## 14. Success criteria

The project succeeds if it produces defensible answers to:

1. Does audio state improve **calibration** (not merely accuracy) over a controlled text-only baseline, stratified by ambiguity?
2. Does CE+Brier beat post-hoc temperature scaling — i.e. is calibration distributed or scalar?
3. Do prosodic interventions causally move output confidence, with a dose–response relationship?
4. Do entropy neurons exist in this readout, and do they coincide with F0-responsive units?

**A clean null on (1) or (3) is a valid outcome** — provided the diagnostic arms (explicit-prosody channel, Whisper-vs-WavLM) can attribute it to a cause rather than leave it ambiguous. That attribution capability is why those arms are in scope.

## 14b. API access (resolved 2026-09-19)

Jev is available via **Vercel AI Gateway** (`typesafe-ai/jev`), removing the TypeSafe early-access dependency. 32k context, $0.042/1M input tokens, output free. Interface matches the HTTP API (`boolean` in the AI SDK where the HTTP API says `noul`):

```ts
import { experimental_evaluate as evaluate } from 'ai';
const result = await evaluate({
  model: 'typesafe-ai/jev',
  state: '...',
  questions: { refunded: { type: 'boolean', instructions: 'Was a refund issued?' } },
})
```

Two consequences:
1. **§7's practical baseline becomes real Jev on transcripts**, not a prompted LLM stand-in. A far stronger comparison — it isolates modality against the actual System One model.
2. **§13's CANDOR pseudo-labeling extension becomes cheap.** At $0.042/MTok with free output, Jev is a viable teacher for breadth labels.

Does not change the core build.

## 15. Sources

**Jev / System One**
- [Introducing System One Models & Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev) — TypeSafe AI launch post
- [Jev's Architecture Unmasked](https://archerhume.com/posts/jevs-architecture-unmasked/) — Archer Hume; source of the `z = Wh + b` readout reconstruction, the shared-state/isolated-branch description, and the **ECE 0.0313 on 1,200 MMLU items** figure. Explicitly speculative black-box reverse engineering.
- [TypeSafe HTTP API reference](https://docs.typesafe.ai/api.md) — `state`/`questions` schema, Choice/Score/Noul definitions

**Ecosystem (surveyed for the audio gap)**
- [awesome-jev](https://github.com/cobanov/awesome-jev) — ~60 projects, no audio
- [NanoJev](https://github.com/TianyuCodings/NanoJev) — Qwen3-0.6B + decision heads; dynamic-candidate set attention
- [openjev-sglang](https://github.com/ekzhang/openjev-sglang) — prefill + first-token logprob readout
- [jevmlx](https://github.com/bnsd55/jevmlx) — same approach on Apple Silicon / MLX
- [Verdict-open-jev](https://github.com/Heman10x-NGU/Verdict-open-jev) — ModernBERT encoder + classification head
- [jev-benchmarks](https://github.com/AbdelStark/jev-benchmarks) / [jevcal](https://github.com/abhixhek/jevcal) — calibration and coverage methodology

**Data**
- [HarperValleyBank](https://arxiv.org/abs/2010.13929) — Wu et al., 2020. Public domain.
- [CANDOR corpus](https://www.science.org/doi/10.1126/sciadv.adf3197) — Reece et al., *Science Advances*. Deferred; see §13.

**Related prior work by the author**
- [TinyAya Expedition](https://alperiox.dev/posts/tinyaya-expedition-retrospective/) — source of the silent-failure lessons encoded in §11.

A Sophion knowledge base (`jev`) holds compiled articles for all Jev sources above.
