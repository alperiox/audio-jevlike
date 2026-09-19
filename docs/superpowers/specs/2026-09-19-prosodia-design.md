# Prosodia — Design Spec

**Date:** 2026-09-19
**Status:** Approved design, pending implementation plan

---

## 0. TL;DR

**Build** an audio-native Jev-shaped decision model — speech + context in, typed `Choice`/`Score`/`Noul` out, one prefill-only forward pass — and use it to answer whether **prosody carries calibration information, and where that calibration physically lives**.

| Decision | Choice |
|---|---|
| Data | **MELD** (build corpus, instant) → **IEMOCAP** (thesis corpus, registration pending) |
| Thesis tasks | MELD: sentiment (`Score`, ordered), emotion (`Choice`). IEMOCAP: **arousal vs valence contrast** |
| Architecture | Frozen speech encoder + our state encoder, isolated branch layers, 3 heads, **linear readout** |
| State | audio + structured context, with modality dropout |
| Questions | Frozen sentence encoder + paraphrase/candidate augmentation + held-out-question split |
| Ablation | 3 losses (CE / CE+Brier / CE+temp-scaling) x 3 encoders (WavLM / Whisper / explicit-prosody) = 9 runs, **+ 3 text-only baseline runs** (one per loss regime; encoder axis collapses since audio is absent) = 12 runs total |
| Compute | Local M2 Pro (16GB, 230GB free), ~20GB footprint; `ssh mac` / Colab as overflow |
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
- Audio-native decision model over **MELD** (build corpus), then **IEMOCAP** (thesis corpus)
- Corpus abstraction making loaders interchangeable (§4.3)
- Loss ablation (CE / CE+Brier / CE+temperature) as both the calibration experiment and the interp setup
- Encoder ablation (WavLM / Whisper / explicit-prosody)
- Prosodic intervention experiments
- Entropy-neuron localization
- Live state-panel demo

**Explicitly out of scope:**
- CANDOR (see §13)
- HarperValleyBank as a thesis vehicle — its affect labels are model outputs (§4.0); retained only as an optional non-affect `Choice` task
- Tier-2 LLM-teacher labels for any thesis-testing claim
- Turn-completion as a question (leak-prone; see §11)
- Dynamic per-request question *training* beyond paraphrase/candidate augmentation
- Any claim of production-readiness

## 4. Data

### 4.0 Label provenance is verified per label, not assumed

Corpus summaries say "annotated with X" for both human annotation and model output. Provenance is verified in the source paper, **per label**, before any label becomes a thesis vehicle. Results of that check:

| Corpus | Label | Provenance | Verdict |
|---|---|---|---|
| HarperValleyBank | Intent (8) | *"derived automatically from the tasks assigned to callers"* | Gold by construction |
| HarperValleyBank | Dialog actions (16) | *"produced using a Gridspace API rather than human annotation"* | **Rejected — model output** |
| HarperValleyBank | Emotional valence (3) | *"Gridspace Speech API trained on a large corpus of proprietary data"* | **Rejected — model output** |
| MELD | Emotion (7), Sentiment (3) | *"re-annotate all the utterances by asking the three annotators to also look at the available video clip"* | **Accepted — human, multimodal** |
| IEMOCAP | Categorical + V/A/D | ≥2 human raters per utterance | **Accepted — human** |

**HarperValleyBank is therefore off the critical path.** Its valence labels are a proprietary *audio* model's outputs, which would have made "audio beats text on valence" circular — we would have measured successful distillation of Gridspace's classifier and reported it as a prosody finding. Its one gold label (intent) is retained as an optional non-affect `Choice` task; nothing depends on it.

### 4.1 MELD — build corpus (available now)

| Property | Value |
|---|---|
| Source | *Friends*, multi-party dialogues |
| Dialogues | 1,039 / 114 / 280 (train/dev/test) |
| Utterances | 9,989 / 1,109 / 2,610 |
| Labels | Emotion (7) → `Choice`; Sentiment (neg/neu/pos) → **`Score`, ordered** |
| Annotation | 3 annotators, majority vote, **Fleiss κ = 0.43** (vs 0.34 text-only) |
| Audio | 16-bit PCM WAV, extracted from episode video |
| Access | Free, immediate |
| Splits | **Dialogue-disjoint, speaker-shared** — see caveat below |

Sentiment being *ordered* (negative < neutral < positive) maps onto the `Score` primitive exactly — an ordered rubric read out as a probability-weighted expectation, structurally identical to the API docs' `["Calm", "Frustrated", "Very angry"]` example.

A **Dyadic MELD** variant (contiguous dyadic sub-dialogues) ships with the corpus and is preferred where two-party structure simplifies context construction.

**MELD's shipped splits are speaker-shared, not speaker-disjoint (owner Decision 1, 2026-09-19).** MELD is *Friends*: the six leads appear in train, dev, and test alike. The shipped train/dev/test CSVs are only **dialogue**-disjoint. We do not re-split MELD to fix this — *Friends* is dominated by those six characters, so a speaker-disjoint split would shred both data volume and class balance, and MELD was always the build/pipeline-validation corpus, not the evidence corpus. Because speaker identity is far easier to recover from acoustics than from text, and per-character emotion priors in *Friends* are strong, this specifically inflates the **audio** arm — exactly the confound that would manufacture an "audio beats text on calibration" finding. Consequently: **MELD results are pipeline validation only and cannot support an audio-improves-calibration claim.** `assert_speaker_disjoint` (§4.3) is deliberately never called on MELD, since it would fail by design; `scripts/run_ablation.py` prints an unmissable startup warning and injects the same text into every arm's W&B config instead. The real claim rests on IEMOCAP (§4.2), whose leave-one-session-out protocol is genuinely speaker-disjoint.

### 4.2 IEMOCAP — thesis corpus (registration pending)

| Property | Value |
|---|---|
| Content | 5 sessions of dyadic dialogues, 10 actors (5M/5F) |
| Size | ~12h, 10,039 utterances (5,255 scripted / 4,784 improvised) |
| Labels | Categorical → `Choice`; **5-point valence / arousal / dominance** → `Score` |
| Annotation | ≥2 raters per utterance, **per-rater votes available** |
| Audio | 16kHz studio |
| Protocol | Leave-one-session-out 5-fold CV (2 speakers per fold) |

IEMOCAP carries the headline experiment (§7.1). Thesis-critical evaluation is restricted to the **improvised** portion (4,784 utterances), with scripted reported separately — improvised speech is considerably more natural.

### 4.3 Corpus abstraction

A `Corpus` protocol yields `(audio, context, {question_key: (label, tier)})`, making MELD, IEMOCAP and HarperValleyBank interchangeable loaders. This is required from Task 1: it decouples the build from IEMOCAP's registration lead time, so work proceeds on MELD and IEMOCAP drops in with no other change.

`schema.assert_speaker_disjoint(splits)` makes the speaker-disjointness constraint enforceable rather than aspirational: it raises, naming the offending speakers, if any speaker appears in more than one split. It is exercised by IEMOCAP's leave-one-session-out splits and deliberately never called on MELD (§4.1) — MELD would fail it by design.

### 4.4 Question-side augmentation (free — no new labels)

- **Paraphrase:** ~20 rephrasings per question. Same audio, same label.
- **Candidate-set:** subsample and permute option sets; rewrite descriptions. Defeats positional shortcuts.
- **Held-out questions:** train on most question types, evaluate on unseen ones — a zero-shot generalization claim.

**Hard rule:** augment the *question* side freely; **never** the waveform. Speed perturbation and pitch shifting are label-preserving only by assumption, and here they would mangle the exact cues under study.

### 4.5 Known data limitations

- **MELD:** ~42% of utterances are under five words (thin prosodic material); heavy neutral imbalance (~47%); TV audio carries music and laugh track; only majority labels ship, so ambiguity strata must come from another source (text-model entropy) until IEMOCAP lands.
- **IEMOCAP:** only 10 speakers; acted.
- Both are performed rather than spontaneous affect. No claim of transfer to real traffic.

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

**Correction (I9, final-review fix, 2026-09-20):** Arm C is DERIVED from Arm A's trained checkpoint, not independently retrained — `scripts/run_ablation.py`'s `run_derived_arm_c` loads Arm A's saved weights and applies only the post-hoc temperature-scaling step. Arm C's training previously ran as a second, independent execution of Arm A's exact config, so its identity with Arm A rested on `torch.manual_seed` plus an identical op sequence giving bit-identical results across two separate runs — likely, but never guaranteed or asserted, at an expected B-vs-C effect size (~0.01 ECE) that ordinary training noise could fully absorb. Deriving Arm C makes that identity exact by construction. See the plan doc's Task 15 Correction note and `tests/test_run_ablation.py`'s bit-identity test.

**Encoder axis:**

| Arm | Role |
|---|---|
| WavLM-large | Primary — SSL objective retains paralinguistic information |
| Whisper encoder | Control — ASR objective may discard prosody at the feature boundary |
| Explicit F0/energy/voicing channel | Diagnostic — distinguishes encoder failure from task failure on a null result |

**12 runs** (the 9-arm encoder grid + 3 text-only baseline arms, one per loss regime — §7.1 "Two baselines"). Fixed seeds, **Weights & Biases for every run**, checkpoints local (~13GB total), Drive as backup. Splits are **session-disjoint on IEMOCAP** (leave-one-session-out, enforced by `assert_speaker_disjoint`) but only **dialogue-disjoint and speaker-shared on MELD** — never utterance-random on either. See §4.1 for why MELD is not re-split and what that means for interpreting its results.

**Correction (I6, final-review fix, 2026-09-20):** "minutes each over cached features" was wrong by ~30x and is removed above rather than corrected in place, because the original number rested on an assumption (`ProsodiaModel.forward` running batched) that was false at the time it was written — `forward` un-batched `collate_batch`'s padded `(B, T, D)` batch into a per-example Python loop, measured at ~88% of forward's cost. The I6 fix batches `forward` (see `src/prosodia/model/prosodia.py` and its plan-doc Task 11 correction) and measured, on this machine (MPS, `in_dim=1024, d_model=256, B=16, T=150`): forward+backward per batch 122.9ms → 43.4ms (~2.8x). Per-epoch/per-arm/grid wall time depends on dataset size (not yet run — feature extraction and the ablation grid are both still pending), so this section deliberately no longer states a wall-clock estimate; see the I6/I2 fix report for the full profiling detail instead of restating a number likely to go stale again.

## 7. Evaluation

Calibration metrics are first-class, not an afterthought.

- **Discrimination:** accuracy, macro-F1
- **Calibration:** ECE, Brier, NLL, reliability diagrams
- **Accuracy-vs-coverage curves** — at threshold *t*, what coverage and what error rate. The practical artifact. `evaluation.metrics.coverage_curve` computes it; `scripts/run_ablation.py` logs one per question per arm to W&B as a table + line plot (`_log_coverage_curves`), not bare tensors.
- **Latency**
### 7.1 The differential prediction (IEMOCAP)

IEMOCAP annotates **valence and arousal on the same utterances**, and they relate to acoustics differently: arousal is carried by loudness, pitch range and speech rate; valence is substantially lexical. The prediction is therefore **directional, not a single comparison**:

> Audio should beat text decisively on **arousal** calibration, and roughly tie on **valence**.

Same audio, same model, same annotation scheme, same raters — only the target dimension changes. Nearly every confound that could inflate audio performance (speaker leakage, session artifacts, channel cues, encoder capacity) is blind to which dimension is being predicted, so an effect present on arousal and absent on valence is very hard to explain away. A single audio-beats-text number always has alternative explanations; this design removes most of them.

On MELD (no dimensional labels) this experiment cannot be run; MELD validates the pipeline end-to-end and provides an out-of-distribution domain.

- **Stratified by ambiguity** — the thesis-critical view. Define ambiguity independently — **inter-rater disagreement on IEMOCAP** (per-rater votes are available); text-model entropy on MELD, where only majority labels ship. Then report calibration *within* strata. Aggregate numbers will bury the effect.

**Two baselines, doing different jobs:**

| Baseline | Purpose |
|---|---|
| Same architecture, text-only state | **Controlled** — isolates modality, holds everything else fixed |
| Whisper → **real Jev** (Vercel AI Gateway) | **Practical** — the actual text-state System One model, not a prompted stand-in |

The controlled baseline is wired in as three `ARMS` entries (`evaluation.baselines.TextOnlyBaseline`), one per loss regime (A/B/C), not one per encoder: `TextOnlyBaseline.mute_audio` forces `audio_present` False for every example, so the encoder axis is meaningless there — WavLM, Whisper, and the explicit-prosody channel would train bit-identically once audio never reaches the model. The loss axis is not meaningless: success criterion #1 diffs each encoder arm's calibration against a text-only arm trained under the *same* loss regime, so that a gain can be attributed to audio rather than conflated with whichever calibration treatment (Brier, temperature scaling) happened to be active.

**Modality isolation is enforced on both content and duration (owner Decision 3, 2026-09-19).** `mute_audio` only sets `audio_present` False; the actual isolation happens in `ProsodiaModel._encode_state`, which gates on it twice — once on the audio state's content (`torch.where`, replacing it with the learned `audio_absent` vector) and once on `mask` (`mask & audio_present`, excluding those positions from `IsolatedBranches`' attention entirely). The second gate is necessary because `audio_mask` is built from each example's real cached-feature length: without it, a muted row's constant content still receives a softmax weight in cross-attention that is a function of how many audio positions are valid — i.e. of the real, un-muted audio duration — a channel independent of, and measured larger than, the content leak the first gate alone closes. Enforced once, at `_encode_state`, rather than in `mute_audio` too: `_encode_state` is the only place both `audio_present` and the mask are already in hand, and every batch — however constructed — passes through it, so a single enforcement point covers `TextOnlyBaseline` and any other caller alike.

## 8. Interpretability study

1. **Prosodic interventions.** F0 flattening, pause lengthening, disfluency splicing via `praat-parselmouth`/PSOLA. Swept continuously (0/25/50/100%) for dose–response curves, with lexical content held exactly fixed. This is the signature experiment: a causal intervention on a physically meaningful, continuously parameterized input axis — an affordance text interpretability does not have.
2. **Entropy-neuron scan.** Ablate each late-layer unit; plot Δentropy against Δargmax-rate. Entropy neurons are the high-Δentropy, near-zero-Δargmax cluster.
3. **The intersection.** Are units responsive to F0 flattening the same units that are entropy neurons? If yes: **an entropy neuron with an identified physical cause.** Existing entropy-neuron work is all on autoregressive LMs where the cause is unknown.
4. **Linear readout probes.** Project `h` onto confidence-relevant directions directly — available because the readout stayed linear.

Tooling detail to be settled at implementation time.

## 9. Demo

Live conversational-state panel: audio in, question bank lighting up with calibrated probabilities in real time, plus a free-text box to pose a new question and watch it answer. Dynamic questions are what make this more than a static panel of lights.

## 10. Compute and environment

**Primary: the local machine** (`CL-MAC238`) — Apple M2 Pro, 16GB unified, **230GB disk free**, `uv` present. Chosen over `ssh mac` because the git repo lives here, so code and compute stay together with no sync layer, and 230GB accommodates the feature cache far more comfortably than the remote's 73GB. Capacity is not the constraint: the trainable model is 10-30M params over precomputed features (~480MB including Adam state) and WavLM-large extraction needs ~1.2GB.

**Overflow: `ssh mac`** — Apple M4 Pro, 24GB unified (~14GB free; `llama-server` holds ~4GB), 73GB disk, `uv 0.12.16`, Python 3.11, `ffmpeg`, `brew`, `git`. Available if a run needs more headroom. Because features are precomputed and all code is device-agnostic, moving a run there costs a file copy and nothing else.

**Fallback: Google Colab** (4TB Drive available). Write device-agnostic code (`mps` → `cuda` → `cpu`). Because features are precomputed, the training loop never touches the audio stack, so device differences are minimal.

The Mac is primary because the interp work is interactive and long-running — ablation sweeps and intervention curves want a kernel alive for hours, which Colab's session limit actively fights.

**Budget:**

| Item | Size |
|---|---|
| Feature cache (WavLM-large, 50Hz, fp16, 23h) | ~8.5 GB |
| Checkpoints (12 runs × ~3) | ~13 GB |
| **Total** | **~22 GB** of 73 GB free |

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
| 9 | **Label provenance assumed from a dataset summary** | Choice of thesis vehicle | Verify per label in the source paper *before* the design freezes (§4.0). An *audio*-model teacher is worse than a text one: it makes "audio beats text" circular |
| 8 | **Upstream pre-aggregated features** (e.g. CANDOR's 1s bins) are prosody-blind | Any third-party feature set | Compute prosody from raw audio; never trust packaged aggregates |

Trap 9 caught HarperValleyBank's valence labels only after the design had frozen; the single visible thread was `emotion` being stored as *softmax probabilities*, which humans do not produce. Traps 1–3, 5, 6 produce **false nulls**. Trap 4 produces a **false positive so strong it also suppresses the real effect** — a punctuation detector scoring 97% would show no response to F0 flattening, indistinguishable from a genuine null. Leaks must be closed by construction, not detected afterwards.

**Correction (I8, final-review fix, 2026-09-20):** trap 4's guardrail is only partially implemented. `MeldCorpus.build_context` (`src/prosodia/corpora/meld.py`) now strips sentence punctuation and speaker names — the two channels a punctuation-and-identity detector could exploit as the "false positive so strong it suppresses the real effect" described above — but context is still built from MELD's gold CSV transcripts, not streaming ASR output. Running real ASR over the corpus is a separate, multi-hour pass and out of scope for this fix; see §12's limitations entry below for the remaining deviation.

**Enforcement:** label provenance (tier) is a field in the dataset schema, so a Tier-2 row cannot physically enter a thesis-testing eval split.

## 12. Limitations

- **Small corpora.** MELD ~13k utterances; IEMOCAP ~10k across only 10 speakers. Bounds calibration resolution.
- **Performed, not spontaneous, affect** in both corpora. No claim of transfer to real traffic.
- **MELD's κ evidence is multimodal, not prosodic.** Annotators saw video; part of the 0.34→0.43 gain is facial. Motivation for the thesis, not evidence for it — our model receives audio only.
- **MELD's splits are speaker-shared, not speaker-disjoint (owner Decision 1, 2026-09-19).** The shipped train/dev/test CSVs are only dialogue-disjoint; the six recurring *Friends* leads appear in every split. Speaker identity is easier to recover from acoustics than from text, so this inflates the audio arm specifically — MELD results are pipeline validation only and cannot support an audio-improves-calibration claim. Not re-split (§4.1): a speaker-disjoint split would shred *Friends*' already character-dominated data volume and class balance, and MELD was always the build corpus, not the evidence corpus. `assert_speaker_disjoint` is deliberately not called on it. The real claim rests on IEMOCAP's leave-one-session-out protocol, which is genuinely speaker-disjoint.
- **Context is built from gold transcripts, not streaming ASR (owner-approved final-review fix I8, 2026-09-20).** Trap 4's guardrail (§11) calls for ASR-derived context with punctuation and segmentation stripped. `build_context` now strips sentence punctuation and speaker names — closing the affect-cue leak (punctuation is itself affect-bearing) and the identity leak (speaker names are a direct key into MELD's speaker-shared splits, Decision 1) — but the underlying text is still MELD's gold CSV transcripts. Running real ASR over the corpus is a separate, multi-hour pass, deliberately deferred rather than folded into this fix. Gold transcripts are cleaner (fewer errors, no ASR-specific artifacts) than what a deployed streaming-ASR context would contain, so this remains a residual gap between the guardrail's intent and the implementation, distinct from (and narrower than) the punctuation/identity leaks this fix closes.
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
2. **(IEMOCAP, headline)** Does the audio advantage appear on **arousal** and not on **valence**, as §7.1 predicts? The contrast is the claim; a uniform gain across both dimensions is a weaker and more confoundable result.
3. Does CE+Brier beat post-hoc temperature scaling — i.e. is calibration distributed or scalar?
4. Do prosodic interventions causally move output confidence, with a dose–response relationship?
5. Do entropy neurons exist in this readout, and do they coincide with F0-responsive units?

**A clean null on (1), (2) or (4) is a valid outcome** — provided the diagnostic arms (explicit-prosody channel, Whisper-vs-WavLM) can attribute it to a cause rather than leave it ambiguous. That attribution capability is why those arms are in scope.

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
