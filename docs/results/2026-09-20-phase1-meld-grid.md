# Phase 1 results — 12-arm MELD grid

Run completed 2026-09-20 07:06 on `ssh mac`. All 12 arms exit 0; WavLM,
Whisper and explicit-prosody caches each hold exactly 13,706 tensors.
Test split n = 2,610. Suite 141/141.

Spec: `docs/superpowers/specs/2026-09-19-prosodia-design.md`
Plan: `docs/superpowers/plans/2026-09-19-prosodia-phase1.md`

## Base-rate anchor (added post-hoc, corrected)

A constant predictor emitting the class marginal. **Fit on train, scored on
test** — an earlier version fit it on test, which forces ECE to 0 by
construction and is not a defensible anchor.

| question | majority acc | Brier | ECE |
|---|---|---|---|
| emotion (7-way) | 0.4812 | 0.7096 | 0.0097 |
| sentiment (3-way) | 0.4812 | 0.6286 | 0.0097 |
| is_negative (2-way) | 0.6808 | 0.4358 | 0.0243 |

**ECE cannot rank these arms.** Rank all 12 by ECE, rank them again by
Brier, and the two orderings are near-independent: Spearman rho = +0.147
(emotion), -0.294 (sentiment), -0.266 (is_negative). Not reversed —
unrelated.

The mechanism: calibration is a *marginal* property, so a constant
predictor emitting the marginal is near-perfectly calibrated while knowing
nothing about the input. On the two multi-class questions all 12 arms are
worse-calibrated than that constant (best arm ECE 0.0404 emotion, 0.0213
sentiment); on the binary question 7 of 12 beat it. Class imbalance makes
the degenerate solution look better on ECE *and* accuracy simultaneously.

Report Brier/NLL as headline; keep ECE as a diagnostic printed beside this
anchor.

## Brier (strictly proper; lower better)

| arm | emotion | sentiment | is_negative |
|---|---|---|---|
| base rate (train-fit) | 0.7096 | 0.6286 | 0.4358 |
| wavlm A-ce | 0.6944 | 0.5964 | 0.4054 |
| wavlm B-brier | 0.6858 | 0.5932 | 0.3963 |
| wavlm C-temp | 0.6935 | 0.5957 | 0.4054 |
| whisper A-ce | 0.6707 | 0.5673 | 0.3864 |
| whisper B-brier | 0.6722 | 0.5685 | 0.3838 |
| **whisper C-temp** | **0.6670** | **0.5603** | **0.3792** |
| prosody A-ce | 0.7177 | 0.6274 | 0.4239 |
| prosody B-brier | 0.7247 | 0.6281 | 0.4257 |
| prosody C-temp | 0.7170 | 0.6265 | 0.4237 |
| text_only A-ce | 0.7119 | 0.6197 | 0.4238 |
| text_only B-brier | 0.7119 | 0.6205 | 0.4247 |
| text_only C-temp | 0.7122 | 0.6195 | 0.4240 |

`whisper C-temp` wins all three. Against the corrected anchor the failures
concentrate on **emotion**, where both the prosody arms and the text_only
arms lose to a constant; on sentiment the prosody arms sit within 0.002 of
it (a wash). Only Whisper and WavLM clear the floor on every question.

## Accuracy / macro-F1 (Arm A-ce)

| encoder | emo acc | emo F1 | sent acc | sent F1 | neg acc | neg F1 |
|---|---|---|---|---|---|---|
| wavlm | 0.480 | 0.151 | 0.534 | 0.405 | 0.703 | 0.630 |
| whisper | 0.493 | 0.221 | 0.567 | 0.500 | 0.713 | 0.655 |
| prosody | 0.453 | 0.149 | 0.481 | 0.405 | 0.674 | 0.551 |
| text_only | 0.471 | 0.116 | 0.487 | 0.304 | 0.672 | 0.507 |
| majority | 0.481 | 0.093 | 0.481 | 0.217 | 0.681 | 0.405 |

Whisper (the *control*) beats WavLM (the *primary*) on every cell. Whisper's
encoder is supervised-ASR-trained and carries heavy lexical content; read
with the prosody arm's collapse, the economical reading is that the winning
signal on MELD is lexical, not prosodic.

Emotion accuracy brackets the 0.481 majority rate for every arm — macro-F1
is the only column where emotion shows signal.

## Exit criterion: Arm B vs Arm C (ECE delta, negative favors B)

| question | wavlm | whisper | prosody | text_only | B wins |
|---|---|---|---|---|---|
| emotion | −0.0161 | −0.0043 | −0.0074 | −0.0012 | 4/4 |
| sentiment | +0.0138 | +0.0107 | −0.0059 | +0.0025 | 1/4 |
| is_negative | −0.0107 | +0.0162 | +0.0053 | −0.0038 | 2/4 |

7/12 overall — a wash in aggregate, but it splits by label cardinality, not
by encoder. A single scalar temperature rescales a distribution uniformly
and cannot fix per-class miscalibration, so learned calibration earns its
keep as cardinality rises and a free scalar suffices where it does not.

(Magnitudes inherit the ECE caveat above; the cardinality *pattern* is the
durable part.)

## Jev's training strategy was not used

None of the 12 arms use RLCD. There is no reward model, no preference data,
no contrastive distillation. "RLCD" appears in exactly two places in the
tree -- a spec table cell and a comment in `train/losses.py` -- both reading
"RLCD stand-in". The three regimes are CE, CE+Brier, and post-hoc
temperature scaling.

What was borrowed from Jev is the **architecture shape** (state encoded
once, questions branching in structural isolation, linear readout per
question). Supervised proper-scoring training was substituted for the part
of Jev that actually produces its calibration. This grid therefore tests a
Jev-shaped *model*, not Jev's *method*.

## The explicit-prosody arm is a diagnostic, not a contender

Three scalars per frame at 50Hz: F0 (`librosa.pyin`), RMS energy, voicing
probability. Deliberately not Jev-shaped -- Jev's premise is a strong
pretrained representation read out linearly, and hand-built pitch contours
are the opposite. Its job is attribution (spec: "a clean null is a valid
outcome -- provided the diagnostic arms can attribute it to a cause"), and
it delivered one: hand-built prosodic descriptors carry essentially no
usable signal on MELD, which is what makes the Whisper result interpretable
rather than ambiguous.

## Scope limits on these numbers

1. **No arm sees the utterance's words** -- by design, not by defect. The
   point of an audio-native decision model is to avoid paying for ASR on the
   current turn. Whisper's win shows the linguistic content is reachable in
   latent space without decoding. `Example` has `audio_path` and
   `context` and no transcript field; `build_context` serializes only
   preceding turns (spec §11 trap 4). The text-only arm mutes audio, so it
   has dialogue history alone. The comparison run is therefore
   *audio + history vs history*, NOT audio vs transcript. Matches the spec
   as written; the phrase "text-only" is narrower than it sounds.
2. **MELD splits are speaker-shared** — pipeline validation only; cannot
   support an audio-improves-calibration claim (spec Decision 1).
3. **Emotion sits at the majority baseline** for every arm.
4. 30s audio cap; gold transcripts rather than streaming ASR in context.

## Recommended next steps

- Switch headline reporting to Brier/NLL; print ECE beside the base-rate
  anchor. Reporting change only — both metrics are already logged.
- Swap the primary encoder to an ASR-pretrained multilingual model (XLS-R
  or MMS). Whisper beating WavLM on every cell is direct evidence, and it
  keeps the transcript out of the loop.
- Add 3 transcript arms as a comparison floor (not as a shipped model).
  ~20 min/arm.
- Decide on RLCD: implement it, or state plainly that the project tests
  Jev's architecture and not its method.
- IEMOCAP: leave-one-session-out is genuinely speaker-disjoint and carries
  §7.1's arousal-vs-valence contrast. MELD cannot answer the thesis.

---

## Addendum: EmotionLines stratification (re-analysis, no retraining)

EmotionLines preserves the raw 5-annotator vote string per utterance and
joins to **2610/2610** MELD test rows on normalised speaker + text. Those
annotators saw text only, so vote entropy `H(q_text)` measures transcript
ambiguity. See `analysis/README.md`.

Pipeline was validated first: all 24 arm x question aggregate metrics
recomputed from dumped per-uid predictions reproduce the published grid to
4 decimals (`analysis/verify.py`).

Split at median H: where the 5 readers agree, MELD's AV label agrees 95.8%
of the time; where they disagree, 58.9%.

### What the raw deltas showed (and why they mislead)

Brier vs the text-only baseline looked like the thesis: prosody +0.0360 on
the clear stratum (worse than text) and -0.0776 on the misleading stratum
(better). Whisper showed the opposite profile.

**This was overstated.** Two problems:

1. **Base rates differ enormously between strata** -- emotion is 59.4%
   neutral in CLEAR and 29.7% in AMBIG. A flatter model looks good on a
   flatter stratum regardless of what it hears.
2. The MISLEADING stratum is partly circular: it is *defined* by text
   readers disagreeing with the label being scored against.

### Base-rate-controlled result (Brier skill score vs each stratum's own marginal)

| arm | emo CLEAR | emo AMBIG | sent CLEAR | sent AMBIG |
|---|---|---|---|---|
| whisper C-temp | **+0.048** | -0.029 | **+0.087** | -0.024 |
| wavlm B-brier | -0.007 | -0.023 | -0.005 | -0.026 |
| prosody B-brier | -0.110 | -0.025 | -0.083 | -0.060 |
| text_only B-brier | -0.050 | -0.056 | -0.040 | -0.091 |

**Prosody never beats a constant anywhere.** Only whisper clears its
stratum base rate, and only on the CLEAR stratum (plus is_negative).

What survives: the *direction* of the dissociation. Prosody is relatively
better on AMBIG (+0.085 skill), whisper relatively better on CLEAR
(-0.078). The encoders' errors are distributed differently. That is NOT
sufficient to claim prosody carries information a transcript destroys.

## Addendum: absolute model quality

Emotion weighted-F1, best arm (whisper A-ce): **0.437**, against roughly
0.57-0.60 for published text-only MELD baselines. Per class:

| class | train n | test F1 |
|---|---|---|
| neutral | 4710 | 0.679 |
| surprise | 1205 | 0.314 |
| anger | 1109 | 0.311 |
| joy | 1743 | 0.216 |
| sadness | 683 | **0.027** |
| disgust | 271 | **0.000** |
| fear | 268 | **0.000** |

Fear/disgust at ~270 could be scarcity. **Sadness at 683 cannot** -- head
classes fine and everything else at the floor is the signature of
unweighted CE collapsing, not of a representation that cannot hold the
concept. Arm D (`<enc>__D-balanced`, inverse-frequency class weights, in
`DIAGNOSTIC_ARMS`) is the diagnostic for this.

## The architecture was never tested

All 12 grid arms share the same architecture; only encoder and loss vary.
There is no non-Jev-shaped control, so nothing in this grid bears on
whether the Jev shape helps. Separately, the trainable stack is **3.59M
parameters** trained from scratch on ~10k clips -- Jev's premise is a
pretrained backbone with the decode loop replaced by a readout, and
NanoJev kept a 0.6B Qwen3 backbone. We reproduced the shape without the
substance.

## Arm D verdict: the collapse had TWO causes, not one

Inverse-frequency class weighting (6.9x sadness:neutral, 17.6x
fear:neutral), three encoders. Macro-F1 on emotion: whisper 0.221 ->
0.198, prosody 0.149 -> 0.147, wavlm 0.151 -> 0.163. Flat to slightly
down -- weighting did NOT fix the collapse.

Per class it is not flat at all. Three distinct behaviours:

**Optimization-limited (recovered for real).** joy, 1743 train examples,
was being suppressed by unweighted CE despite being the 2nd commonest
class:

| encoder | A-ce F1 | D F1 | recall A -> D |
|---|---|---|---|
| whisper | 0.216 | 0.310 | 0.152 -> 0.328 |
| prosody | 0.149 | 0.267 | 0.112 -> 0.391 |
| wavlm | 0.045 | 0.188 | 0.025 -> 0.169 |

**Representation-limited (did NOT recover).** sadness, 683 examples, given
a 6.9x upweight:

| encoder | A-ce F1 | D F1 | D recall | D precision |
|---|---|---|---|---|
| whisper | 0.027 | 0.018 | 0.010 | 0.182 |
| prosody | 0.072 | 0.000 | 0.000 | - |
| wavlm | 0.000 | 0.046 | 0.029 | 0.113 |

Sadness recall never exceeds 2.9% on any encoder. The model was told
loudly to predict sadness and still would not. With 683 training examples
that is not scarcity and not an objective problem.

**Recovered in name only.** fear and disgust move off 0.000 to F1
0.03-0.08, but at precision **0.018-0.051** -- 95-97% of those predictions
are wrong. This is the recall-bought-at-zero-precision case; do not read it
as recovery. (The `<-- recovered` flag in `analysis/perclass.py` thresholds
on F1 delta alone and mislabels these; read the precision column.)

**Collateral damage.** surprise (1205 examples) fell 0.314 -> 0.132 on
whisper; neutral fell ~0.11-0.17 everywhere. Aggressive weighting damaged
classes that were working.

### Consequence

The tail collapse is not primarily an optimization artifact. Part of it was
(joy), and that part was cheap to close. The rest -- sadness especially --
survived a 6.9x incentive to change, which is direct evidence that the
frozen-features + 3.59M-parameter stack cannot support the distinction.

That converts the backbone from an assumption into an evidence-backed next
step. Caveat: one seed, one weighting scheme (full inverse frequency, which
is aggressive). "Reweighting does not help" is established for THIS scheme;
"the representation cannot hold sadness" is the inference from recall never
moving.
