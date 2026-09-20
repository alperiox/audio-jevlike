# EmotionLines ambiguity stratification

Re-analysis of the Phase 1 checkpoints. No retraining.

MELD ships one aggregated label per utterance. **EmotionLines** — which MELD
was built from — preserves the raw 5-annotator vote string (`"2000030"` over
`[neutral, joy, sadness, fear, anger, surprise, disgust]`). Joining on
normalised speaker + utterance text matches **2610/2610** MELD test rows.

Those five annotators saw **text only**; MELD's three re-annotators had
audio-visual. So the vote distribution `q_text` measures how ambiguous an
utterance is *from the transcript alone* — the ambiguity stratifier spec
§7.1 asks for and the project never had.

## Order of operations

1. `scripts/dump_predictions.py` (on the run host) — per-uid logits+targets
   for the 8 trained arms. Targets travel with the logits because
   `permute_candidates` randomises option order per example, so logit index
   is not a class label.
2. `verify.py` — recomputes aggregate test metrics from the dumps and checks
   them against the published grid. All 24 arm×question cells reproduce to
   4 decimals. Run this before trusting anything downstream.
3. `build_qtext.py` — builds and characterises `q_text`.
4. `stratify.py` — per-stratum Brier vs the text-only baseline.
5. `interaction.py` — the actual test: difference-in-differences with a
   bootstrap CI. Two separately-significant strata do not establish that
   they differ from each other.

## Two stratifiers, only one of them clean

- **H-split** (entropy of `q_text`, median split) — defined without
  reference to the MELD label. Use this one.
- **MISLEADING vs SUFFICIENT** (does `argmax q_text` match the MELD AV
  label) — partly circular: the stratum is *defined* by text readers
  disagreeing with the label being scored against, which handicaps any
  lexical model there by construction. Reported as corroboration only.

The headline interaction is significant under the clean stratifier too.
