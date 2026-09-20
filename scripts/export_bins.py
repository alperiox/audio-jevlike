"""Persist the TRAIN-fitted tertile boundaries next to a checkpoint.

The live demo has to bin a fresh recording against the same thresholds the
model was trained against. Refitting on live audio would move the goalposts
per recording and make "the model agrees with the truth" unfalsifiable.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prosodia.corpora.meld import MeldCorpus  # noqa: E402
from prosodia.features import FeatureCache  # noqa: E402
from prosodia.labels.acoustic import (  # noqa: E402
    LOUDNESS_NAMES, PITCH_NAMES, RATE_NAMES, TertileBinner, mean_loudness_db,
    pitch_slope_semitones_per_second, speaking_rate_wps,
)
from prosodia.labels.bank import timing_is_plausible  # noqa: E402
from train_demo import load_declared_seconds, load_transcripts  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--corpus-root", type=Path, required=True)
ap.add_argument("--cache-root", type=Path, required=True)
ap.add_argument("--out", type=Path, required=True)
a = ap.parse_args()

corpus = MeldCorpus(a.corpus_root)
cache = FeatureCache(a.cache_root / "prosody")
transcripts = load_transcripts(a.corpus_root)
declared = load_declared_seconds(a.corpus_root)

slopes, rates, louds = [], [], []
for ex in corpus.iter_examples("train"):
    if ex.uid not in cache:
        continue
    if not timing_is_plausible(declared.get(ex.uid), transcripts.get(ex.uid, "")):
        continue
    p = cache.read(ex.uid)
    s = pitch_slope_semitones_per_second(p)
    if s is not None:
        slopes.append(s)
    r = speaking_rate_wps(p, transcripts.get(ex.uid, ""))
    if r is not None:
        rates.append(r)
    louds.append(mean_loudness_db(p))

out = {}
for key, vals, names in (("pitch", slopes, PITCH_NAMES),
                         ("rate", rates, RATE_NAMES),
                         ("loud", louds, LOUDNESS_NAMES)):
    b = TertileBinner.fit(vals, names)
    out[key] = [b.low, b.high]
    print(f"  {key:<6} n={len(vals):<6} low={b.low:.4f} high={b.high:.4f}")

a.out.parent.mkdir(parents=True, exist_ok=True)
a.out.write_text(json.dumps(out))
print(f"wrote {a.out}")
