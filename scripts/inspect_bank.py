"""Feasibility check: what does the demo question bank actually look like?"""
import argparse, collections, csv, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prosodia.corpora.meld import MeldCorpus          # noqa: E402
from prosodia.features import FeatureCache             # noqa: E402
from prosodia.labels.bank import attach_demo_labels, demo_question_specs  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--corpus-root", type=Path, required=True)
ap.add_argument("--cache-root", type=Path, required=True)
a = ap.parse_args()

corpus = MeldCorpus(a.corpus_root)
splits = {s: list(corpus.iter_examples(s)) for s in ("train", "dev", "test")}

transcripts = {}
for split, fname in (("train", "train_sent_emo.csv"), ("dev", "dev_sent_emo.csv"),
                     ("test", "test_sent_emo.csv")):
    p = a.corpus_root / "MELD.Raw" / fname
    for r in csv.DictReader(open(p, encoding="utf-8", errors="replace")):
        transcripts[f"meld-{split}-{r['Dialogue_ID']}-{r['Utterance_ID']}"] = r["Utterance"]

cache = FeatureCache(a.cache_root / "prosody")
out = attach_demo_labels(splits, cache, transcripts)

keys = [s.key for s in corpus.question_specs()] + [s.key for s in demo_question_specs()]
print(f"{'question':<18}" + "".join(f"{s:>22}" for s in ("train", "dev", "test")))
print("-" * 84)
for k in keys:
    cells = []
    for s in ("train", "dev", "test"):
        n = sum(1 for ex in out[s] if k in ex.labels)
        cells.append(f"{n:>9} ({n/len(out[s])*100:4.1f}%)")
    print(f"  {k:<16}" + "".join(f"{c:>22}" for c in cells))

print("\nclass balance on TEST (the demo's own answers):")
for k in ("speaker", "pitch_direction", "speaking_rate", "loudness"):
    c = collections.Counter(ex.labels[k].value for ex in out["test"] if k in ex.labels)
    tot = sum(c.values())
    if not tot:
        print(f"  {k:<16} EMPTY"); continue
    parts = "  ".join(f"{v}={n} ({n/tot*100:.0f}%)" for v, n in sorted(c.items(), key=lambda kv: str(kv[0])))
    print(f"  {k:<16} n={tot:<5} {parts}")
