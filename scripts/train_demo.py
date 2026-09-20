"""Train the demo model: seven typed questions over one audio state.

Reuses `run_arm` and `_build_loaders` from run_ablation rather than
reimplementing a training loop -- a second, independently-written loop is how
this project previously reintroduced a target/logit alignment bug.

Not part of the ablation grid: `ARMS` stays exactly the published twelve.
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import run_ablation as RA  # noqa: E402
from prosodia.config import RunConfig  # noqa: E402
from prosodia.corpora.meld import MeldCorpus  # noqa: E402
from prosodia.features import FeatureCache  # noqa: E402
from prosodia.labels.bank import attach_demo_labels, demo_question_specs  # noqa: E402
from prosodia.schema import assert_thesis_safe  # noqa: E402

_CSV = {"train": "train_sent_emo.csv", "dev": "dev_sent_emo.csv",
        "test": "test_sent_emo.csv"}


def load_transcripts(corpus_root: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for split, fname in _CSV.items():
        path = corpus_root / "MELD.Raw" / fname
        for r in csv.DictReader(open(path, encoding="utf-8", errors="replace")):
            uid = f"meld-{split}-{r['Dialogue_ID']}-{r['Utterance_ID']}"
            out[uid] = r["Utterance"]
    return out


_TIME_RE = re.compile(r"(\d+):(\d+):(\d+)[,.](\d+)")


def _to_seconds(stamp: str) -> float | None:
    m = _TIME_RE.match(stamp.strip())
    if not m:
        return None
    h, mi, s, ms = (int(x) for x in m.groups())
    return h * 3600 + mi * 60 + s + ms / 1000.0


def load_declared_seconds(corpus_root: Path) -> dict[str, float]:
    """MELD's own declared utterance spans, used to reject impossible ones."""
    out: dict[str, float] = {}
    for split, fname in _CSV.items():
        path = corpus_root / "MELD.Raw" / fname
        for r in csv.DictReader(open(path, encoding="utf-8", errors="replace")):
            a, b = _to_seconds(r["StartTime"]), _to_seconds(r["EndTime"])
            if a is None or b is None:
                continue
            out[f"meld-{split}-{r['Dialogue_ID']}-{r['Utterance_ID']}"] = b - a
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-root", type=Path, required=True)
    ap.add_argument("--cache-root", type=Path, required=True)
    ap.add_argument("--ckpt-root", type=Path, required=True)
    ap.add_argument("--encoder", default="whisper")
    ap.add_argument("--name", default="demo__whisper")
    ap.add_argument("--epochs", type=int, default=20)
    args = ap.parse_args()

    corpus = MeldCorpus(args.corpus_root)
    specs = corpus.question_specs() + demo_question_specs()
    splits = {s: list(corpus.iter_examples(s)) for s in ("train", "dev", "test")}

    # Acoustic statistics are read from the EXPLICIT-PROSODY cache regardless
    # of which encoder the model trains on -- F0/RMS/voicing is what they are
    # computed from. Passing the whisper cache here would silently compute
    # nonsense from the wrong feature space.
    splits = attach_demo_labels(
        splits, FeatureCache(args.cache_root / "prosody"),
        load_transcripts(args.corpus_root),
        declared_seconds=load_declared_seconds(args.corpus_root))

    assert_thesis_safe(splits["test"], [s.key for s in specs])
    RA._print_meld_warning_if_applicable(corpus)

    cfg = RunConfig(name=args.name, encoder=args.encoder,
                    brier_weight=0.5, epochs=args.epochs)
    cache = FeatureCache(args.cache_root / cfg.encoder)
    loaders = RA._build_loaders(splits, specs, cache, cfg)
    in_dim = next(iter(loaders["train"]))["audio"].shape[-1]

    for name, exs in splits.items():
        counts = {s.key: sum(1 for e in exs if s.key in e.labels) for s in specs}
        print(f"  {name}: {len(exs)} examples, label coverage {counts}")

    stats = RA.run_arm(cfg, loaders, in_dim, ckpt_root=args.ckpt_root, run=None)
    for q, m in stats.items():
        print(f"{q:<16} acc={m['accuracy']:.4f} macroF1={m['macro_f1']:.4f} "
              f"ece={m['ece']:.4f} brier={m['brier']:.4f}")


if __name__ == "__main__":
    main()
