"""Dataset and collation.

Modality dropout (spec §5) yields three eval conditions — full / audio-only /
context-only — from one checkpoint, and simultaneously prevents the model from
learning to ignore audio whenever context is informative. It never drops both:
an example with no state at all carries no signal.
"""
from __future__ import annotations

import random
import sys
from typing import Any, Sequence

import torch
from torch.utils.data import Dataset

from prosodia.features import FeatureCache
from prosodia.questions import paraphrase, permute_candidates
from prosodia.schema import Example, QuestionSpec

# I2: `FeatureCache.__init__` does `mkdir(parents=True, exist_ok=True)`, so a
# wrong or partially-extracted cache path is CREATED rather than rejected --
# nothing at the cache layer distinguishes "this corpus genuinely has a
# handful of undecodable clips" from "extraction for this arm never
# finished." 0.98 is generous enough to absorb the former (MeldCorpus's own
# loader already skips its "handful of undecodable clips" before an example
# ever reaches here, so healthy caches should see ~0 drops in practice) while
# still catching the latter, which routinely drops far more than 2% of a
# corpus. Overridable per call site since a smaller pilot cache or a corpus
# with genuinely more decode failures may need a lower floor.
DEFAULT_MIN_CACHE_COVERAGE = 0.98


class ProsodiaDataset(Dataset):
    """Callers must invoke `set_epoch(epoch)` before each training epoch —
    the per-item RNG is seeded deterministically from `(rng_seed, epoch,
    idx)`, so without advancing the epoch every pass over the data would
    draw the exact same augmentation and modality-dropout choices."""

    def __init__(
        self,
        examples: Sequence[Example],
        specs: Sequence[QuestionSpec],
        cache: FeatureCache,
        rng_seed: int = 0,
        augment: bool = True,
        modality_dropout: float = 0.15,
        min_cache_coverage: float = DEFAULT_MIN_CACHE_COVERAGE,
    ) -> None:
        examples = list(examples)
        self.dropped_uids: list[str] = [e.uid for e in examples if e.uid not in cache]
        self.examples = [e for e in examples if e.uid in cache]
        self.n_total_examples = len(examples)
        self.n_dropped = len(self.dropped_uids)
        # No denominator to be "a fraction of" when the caller supplied zero
        # examples -- treat that as full (vacuous) coverage rather than 0/0.
        self.cache_coverage = (
            1.0 if self.n_total_examples == 0
            else (self.n_total_examples - self.n_dropped) / self.n_total_examples
        )

        # I2: a cache whose coverage silently determines the training set is
        # exactly the failure the final review flagged -- a half-finished
        # extraction for one arm quietly shrinks (and changes the CONTENTS
        # of) that arm's training set relative to every other arm in the
        # grid, and the grid then compares arms trained on different data
        # with nothing in the logs to say so. Below the floor: refuse to
        # proceed at all. Above it but still non-zero: proceed, but never
        # silently -- print an unmissable warning (this project's existing
        # idiom for "this needs a human's attention" -- see
        # `scripts/run_ablation.py`'s MELD speaker-leakage banner -- rather
        # than `warnings.warn`, which `-W error` would turn into a hard
        # failure for a condition this function is deliberately choosing to
        # tolerate).
        if self.n_dropped and self.cache_coverage < min_cache_coverage:
            sample = self.dropped_uids[:10]
            raise ValueError(
                f"feature cache at {cache.path} covers only "
                f"{self.cache_coverage:.1%} of the {self.n_total_examples} "
                f"examples supplied (missing {self.n_dropped}), below the "
                f"required floor of {min_cache_coverage:.1%}. This usually "
                "means the cache is wrong, empty, or a partially-finished "
                "extraction -- refusing to silently train on a shrunken "
                f"dataset. Sample of missing uids: {sample}"
            )
        if self.n_dropped:
            banner = "!" * 78
            sample = self.dropped_uids[:10]
            print(
                f"\n{banner}\nFEATURE-CACHE COVERAGE WARNING\n"
                f"{self.n_dropped} of {self.n_total_examples} examples "
                f"({1 - self.cache_coverage:.1%}) are missing from the "
                f"feature cache at {cache.path} and were DROPPED from this "
                "dataset. If this dataset is one arm of a multi-arm "
                "comparison, a different-sized training set silently voids "
                "that comparison even though loss will still descend and "
                f"metrics will still look plausible. Sample of missing "
                f"uids: {sample}\n{banner}\n",
                file=sys.stderr,
            )

        self.specs = list(specs)
        self.cache = cache
        self.augment = augment
        self.modality_dropout = modality_dropout
        self._seed = rng_seed
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Advance the RNG stream for a new pass over the data (see class
        docstring). Mirrors `DistributedSampler.set_epoch`."""
        self._epoch = epoch

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        ex = self.examples[idx]
        # Seeded from caller-controlled state only — (seed, epoch, idx) — so
        # the same triple reproduces the same draw in any process. Mixing in
        # global `random` entropy here would make the ablation grid's arms
        # differ by augmentation/dropout noise, not just by loss or encoder.
        rng = random.Random(hash((self._seed, self._epoch, idx)))

        audio_present, context_present = True, True
        if self.modality_dropout > 0 and rng.random() < self.modality_dropout:
            # drop exactly one modality, never both
            if rng.random() < 0.5:
                audio_present = False
            else:
                context_present = False

        questions: dict[str, Any] = {}
        targets: dict[str, Any] = {}
        for spec in self.specs:
            label = ex.labels.get(spec.key)
            if label is None:
                continue
            active = spec
            if self.augment:
                active = paraphrase(active, rng)
                gold = label.value if spec.qtype == "choice" else None
                active, _ = permute_candidates(active, rng, keep=gold)

            options = active.options
            if spec.qtype == "choice":
                # permute_candidates guarantees the gold option survives
                target = options.index(label.value)
            elif spec.qtype == "score":
                target = int(label.value)
            else:
                target = int(label.value)

            questions[spec.key] = {
                "instructions": active.instructions,
                "options": options,
                "qtype": spec.qtype,
            }
            targets[spec.key] = target

        return {
            "uid": ex.uid,
            "audio": self.cache.read(ex.uid),
            "audio_present": audio_present,
            "context": ex.context,
            "context_present": context_present,
            "questions": questions,
            "targets": targets,
            "speaker": ex.speaker,
        }


def collate_batch(items: Sequence[dict[str, Any]]) -> dict[str, Any]:
    lengths = [it["audio"].shape[0] for it in items]
    tmax, dim = max(lengths), items[0]["audio"].shape[1]
    audio = torch.zeros(len(items), tmax, dim)
    mask = torch.zeros(len(items), tmax, dtype=torch.bool)
    for i, it in enumerate(items):
        n = it["audio"].shape[0]
        audio[i, :n] = it["audio"]
        mask[i, :n] = True
    return {
        "uid": [it["uid"] for it in items],
        "audio": audio,
        "audio_mask": mask,
        "audio_present": torch.tensor([it["audio_present"] for it in items]),
        "context": [it["context"] for it in items],
        "context_present": torch.tensor([it["context_present"] for it in items]),
        "questions": [it["questions"] for it in items],
        "targets": [it["targets"] for it in items],
        "speaker": [it["speaker"] for it in items],
    }
