"""Run the 9-arm grid. Resumes from checkpoints, logs every arm to W&B.

Arm C (`cfg.temperature_scale`) is not a separate training run: its
training is identical to Arm A (`brier_weight=0.0`). Only the post-training
scoring step differs -- see `_calibrated_test_metrics`, which fits a
`TemperatureScaler` on the DEV split's logits and applies it to the TEST
split's logits. Fitting on TEST instead would leak the test set into the
calibration step and invalidate the Arm B vs Arm C comparison this whole
grid exists to produce.

MELD speaker leakage (owner Decision 1, 2026-09-19): MELD's shipped splits
are dialogue-disjoint, not speaker-disjoint -- the six *Friends* leads
appear in train, dev, and test. Speaker identity is far easier to recover
from acoustics than from text, so this inflates the audio arm specifically.
We do NOT re-split MELD (it was always the build corpus, not the evidence;
a speaker-disjoint split would shred both data volume and class balance) or
call `assert_speaker_disjoint` on it (it would fail, by design). Instead
`_meld_speaker_leakage_warning` below prints an unmissable startup warning
and the same text rides along in every arm's W&B config, so it travels with
the numbers instead of living only in a terminal someone scrolled past.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from prosodia.config import RunConfig
from prosodia.corpora.meld import MeldCorpus
from prosodia.data import ProsodiaDataset, collate_batch
from prosodia.evaluation.baselines import ARMS
from prosodia.evaluation.metrics import compute_metrics
from prosodia.features import FeatureCache
from prosodia.model.prosodia import ProsodiaModel
from prosodia.schema import assert_thesis_safe
from prosodia.train.calibrate import TemperatureScaler
from prosodia.train.loop import evaluate, save_checkpoint, train_one_epoch

MELD_SPEAKER_LEAKAGE_WARNING = (
    "MELD splits are dialogue-disjoint, NOT speaker-disjoint: the six "
    "recurring Friends leads appear in train, dev, AND test. Speaker "
    "identity is far easier to recover from acoustics than from text, so "
    "speaker leakage inflates the AUDIO arm specifically -- exactly the "
    "confound that would manufacture an 'audio beats text on calibration' "
    "finding. MELD results are PIPELINE VALIDATION ONLY and cannot support "
    "an audio-improves-calibration claim; that claim rests on IEMOCAP's "
    "leave-one-session-out protocol, which is genuinely speaker-disjoint."
)


def _meld_speaker_leakage_warning(corpus: Any) -> str | None:
    """Returns the leakage warning text when `corpus` is MELD, else None.

    Factored out (rather than inlined in `__main__`) so it is testable
    without wandb, network, or the real corpus -- a stub object with the
    right type is enough.
    """
    if isinstance(corpus, MeldCorpus):
        return MELD_SPEAKER_LEAKAGE_WARNING
    return None


def _print_meld_warning_if_applicable(corpus: Any) -> None:
    warning = _meld_speaker_leakage_warning(corpus)
    if warning is None:
        return
    banner = "!" * 78
    print(f"\n{banner}\nMELD SPEAKER-LEAKAGE WARNING\n{warning}\n{banner}\n",
          file=sys.stderr)


def _calibrated_test_metrics(
    model: Any, dev_loader: DataLoader, test_loader: DataLoader,
) -> dict[str, dict[str, float]]:
    """Arm C: fit a per-question `TemperatureScaler` on the DEV split's
    logits/targets, then transform the TEST split's logits before scoring.

    `evaluate(..., return_logits=True)` hands back the exact per-question
    logits/targets it already collected and aligned, so this reuses that one
    corrected path instead of re-deriving it -- see its docstring for why a
    second, independent collection loop would be risky here.
    """
    _, dev_logits, dev_targets = evaluate(model, dev_loader, return_logits=True)
    _, test_logits, test_targets = evaluate(model, test_loader, return_logits=True)

    results: dict[str, dict[str, float]] = {}
    for key, logits in test_logits.items():
        scaler = TemperatureScaler().fit(dev_logits[key], dev_targets[key])
        scaled = scaler.transform(logits)
        probs = torch.softmax(scaled, dim=-1)
        results[key] = compute_metrics(probs, test_targets[key])
    return results


def run_arm(
    cfg: RunConfig,
    loaders: dict[str, DataLoader],
    in_dim: int,
    ckpt_root: Path,
    run: Any = None,
) -> dict[str, dict[str, float]]:
    """Trains one arm end-to-end and returns its test-set metrics.

    Reads `cfg.temperature_scale` (Arm C): training is identical to Arm A,
    and only the final scoring step branches, via `_calibrated_test_metrics`.
    """
    torch.manual_seed(cfg.seed)
    model = ProsodiaModel(in_dim=in_dim, d_model=cfg.d_model,
                          state_layers=cfg.state_layers,
                          branch_layers=cfg.branch_layers,
                          n_heads=cfg.n_heads, stride=cfg.stride)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                            weight_decay=cfg.weight_decay)

    for epoch in range(cfg.epochs):
        train_stats = train_one_epoch(model, loaders["train"], opt, cfg, epoch)
        dev_stats = evaluate(model, loaders["dev"])
        if run is not None:
            run.log({"epoch": epoch, **{f"train/{k}": v for k, v in train_stats.items()},
                     **{f"dev/{q}/{m}": v for q, mm in dev_stats.items()
                        for m, v in mm.items()}})
        save_checkpoint(ckpt_root / cfg.name / f"epoch{epoch}.pt",
                        model, opt, epoch, cfg)

    if cfg.temperature_scale:
        test_stats = _calibrated_test_metrics(model, loaders["dev"], loaders["test"])
    else:
        test_stats = evaluate(model, loaders["test"])

    if run is not None:
        run.log({f"test/{q}/{m}": v for q, mm in test_stats.items()
                 for m, v in mm.items()})
    return test_stats


def _build_loaders(
    splits: dict[str, list], specs, cache: FeatureCache, cfg: RunConfig,
) -> dict[str, DataLoader]:
    return {
        name: DataLoader(
            ProsodiaDataset(exs, specs, cache, rng_seed=cfg.seed,
                            augment=(name == "train"),
                            modality_dropout=cfg.modality_dropout if name == "train" else 0.0),
            batch_size=cfg.batch_size, shuffle=(name == "train"),
            collate_fn=collate_batch)
        for name, exs in splits.items()
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-root", type=Path, required=True)
    ap.add_argument("--cache-root", type=Path, required=True)
    ap.add_argument("--ckpt-root", type=Path, default=Path.home() / "prosodia-ckpt")
    ap.add_argument("--only", nargs="*", help="arm names to run; default all")
    args = ap.parse_args()

    corpus = MeldCorpus(args.corpus_root)
    specs = corpus.question_specs()
    keys = [s.key for s in specs]

    splits = {s: list(corpus.iter_examples(s)) for s in ("train", "dev", "test")}
    # Guardrail: no model-output labels may back a thesis-testing split.
    assert_thesis_safe(splits["test"], keys)
    # NOT `assert_speaker_disjoint(splits)` here -- MELD would fail it, by
    # design (Decision 1). Warn instead; see the module docstring.
    _print_meld_warning_if_applicable(corpus)
    meld_warning = _meld_speaker_leakage_warning(corpus)

    import wandb  # lazy: keeps this script importable (e.g. by tests) offline

    for cfg in ARMS:
        if args.only and cfg.name not in args.only:
            continue
        cache = FeatureCache(args.cache_root / cfg.encoder)
        loaders = _build_loaders(splits, specs, cache, cfg)

        in_dim = next(iter(loaders["train"]))["audio"].shape[-1]

        wandb_config = cfg.as_dict()
        if meld_warning is not None:
            # Rides along with every arm's config so the caveat travels
            # with the numbers instead of living only in a terminal
            # someone scrolled past.
            wandb_config["meld_speaker_leakage_warning"] = meld_warning

        run = wandb.init(project=cfg.wandb_project, name=cfg.name,
                         config=wandb_config, reinit=True)
        test_stats = run_arm(cfg, loaders, in_dim, ckpt_root=args.ckpt_root, run=run)
        print(cfg.name, test_stats)
        run.finish()
