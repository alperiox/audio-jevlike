"""Run the 12-arm grid (9-arm encoder ablation + 3-arm text-only baseline).
Resumes from checkpoints, logs every arm to W&B.

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
from typing import Any, Sequence

import torch
from torch.utils.data import DataLoader

from prosodia.config import RunConfig
from prosodia.corpora.meld import MeldCorpus
from prosodia.data import ProsodiaDataset, collate_batch
from prosodia.evaluation.baselines import ARMS, TextOnlyBaseline
from prosodia.evaluation.metrics import compute_metrics, coverage_curve
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


def _cache_uid_coverage(examples: Sequence[Any], cache: FeatureCache) -> set[str]:
    return {ex.uid for ex in examples if ex.uid in cache}


def assert_uniform_cache_coverage(
    examples: Sequence[Any], encoders: Sequence[str], cache_root: Path,
) -> None:
    """I2, the cross-arm half of the fix: `ProsodiaDataset`'s own coverage
    floor (see `data.py`) protects any ONE arm from silently training on a
    badly incomplete cache, but says nothing about whether two arms are
    training on the SAME data -- and a comparison between arms trained on
    different data is void even when each arm's own cache individually
    clears the floor (e.g. two 99%-covered caches that are missing a
    *different* 1%). This is the check that actually protects the grid's
    comparison: it fails loudly, before any arm trains, if the encoder
    caches this run will use do not all cover the identical uid set.

    This has been more load-bearing since the text-only baseline landed
    (Decision 2): those arms borrow `TEXT_ONLY_CACHE_ENCODER`'s cache
    purely for shape/`in_dim` (`evaluation/baselines.py`), but cache
    COVERAGE still gates which examples their dataset contains at all --
    so a partial WavLM cache silently changes the text-only baseline's
    training set relative to the very audio arms it exists to control for.

    Deliberately takes `encoders` (the set actually in play for this run,
    honoring `--only`) rather than re-deriving it from `ARMS`, and takes
    `examples` as a flat sequence rather than the `{split: [...]}` dict --
    the check is about the UNION of uids each cache could be asked to
    supply across train/dev/test, not about any one split.
    """
    if len(encoders) <= 1:
        # A single encoder in play (e.g. `--only wavlm__A-ce`) has nothing
        # to be inconsistent WITH; only compare when there's more than one.
        return

    coverage: dict[str, set[str]] = {
        enc: _cache_uid_coverage(examples, FeatureCache(cache_root / enc))
        for enc in encoders
    }
    reference = coverage[encoders[0]]
    if all(cov == reference for cov in coverage.values()):
        return

    union = set().union(*coverage.values())
    lines = [
        f"encoder caches under {cache_root} do NOT all cover the same "
        "examples -- the ablation grid would compare arms TRAINED ON "
        "DIFFERENT DATA, silently voiding the comparison. Coverage per "
        "encoder:",
    ]
    for enc in encoders:
        cov = coverage[enc]
        missing = sorted(union - cov)
        lines.append(f"  {enc!r}: {len(cov)}/{len(union)} examples cached")
        if missing:
            lines.append(f"    missing (sample): {missing[:10]}")
    raise ValueError("\n".join(lines))


def _calibrated_test_metrics(
    model: Any, dev_loader: DataLoader, test_loader: DataLoader,
    return_probs: bool = False,
) -> dict[str, dict[str, float]] | tuple[
    dict[str, dict[str, float]], dict[str, torch.Tensor], dict[str, torch.Tensor]
]:
    """Arm C: fit a per-question `TemperatureScaler` on the DEV split's
    logits/targets, then transform the TEST split's logits before scoring.

    `evaluate(..., return_logits=True)` hands back the exact per-question
    logits/targets it already collected and aligned, so this reuses that one
    corrected path instead of re-deriving it -- see its docstring for why a
    second, independent collection loop would be risky here.

    `return_probs`, if True, additionally returns the calibrated (post
    temperature-scaling) per-question TEST probabilities and targets --
    exactly what `coverage_curve` needs, and exactly what Arm C actually
    reports, so its coverage curve reflects the calibrated probabilities
    rather than the raw ones.
    """
    _, dev_logits, dev_targets = evaluate(model, dev_loader, return_logits=True)
    _, test_logits, test_targets = evaluate(model, test_loader, return_logits=True)

    results: dict[str, dict[str, float]] = {}
    probs_by_q: dict[str, torch.Tensor] = {}
    for key, logits in test_logits.items():
        scaler = TemperatureScaler().fit(dev_logits[key], dev_targets[key])
        scaled = scaler.transform(logits)
        probs = torch.softmax(scaled, dim=-1)
        results[key] = compute_metrics(probs, test_targets[key])
        probs_by_q[key] = probs
    if return_probs:
        return results, probs_by_q, test_targets
    return results


def _plain_test_metrics(
    model: Any, test_loader: DataLoader, return_probs: bool = False,
) -> dict[str, dict[str, float]] | tuple[
    dict[str, dict[str, float]], dict[str, torch.Tensor], dict[str, torch.Tensor]
]:
    """Arms A and B: plain `evaluate()`, no post-hoc calibration step.

    `return_probs`, if True, additionally returns the per-question TEST
    probabilities/targets `evaluate(..., return_logits=True)` already
    collected, softmax'd -- the same shape `_calibrated_test_metrics`
    returns, so `run_arm` can log coverage curves identically regardless of
    which branch produced the arm's test metrics.
    """
    results, logits_out, targets_out = evaluate(model, test_loader, return_logits=True)
    if return_probs:
        probs_by_q = {k: torch.softmax(v, dim=-1) for k, v in logits_out.items()}
        return results, probs_by_q, targets_out
    return results


def _log_coverage_curves(
    run: Any, probs_by_q: dict[str, torch.Tensor], targets_by_q: dict[str, torch.Tensor],
    prefix: str = "test",
) -> None:
    """Logs each question's accuracy-vs-coverage curve as a W&B table + line
    plot -- spec §7 calls this "the practical artifact": at confidence
    threshold t, what fraction of traffic is automated and at what error
    rate. Logged as a plotted table, not three bare tensors, so it is
    actually readable later rather than requiring a caller to know
    `coverage_curve`'s tuple order to make sense of it.
    """
    import wandb

    for key, probs in probs_by_q.items():
        targets = targets_by_q[key]
        thresholds, coverage, error = coverage_curve(probs, targets)
        table = wandb.Table(
            columns=["threshold", "coverage", "error_rate"],
            data=[[t, c, e] for t, c, e in
                  zip(thresholds.tolist(), coverage.tolist(), error.tolist())],
        )
        run.log({
            f"{prefix}/{key}/coverage_curve": wandb.plot.line(
                table, "coverage", "error_rate",
                title=f"{key}: error rate vs. coverage",
            ),
        })


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
        test_stats, test_probs, test_targets_by_q = _calibrated_test_metrics(
            model, loaders["dev"], loaders["test"], return_probs=True)
    else:
        test_stats, test_probs, test_targets_by_q = _plain_test_metrics(
            model, loaders["test"], return_probs=True)

    if run is not None:
        run.log({f"test/{q}/{m}": v for q, mm in test_stats.items()
                 for m, v in mm.items()})
        _log_coverage_curves(run, test_probs, test_targets_by_q)
    return test_stats


def _build_loaders(
    splits: dict[str, list], specs, cache: FeatureCache, cfg: RunConfig,
) -> dict[str, DataLoader]:
    """cfg.text_only (the controlled text-only baseline, Decision 2) wraps
    `collate_batch` with `TextOnlyBaseline.mute_audio` so every batch's
    `audio_present` is forced False after collation -- same architecture,
    same training, same data, audio permanently absent. This happens at
    collate time (not by editing `ProsodiaDataset`) so the same cached
    audio tensors, augmentation, and modality-dropout logic are shared with
    every other arm; only the final audio_present flag differs.
    """
    collate_fn = collate_batch
    if cfg.text_only:
        def collate_fn(items, _collate=collate_batch):
            return TextOnlyBaseline.mute_audio(_collate(items))

    return {
        name: DataLoader(
            ProsodiaDataset(exs, specs, cache, rng_seed=cfg.seed,
                            augment=(name == "train"),
                            modality_dropout=cfg.modality_dropout if name == "train" else 0.0),
            batch_size=cfg.batch_size, shuffle=(name == "train"),
            collate_fn=collate_fn)
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

    # I2, cross-arm half: fail loudly, before ANY arm trains, if the
    # encoder caches this run will actually use disagree on which examples
    # they cover. Checked over the union of every split (train/dev/test)
    # since ProsodiaDataset is built per-split per-arm and any of the three
    # could be the one that diverges.
    arms_to_run = [cfg for cfg in ARMS if not args.only or cfg.name in args.only]
    encoders_in_play = sorted({cfg.encoder for cfg in arms_to_run})
    all_examples = [ex for split_exs in splits.values() for ex in split_exs]
    assert_uniform_cache_coverage(all_examples, encoders_in_play, args.cache_root)

    import wandb  # lazy: keeps this script importable (e.g. by tests) offline

    for cfg in arms_to_run:
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
