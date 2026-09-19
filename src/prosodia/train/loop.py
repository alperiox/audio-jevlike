"""Training loop, evaluation, and checkpointing (spec §8).

Epoch handling: `ProsodiaDataset` seeds its per-item RNG from
`(rng_seed, epoch, idx)` alone (Task 6) so that the 12-arm ablation grid
(9-arm encoder grid + 3-arm text-only baseline) compares arms that differ
only in loss/encoder/modality, never in incidental RNG entropy. The consequence is that *something* must call `dataset.set_epoch`
before every training pass, or every epoch draws the identical
augmentation/modality-dropout pattern -- a silent failure: loss still
descends, metrics still look plausible, the model just never sees the
augmentation diversity the design calls for.

`train_one_epoch` owns this: `epoch` is a required argument (no default),
and the first thing the function does with it is
`loader.dataset.set_epoch(epoch)`. Making it required rather than an
internal counter means a caller cannot invoke this function without
deciding what epoch it is. `evaluate()` never calls `set_epoch` -- eval
loaders are built with `augment=False`, so there is nothing to advance and
doing so would only make eval depend on call history.

W&B logging is opt-in and lazily imported (see `init_wandb`): nothing in
this module imports `wandb` at module scope, and no code path exercised by
the test suite ever imports it or touches the network.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from prosodia.config import RunConfig
from prosodia.device import get_device
from prosodia.evaluation.metrics import compute_metrics
from prosodia.train.losses import composite_loss


def _move(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out = dict(batch)
    for key in ("audio", "audio_mask", "audio_present", "context_present"):
        out[key] = batch[key].to(device)
    return out


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    cfg: RunConfig,
    epoch: int,
    run: Any = None,
) -> dict[str, float]:
    """Runs one training epoch over `loader`.

    `epoch` is mandatory: see the module docstring for why. `run`, if given
    (from `init_wandb`), receives `train/loss` and `epoch` each call.
    """
    loader.dataset.set_epoch(epoch)

    device = get_device()
    model.to(device).train()
    total, n = 0.0, 0

    for batch in loader:
        batch = _move(batch, device)
        outputs = model(batch)
        loss = torch.zeros((), device=device)
        count = 0
        for out, targets in zip(outputs, batch["targets"]):
            for key, logits in out.items():
                target = torch.tensor([targets[key]], device=device)
                loss = loss + composite_loss(logits.unsqueeze(0), target,
                                             cfg.brier_weight)
                count += 1
        if count == 0:
            continue
        loss = loss / count

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total += loss.item()
        n += 1

    metrics = {"loss": total / max(n, 1)}
    if run is not None:
        run.log({"train/loss": metrics["loss"], "epoch": epoch})
    return metrics


@torch.no_grad()
def evaluate(
    model: nn.Module, loader: DataLoader, run: Any = None, epoch: int | None = None,
    return_logits: bool = False,
) -> dict[str, dict[str, float]] | tuple[
    dict[str, dict[str, float]], dict[str, torch.Tensor], dict[str, torch.Tensor]
]:
    """Returns per-question metrics. Probabilities are computed in fp32 on CPU.

    Does NOT call `set_epoch` -- see the module docstring.

    `return_logits`, if True, additionally returns the per-question raw
    logits and targets this call already collected and aligned -- the same
    tensors the metrics above are computed from, not a second collection
    pass. Arm C (post-hoc temperature scaling) needs exactly these: fit a
    scaler on the DEV split's logits/targets, then transform the TEST
    split's logits before scoring. Reusing this path (rather than a fresh
    loop over the loader) matters because the target/logit alignment here
    was previously fixed after a bug that paired targets *positionally*
    with logits after filtering mismatched-width rows -- a second,
    independently-written collection loop would be an easy way to
    reintroduce exactly that.
    """
    device = get_device()
    model.to(device).eval()
    logits_by_q: dict[str, list[torch.Tensor]] = defaultdict(list)
    targets_by_q: dict[str, list[int]] = defaultdict(list)

    for batch in loader:
        batch = _move(batch, device)
        for out, targets in zip(model(batch), batch["targets"]):
            for key, logits in out.items():
                logits_by_q[key].append(logits.detach().float().cpu())
                targets_by_q[key].append(targets[key])

    results: dict[str, dict[str, float]] = {}
    logits_out: dict[str, torch.Tensor] = {}
    targets_out: dict[str, torch.Tensor] = {}
    for key, rows in logits_by_q.items():
        targets_list = targets_by_q[key]
        widths = [r.shape[-1] for r in rows]
        modal_width, modal_count = Counter(widths).most_common(1)[0]

        if modal_count < len(rows):
            kept_frac = modal_count / len(rows)
            if kept_frac < 0.5:
                # Augmentation is expected to be off for eval loaders, so
                # varying widths at all is already surprising. Refuse rather
                # than silently score a minority subset of the data.
                raise ValueError(
                    f"evaluate(): question {key!r} has option-count widths "
                    f"{sorted(set(widths))} across {len(rows)} rows; the modal "
                    f"width {modal_width} covers only {kept_frac:.0%} of them. "
                    "Refusing to silently discard the majority -- confirm the "
                    "eval loader was built with augment=False."
                )
            # Filter logits and targets TOGETHER so a dropped row can never
            # leave the two lists misaligned (filtering only `rows` and then
            # slicing `targets_list` by position would do exactly that).
            paired = [(r, t) for r, t in zip(rows, targets_list) if r.shape[-1] == modal_width]
            rows = [r for r, _ in paired]
            targets_list = [t for _, t in paired]

        logits = torch.stack(rows)
        targets = torch.tensor(targets_list)
        probs = torch.softmax(logits, dim=-1)
        results[key] = compute_metrics(probs, targets)
        if return_logits:
            logits_out[key] = logits
            targets_out[key] = targets
        if run is not None:
            log = {f"eval/{key}/{m}": v for m, v in results[key].items()}
            if epoch is not None:
                log["epoch"] = epoch
            run.log(log)
    if return_logits:
        return results, logits_out, targets_out
    return results


# I5b: which dev metric selects the checkpoint that test scoring reads
# from (`run_arm` in `scripts/run_ablation.py`). NLL, not ECE:
# `evaluation.metrics.expected_calibration_error` is a BINNED statistic
# (10 bins) and is genuinely noisy on a dev split this size -- picking the
# epoch whose dev ECE happens to be lowest, then reporting THAT epoch's
# test ECE, is a subtle circularity: a low dev ECE can be partly luck
# (which bin each borderline-confidence example happened to land in), and
# luck does not carry over to the test split, so the reported test ECE
# would be biased optimistic relative to a metric that did not select on
# it. Dev loss (the actual training objective) sidesteps that circularity
# even more cleanly, but is not what this phase measures, does not exist
# as a per-question quantity `evaluate()` already reports, and for Arm A
# (brier_weight=0.0) collapses to plain cross-entropy, which rewards
# confidence rather than calibration. NLL is the standard compromise for
# exactly this reason (Guo et al. 2017 fit/select temperature scaling on
# NLL, not ECE): a strictly proper scoring rule, sensitive to both
# calibration and discrimination, computed per-example rather than binned
# -- far less noisy than ECE on a small split -- and already present in
# `evaluate()`'s per-question output at zero extra cost.
DEV_SELECTION_METRIC = "nll"


def dev_selection_score(dev_stats: dict[str, dict[str, float]]) -> float:
    """Reduces `evaluate()`'s per-question dev metrics to the single scalar
    `run_arm` selects the `best` checkpoint on: the unweighted mean of
    `DEV_SELECTION_METRIC` across question keys. Lower is better.

    Unweighted mean, not weighted by per-question example count: the exit
    criterion treats each question as one calibration judgment, and
    `evaluate()` makes no claim about per-question example-count parity
    across questions.
    """
    values = [q[DEV_SELECTION_METRIC] for q in dev_stats.values()]
    if not values:
        raise ValueError("dev_selection_score(): dev_stats has no question keys")
    return sum(values) / len(values)


def init_wandb(cfg: RunConfig) -> Any:
    """Lazily imports and starts a Weights & Biases run for `cfg`.

    The import lives inside this function, not at module scope, so that
    importing/testing `prosodia.train.loop` never requires `wandb`, network
    access, or credentials. Pass the returned run to `train_one_epoch` /
    `evaluate` as `run=` to enable logging.
    """
    import wandb

    return wandb.init(project=cfg.wandb_project, name=cfg.name, config=cfg.as_dict())


# I5a: `ProsodiaModel.question_encoder._st` is `SentenceTransformer(
# "all-MiniLM-L6-v2")`, frozen (`requires_grad_(False)`, see `qencoder.py`)
# and reconstructed from the HuggingFace hub by `QuestionEncoder.__init__`
# on every fresh model instantiation -- `load_checkpoint` never needs to
# restore it, because whatever `model` it is handed already has it, built
# the identical way. Measured at the grid's config (in_dim=1024,
# d_model=256): `_st` accounts for 22.71M of the model's 26.37M state_dict
# elements (86.1%); the remaining 3.65M are exactly the trainable
# parameters. Saving the frozen 86% anyway, every epoch, across 12 arms x
# <=20 epochs, is pure waste with zero information content -- the I5a
# finding that motivates filtering it out below.
_FROZEN_STATE_DICT_PREFIX = "question_encoder._st."


def _trainable_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """`model.state_dict()` minus the frozen sentence-transformer's own
    parameters/buffers (see `_FROZEN_STATE_DICT_PREFIX` above)."""
    return {k: v for k, v in model.state_dict().items()
            if not k.startswith(_FROZEN_STATE_DICT_PREFIX)}


def save_checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer,
                    epoch: int, cfg: RunConfig) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": _trainable_state_dict(model), "optimizer": optimizer.state_dict(),
                "epoch": epoch, "config": cfg.as_dict()}, path)


def load_checkpoint(path: Path, model: nn.Module,
                    optimizer: torch.optim.Optimizer | None = None) -> int:
    """Restores weights (and optimizer state, if given) and returns the epoch.

    Loads onto `get_device()` and, for the optimizer, explicitly relocates
    every state tensor there too. `Optimizer.load_state_dict` does not
    reliably move its state tensors to match the parameters' device, so
    without this a resumed run can end up with accelerator-resident
    parameters paired with CPU-resident Adam moment buffers -- silently
    correct until the very first post-resume `.step()`, which then raises a
    device-mismatch error deep inside the optimizer.

    `ckpt["model"]` (I5a) no longer contains `question_encoder._st.*`: it
    was never trained and is excluded from `save_checkpoint`'s state_dict
    on purpose (see `_trainable_state_dict`). `model` already has those
    weights -- `QuestionEncoder.__init__` pulls them from the HuggingFace
    hub the same way regardless of what any checkpoint restores -- so
    `load_state_dict` is called with `strict=False` and the *expected*
    missing keys (exactly the `_FROZEN_STATE_DICT_PREFIX` ones) are
    swallowed. Anything else -- an unexpected key, or a missing key that is
    NOT part of the frozen encoder -- means the checkpoint and `model`
    disagree about the TRAINABLE weights, which is a real bug (e.g. a stale
    checkpoint from a differently-shaped model, or a future change that
    accidentally excludes a trainable parameter too), so that case still
    fails loudly rather than silently leaving part of the model
    uninitialized.
    """
    device = get_device()
    ckpt = torch.load(path, map_location=device)
    result = model.load_state_dict(ckpt["model"], strict=False)
    unexpected = list(result.unexpected_keys)
    bad_missing = [k for k in result.missing_keys
                   if not k.startswith(_FROZEN_STATE_DICT_PREFIX)]
    if unexpected or bad_missing:
        raise RuntimeError(
            f"load_checkpoint({path}): state_dict mismatch beyond the "
            f"expected frozen-sentence-transformer keys. Unexpected keys: "
            f"{unexpected}. Missing (non-frozen) keys: {bad_missing}. This "
            "means the checkpoint and the model disagree about the "
            "TRAINABLE weights -- not the expected, harmless absence of "
            f"{_FROZEN_STATE_DICT_PREFIX!r} entries (I5a)."
        )
    model.to(device)
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
    return int(ckpt["epoch"])
