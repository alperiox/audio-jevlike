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


def init_wandb(cfg: RunConfig) -> Any:
    """Lazily imports and starts a Weights & Biases run for `cfg`.

    The import lives inside this function, not at module scope, so that
    importing/testing `prosodia.train.loop` never requires `wandb`, network
    access, or credentials. Pass the returned run to `train_one_epoch` /
    `evaluate` as `run=` to enable logging.
    """
    import wandb

    return wandb.init(project=cfg.wandb_project, name=cfg.name, config=cfg.as_dict())


def save_checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer,
                    epoch: int, cfg: RunConfig) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
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
    """
    device = get_device()
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
    return int(ckpt["epoch"])
