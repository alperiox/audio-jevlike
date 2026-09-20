"""Dump per-uid test/dev logits for every trained arm.

Re-analysis support (EmotionLines ambiguity stratification). Deliberately
does NOT reuse `evaluate()`'s return_logits path: that path aligns logits
to targets POSITIONALLY after dropping mismatched-width rows, which is safe
for aggregate metrics but cannot be joined back to utterance ids. Here we
carry `batch["uid"]` through the forward pass so every row is keyed by the
utterance it came from, and a dropped row is simply absent rather than
silently shifting its neighbours.

Loader construction is imported from `run_ablation` rather than reimplemented
so eval-time settings (shuffle off, augmentation off, modality dropout off,
text-only audio muting) cannot drift from the grid that produced the
checkpoints.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import run_ablation as RA  # noqa: E402
from prosodia.corpora.meld import MeldCorpus  # noqa: E402
from prosodia.evaluation.baselines import SELECTABLE_ARMS  # noqa: E402
from prosodia.features import FeatureCache  # noqa: E402
from prosodia.model.prosodia import ProsodiaModel  # noqa: E402
from prosodia.train.loop import _move, load_checkpoint  # noqa: E402
from prosodia.device import get_device  # noqa: E402


@torch.no_grad()
def dump_split(model, loader, device) -> dict[str, dict[str, list[float]]]:
    """uid -> question -> {"logits": [...], "target": int}.

    Targets are dumped alongside the logits because `permute_candidates`
    randomises option order PER EXAMPLE -- logit index does not correspond
    to a fixed class label, so a downstream consumer that assumed a fixed
    ordering would score against the wrong option. The target index is the
    only thing that identifies the correct slot for that row.
    """
    model.to(device).eval()
    out: dict[str, dict[str, dict]] = {}
    for batch in loader:
        uids = batch["uid"]
        moved = _move(batch, device)
        for uid, per_q, tgt in zip(uids, model(moved), batch["targets"]):
            out[uid] = {
                k: {"logits": v.detach().float().cpu().tolist(), "target": int(tgt[k])}
                for k, v in per_q.items() if k in tgt
            }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-root", type=Path, required=True)
    ap.add_argument("--cache-root", type=Path, required=True)
    ap.add_argument("--ckpt-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    corpus = MeldCorpus(args.corpus_root)
    specs = corpus.question_specs()
    splits = {s: list(corpus.iter_examples(s)) for s in ("train", "dev", "test")}
    device = get_device()
    args.out.mkdir(parents=True, exist_ok=True)

    # Only the TRAINED arms have checkpoints; Arm C is derived downstream
    # from its companion Arm A's dev logits, exactly as the grid does.
    trained = [c for c in SELECTABLE_ARMS if not c.temperature_scale]

    for cfg in trained:
        ckpt = args.ckpt_root / cfg.name / RA.BEST_CKPT_NAME
        if not ckpt.is_file():
            print(f"SKIP {cfg.name}: no {ckpt}", file=sys.stderr)
            continue
        cache = FeatureCache(args.cache_root / cfg.encoder)
        loaders = RA._build_loaders(splits, specs, cache, cfg)
        in_dim = next(iter(loaders["train"]))["audio"].shape[-1]

        model = ProsodiaModel(in_dim=in_dim, d_model=cfg.d_model,
                              state_layers=cfg.state_layers,
                              branch_layers=cfg.branch_layers,
                              n_heads=cfg.n_heads, stride=cfg.stride)
        load_checkpoint(ckpt, model)

        payload = {s: dump_split(model, loaders[s], device) for s in ("dev", "test")}
        path = args.out / f"{cfg.name}.json"
        path.write_text(json.dumps(payload))
        print(f"{cfg.name}: dev={len(payload['dev'])} test={len(payload['test'])} -> {path}")


if __name__ == "__main__":
    main()
