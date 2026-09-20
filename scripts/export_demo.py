"""Export a self-contained payload for the demo page.

Runs the trained demo model over a stratified sample of TEST clips and emits
one JSON blob carrying, per clip: base64 mp3 audio, the model's distribution
for each of the seven questions, and the gold answer.

The page is a static artifact with no network access, so the audio has to
travel inside the payload. mp3 at 64kbps mono keeps ~24 clips well under a
megabyte.
"""
from __future__ import annotations

import argparse, base64, csv, json, os, random, shutil, subprocess, sys, tempfile
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import run_ablation as RA  # noqa: E402
from prosodia.config import RunConfig  # noqa: E402
from prosodia.corpora.meld import MeldCorpus  # noqa: E402
from prosodia.device import get_device  # noqa: E402
from prosodia.features import FeatureCache  # noqa: E402
from prosodia.labels.acoustic import LOUDNESS_NAMES, RATE_NAMES  # noqa: E402
from prosodia.labels.bank import attach_demo_labels, demo_question_specs  # noqa: E402
from prosodia.model.prosodia import ProsodiaModel  # noqa: E402
from prosodia.train.loop import _move, load_checkpoint  # noqa: E402
from train_demo import load_declared_seconds, load_transcripts  # noqa: E402

SENTIMENT_NAMES = ["negative", "neutral", "positive"]
NOUL_NAMES = ["no", "yes"]


def _ffmpeg() -> str:
    """Resolve ffmpeg explicitly.

    A non-interactive ssh session does not source the login shell, so
    /opt/homebrew/bin is absent from PATH and a bare "ffmpeg" raises
    FileNotFoundError mid-export -- after the model has already run.
    """
    found = shutil.which("ffmpeg")
    if found:
        return found
    for cand in ("/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg"):
        if os.path.exists(cand):
            return cand
    raise FileNotFoundError(
        "ffmpeg not found on PATH or in the usual Homebrew locations; "
        "the demo payload needs it to encode clip audio"
    )


def to_mp3_b64(wav_path: str) -> str | None:
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=True) as tmp:
        r = subprocess.run(
            [_ffmpeg(), "-y", "-i", wav_path, "-ac", "1", "-ar", "22050",
             "-b:a", "64k", tmp.name],
            capture_output=True)
        if r.returncode != 0:
            return None
        data = Path(tmp.name).read_bytes()
    return base64.b64encode(data).decode("ascii")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-root", type=Path, required=True)
    ap.add_argument("--cache-root", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=24)
    args = ap.parse_args()

    corpus = MeldCorpus(args.corpus_root)
    specs = corpus.question_specs() + demo_question_specs()
    splits = {s: list(corpus.iter_examples(s)) for s in ("train", "dev", "test")}
    splits = attach_demo_labels(
        splits, FeatureCache(args.cache_root / "prosody"),
        load_transcripts(args.corpus_root),
        declared_seconds=load_declared_seconds(args.corpus_root))

    cfg = RunConfig(name="demo__whisper", encoder="whisper", brier_weight=0.5)
    cache = FeatureCache(args.cache_root / cfg.encoder)
    loaders = RA._build_loaders(splits, specs, cache, cfg)
    in_dim = next(iter(loaders["train"]))["audio"].shape[-1]

    model = ProsodiaModel(in_dim=in_dim, d_model=cfg.d_model,
                          state_layers=cfg.state_layers,
                          branch_layers=cfg.branch_layers,
                          n_heads=cfg.n_heads, stride=cfg.stride)
    load_checkpoint(args.ckpt, model)
    device = get_device(); model.to(device).eval()

    # option order per question, exactly as the eval loader presents it
    option_names = {}
    for s in specs:
        if s.qtype == "choice":
            option_names[s.key] = list(s.criteria.keys())
        elif s.qtype == "score":
            option_names[s.key] = list(s.criteria)
        else:
            option_names[s.key] = NOUL_NAMES

    by_uid = {ex.uid: ex for ex in splits["test"]}
    preds: dict[str, dict] = {}
    with torch.no_grad():
        for batch in loaders["test"]:
            moved = _move(batch, device)
            for uid, per_q in zip(batch["uid"], model(moved)):
                preds[uid] = {k: torch.softmax(v.detach().float().cpu(), -1).tolist()
                              for k, v in per_q.items()}

    # stratify: spread across speakers and pitch classes so the page is not
    # 24 clips of Ross saying something neutral
    rng = random.Random(0)
    buckets = defaultdict(list)
    for uid, ex in by_uid.items():
        if uid not in preds or "speaker" not in ex.labels:
            continue
        buckets[(ex.labels["speaker"].value,
                 ex.labels.get("pitch_direction", type("x", (), {"value": "?"})).value)].append(uid)
    chosen: list[str] = []
    keys = sorted(buckets)
    rng.shuffle(keys)
    for k in keys:
        if len(chosen) >= args.n:
            break
        chosen.append(rng.choice(buckets[k]))

    gold_name = {
        "sentiment": SENTIMENT_NAMES, "is_negative": NOUL_NAMES,
        "speaking_rate": list(RATE_NAMES), "loudness": list(LOUDNESS_NAMES),
    }
    clips = []
    transcripts = load_transcripts(args.corpus_root)
    for uid in chosen:
        ex = by_uid[uid]
        b64 = to_mp3_b64(ex.audio_path)
        if b64 is None:
            continue
        qs = {}
        for s in specs:
            if s.key not in preds[uid] or s.key not in ex.labels:
                continue
            raw = ex.labels[s.key].value
            gold = gold_name[s.key][int(raw)] if s.key in gold_name and not isinstance(raw, str) else str(raw)
            qs[s.key] = {
                "instructions": s.instructions,
                "options": option_names[s.key],
                "probs": preds[uid][s.key],
                "gold": gold,
            }
        clips.append({"uid": uid, "speaker": ex.speaker,
                      "transcript": transcripts.get(uid, ""),
                      "audio": b64, "questions": qs})

    args.out.write_text(json.dumps({"clips": clips}))
    size = args.out.stat().st_size / 1e6
    print(f"wrote {len(clips)} clips to {args.out} ({size:.2f} MB)")


if __name__ == "__main__":
    main()
