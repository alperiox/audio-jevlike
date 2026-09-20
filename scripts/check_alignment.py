"""Measure how often a MELD clip actually contains its transcript.

Metadata self-consistency (is the declared span long enough for the words?)
is only a LOWER BOUND on misalignment -- it cannot see a clip that is the
wrong segment at a plausible length, or offset by a second, or carrying the
neighbouring line. This runs real ASR and compares.

Whisper-small is already cached (the feature pipeline uses its encoder); the
decoder is unused elsewhere and is exactly what is needed here.

Reading the result: noisy sitcom audio with a laugh track produces real WER
even on correctly aligned clips, so the absolute level is not the signal.
Look for BIMODALITY -- aligned clips cluster low, misaligned clips pile up
near or above 1.0 because the reference words simply are not present.
"""
from __future__ import annotations

import argparse, csv, json, random, re, sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from prosodia.features import SAMPLE_RATE  # noqa: E402

_PUNCT = re.compile(r"[^\w\s']")


def norm(t: str) -> list[str]:
    t = t.replace("\x92", "'").replace("’", "'")
    return _PUNCT.sub(" ", t.lower()).split()


def wer(ref: list[str], hyp: list[str]) -> float:
    if not ref:
        return 0.0 if not hyp else 1.0
    d = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        prev, d[0] = d[0], i
        for j, h in enumerate(hyp, 1):
            cur = d[j]
            d[j] = min(d[j] + 1, d[j - 1] + 1, prev + (r != h))
            prev = cur
    return d[len(hyp)] / len(ref)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-root", type=Path, required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    import librosa
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    name = "openai/whisper-small"
    proc = WhisperProcessor.from_pretrained(name)
    model = WhisperForConditionalGeneration.from_pretrained(name)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model.to(dev).eval()
    forced = proc.get_decoder_prompt_ids(language="english", task="transcribe")

    csv_path = args.corpus_root / "MELD.Raw" / f"{args.split}_sent_emo.csv"
    audio_dir = args.corpus_root / "MELD.Raw" / {
        "train": "train_splits", "dev": "dev_splits_complete",
        "test": "output_repeated_splits_test"}[args.split]
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8", errors="replace")))
    rng = random.Random(0)
    rng.shuffle(rows)

    out = []
    for r in rows:
        if len(out) >= args.n:
            break
        wav_p = audio_dir / f"dia{r['Dialogue_ID']}_utt{r['Utterance_ID']}.wav"
        if not wav_p.exists():
            continue
        try:
            wav, _ = librosa.load(str(wav_p), sr=SAMPLE_RATE, mono=True)
        except Exception:
            continue
        if len(wav) < SAMPLE_RATE // 20:
            hyp_text = ""
        else:
            feats = proc(wav[: SAMPLE_RATE * 30], sampling_rate=SAMPLE_RATE,
                         return_tensors="pt").input_features.to(dev)
            with torch.no_grad():
                ids = model.generate(feats, forced_decoder_ids=forced, max_new_tokens=80)
            hyp_text = proc.batch_decode(ids, skip_special_tokens=True)[0]
        ref, hyp = norm(r["Utterance"]), norm(hyp_text)
        out.append({
            "uid": f"meld-{args.split}-{r['Dialogue_ID']}-{r['Utterance_ID']}",
            "ref": " ".join(ref), "hyp": " ".join(hyp),
            "wer": wer(ref, hyp), "n_ref_words": len(ref),
            "dur_s": len(wav) / SAMPLE_RATE,
        })
        if len(out) % 50 == 0:
            print(f"  {len(out)}/{args.n}", flush=True)

    args.out.write_text(json.dumps(out))
    ws = sorted(x["wer"] for x in out)
    n = len(ws)
    print(f"\ntranscribed {n} clips from {args.split}")
    print(f"  WER  p10={ws[n//10]:.2f}  p25={ws[n//4]:.2f}  median={ws[n//2]:.2f} "
          f" p75={ws[3*n//4]:.2f}  p90={ws[9*n//10]:.2f}")
    for thr in (0.3, 0.5, 0.8, 1.0):
        c = sum(1 for w in ws if w <= thr)
        print(f"  WER <= {thr:.1f} : {c:>4} ({c/n*100:5.1f}%)")
    empty = sum(1 for x in out if not x["hyp"])
    print(f"  ASR returned nothing: {empty} ({empty/n*100:.1f}%)")


if __name__ == "__main__":
    main()
