"""One-time feature extraction. On the order of an hour per encoder arm."""
from __future__ import annotations

import argparse
from pathlib import Path

import soundfile as sf

from prosodia.corpora.meld import MeldCorpus
from prosodia.device import get_device
from prosodia.features import SAMPLE_RATE, FeatureCache, FeatureExtractor

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-root", type=Path, required=True)
    ap.add_argument("--cache-root", type=Path, required=True)
    ap.add_argument("--encoder", choices=["wavlm", "whisper", "prosody"], required=True)
    ap.add_argument("--splits", nargs="+", default=["train", "dev", "test"])
    args = ap.parse_args()

    extractor = FeatureExtractor(args.encoder, get_device())
    cache = FeatureCache(args.cache_root / args.encoder)
    corpus = MeldCorpus(args.corpus_root)

    for split in args.splits:
        done = skipped = 0
        for ex in corpus.iter_examples(split):
            if ex.uid in cache:
                skipped += 1
                continue
            wav, sr = sf.read(ex.audio_path, dtype="float32")
            assert sr == SAMPLE_RATE, f"{ex.audio_path} is {sr}Hz, expected 16000"
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
            cache.write(ex.uid, extractor.encode(wav))
            done += 1
            if done % 250 == 0:
                print(f"{split}: {done} extracted, {skipped} cached")
        print(f"{split}: DONE {done} extracted, {skipped} already cached")
