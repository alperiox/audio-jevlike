#!/usr/bin/env bash
set -euo pipefail
ROOT="${1:-$HOME/prosodia-data/meld}"
mkdir -p "$ROOT" && cd "$ROOT"
[ -f MELD.Raw.tar.gz ] || wget https://web.eecs.umich.edu/~mihalcea/downloads/MELD.Raw.tar.gz
[ -d MELD.Raw ] || tar -xzf MELD.Raw.tar.gz
for s in train dev test; do
  case "$s" in
    train) marker="MELD.Raw/train_splits" ;;
    dev) marker="MELD.Raw/dev_splits_complete" ;;
    test) marker="MELD.Raw/output_repeated_splits_test" ;;
  esac
  [ -d "$marker" ] || tar -xzf "MELD.Raw/${s}.tar.gz" -C MELD.Raw/ || true
done
# Extract 16kHz mono wav from each mp4 (spec: all audio 16kHz before the encoder)
find MELD.Raw -name '*.mp4' | while read -r f; do
  out="${f%.mp4}.wav"
  [ -f "$out" ] || ffmpeg -loglevel error -i "$f" -ac 1 -ar 16000 -vn "$out"
done
echo "MELD ready at $ROOT"
