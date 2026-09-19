#!/usr/bin/env bash
set -euo pipefail
ROOT="${1:-$HOME/prosodia-data/meld}"
mkdir -p "$ROOT" && cd "$ROOT"

command -v ffmpeg >/dev/null 2>&1 || { echo "fetch_meld.sh: ffmpeg not found on PATH" >&2; exit 1; }

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

# Extract 16kHz mono wav from each mp4 (spec: all audio 16kHz before the encoder).
# MELD ships at least one corrupt clip (e.g. dia125_utt3.mp4: "moov atom not
# found"). A single bad clip must not abort the loop or block later splits —
# it is skipped, counted, and reported, mirroring the loader's own tolerance
# for missing .wav files. ffmpeg writes to a temp path first and only the
# temp file is renamed into place on success, so a clip that fails partway
# through never leaves a truncated .wav that a later idempotent run would
# mistake for a completed conversion.
converted=0
skipped=0
failed=()
while IFS= read -r -d '' f; do
  out="${f%.mp4}.wav"
  if [ -f "$out" ]; then
    skipped=$((skipped + 1))
    continue
  fi
  tmp="${out}.part"
  if ffmpeg -loglevel error -i "$f" -ac 1 -ar 16000 -vn -f wav -y "$tmp" 2>/dev/null; then
    mv "$tmp" "$out"
    converted=$((converted + 1))
  else
    rm -f "$tmp"
    failed+=("$f")
  fi
done < <(find MELD.Raw -name '*.mp4' -print0)

echo "Converted: $converted  Already present: $skipped  Failed: ${#failed[@]}"
if [ "${#failed[@]}" -gt 0 ]; then
  echo "Failed clips (undecodable, skipped — not treated as fatal):"
  printf '  %s\n' "${failed[@]}"
fi

echo "MELD ready at $ROOT"
