"""MELD loader.

Labels are HUMAN tier: MELD re-annotated all EmotionLines utterances with
three annotators who had the video clip available (Fleiss kappa 0.43 vs 0.34
text-only). Verified in aclanthology.org/P19-1050 section 3.1.
"""
from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Iterator, Sequence

from prosodia.schema import Example, Label, LabelTier, QuestionSpec

EMOTIONS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
SENTIMENTS = ["negative", "neutral", "positive"]  # ordered — maps to Score

_SPLIT_DIRS = {"train": "train_splits", "dev": "dev_splits_complete",
               "test": "output_repeated_splits_test"}
_SPLIT_CSVS = {"train": "train_sent_emo.csv", "dev": "dev_sent_emo.csv",
               "test": "test_sent_emo.csv"}

# I8: strip sentence punctuation from context text while preserving
# word-internal apostrophes (contractions like "don't", possessives like
# "Ross's"). Rule, applied in order:
#   1. Replace any character that is not a word character, whitespace, or
#      an apostrophe with a SPACE (not empty) -- this removes `!?.,;:"()`
#      etc. without gluing the words on either side together (e.g.
#      "Wait...what?!" must become "Wait what", never "Waitwhat").
#   2. Delete any SURVIVING apostrophe that is not flanked by a word
#      character on both sides -- this catches quote-mark apostrophes
#      (leading/trailing) without touching one sitting mid-word. Deleting
#      (not spacing) is correct here: a quote apostrophe already sits next
#      to a real space on its outer side, so removing it cannot merge two
#      words.
#   3. Collapse whitespace runs the substitutions may have introduced and
#      strip the ends.
_STRIP_CHARS_RE = re.compile(r"[^\w\s']")
_STRAY_APOSTROPHE_RE = re.compile(r"(?<!\w)'|'(?!\w)")


def _strip_punctuation(text: str) -> str:
    text = _STRIP_CHARS_RE.sub(" ", text)
    text = _STRAY_APOSTROPHE_RE.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


def build_context(rows: Sequence[dict], idx: int, max_turns: int = 6) -> str:
    """Serialize preceding turns of the same dialogue.

    Excludes the current utterance and everything after it. Leaking either
    would make the task trivial (spec §11 trap 4).

    I8: neither the speaker name nor sentence punctuation survive into the
    serialized context. MELD's splits are speaker-SHARED (Decision 1) --
    the six recurring *Friends* leads appear in train, dev, and test alike
    -- so a speaker name prefix in the context handed the text side a
    direct identity key, a second leakage channel alongside the acoustic
    one `assert_speaker_disjoint`/the MELD warning already flag. Gold
    punctuation (`!`, `?`, ...) is itself affect-bearing and not what a
    deployed streaming-ASR context would contain, so it inflates the
    text-only baseline's floor relative to what audio has to beat. See
    `_strip_punctuation` for the exact rule (word-internal apostrophes
    like "don't" survive).

    This still serializes gold transcript text, not streaming ASR output
    (spec §11 trap 4's guardrail, §12 limitations) -- the acute leak this
    function already closes (current/future utterances, cross-dialogue
    rows) and the two channels closed here (speaker identity, punctuation)
    are not the same fix as running real ASR, which remains out of scope.
    """
    dialogue = rows[idx]["Dialogue_ID"]
    prior = [r for r in rows[:idx] if r["Dialogue_ID"] == dialogue]
    return "\n".join(_strip_punctuation(r["Utterance"]) for r in prior[-max_turns:])


class MeldCorpus:
    name = "meld"

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def question_specs(self) -> list[QuestionSpec]:
        return [
            QuestionSpec(
                key="emotion", qtype="choice",
                instructions="Which emotion is the speaker expressing?",
                criteria={e: None for e in EMOTIONS},
            ),
            QuestionSpec(
                key="sentiment", qtype="score",
                instructions="Rate the sentiment the speaker conveys.",
                criteria=list(SENTIMENTS),
            ),
            QuestionSpec(
                key="is_negative", qtype="noul",
                instructions="Is the speaker expressing something negative?",
                criteria={"true": "Negative sentiment", "false": "Neutral or positive"},
            ),
        ]

    def iter_examples(self, split: str) -> Iterator[Example]:
        csv_path = self.root / "MELD.Raw" / _SPLIT_CSVS[split]
        audio_dir = self.root / "MELD.Raw" / _SPLIT_DIRS[split]
        with csv_path.open(newline="") as fh:
            rows = list(csv.DictReader(fh))

        for idx, row in enumerate(rows):
            wav = audio_dir / f"dia{row['Dialogue_ID']}_utt{row['Utterance_ID']}.wav"
            if not wav.exists():
                continue  # MELD ships a handful of undecodable clips
            sentiment_idx = SENTIMENTS.index(row["Sentiment"].strip().lower())
            yield Example(
                uid=f"meld-{split}-{row['Dialogue_ID']}-{row['Utterance_ID']}",
                corpus=self.name,
                audio_path=str(wav),
                context=build_context(rows, idx),
                speaker=row["Speaker"].strip(),
                labels={
                    "emotion": Label(row["Emotion"].strip().lower(), LabelTier.HUMAN),
                    "sentiment": Label(sentiment_idx, LabelTier.HUMAN),
                    "is_negative": Label(int(sentiment_idx == 0), LabelTier.HUMAN),
                },
            )
