"""MELD loader.

Labels are HUMAN tier: MELD re-annotated all EmotionLines utterances with
three annotators who had the video clip available (Fleiss kappa 0.43 vs 0.34
text-only). Verified in aclanthology.org/P19-1050 section 3.1.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterator, Sequence

from prosodia.schema import Example, Label, LabelTier, QuestionSpec

EMOTIONS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
SENTIMENTS = ["negative", "neutral", "positive"]  # ordered — maps to Score

_SPLIT_DIRS = {"train": "train_splits", "dev": "dev_splits_complete",
               "test": "output_repeated_splits_test"}
_SPLIT_CSVS = {"train": "train_sent_emo.csv", "dev": "dev_sent_emo.csv",
               "test": "test_sent_emo.csv"}


def build_context(rows: Sequence[dict], idx: int, max_turns: int = 6) -> str:
    """Serialize preceding turns of the same dialogue.

    Excludes the current utterance and everything after it. Leaking either
    would make the task trivial (spec §11 trap 4).
    """
    dialogue = rows[idx]["Dialogue_ID"]
    prior = [r for r in rows[:idx] if r["Dialogue_ID"] == dialogue]
    return "\n".join(f"{r['Speaker']}: {r['Utterance']}" for r in prior[-max_turns:])


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
