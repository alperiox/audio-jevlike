import csv
from pathlib import Path

import pytest
from prosodia.corpora.meld import MeldCorpus, build_context
from prosodia.schema import LabelTier

ROWS = [
    {"Utterance": "You liked it?", "Speaker": "Joey", "Emotion": "surprise",
     "Sentiment": "positive", "Dialogue_ID": "0", "Utterance_ID": "0"},
    {"Utterance": "Oh yeah!", "Speaker": "Chandler", "Emotion": "joy",
     "Sentiment": "positive", "Dialogue_ID": "0", "Utterance_ID": "1"},
    {"Utterance": "You fell asleep!", "Speaker": "Joey", "Emotion": "anger",
     "Sentiment": "negative", "Dialogue_ID": "0", "Utterance_ID": "2"},
]


@pytest.fixture
def meld_root(tmp_path: Path) -> Path:
    split_dir = tmp_path / "MELD.Raw" / "train_splits"
    split_dir.mkdir(parents=True)
    for r in ROWS:
        (split_dir / f"dia{r['Dialogue_ID']}_utt{r['Utterance_ID']}.wav").write_bytes(b"RIFF")
    csv_path = tmp_path / "MELD.Raw" / "train_sent_emo.csv"
    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(ROWS[0]))
        w.writeheader()
        w.writerows(ROWS)
    return tmp_path


def test_context_excludes_current_and_future_turns():
    ctx = build_context(ROWS, idx=2)
    assert "You liked it?" in ctx and "Oh yeah!" in ctx
    assert "You fell asleep!" not in ctx  # never leak the current utterance


def test_context_is_empty_for_first_turn():
    assert build_context(ROWS, idx=0) == ""


def test_iter_examples_yields_human_tier_labels(meld_root: Path):
    exs = list(MeldCorpus(meld_root).iter_examples("train"))
    assert len(exs) == 3
    ex = exs[2]
    assert ex.labels["emotion"].value == "anger"
    assert ex.labels["sentiment"].value == 0  # negative=0, neutral=1, positive=2 -> ordered
    assert ex.labels["emotion"].tier is LabelTier.HUMAN
    assert ex.speaker == "Joey"


def test_sentiment_is_ordered_for_the_score_primitive(meld_root: Path):
    corpus = MeldCorpus(meld_root)
    spec = {s.key: s for s in corpus.question_specs()}["sentiment"]
    assert spec.qtype == "score"
    assert spec.criteria == ["negative", "neutral", "positive"]
