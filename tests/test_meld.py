import csv
from pathlib import Path

import pytest
from prosodia.corpora.meld import MeldCorpus, build_context
from prosodia.schema import LabelTier

ROWS = [
    {"Utterance": "You liked it?", "Speaker": "Joey", "Emotion": "surprise",
     "Sentiment": "positive", "Dialogue_ID": "0", "Utterance_ID": "0"},
    {"Utterance": "Hi there.", "Speaker": "Ross", "Emotion": "neutral",
     "Sentiment": "neutral", "Dialogue_ID": "1", "Utterance_ID": "0"},
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
    # I8: context is now unpunctuated, so assertions match the stripped
    # form ("You liked it", not "You liked it?") -- see the I8 tests below
    # for the stripping behavior itself.
    ctx = build_context(ROWS, idx=3)
    assert "You liked it" in ctx and "Oh yeah" in ctx
    assert "You fell asleep" not in ctx  # never leak the current utterance


def test_context_excludes_other_dialogues():
    # ROWS[1] (Dialogue_ID="1") sits at CSV position 1, strictly before idx=3,
    # so a naive rows[:idx] slice without the dialogue filter would include
    # it. The filter must exclude it even though it precedes the current row.
    ctx = build_context(ROWS, idx=3)
    assert "Hi there" not in ctx  # different Dialogue_ID — must never bleed in
    assert "You liked it" in ctx and "Oh yeah" in ctx  # same-dialogue turns still present


def test_context_is_empty_for_first_turn():
    assert build_context(ROWS, idx=0) == ""


# --- I8: speaker names and sentence punctuation must not survive ----------

def test_build_context_strips_speaker_names():
    """Fault this catches: MELD's splits are speaker-SHARED (Decision 1) --
    the six recurring Friends leads appear in train, dev, and test alike --
    so a speaker name left in the context handed the text side a direct
    identity key, a second leakage channel alongside the acoustic one. Any
    of the three speakers appearing in ROWS' prior turns (Joey, Chandler)
    must not appear as a name in the built context."""
    ctx = build_context(ROWS, idx=3)
    for name in ("Joey", "Ross", "Chandler"):
        assert name not in ctx, f"{name!r} leaked into context: {ctx!r}"


def test_build_context_strips_sentence_punctuation():
    """Fault this catches: punctuated gold text (`!`, `?`, ...) is itself
    affect-bearing and raises the text-only baseline's floor relative to
    what a deployed streaming-ASR context would actually contain (spec §11
    trap 4). None of `!?.,` may survive into the built context."""
    ctx = build_context(ROWS, idx=3)
    assert not any(ch in ctx for ch in "!?.,")
    assert "You liked it" in ctx
    assert "Oh yeah" in ctx


def test_build_context_preserves_word_internal_apostrophes():
    """Companion to the punctuation-stripping test: word-internal
    apostrophes (contractions, possessives) are not sentence punctuation
    and must survive intact -- "don't" must never become "dont" or "don t".
    Fault this catches: a blunt strip-all-punctuation implementation that
    treats every apostrophe the same as `!` or `?`."""
    rows = [
        {"Utterance": "I don't know, Ross's dog!", "Speaker": "Joey",
         "Dialogue_ID": "0", "Utterance_ID": "0"},
        {"Utterance": "current turn", "Speaker": "Ross",
         "Dialogue_ID": "0", "Utterance_ID": "1"},
    ]
    ctx = build_context(rows, idx=1)
    assert "don't" in ctx
    assert "Ross's" in ctx
    assert "!" not in ctx and "," not in ctx
    # and no merged-word artifact from naive punctuation deletion
    assert "dont" not in ctx and "Rosss" not in ctx


def test_iter_examples_yields_human_tier_labels(meld_root: Path):
    exs = list(MeldCorpus(meld_root).iter_examples("train"))
    assert len(exs) == 4
    ex = exs[3]
    assert ex.labels["emotion"].value == "anger"
    assert ex.labels["sentiment"].value == 0  # negative=0, neutral=1, positive=2 -> ordered
    assert ex.labels["emotion"].tier is LabelTier.HUMAN
    assert ex.speaker == "Joey"


def test_sentiment_is_ordered_for_the_score_primitive(meld_root: Path):
    corpus = MeldCorpus(meld_root)
    spec = {s.key: s for s in corpus.question_specs()}["sentiment"]
    assert spec.qtype == "score"
    assert spec.criteria == ["negative", "neutral", "positive"]
