# Prosodia Phase 1 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an audio-native Jev-shaped decision model (speech + context → typed `Choice`/`Score`/`Noul` with calibrated probabilities) and run the 12-arm grid (9-arm encoder ablation + 3-arm controlled text-only baseline, one per loss regime — Decision 2) to completion on MELD, producing evaluated checkpoints.

**Architecture:** A frozen speech encoder produces frame features, cached once to disk. A small trainable state encoder pools those frames with attention (never mean — contours must survive) and fuses serialized context. Each question, embedded by a frozen sentence encoder, gets an isolated cross-attention branch over the shared state — branches cannot attend to one another. Three heads read out through a strictly linear layer.

**Tech Stack:** Python 3.12 (uv), PyTorch (MPS), HuggingFace transformers (WavLM / Whisper / sentence-transformers), ffmpeg, Weights & Biases, pytest.

**Spec:** `docs/superpowers/specs/2026-09-19-prosodia-design.md`

**Phase 2 (separate plan, later):** interpretability tooling (§8) and the live demo (§9). Both require Phase 1 checkpoints to exist.

**Deferred, not forgotten — the IEMOCAP loader.** Spec §4.2 makes IEMOCAP the thesis corpus, but registration is still pending and the shipped file layout is not known precisely enough to write real steps against. Task 3 delivers the `Corpus` protocol specifically so this is a single additional loader and nothing else changes: implement `IemocapCorpus` against the same interface, with `question_specs()` returning categorical emotion (`Choice`) plus 5-point valence / arousal / dominance (`Score`), and session-disjoint splits (leave-one-session-out, 5 folds) — genuinely speaker-disjoint, unlike MELD's dialogue-disjoint-but-speaker-shared splits (see the corrected Global Constraints note below and spec §4.1). The arousal-vs-valence contrast in spec §7.1 runs only once that loader exists.

**Correction (owner Decision 1, 2026-09-19):** this plan and the spec originally claimed splits were "speaker-disjoint (MELD) or session-disjoint (IEMOCAP)." That is false for MELD: `MeldCorpus.iter_examples` uses MELD's shipped train/dev/test CSVs, which are **dialogue**-disjoint only — MELD is *Friends*, and the six leads appear in every split. Not re-split (a speaker-disjoint split would shred both data volume and class balance for a corpus dominated by six characters, and MELD was always the build/pipeline-validation corpus, not the evidence corpus). `schema.assert_speaker_disjoint(splits)` (Task 2) makes the constraint enforceable rather than aspirational, is exercised by IEMOCAP's session-disjoint splits, and is deliberately never called on MELD — it would fail by design. `scripts/run_ablation.py` (Task 15) prints an unmissable startup warning and injects the same text into every arm's W&B config when the corpus is MELD, so the caveat travels with the numbers. See spec §4.1 and §12.

**Correction (owner Decision 3, 2026-09-19):** the text-only baseline (Decision 2) was not actually modality-isolated. `TextOnlyBaseline.mute_audio` and `ProsodiaModel._encode_state`'s `torch.where` gate neutralized the audio state's *content* when muted, but the attention mask passed alongside it still carried each example's real, un-muted audio duration, which reached `IsolatedBranches`' cross-attention as a length-dependent softmax-weight channel independent of content (measured leak up to 4.7e-3, larger than the 2.0e-3 content leak the existing gate guards against). Fixed at the single point where both signals are already in hand: `_encode_state` now also zeroes `mask` for muted rows (`mask = mask & audio_present.view(-1, 1)`), covering the C2 stat-token positions for free since they live in the same mask tensor. See Task 11's correction note and spec §7.1.

## Global Constraints

- **Python 3.12**, managed by `uv`. Target machine: **this machine** (Apple M2 Pro, 16GB unified, 230GB free) — the repo and the compute live together, no sync step. `ssh mac` (M4 Pro, 24GB) remains available as overflow; all code is device-agnostic so nothing changes if a run moves.
- **Device order:** `mps` → `cuda` → `cpu`. All code device-agnostic.
- **Interp and calibration measurements run in fp32.** Training may use fp16; metrics may not.
- **Readout is strictly linear** (`z = Wh + b`). No MLP head. Non-negotiable — §8 depends on it.
- **Pooling is attention or strided; never mean.** Mean pooling destroys prosodic contours.
- **Never augment the waveform.** No speed perturbation, pitch shift, or noise. Question-side augmentation only.
- **Every label carries a `LabelTier`.** Tier `MODEL_OUTPUT` must never enter a thesis-testing split; this is enforced in code, not by convention.
- **All audio resampled to 16kHz mono** before the frozen encoder.
- **All training runs log to Weights & Biases**, project `prosodia`.
- **Splits are session-disjoint on IEMOCAP** (leave-one-session-out; genuinely speaker-disjoint, enforced by `assert_speaker_disjoint`) but only **dialogue-disjoint and speaker-shared on MELD** (see the Decision 1 correction above and spec §4.1 — MELD results are pipeline validation only, not evidence for an audio-improves-calibration claim). Never utterance-random on either.
- Feature cache: fp16 on disk, **fp32 at measurement time**.

---

### Task 1: Project scaffold, device selection, numerical guard

**Files:**
- Create: `pyproject.toml`
- Create: `src/prosodia/__init__.py`
- Create: `src/prosodia/device.py`
- Test: `tests/test_device.py`

**Interfaces:**
- Consumes: nothing
- Produces: `get_device() -> torch.device`; `assert_close_across_devices(fn, *args, atol=1e-4) -> None`

- [ ] **Step 1: Create the uv project**

Initialise in place, inside the existing repo, so the code is version-controlled alongside the spec and plan:

```bash
cd /Users/alperbalbay/coding/jev/.worktrees/prosodia-phase1
uv init --python 3.12 --lib --name prosodia
uv add torch torchaudio transformers sentence-transformers soundfile librosa numpy scipy scikit-learn pandas wandb
uv add --dev pytest pytest-cov
```

Verify `ffmpeg` is available (Task 3's fetch script needs it to extract audio from MELD's mp4s):

```bash
ffmpeg -version >/dev/null 2>&1 && echo "ffmpeg ok" || brew install ffmpeg
```

- [ ] **Step 2: Write the failing test**

```python
# tests/test_device.py
import torch
from prosodia.device import get_device, assert_close_across_devices


def test_get_device_returns_a_real_device():
    dev = get_device()
    assert dev.type in {"mps", "cuda", "cpu"}
    torch.zeros(4, device=dev)  # must actually allocate


def test_assert_close_across_devices_passes_for_stable_op():
    def softmax_entropy(x):
        p = torch.softmax(x, dim=-1)
        return -(p * p.log()).sum(-1)

    x = torch.randn(8, 16, dtype=torch.float32)
    assert_close_across_devices(softmax_entropy, x)


def test_assert_close_across_devices_raises_on_divergence():
    def bad(x):
        # deliberately device-dependent
        return x.sum() + (0.0 if x.device.type == "cpu" else 1.0)

    x = torch.randn(4, dtype=torch.float32)
    dev = get_device()
    if dev.type == "cpu":
        return  # nothing to compare against
    try:
        assert_close_across_devices(bad, x)
    except AssertionError:
        return
    raise AssertionError("expected divergence to be caught")
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/test_device.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'prosodia.device'`

- [ ] **Step 4: Implement**

```python
# src/prosodia/device.py
"""Device selection and the numerical-fidelity guard.

Prosodia claims small effects (ECE deltas ~0.01, entropy shifts from single
ablations). Those are exactly the magnitudes an MPS backend inconsistency can
manufacture. Measurement code runs fp32 and is checked against CPU.
"""
from __future__ import annotations

import torch


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def assert_close_across_devices(fn, *args, atol: float = 1e-4, **kwargs) -> None:
    """Run `fn` on the accelerator and on CPU in fp32; assert agreement.

    No-op when the accelerator *is* CPU. Use on every metric and every
    interpretability measurement.
    """
    dev = get_device()
    if dev.type == "cpu":
        return

    def _to(obj, d):
        if isinstance(obj, torch.Tensor):
            if torch.is_floating_point(obj):
                return obj.to(device=d, dtype=torch.float32)
            return obj.to(device=d)
        return obj

    acc = fn(*[_to(a, dev) for a in args], **{k: _to(v, dev) for k, v in kwargs.items()})
    cpu = fn(*[_to(a, "cpu") for a in args], **{k: _to(v, "cpu") for k, v in kwargs.items()})
    acc_t = acc.detach().to("cpu", torch.float32)
    cpu_t = cpu.detach().to(torch.float32)
    max_diff = (acc_t - cpu_t).abs().max().item()
    if max_diff > atol:
        raise AssertionError(
            f"{getattr(fn, '__name__', fn)} diverges across devices: "
            f"max|{dev.type} - cpu| = {max_diff:.3e} > atol={atol:.3e}"
        )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_device.py -v`
Expected: 5 passed

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml src/prosodia/ tests/test_device.py
git commit -m "feat: project scaffold, device selection, cross-device numerical guard"
```

---

### Task 2: Schema with label-tier enforcement

**Files:**
- Create: `src/prosodia/schema.py`
- Test: `tests/test_schema.py`

**Interfaces:**
- Consumes: nothing
- Produces: `LabelTier` (enum: `GOLD`, `HUMAN`, `MODEL_OUTPUT`); `Label(value, tier)`; `QuestionSpec(key, qtype, instructions, criteria)` with `qtype in {"noul","choice","score"}`; `Example(uid, corpus, audio_path, context, labels, speaker=None)`; `assert_thesis_safe(examples, question_keys) -> None`; `assert_speaker_disjoint(splits: dict[str, list[Example]]) -> None` (owner Decision 1, 2026-09-19 — see the corrected Global Constraints note above)

- [ ] **Step 1: Write the failing test**

*(Synced to the actual, current `tests/test_schema.py` — 12 tests: the original 5 plus post-review fixes for the `score` bare-string hole and `Label`'s tier coercion/rejection paths, plus 3 new tests for `assert_speaker_disjoint` added under Decision 1.)*

```python
# tests/test_schema.py
import pytest
from prosodia.schema import (
    Example, Label, LabelTier, QuestionSpec, assert_speaker_disjoint, assert_thesis_safe,
)


def _ex(uid, tier):
    return Example(
        uid=uid, corpus="meld", audio_path=f"/tmp/{uid}.wav",
        context="SPEAKER: hello", labels={"sentiment": Label(1, tier)},
    )


def _speaker_ex(uid, speaker):
    return Example(
        uid=uid, corpus="iemocap", audio_path=f"/tmp/{uid}.wav",
        context="SPEAKER: hello", labels={"sentiment": Label(1, LabelTier.GOLD)},
        speaker=speaker,
    )


def test_choice_spec_requires_at_least_two_options():
    with pytest.raises(ValueError):
        QuestionSpec("x", "choice", "Pick one", {"only": None})


def test_score_spec_requires_ordered_levels():
    spec = QuestionSpec("sent", "score", "Rate sentiment",
                        ["negative", "neutral", "positive"])
    assert spec.n_options == 3


def test_score_spec_rejects_a_bare_string_as_criteria():
    """A `str` IS a `Sequence` in Python, so `isinstance(criteria, Sequence)`
    alone lets a bare string slip through validation as if it were a list of
    single-character levels -- e.g. criteria="ab" would validate with levels
    ["a", "b"]. That is silent label corruption for any caller who passes a
    string by mistake instead of a list/tuple of level names."""
    with pytest.raises(ValueError):
        QuestionSpec("s", "score", "Rate it", "ab")


def test_noul_spec_has_two_implicit_options():
    assert QuestionSpec("q", "noul", "Is it urgent?", None).n_options == 2


def test_thesis_safe_accepts_gold_and_human():
    assert_thesis_safe([_ex("a", LabelTier.GOLD), _ex("b", LabelTier.HUMAN)],
                       ["sentiment"])


def test_thesis_safe_rejects_model_output():
    with pytest.raises(ValueError, match="MODEL_OUTPUT"):
        assert_thesis_safe([_ex("a", LabelTier.MODEL_OUTPUT)], ["sentiment"])


def test_label_coerces_string_tier():
    """Label should coerce raw string tier values to LabelTier."""
    lab = Label(value=1, tier="gold")
    assert lab.tier == LabelTier.GOLD
    assert isinstance(lab.tier, LabelTier)


def test_label_rejects_invalid_tier():
    """Label should raise ValueError on invalid tier value."""
    with pytest.raises(ValueError, match="tier must be"):
        Label(value=1, tier="invalid_tier")


def test_assert_thesis_safe_rejects_missing_key():
    """assert_thesis_safe should raise if a question_key never appears in examples."""
    with pytest.raises(ValueError, match="unrecognized keys"):
        assert_thesis_safe([_ex("a", LabelTier.GOLD)], ["sentiment", "typo_key"])


def test_assert_speaker_disjoint_passes_when_no_speaker_repeats():
    splits = {
        "train": [_speaker_ex("a", "Ses01"), _speaker_ex("b", "Ses02")],
        "dev": [_speaker_ex("c", "Ses03")],
        "test": [_speaker_ex("d", "Ses04")],
    }
    assert_speaker_disjoint(splits) is None  # must not raise


def test_assert_speaker_disjoint_raises_and_names_the_overlapping_speaker():
    """Fault it exists to catch: MELD-style splits where a speaker (e.g. one
    of the six recurring leads) shows up in more than one split -- the exact
    confound Decision 1 documents rather than silently re-splits away."""
    splits = {
        "train": [_speaker_ex("a", "Joey"), _speaker_ex("b", "Ross")],
        "dev": [_speaker_ex("c", "Joey")],
        "test": [_speaker_ex("d", "Chandler")],
    }
    with pytest.raises(ValueError, match="Joey"):
        assert_speaker_disjoint(splits)


def test_assert_speaker_disjoint_ignores_examples_with_no_speaker():
    """Corpora that never populate `Example.speaker` (speaker=None) must not
    false-positive just because multiple splits share the same None value."""
    splits = {
        "train": [_ex("a", LabelTier.GOLD)],
        "dev": [_ex("b", LabelTier.GOLD)],
    }
    assert_speaker_disjoint(splits) is None  # must not raise
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_schema.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'prosodia.schema'`

- [ ] **Step 3: Implement**

*(Synced to the actual, current `src/prosodia/schema.py`: `QuestionSpec.__post_init__`'s `score` branch excludes `str` explicitly — a bare string is a `Sequence` too, and would otherwise validate as a list of single-character levels — and `assert_speaker_disjoint` is added per Decision 1.)*

```python
# src/prosodia/schema.py
"""Core types. Label provenance is a first-class field, not a comment.

Spec §4.0 / §11 trap 9: HarperValleyBank's valence labels turned out to be a
proprietary audio model's outputs. Training a prosody thesis on them would
have been circular. The tier travels with every label so that failure mode
is impossible to repeat by accident.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Sequence


class LabelTier(str, Enum):
    GOLD = "gold"                  # true by construction (e.g. assigned task)
    HUMAN = "human"                # human annotators
    MODEL_OUTPUT = "model_output"  # another model's predictions — breadth only


THESIS_SAFE_TIERS = frozenset({LabelTier.GOLD, LabelTier.HUMAN})

QType = str  # "noul" | "choice" | "score"


@dataclass(frozen=True)
class Label:
    value: Any
    tier: LabelTier

    def __post_init__(self) -> None:
        if not isinstance(self.tier, LabelTier):
            try:
                coerced = LabelTier(self.tier)
                object.__setattr__(self, "tier", coerced)
            except (ValueError, KeyError):
                raise ValueError(
                    f"tier must be a LabelTier or valid tier value, got {self.tier!r}"
                )


@dataclass(frozen=True)
class QuestionSpec:
    key: str
    qtype: QType
    instructions: str
    criteria: Any = None  # choice: dict[str, str|None]; score: ordered list; noul: dict|None

    def __post_init__(self) -> None:
        if self.qtype not in {"noul", "choice", "score"}:
            raise ValueError(f"unknown qtype {self.qtype!r}")
        if self.qtype == "choice":
            if not isinstance(self.criteria, dict) or len(self.criteria) < 2:
                raise ValueError("choice requires a dict of >= 2 options")
        if self.qtype == "score":
            if (
                not isinstance(self.criteria, Sequence)
                or isinstance(self.criteria, str)
                or len(self.criteria) < 2
            ):
                raise ValueError("score requires an ordered sequence of >= 2 levels")

    @property
    def options(self) -> list[str]:
        if self.qtype == "noul":
            return ["false", "true"]
        if self.qtype == "choice":
            return list(self.criteria.keys())
        return [str(c) for c in self.criteria]

    @property
    def n_options(self) -> int:
        return len(self.options)


@dataclass(frozen=True)
class Example:
    uid: str
    corpus: str
    audio_path: str
    context: str
    labels: dict[str, Label] = field(default_factory=dict)
    speaker: str | None = None


def assert_speaker_disjoint(splits: dict[str, list[Example]]) -> None:
    """Raise if any speaker appears in more than one split.

    Speaker leakage across train/dev/test lets a model key off speaker
    identity instead of the signal a question actually asks about, and it
    is far easier to recover identity from acoustics than from text — so an
    audio arm evaluated on leaked speakers looks artificially strong
    specifically on the axis this project's thesis depends on.

    Intentionally NOT called on `MeldCorpus`: MELD's shipped splits are
    dialogue-disjoint, not speaker-disjoint (the six *Friends* leads appear
    in train, dev, and test), and would fail this by design. See
    `scripts/run_ablation.py`'s startup warning for that corpus. IEMOCAP's
    leave-one-session-out protocol is genuinely speaker-disjoint and is
    expected to satisfy this guard.
    """
    speaker_to_splits: dict[str, set[str]] = {}
    for split_name, examples in splits.items():
        for ex in examples:
            if ex.speaker is None:
                continue
            speaker_to_splits.setdefault(ex.speaker, set()).add(split_name)

    overlapping = {
        speaker: sorted(names)
        for speaker, names in speaker_to_splits.items()
        if len(names) > 1
    }
    if overlapping:
        detail = "; ".join(
            f"{speaker!r} in {names}" for speaker, names in sorted(overlapping.items())
        )
        raise ValueError(
            "splits are not speaker-disjoint — the following speakers appear "
            f"in more than one split: {detail}"
        )


def assert_thesis_safe(examples: Iterable[Example], question_keys: Sequence[str]) -> None:
    """Raise if any label backing a thesis-testing question is a model output."""
    examples_list = list(examples)

    # Check that all question_keys appear in at least one example.
    all_keys: set[str] = set()
    for ex in examples_list:
        all_keys.update(ex.labels.keys())

    missing_keys = set(question_keys) - all_keys
    if missing_keys:
        raise ValueError(
            f"question_keys contain unrecognized keys — {sorted(missing_keys)} "
            "do not appear in any example. Possible typo?"
        )

    # Check for non-safe tiers.
    offenders: set[str] = set()
    for ex in examples_list:
        for key in question_keys:
            lab = ex.labels.get(key)
            if lab is not None and lab.tier not in THESIS_SAFE_TIERS:
                offenders.add(f"{key}:{lab.tier.value}")
    if offenders:
        raise ValueError(
            "thesis-testing split contains non-gold/human labels — "
            f"MODEL_OUTPUT present for {sorted(offenders)}. See spec §4.0."
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_schema.py -v`
Expected: 12 passed

- [ ] **Step 5: Commit**

```bash
git add src/prosodia/schema.py tests/test_schema.py
git commit -m "feat: schema with label-tier provenance enforcement"
```

---

### Task 3: Corpus protocol and MELD loader

**Correction (I8, final-review fix, 2026-09-20):** the `build_context` code block below is superseded. It serialized `f"{Speaker}: {Utterance}"` straight from MELD's gold CSV, which left two leakage channels open past what the acute current-utterance/future-turn/cross-dialogue guards (Step 6's tests) cover: (1) speaker names — MELD's splits are speaker-SHARED (Decision 1), so a name in the context handed the text side a direct identity key, a second leakage channel alongside the acoustic one; (2) sentence punctuation (`!`, `?`, ...) — itself affect-bearing, and not what a deployed streaming-ASR context would contain (spec §11 trap 4's guardrail), so it inflated the text-only baseline's floor relative to what audio has to beat. The current `src/prosodia/corpora/meld.py`'s `build_context` strips both: no speaker prefix, and `_strip_punctuation` removes sentence punctuation while preserving word-internal apostrophes (contractions like "don't", possessives like "Ross's") — replace-with-space for general punctuation (so `"Wait...what?!"` becomes `"Wait what"`, never `"Waitwhat"`), then delete any apostrophe not flanked by a word character on both sides, then collapse whitespace. Turn structure (one line per prior utterance, `max_turns` respected, current/future/cross-dialogue exclusion) is unchanged. This is still gold-transcript text, not streaming ASR output — see the spec §12 limitations update; running real ASR over the corpus remains out of scope. See `tests/test_meld.py::test_build_context_strips_speaker_names`, `::test_build_context_strips_sentence_punctuation`, and `::test_build_context_preserves_word_internal_apostrophes` for the regression coverage.

**Files:**
- Create: `src/prosodia/corpora/__init__.py`
- Create: `src/prosodia/corpora/base.py`
- Create: `src/prosodia/corpora/meld.py`
- Create: `scripts/fetch_meld.sh`
- Test: `tests/test_meld.py`

**Interfaces:**
- Consumes: `Example`, `Label`, `LabelTier`, `QuestionSpec` from Task 2
- Produces: `Corpus` protocol with `.name: str`, `.question_specs() -> list[QuestionSpec]`, `.iter_examples(split: str) -> Iterator[Example]`; `MeldCorpus(root: Path)`; `build_context(rows, idx, max_turns=6) -> str`

- [ ] **Step 1: Write the fetch script**

```bash
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
```

- [ ] **Step 2: Write the failing test**

```python
# tests/test_meld.py
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
    ctx = build_context(ROWS, idx=3)
    assert "You liked it?" in ctx and "Oh yeah!" in ctx
    assert "You fell asleep!" not in ctx  # never leak the current utterance


def test_context_excludes_other_dialogues():
    # ROWS[1] (Dialogue_ID="1") sits at CSV position 1, strictly before idx=3,
    # so a naive rows[:idx] slice without the dialogue filter would include
    # it. The filter must exclude it even though it precedes the current row.
    ctx = build_context(ROWS, idx=3)
    assert "Hi there." not in ctx  # different Dialogue_ID — must never bleed in
    assert "You liked it?" in ctx and "Oh yeah!" in ctx  # same-dialogue turns still present


def test_context_is_empty_for_first_turn():
    assert build_context(ROWS, idx=0) == ""


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
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/test_meld.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'prosodia.corpora'`

- [ ] **Step 4: Implement the protocol**

```python
# src/prosodia/corpora/base.py
from __future__ import annotations

from typing import Iterator, Protocol, runtime_checkable

from prosodia.schema import Example, QuestionSpec


@runtime_checkable
class Corpus(Protocol):
    """Loaders are interchangeable so the build is decoupled from IEMOCAP
    registration lead time (spec §4.3)."""

    name: str

    def question_specs(self) -> list[QuestionSpec]: ...
    def iter_examples(self, split: str) -> Iterator[Example]: ...
```

- [ ] **Step 5: Implement the MELD loader**

```python
# src/prosodia/corpora/meld.py
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
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/test_meld.py -v`
Expected: 5 passed originally; 8 passed after I8's speaker-name/punctuation-stripping tests were added (see the Correction note above)

- [ ] **Step 7: Fetch the real corpus and smoke-check counts**

```bash
bash scripts/fetch_meld.sh ~/prosodia-data/meld
uv run python -c "
from pathlib import Path
from prosodia.corpora.meld import MeldCorpus
c = MeldCorpus(Path.home()/'prosodia-data/meld')
for s in ('train','dev','test'):
    print(s, sum(1 for _ in c.iter_examples(s)))
"
```

Expected: roughly `train 9989`, `dev 1109`, `test 2610` (slightly lower is fine — undecodable clips are skipped).

- [ ] **Step 8: Commit**

```bash
git add src/prosodia/corpora/ scripts/fetch_meld.sh tests/test_meld.py
git commit -m "feat: corpus protocol and MELD loader with dialogue context"
```

---

### Task 4: Question bank and label-preserving augmentation

**Files:**
- Create: `src/prosodia/questions.py`
- Test: `tests/test_questions.py`

**Interfaces:**
- Consumes: `QuestionSpec` from Task 2
- Produces: `PARAPHRASES: dict[str, list[str]]`; `paraphrase(spec, rng) -> QuestionSpec`; `permute_candidates(spec, rng, min_options=2) -> tuple[QuestionSpec, dict[str,str]]`; `holdout_split(specs, held_out_keys) -> tuple[list, list]`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_questions.py
import ast
import random
from pathlib import Path

import pytest

from prosodia.corpora.meld import MeldCorpus
from prosodia.questions import (
    holdout_split, paraphrase, permute_candidates,
)
from prosodia.schema import QuestionSpec

CHOICE = QuestionSpec("emotion", "choice", "Which emotion is the speaker expressing?",
                      {"anger": None, "joy": None, "neutral": None, "sadness": None})
SCORE = QuestionSpec("sentiment", "score", "Rate the sentiment the speaker conveys.",
                     ["negative", "neutral", "positive"])


def test_paraphrase_changes_wording_but_not_identity():
    rng = random.Random(0)
    p = paraphrase(CHOICE, rng)
    assert p.key == CHOICE.key
    assert p.qtype == CHOICE.qtype
    assert p.options == CHOICE.options
    assert p.instructions != CHOICE.instructions


def test_permute_candidates_always_keeps_the_gold_option():
    """Dropping the gold option would make the question unanswerable, so the
    example would be silently skipped — quietly biasing the training set
    toward whichever classes survive sampling most often."""
    rng = random.Random(1)
    for _ in range(100):
        spec, mapping = permute_candidates(CHOICE, rng, keep="joy")
        assert "joy" in spec.options
        assert 2 <= spec.n_options <= CHOICE.n_options
        assert set(mapping).issubset(set(CHOICE.options))
        assert len(set(mapping.values())) == len(mapping)


def test_permute_candidates_respects_min_options_with_no_keep():
    """The >= min_options invariant must hold unconditionally, not only when
    `keep` is supplied. The naive lower bound
    `max(min_options - 1, 0)` assumes `keep` will always be appended back in,
    so with keep=None it can sample down to a single option, and
    QuestionSpec.__post_init__ rejects a choice spec with < 2 options.
    Simulating that formula over 2000 seeds fails 519 times (~26%); 500 seeds
    is plenty to catch it reliably."""
    for seed in range(500):
        rng = random.Random(seed)
        spec, mapping = permute_candidates(CHOICE, rng)
        assert spec.n_options >= 2
        assert len(mapping) >= 2


def test_permute_candidates_raises_when_min_options_exceeds_available_options():
    """The docstring promises the result always has at least `min_options`
    options. When that's impossible — `min_options` exceeds the spec's own
    option count — the only honest response is a loud, diagnosable error.
    Silently returning fewer options than promised (the old clamped
    behaviour) would be a contract violation in a module whose entire job is
    provable label-preserving transforms."""
    rng = random.Random(3)
    with pytest.raises(ValueError):
        permute_candidates(CHOICE, rng, keep="joy", min_options=10)


def test_permute_never_applies_to_score_questions():
    # Score levels are ORDERED; subsetting or shuffling them destroys the label.
    rng = random.Random(2)
    spec, mapping = permute_candidates(SCORE, rng, keep="neutral")
    assert spec.criteria == SCORE.criteria
    assert mapping == {o: o for o in SCORE.options}


def test_holdout_split_is_disjoint():
    specs = MeldCorpus(".").question_specs()
    train, held = holdout_split(specs, ["is_negative"])
    assert {s.key for s in train}.isdisjoint({s.key for s in held})
    assert {s.key for s in held} == {"is_negative"}


_BANNED_WAVEFORM_IDENTIFIERS = frozenset({
    "pitch_shift", "time_stretch", "speed", "speed_perturb", "resample",
    "add_noise",
})
_BANNED_WAVEFORM_MODULES = ("librosa.effects", "torchaudio.sox_effects")


def _dotted_attr_chain(node: ast.expr) -> str | None:
    """Reconstruct 'a.b.c' from a Name/Attribute chain, else None."""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def _waveform_augmentation_offenders(src_root: Path) -> list[str]:
    """Scan every .py file under `src_root` at the AST level for waveform
    augmentation: function/method definitions named like a waveform op, calls
    to one (by bare name or dotted attribute chain), and imports of one --
    whether from `prosodia.questions` or anywhere else in the package."""
    offenders: list[str] = []
    for path in sorted(src_root.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        rel = path.relative_to(src_root.parent)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name in _BANNED_WAVEFORM_IDENTIFIERS:
                    offenders.append(f"{rel}:{node.lineno} defines {node.name!r}")
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    chain = node.func.id
                else:
                    chain = _dotted_attr_chain(node.func)
                if chain is not None and (
                    chain.split(".")[-1] in _BANNED_WAVEFORM_IDENTIFIERS
                    or any(chain == m or chain.startswith(m + ".")
                           for m in _BANNED_WAVEFORM_MODULES)
                ):
                    offenders.append(f"{rel}:{node.lineno} calls {chain!r}")
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                for alias in node.names:
                    full = f"{mod}.{alias.name}" if mod else alias.name
                    if (
                        alias.name in _BANNED_WAVEFORM_IDENTIFIERS
                        or mod in _BANNED_WAVEFORM_MODULES
                        or any(full.startswith(m + ".") or full == m
                               for m in _BANNED_WAVEFORM_MODULES)
                    ):
                        offenders.append(f"{rel}:{node.lineno} imports {full!r}")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if any(alias.name == m or alias.name.startswith(m + ".")
                           for m in _BANNED_WAVEFORM_MODULES):
                        offenders.append(f"{rel}:{node.lineno} imports {alias.name!r}")
    return offenders


def test_no_waveform_augmentation_anywhere_in_src():
    """Spec §11 trap 6: waveform augmentation (pitch/speed/time perturbation)
    mangles the exact prosodic cues the thesis is about, so it must never
    exist anywhere in the package -- not just be absent from
    `prosodia.questions`'s public names.

    The old version of this test (`test_no_waveform_augmentation_is_exported`)
    checked a fixed list of banned NAMES against `dir(prosodia.questions)`.
    That is a guardrail test in the same broken family as C1's pooling test
    (2026-09-19 final review): it is defeated by adding a differently-named
    function, or by adding the banned functionality to any module other than
    `questions.py`. This version AST-scans every file under `src/` for
    function/method DEFINITIONS, CALLS (by bare name or dotted attribute
    chain), and IMPORTS matching known waveform-augmentation identifiers or
    modules (`librosa.effects.*`, `torchaudio.sox_effects.*`), so a
    newly-added waveform op is caught regardless of which module or name it
    lands under."""
    src_root = Path(__file__).resolve().parent.parent / "src"
    offenders = _waveform_augmentation_offenders(src_root)
    assert not offenders, "waveform augmentation found in src/:\n" + "\n".join(offenders)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_questions.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'prosodia.questions'`

- [ ] **Step 3: Implement**

```python
# src/prosodia/questions.py
"""Question-side augmentation.

Spec §4.4: labels are expensive, question phrasings are free. Every transform
here provably preserves the target — rephrasing a question does not change
which emotion was expressed, and permuting option order does not change which
option is correct.

There is deliberately NO waveform augmentation in this codebase. Speed
perturbation and pitch shifting are label-preserving only by assumption, and
they mangle the exact cues the thesis is about (spec §11 trap 6).
"""
from __future__ import annotations

import random
from typing import Sequence

from prosodia.schema import QuestionSpec

PARAPHRASES: dict[str, list[str]] = {
    "emotion": [
        "Which emotion is the speaker expressing?",
        "What emotion comes through in this utterance?",
        "Identify the speaker's emotional state.",
        "How does the speaker feel here?",
        "Which of these emotions best fits what you hear?",
    ],
    "sentiment": [
        "Rate the sentiment the speaker conveys.",
        "How positive or negative is this utterance?",
        "Where does this fall on a negative-to-positive scale?",
        "Judge the overall sentiment expressed.",
        "Rate how favourable the speaker sounds.",
    ],
    "is_negative": [
        "Is the speaker expressing something negative?",
        "Does this utterance carry negative sentiment?",
        "Would you describe the speaker as sounding negative?",
        "Is there negativity in what the speaker says?",
        "Does the speaker come across as unhappy or critical?",
    ],
}


def paraphrase(spec: QuestionSpec, rng: random.Random) -> QuestionSpec:
    """Return the same question with different wording."""
    options = PARAPHRASES.get(spec.key)
    if not options:
        return spec
    alternatives = [o for o in options if o != spec.instructions] or options
    return QuestionSpec(spec.key, spec.qtype, rng.choice(alternatives), spec.criteria)


def permute_candidates(
    spec: QuestionSpec, rng: random.Random, keep: str | None = None,
    min_options: int = 2,
) -> tuple[QuestionSpec, dict[str, str]]:
    """Subsample and shuffle a Choice option set.

    `keep` is the gold option and, when given, is always retained — a
    question whose correct answer is not on the menu is unanswerable, and
    silently skipping those examples would bias the training set toward
    frequently-sampled classes.

    The result always has at least `min_options` options, whether or not
    `keep` is supplied. This must hold unconditionally: computing the lower
    sample bound as if `keep` will always be appended back in (regardless of
    whether it actually is) lets the count fall below `min_options` whenever
    `keep` is None or not among the spec's options — silently producing a
    single-option QuestionSpec that QuestionSpec.__post_init__ rejects.

    Raises `ValueError` if `min_options` exceeds the number of options the
    spec itself has — the guarantee above would otherwise be impossible to
    keep, and returning fewer options than promised without saying so would
    be silent, undiagnosable label corruption in exactly the module whose
    job is to make transforms provably label-preserving.

    Score questions are returned untouched: their levels are ORDERED, so
    subsetting or shuffling would corrupt the target. `min_options` is not
    validated against them for this reason.
    """
    identity = {o: o for o in spec.options}
    if spec.qtype != "choice":
        return spec, identity

    options = list(spec.criteria.keys())
    if min_options > len(options):
        raise ValueError(
            f"min_options={min_options} exceeds the {len(options)} options "
            f"available on spec {spec.key!r}"
        )
    keep_present = keep is not None and keep in options
    pool = [o for o in options if o != keep] if keep_present else list(options)

    # `keep_present` options are added back after sampling, so the pool only
    # needs to supply `min_options - 1` of them; otherwise the pool alone
    # must supply the full `min_options`. The validation above guarantees
    # this lower bound never exceeds len(pool), so no clamp is needed here.
    low = max(min_options - 1, 0) if keep_present else min_options
    k = rng.randint(low, len(pool))
    kept = rng.sample(pool, k)
    if keep_present:
        kept.append(keep)
    rng.shuffle(kept)
    return (
        QuestionSpec(spec.key, "choice", spec.instructions,
                     {o: spec.criteria[o] for o in kept}),
        {o: o for o in kept},
    )


def holdout_split(
    specs: Sequence[QuestionSpec], held_out_keys: Sequence[str]
) -> tuple[list[QuestionSpec], list[QuestionSpec]]:
    """Split the bank into seen and never-seen questions (zero-shot eval)."""
    held = set(held_out_keys)
    return ([s for s in specs if s.key not in held],
            [s for s in specs if s.key in held])
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_questions.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add src/prosodia/questions.py tests/test_questions.py
git commit -m "feat: question bank with label-preserving augmentation only"
```

---

### Task 5: Frozen encoder feature extraction and shard cache

**Correction (duration-cap fix, owner-reported, 2026-09-20):** a real extraction run over MELD died mid-corpus (11,444 of 13,706 utterances) with `RuntimeError: Invalid buffer size: 13.85 GiB`. WavLM's gated relative-position bias builds a T×T index tensor, so memory scales with the *square* of clip length; MELD contains two multi-minute segmentation artifacts (`dia38_utt4.wav`, 304.9s for the transcript *"Oh it's great, it's a role on"*; `dia220_utt0.wav`, 235.1s for *"What's that smell?"*) plus one genuinely long turn in the 20–41s range. The code block below is superseded: `FeatureExtractor.__init__` now takes `max_audio_seconds` (default `DEFAULT_MAX_AUDIO_SECONDS = 30.0`), and `encode` calls a new module function `truncate_to_max_seconds(wav, max_seconds)` as its first line, before dispatching to any of the three encoder arms — uniform treatment matters because `assert_uniform_cache_coverage` (Task 15) requires every encoder's cache to cover an identical uid set, so a cap that behaved differently per arm would break the cross-arm comparison. `encode` sets `self.last_truncated` so `scripts/extract_features.py` can report every truncated uid loudly at the end of a run rather than truncating silently. At the default 30s, the cap coincides exactly with `WhisperFeatureExtractor`'s own fixed 30s mel window (480,000 samples either way), so it is a no-op for the whisper arm specifically and only changes behavior for wavlm/prosody, which have no internal bound of their own; tuning the cap *above* 30s would silently stop matching Whisper's own window (Whisper still can't see past 30s), so it is only safe to tune it down, not up, without also revisiting that interaction. This affects 3 of 13,706 MELD utterances (0.02%) and is recorded in spec §12's limitations. See `tests/test_features.py`'s three new `test_duration_cap_*` tests (fault-injected: with the truncation call removed, the long-clip test measures more frames than the capped reference and the whisper-arm test finds `last_truncated is False`) and the current `src/prosodia/features.py`/`scripts/extract_features.py` for the real implementation, which supersedes the code blocks below.

**Files:**
- Create: `src/prosodia/features.py`
- Create: `scripts/extract_features.py`
- Test: `tests/test_features.py`

**Interfaces:**
- Consumes: `Example` (Task 2), `Corpus` (Task 3), `get_device` (Task 1)
- Produces: `ENCODERS: dict[str, str]`; `FeatureExtractor(encoder_key, device)` with `.encode(wav: np.ndarray) -> torch.Tensor` of shape `(T, D)`; `FeatureCache(path)` with `.write(uid, tensor)`, `.read(uid) -> torch.Tensor`, `.__contains__`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_features.py
import math

import numpy as np
import torch

from prosodia.features import ENCODERS, SAMPLE_RATE, FeatureCache


def test_encoder_registry_has_the_three_ablation_arms():
    assert set(ENCODERS) == {"wavlm", "whisper", "prosody"}


def test_cache_roundtrip_is_fp16_on_disk_fp32_in_memory(tmp_path):
    cache = FeatureCache(tmp_path / "feats")
    x = torch.randn(37, 16, dtype=torch.float32)
    cache.write("utt-1", x)
    assert "utt-1" in cache
    back = cache.read("utt-1")
    assert back.dtype is torch.float32          # fp32 at measurement time
    assert back.shape == x.shape
    torch.testing.assert_close(back, x, rtol=1e-2, atol=1e-2)  # fp16 on disk


def test_cache_reports_missing_keys(tmp_path):
    assert "nope" not in FeatureCache(tmp_path / "feats")


def test_prosody_extractor_returns_explicit_f0_energy_voicing():
    from prosodia.features import FeatureExtractor
    ex = FeatureExtractor("prosody", torch.device("cpu"))
    sr = 16000
    t = np.arange(sr) / sr
    wav = (0.3 * np.sin(2 * np.pi * 150 * t)).astype(np.float32)
    feats = ex.encode(wav)
    assert feats.ndim == 2 and feats.shape[1] == 3  # f0, energy, voicing
    assert feats.shape[0] > 10


def test_whisper_encoder_output_is_trimmed_to_real_audio_frames():
    """Whisper's feature extractor zero-pads every clip to 30s, so its
    encoder always emits 1500 frames. A short MELD-length utterance must
    come back trimmed to the frame count for its real duration (50/sec),
    not the full padded 1500.
    """
    from prosodia.features import FeatureExtractor

    ex = FeatureExtractor("whisper", torch.device("cpu"))
    duration_s = 2.0
    wav = np.zeros(int(duration_s * SAMPLE_RATE), dtype=np.float32)
    feats = ex.encode(wav)

    expected_frames = math.ceil(duration_s * 50)  # 100
    assert feats.shape[0] == expected_frames
    assert feats.shape[0] < 1500


def test_whisper_feature_extractor_is_constructed_once_in_init():
    """WhisperFeatureExtractor.from_pretrained must be hoisted into
    __init__, not called on every encode() invocation, or a 13k-utterance
    extraction pass runs overnight instead of ~1 hour.
    """
    from unittest import mock

    from transformers import WhisperFeatureExtractor

    from prosodia.features import FeatureExtractor

    with mock.patch.object(
        WhisperFeatureExtractor, "from_pretrained",
        wraps=WhisperFeatureExtractor.from_pretrained,
    ) as mocked:
        ex = FeatureExtractor("whisper", torch.device("cpu"))
        assert mocked.call_count == 1

        wav = np.zeros(SAMPLE_RATE, dtype=np.float32)
        ex.encode(wav)
        ex.encode(wav)
        assert mocked.call_count == 1  # not re-constructed per encode() call
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_features.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'prosodia.features'`

- [ ] **Step 3: Implement**

```python
# src/prosodia/features.py
"""Frozen-encoder feature extraction, cached once to disk.

Three ablation arms (spec §6):
  wavlm   — primary; SSL objective retains paralinguistic information
  whisper — control; ASR objective may discard prosody at the feature boundary
  prosody — diagnostic; explicit F0/energy/voicing. Distinguishes "the encoder
            threw prosody away" from "the task does not need prosody" on a null.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch

SAMPLE_RATE = 16_000
HOP_LENGTH = 320  # 20ms at 16kHz -> 50Hz frames, matching WavLM

# Whisper's encoder always emits frames at 50/sec of *input* audio, regardless
# of how long that audio actually is (see the trimming note in `encode` below).
WHISPER_FRAMES_PER_SECOND = 50

ENCODERS: dict[str, str] = {
    "wavlm": "microsoft/wavlm-large",
    "whisper": "openai/whisper-small",
    "prosody": "__explicit__",
}


class FeatureExtractor:
    def __init__(self, encoder_key: str, device: torch.device) -> None:
        if encoder_key not in ENCODERS:
            raise ValueError(f"unknown encoder {encoder_key!r}")
        self.key = encoder_key
        self.device = device
        self._model = None
        self._feature_extractor = None
        if encoder_key == "wavlm":
            from transformers import WavLMModel
            self._model = WavLMModel.from_pretrained(ENCODERS[encoder_key])
        elif encoder_key == "whisper":
            from transformers import WhisperFeatureExtractor, WhisperModel
            self._model = WhisperModel.from_pretrained(ENCODERS[encoder_key]).encoder
            # Hoisted out of encode(): from_pretrained() is a
            # filesystem/network load. Called once per utterance across the
            # ~13k-utterance MELD corpus, that turns a ~1 hour extraction
            # pass into an overnight one.
            self._feature_extractor = WhisperFeatureExtractor.from_pretrained(
                ENCODERS["whisper"]
            )
        if self._model is not None:
            self._model.eval().to(device)
            for p in self._model.parameters():
                p.requires_grad_(False)

    @torch.no_grad()
    def encode(self, wav: np.ndarray) -> torch.Tensor:
        """(samples,) float32 @16kHz -> (T, D) float32 frame features."""
        if self.key == "prosody":
            return _explicit_prosody(wav)
        if self.key == "whisper":
            mel = self._feature_extractor(
                wav, sampling_rate=SAMPLE_RATE, return_tensors="pt"
            )
            out = self._model(mel.input_features.to(self.device)).last_hidden_state
            out = out.squeeze(0).float().cpu()
            # WhisperFeatureExtractor zero-pads every clip to a fixed 30s
            # window, so the encoder always emits a fixed 1500 frames (50/sec
            # * 30s) no matter how short the input actually was. MELD
            # utterances average ~3s (~165 real frames), so left untrimmed:
            #   (a) the cache balloons to ~59GB instead of ~8.5GB, and
            #   (b) every downstream attention mask would mark all 1500
            #       frames valid, so ~89% of what the state encoder attends
            #       over would be silent padding — quietly destroying the
            #       control arm this experiment depends on (see module
            #       docstring: whisper is the "did the ASR objective throw
            #       prosody away" test, and a broken control still produces
            #       a clean-looking comparison).
            # Trim back to the frame count that corresponds to the real
            # audio: Whisper's encoder runs at 50 frames/sec of *original*
            # audio (not of the padded 30s), so that count is
            # ceil(len(wav) / SAMPLE_RATE * 50), clamped to whatever the
            # encoder actually produced (it can't exceed the padded max).
            valid_frames = min(
                math.ceil(len(wav) / SAMPLE_RATE * WHISPER_FRAMES_PER_SECOND),
                out.shape[0],
            )
            return out[:valid_frames]
        x = torch.from_numpy(wav).float().unsqueeze(0).to(self.device)
        out = self._model(x).last_hidden_state
        return out.squeeze(0).float().cpu()


def _explicit_prosody(wav: np.ndarray) -> torch.Tensor:
    """F0, RMS energy and voicing probability at 50Hz.

    Uses librosa.pyin (YIN pitch tracking): F0 is what the interventions in
    Phase 2 manipulate, so the diagnostic arm reads exactly the manipulated
    quantity.
    """
    import librosa

    f0, voiced_flag, voiced_prob = librosa.pyin(
        wav, sr=SAMPLE_RATE, fmin=60, fmax=400, hop_length=HOP_LENGTH,
    )
    f0 = np.nan_to_num(f0, nan=0.0)
    rms = librosa.feature.rms(y=wav, hop_length=HOP_LENGTH).squeeze(0)
    n = min(len(f0), len(rms), len(voiced_prob))
    stack = np.stack([f0[:n], rms[:n], np.nan_to_num(voiced_prob[:n])], axis=-1)
    return torch.from_numpy(stack).float()


class FeatureCache:
    """fp16 on disk (halves an ~8.5GB cache), fp32 in memory (spec §10)."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)

    def _file(self, uid: str) -> Path:
        return self.path / f"{uid}.pt"

    def __contains__(self, uid: str) -> bool:
        return self._file(uid).exists()

    def write(self, uid: str, tensor: torch.Tensor) -> None:
        torch.save(tensor.to(torch.float16).contiguous(), self._file(uid))

    def read(self, uid: str) -> torch.Tensor:
        return torch.load(self._file(uid), map_location="cpu").float()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_features.py -v`
Expected: 6 passed originally; 9 passed after the duration-cap fix added `test_duration_cap_truncates_a_clip_longer_than_the_cap`, `test_duration_cap_leaves_a_clip_shorter_than_the_cap_untouched`, and `test_duration_cap_applies_uniformly_to_the_whisper_arm_too` (see the Correction note above)

- [ ] **Step 5: Write the extraction script**

```python
# scripts/extract_features.py
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
```

- [ ] **Step 6: Extract WavLM features and verify cache size**

```bash
uv run python scripts/extract_features.py \
  --corpus-root ~/prosodia-data/meld --cache-root ~/prosodia-cache --encoder wavlm
du -sh ~/prosodia-cache/wavlm
```

Expected: completes without assertion errors; cache on the order of several GB.

- [ ] **Step 7: Commit**

```bash
git add src/prosodia/features.py scripts/extract_features.py tests/test_features.py
git commit -m "feat: frozen encoder extraction with fp16 shard cache"
```

---

### Task 6: Dataset, collate, and modality dropout

**Correction (I2, final-review fix, 2026-09-20):** the `ProsodiaDataset.__init__` code block below is superseded on one point: `self.examples = [e for e in examples if e.uid in cache]` silently dropped any example missing from the feature cache, with no floor, no record, and no warning. `FeatureCache.__init__` does `mkdir(parents=True, exist_ok=True)`, so a wrong or partially-extracted cache path is *created* rather than rejected — a half-finished extraction for one arm would silently train that arm on a smaller (and different) dataset than the other arms in the grid, and the comparison goes void with nothing in the logs to say so. This became more load-bearing after the text-only baseline landed (Decision 2): those arms borrow the WavLM cache purely for `in_dim`/shape, but cache *coverage* still gates which examples their dataset contains at all, so a partial WavLM cache silently changes the baseline's training set relative to the audio arms it controls for. The fix in `src/prosodia/data.py`'s current `ProsodiaDataset.__init__`: records `dropped_uids`/`n_dropped`/`cache_coverage`; raises `ValueError` (with the coverage percentage and a sample of missing uids) below a `min_cache_coverage` floor (default `DEFAULT_MIN_CACHE_COVERAGE = 0.98`, overridable per call); prints an unmissable stderr banner (this project's established idiom — see `scripts/run_ablation.py`'s MELD warning — rather than `warnings.warn`, which `-W error` would turn into a hard failure for a condition this code deliberately tolerates) when some examples are dropped but coverage still clears the floor. This is the per-arm half of I2; `scripts/run_ablation.py`'s new `assert_uniform_cache_coverage` (Task 15) is the cross-arm half that actually protects the grid's comparison. See `tests/test_data.py::test_dataset_raises_when_cache_coverage_falls_below_the_floor` and its three sibling tests for the regression coverage.

**Files:**
- Create: `src/prosodia/data.py`
- Test: `tests/test_data.py`

**Interfaces:**
- Consumes: `Example`, `QuestionSpec`, `FeatureCache`, `paraphrase`, `permute_candidates`
- Produces: `ProsodiaDataset(examples, specs, cache, rng_seed, augment=True, modality_dropout=0.15)`; `collate_batch(items) -> dict` with keys `audio`, `audio_mask`, `audio_present`, `context`, `context_present`, `questions`, `targets`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_data.py
import torch

from prosodia.data import ProsodiaDataset, collate_batch
from prosodia.features import FeatureCache
from prosodia.schema import Example, Label, LabelTier, QuestionSpec

SPECS = [
    QuestionSpec("emotion", "choice", "Which emotion?",
                 {"anger": None, "joy": None, "neutral": None}),
    QuestionSpec("sentiment", "score", "Rate sentiment.",
                 ["negative", "neutral", "positive"]),
]


def _dataset(tmp_path, cache_dir="c", **kw):
    cache = FeatureCache(tmp_path / cache_dir)
    exs = []
    for i, T in enumerate((20, 35, 12)):
        uid = f"u{i}"
        cache.write(uid, torch.randn(T, 8))
        exs.append(Example(uid, "meld", "/x.wav", f"ctx {i}", {
            "emotion": Label("joy", LabelTier.HUMAN),
            "sentiment": Label(2, LabelTier.HUMAN),
        }, speaker="Joey"))
    return ProsodiaDataset(exs, SPECS, cache, rng_seed=0, **kw)


def test_collate_pads_to_longest_and_masks(tmp_path):
    ds = _dataset(tmp_path, augment=False, modality_dropout=0.0)
    batch = collate_batch([ds[i] for i in range(3)])
    assert batch["audio"].shape[:2] == (3, 35)
    assert batch["audio_mask"][2].sum().item() == 12
    assert batch["audio_mask"][1].sum().item() == 35


def test_modality_dropout_never_drops_both(tmp_path):
    ds = _dataset(tmp_path, augment=False, modality_dropout=0.9)
    for epoch in range(200):
        ds.set_epoch(epoch)
        item = ds[0]
        assert item["audio_present"] or item["context_present"]


def test_augmentation_preserves_the_target(tmp_path):
    ds = _dataset(tmp_path, augment=True, modality_dropout=0.0)
    for epoch in range(100):
        ds.set_epoch(epoch)
        item = ds[0]
        emo = item["questions"]["emotion"]
        # whichever subset/order the options were presented in, the target
        # index must still point at "joy"
        assert emo["options"][item["targets"]["emotion"]] == "joy"


def test_score_options_are_never_reordered(tmp_path):
    ds = _dataset(tmp_path, augment=True, modality_dropout=0.0)
    for _ in range(100):
        assert ds[0]["questions"]["sentiment"]["options"] == \
            ["negative", "neutral", "positive"]


def test_same_seed_epoch_idx_is_reproducible_across_instances(tmp_path):
    """The 9-arm ablation grid compares arms that must differ only in loss
    function or encoder. If two independently-constructed datasets with the
    same rng_seed, epoch, and index drew different augmentation/dropout
    choices, that would inject uncontrolled noise into the comparison."""
    ds1 = _dataset(tmp_path, cache_dir="c1", augment=True, modality_dropout=0.5)
    ds2 = _dataset(tmp_path, cache_dir="c2", augment=True, modality_dropout=0.5)
    ds1.set_epoch(3)
    ds2.set_epoch(3)
    item1, item2 = ds1[0], ds2[0]
    assert item1["questions"] == item2["questions"]
    assert item1["targets"] == item2["targets"]
    assert item1["audio_present"] == item2["audio_present"]
    assert item1["context_present"] == item2["context_present"]


def test_different_epoch_changes_the_draw(tmp_path):
    ds = _dataset(tmp_path, augment=True, modality_dropout=0.5)
    draws = set()
    for epoch in range(20):
        ds.set_epoch(epoch)
        item = ds[0]
        draws.add((
            item["questions"]["emotion"]["instructions"],
            tuple(item["questions"]["emotion"]["options"]),
            item["audio_present"],
            item["context_present"],
        ))
    assert len(draws) > 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_data.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'prosodia.data'`

- [ ] **Step 3: Implement**

```python
# src/prosodia/data.py
"""Dataset and collation.

Modality dropout (spec §5) yields three eval conditions — full / audio-only /
context-only — from one checkpoint, and simultaneously prevents the model from
learning to ignore audio whenever context is informative. It never drops both:
an example with no state at all carries no signal.
"""
from __future__ import annotations

import random
from typing import Any, Sequence

import torch
from torch.utils.data import Dataset

from prosodia.features import FeatureCache
from prosodia.questions import paraphrase, permute_candidates
from prosodia.schema import Example, QuestionSpec


class ProsodiaDataset(Dataset):
    """Callers must invoke `set_epoch(epoch)` before each training epoch —
    the per-item RNG is seeded deterministically from `(rng_seed, epoch,
    idx)`, so without advancing the epoch every pass over the data would
    draw the exact same augmentation and modality-dropout choices."""

    def __init__(
        self,
        examples: Sequence[Example],
        specs: Sequence[QuestionSpec],
        cache: FeatureCache,
        rng_seed: int = 0,
        augment: bool = True,
        modality_dropout: float = 0.15,
    ) -> None:
        self.examples = [e for e in examples if e.uid in cache]
        self.specs = list(specs)
        self.cache = cache
        self.augment = augment
        self.modality_dropout = modality_dropout
        self._seed = rng_seed
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Advance the RNG stream for a new pass over the data (see class
        docstring). Mirrors `DistributedSampler.set_epoch`."""
        self._epoch = epoch

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        ex = self.examples[idx]
        # Seeded from caller-controlled state only — (seed, epoch, idx) — so
        # the same triple reproduces the same draw in any process. Mixing in
        # global `random` entropy here would make the ablation grid's arms
        # differ by augmentation/dropout noise, not just by loss or encoder.
        rng = random.Random(hash((self._seed, self._epoch, idx)))

        audio_present, context_present = True, True
        if self.modality_dropout > 0 and rng.random() < self.modality_dropout:
            # drop exactly one modality, never both
            if rng.random() < 0.5:
                audio_present = False
            else:
                context_present = False

        questions: dict[str, Any] = {}
        targets: dict[str, Any] = {}
        for spec in self.specs:
            label = ex.labels.get(spec.key)
            if label is None:
                continue
            active = spec
            if self.augment:
                active = paraphrase(active, rng)
                gold = label.value if spec.qtype == "choice" else None
                active, _ = permute_candidates(active, rng, keep=gold)

            options = active.options
            if spec.qtype == "choice":
                # permute_candidates guarantees the gold option survives
                target = options.index(label.value)
            elif spec.qtype == "score":
                target = int(label.value)
            else:
                target = int(label.value)

            questions[spec.key] = {
                "instructions": active.instructions,
                "options": options,
                "qtype": spec.qtype,
            }
            targets[spec.key] = target

        return {
            "uid": ex.uid,
            "audio": self.cache.read(ex.uid),
            "audio_present": audio_present,
            "context": ex.context,
            "context_present": context_present,
            "questions": questions,
            "targets": targets,
            "speaker": ex.speaker,
        }


def collate_batch(items: Sequence[dict[str, Any]]) -> dict[str, Any]:
    lengths = [it["audio"].shape[0] for it in items]
    tmax, dim = max(lengths), items[0]["audio"].shape[1]
    audio = torch.zeros(len(items), tmax, dim)
    mask = torch.zeros(len(items), tmax, dtype=torch.bool)
    for i, it in enumerate(items):
        n = it["audio"].shape[0]
        audio[i, :n] = it["audio"]
        mask[i, :n] = True
    return {
        "uid": [it["uid"] for it in items],
        "audio": audio,
        "audio_mask": mask,
        "audio_present": torch.tensor([it["audio_present"] for it in items]),
        "context": [it["context"] for it in items],
        "context_present": torch.tensor([it["context_present"] for it in items]),
        "questions": [it["questions"] for it in items],
        "targets": [it["targets"] for it in items],
        "speaker": [it["speaker"] for it in items],
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_data.py -v`
Expected: 6 passed originally; 10 passed after I2's cache-coverage-floor tests were added (see the Correction note above)

- [ ] **Step 5: Commit**

```bash
git add src/prosodia/data.py tests/test_data.py
git commit -m "feat: dataset, collation, modality dropout"
```

---

### Task 7: Attention pooling and state encoder

**Files:**
- Create: `src/prosodia/model/__init__.py`
- Create: `src/prosodia/model/pooling.py`
- Create: `src/prosodia/model/state.py`
- Test: `tests/test_pooling.py`

**Interfaces:**
- Consumes: nothing from prior tasks
- Produces: `AttentionPool(dim, stride)` with `forward(x, mask) -> (pooled, pooled_mask)`; `speaker_relative_norm(x, mask) -> Tensor`; `StateEncoder(in_dim, d_model, n_layers, n_heads, stride)` with `forward(audio, audio_mask) -> (h, h_mask)`

- [ ] **Step 1: Write the failing test — contour preservation**

```python
# tests/test_pooling.py
import torch

from prosodia.model.pooling import AttentionPool, speaker_relative_norm
from prosodia.model.state import StateEncoder


def _ramp(n, lo, hi):
    return torch.linspace(lo, hi, n).unsqueeze(-1).repeat(1, 4).unsqueeze(0)


# NOTE: `test_attention_pool_distinguishes_a_rise_from_a_fall` used to live
# here, comparing AttentionPool's pooled SEQUENCES element-wise for a rise vs
# a fall ramp. That is not a test of time-order sensitivity: any strided
# reducer (including a plain within-window mean) preserves the time axis, so
# rise and fall differ regardless of whether the surrounding model is
# actually order-sensitive. It passed even after a within-window mean was
# swapped into the fixture, ~1000x past its own threshold (2026-09-19 final
# review, C1). C1's real invariant -- whether the ASSEMBLED MODEL'S OUTPUT
# changes under time reversal -- can only be tested at the model level; see
# `test_model.py::test_model_output_is_not_invariant_to_time_reversal` and
# `test_model.py::test_model_distinguishes_rising_from_falling_pitch_contour`
# in Task 11.


def test_attention_pool_reduces_length_by_stride():
    pool = AttentionPool(dim=4, stride=2)
    x, mask = torch.randn(2, 33, 4), torch.ones(2, 33, dtype=torch.bool)
    out, out_mask = pool(x, mask)
    assert out.shape[1] == 17 and out_mask.shape[1] == 17


def test_attention_pool_ignores_padding():
    pool = AttentionPool(dim=4, stride=2).eval()
    x = torch.randn(1, 20, 4)
    mask = torch.zeros(1, 20, dtype=torch.bool)
    mask[0, :10] = True
    padded = x.clone()
    padded[0, 10:] = 999.0  # garbage in the padded region
    with torch.no_grad():
        a, _ = pool(x, mask)
        b, _ = pool(padded, mask)
    torch.testing.assert_close(a, b)


def test_speaker_relative_norm_centres_within_utterance():
    x = torch.randn(2, 40, 6) * 3 + 10
    mask = torch.ones(2, 40, dtype=torch.bool)
    out = speaker_relative_norm(x, mask)
    assert out.mean(dim=1).abs().max().item() < 1e-5


def test_state_encoder_forward_shapes_and_mask():
    """Direct StateEncoder coverage. Also pins the fix for a UserWarning that
    nn.TransformerEncoder raises on every construction when norm_first=True
    and enable_nested_tensor isn't explicitly disabled — under -W error this
    test fails if that warning returns.

    The sequence is 2 longer than `T / stride` because of the C2 fix: two
    extra positions at the front carry the un-normalized per-channel
    utterance mean/std (see `utterance_statistics`), so the model can recover
    pitch/loudness LEVEL that `speaker_relative_norm` otherwise strips.
    """
    enc = StateEncoder(in_dim=8, d_model=16, n_layers=1, n_heads=2, stride=2).eval()
    audio = torch.randn(2, 20, 8)
    mask = torch.ones(2, 20, dtype=torch.bool)
    mask[1, 12:] = False  # second example is shorter: only 12 valid frames
    with torch.no_grad():
        h, h_mask = enc(audio, mask)
    assert h.shape == (2, 12, 16)  # 2 stat tokens + 10 pooled windows
    assert h_mask.shape == (2, 12)
    assert h_mask[0].all()
    # 2 stat tokens (both examples have >= 1 valid frame) + 6 valid windows
    # (12 valid frames / stride 2) for the shorter example.
    assert h_mask[1].sum().item() == 8
    assert not torch.isnan(h).any()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_pooling.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'prosodia.model'`

- [ ] **Step 3: Implement pooling**

```python
# src/prosodia/model/pooling.py
"""Contour-preserving pooling.

Prosody is supra-segmental: it lives in contours over time, not in frames.
Mean pooling maps a rise and a fall to the same vector, which would delete the
thesis signal in the first layer with no visible symptom (spec §11 trap 3).
AttentionPool learns per-frame weights within each window instead.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def utterance_statistics(x: Tensor, mask: Tensor, eps: float = 1e-5) -> tuple[Tensor, Tensor]:
    """Un-normalized per-channel utterance mean and std, shape (b, 1, d) each.

    C2 fix: `speaker_relative_norm` below divides these back out, which is
    exactly the point of within-utterance normalization -- but it also means
    "loud" or "high-pitched" *for this speaker* becomes unrepresentable
    downstream: a loud utterance and a quiet one with the same contour SHAPE
    normalize to bit-identical tensors. `StateEncoder` appends these raw
    stats as extra state positions so the model can recover level while the
    pooled sequence still carries the normalized contour.
    """
    m = mask.unsqueeze(-1).float()
    n = m.sum(dim=1, keepdim=True).clamp(min=1.0)
    mean = (x * m).sum(dim=1, keepdim=True) / n
    var = (((x - mean) ** 2) * m).sum(dim=1, keepdim=True) / n
    std = (var + eps).sqrt()
    return mean, std


def speaker_relative_norm(x: Tensor, mask: Tensor, eps: float = 1e-5) -> Tensor:
    """Centre and scale within the utterance.

    'High pitch' is only meaningful relative to that speaker's own baseline,
    so normalization is within-utterance, not global (spec §5). This strips
    level by construction -- see `utterance_statistics`, which callers that
    need level (e.g. `StateEncoder`) should also use.
    """
    m = mask.unsqueeze(-1).float()
    mean, std = utterance_statistics(x, mask, eps)
    return (x - mean) / std * m


class AttentionPool(nn.Module):
    """Strided pooling with learned within-window attention weights."""

    def __init__(self, dim: int, stride: int = 2) -> None:
        super().__init__()
        self.stride = stride
        self.score = nn.Linear(dim, 1)

    def forward(self, x: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        b, t, d = x.shape
        s = self.stride
        pad = (-t) % s
        if pad:
            x = F.pad(x, (0, 0, 0, pad))
            mask = F.pad(mask, (0, pad), value=False)
        tw = (t + pad) // s

        xw = x.view(b, tw, s, d)
        mw = mask.view(b, tw, s)

        logits = self.score(xw).squeeze(-1)                      # (b, tw, s)
        logits = logits.masked_fill(~mw, torch.finfo(logits.dtype).min)
        weights = torch.softmax(logits, dim=-1)
        weights = weights * mw.float()
        weights = weights / weights.sum(-1, keepdim=True).clamp(min=1e-9)

        pooled = (xw * weights.unsqueeze(-1)).sum(dim=2)
        pooled_mask = mw.any(dim=-1)
        return pooled * pooled_mask.unsqueeze(-1), pooled_mask
```

- [ ] **Step 4: Implement the state encoder**

```python
# src/prosodia/model/state.py
from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from prosodia.model.pooling import AttentionPool, speaker_relative_norm, utterance_statistics


class SinusoidalPositionalEncoding(nn.Module):
    """Adds absolute-position information to a (B, T, D) sequence.

    C1 fix: without this, `nn.TransformerEncoder` self-attention is
    permutation-equivariant and `IsolatedBranches`' cross-attention is
    permutation-invariant over its keys, so the whole model was exactly
    invariant to reversing the audio's time axis (measured max|logit diff|
    ~1e-9 for a full reversal and for a rise-vs-fall ramp pair; 2026-09-19
    final review). Position is computed fresh at forward time from `x`'s own
    shape/device/dtype so it is always correct regardless of module
    placement, and works for any sequence length the pooled state happens to
    have (including the 2 extra utterance-statistics positions the C2 fix
    prepends).
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model

    def forward(self, x: Tensor) -> Tensor:
        b, t, d = x.shape
        position = torch.arange(t, device=x.device, dtype=x.dtype).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d, 2, device=x.device, dtype=x.dtype)
            * (-math.log(10000.0) / d)
        )
        pe = torch.zeros(t, d, device=x.device, dtype=x.dtype)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        return x + pe.unsqueeze(0)


class StateEncoder(nn.Module):
    """Frozen frame features -> the shared state, encoded ONCE per example."""

    def __init__(
        self, in_dim: int, d_model: int = 256, n_layers: int = 2,
        n_heads: int = 4, stride: int = 2,
    ) -> None:
        super().__init__()
        self.project = nn.Linear(in_dim, d_model)
        self.pool = AttentionPool(d_model, stride=stride)
        self.pos_enc = SinusoidalPositionalEncoding(d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model,
            batch_first=True, norm_first=True,
        )
        # enable_nested_tensor=False: the nested-tensor fast path is already
        # unavailable because norm_first=True (deliberate). Without this,
        # nn.TransformerEncoder emits a UserWarning on every construction.
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=n_layers, enable_nested_tensor=False,
        )

    def forward(self, audio: Tensor, audio_mask: Tensor) -> tuple[Tensor, Tensor]:
        # C2 fix: compute un-normalized level stats BEFORE the normalization
        # that strips them, and carry them forward as two extra state
        # positions (see `utterance_statistics` docstring).
        mean, std = utterance_statistics(audio, audio_mask)
        stat_tokens = self.project(torch.cat([mean, std], dim=1))  # (b, 2, d)
        has_audio = audio_mask.any(dim=1, keepdim=True)             # (b, 1)
        stat_mask = has_audio.expand(-1, 2)                         # (b, 2)

        x = speaker_relative_norm(audio, audio_mask)
        x = self.project(x)
        x, mask = self.pool(x, audio_mask)

        x = torch.cat([stat_tokens, x], dim=1)
        mask = torch.cat([stat_mask, mask], dim=1)

        # C1 fix: position must be injected before the (permutation-
        # equivariant) self-attention encoder for the encoder's output to
        # depend on time order at all.
        x = self.pos_enc(x)
        h = self.encoder(x, src_key_padding_mask=~mask)
        return h, mask
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_pooling.py -v`
Expected: 4 passed

- [ ] **Step 6: Commit**

```bash
git add src/prosodia/model/ tests/test_pooling.py
git commit -m "feat: contour-preserving attention pooling and state encoder"
```

---

### Task 8: Question encoder (frozen sentence embeddings)

**Files:**
- Create: `src/prosodia/model/qencoder.py`
- Test: `tests/test_qencoder.py`

**Interfaces:**
- Consumes: nothing
- Produces: `QuestionEncoder(model_name="sentence-transformers/all-MiniLM-L6-v2", d_model=256)` with `.embed_texts(texts: list[str]) -> Tensor (N, d_model)`; caches by text

- [ ] **Step 1: Write the failing test**

```python
# tests/test_qencoder.py
import torch

from prosodia.model.qencoder import QuestionEncoder


def test_embeddings_are_projected_to_d_model():
    enc = QuestionEncoder(d_model=64)
    out = enc.embed_texts(["Which emotion is the speaker expressing?",
                           "How does the speaker feel here?"])
    assert out.shape == (2, 64)


def test_paraphrases_are_closer_than_unrelated_questions():
    enc = QuestionEncoder(d_model=64)
    v = enc.embed_texts([
        "Which emotion is the speaker expressing?",
        "What emotion comes through in this utterance?",
        "What is the account balance for this customer?",
    ])
    v = torch.nn.functional.normalize(v, dim=-1)
    assert (v[0] @ v[1]) > (v[0] @ v[2])


def test_repeated_text_is_cached_not_recomputed():
    enc = QuestionEncoder(d_model=64)
    enc.embed_texts(["Which emotion is the speaker expressing?"])
    before = enc.cache_misses
    enc.embed_texts(["Which emotion is the speaker expressing?"])
    assert enc.cache_misses == before


def test_only_projection_is_trainable():
    enc = QuestionEncoder(d_model=64)
    assert all(not p.requires_grad for p in enc._st.parameters())
    assert all(p.requires_grad for p in enc.project.parameters())


def test_sentence_transformer_stays_in_eval_mode_when_parent_trains():
    # A future training loop will call .train() on a larger model this
    # encoder is nested in. The frozen sentence-transformer must not
    # flip into train mode (which would enable dropout and make its
    # "frozen" embeddings nondeterministic).
    enc = QuestionEncoder(d_model=64)
    enc.train()
    assert enc._st.training is False
    assert enc.project.training is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_qencoder.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'prosodia.model.qencoder'`

- [ ] **Step 3: Implement**

```python
# src/prosodia/model/qencoder.py
"""Frozen sentence encoder over question instructions and candidate labels.

Frozen is the load-bearing choice: you cannot LEARN a question encoder from
~3 base question types, but you can learn to READ a pretrained semantic space.
Paraphrase augmentation is what teaches that reading (spec §5).
"""
from __future__ import annotations

import torch
from torch import Tensor, nn


class QuestionEncoder(nn.Module):
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        d_model: int = 256,
    ) -> None:
        super().__init__()
        from sentence_transformers import SentenceTransformer

        self._st = SentenceTransformer(model_name)
        for p in self._st.parameters():
            p.requires_grad_(False)
        self._st.eval()
        self.project = nn.Linear(self._st.get_embedding_dimension(), d_model)
        self._cache: dict[str, Tensor] = {}
        self.cache_misses = 0

    def train(self, mode: bool = True) -> "QuestionEncoder":
        # nn.Module.train() recurses into every submodule, including the
        # frozen sentence-transformer. If a caller trains a larger model
        # this encoder is embedded in, `.train()` would otherwise flip the
        # frozen encoder into train mode too -- enabling its dropout layers
        # and making "frozen" embeddings nondeterministic even though their
        # gradients stay off. Keep it pinned to eval regardless.
        super().train(mode)
        self._st.eval()
        return self

    def _raw(self, texts: list[str]) -> Tensor:
        missing = [t for t in texts if t not in self._cache]
        if missing:
            self.cache_misses += len(missing)
            with torch.no_grad():
                vecs = self._st.encode(missing, convert_to_tensor=True,
                                       show_progress_bar=False).cpu()
            for t, v in zip(missing, vecs):
                self._cache[t] = v
        return torch.stack([self._cache[t] for t in texts])

    def embed_texts(self, texts: list[str]) -> Tensor:
        raw = self._raw(texts).to(self.project.weight.device,
                                  self.project.weight.dtype)
        return self.project(raw)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_qencoder.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/prosodia/model/qencoder.py tests/test_qencoder.py
git commit -m "feat: frozen question and candidate encoder with caching"
```

---

### Task 9: Isolated branch layers

**Files:**
- Create: `src/prosodia/model/branches.py`
- Test: `tests/test_branches.py`

**Interfaces:**
- Consumes: nothing
- Produces: `IsolatedBranches(d_model, n_layers, n_heads)` with `forward(q, h_state, state_mask) -> Tensor (B, Q, d_model)`

- [ ] **Step 1: Write the failing test — the isolation property**

```python
# tests/test_branches.py
import torch

from prosodia.model.branches import IsolatedBranches


def test_questions_cannot_see_each_other():
    """The architectural claim (spec §2/§5): branches attend to the shared
    state but NOT to one another. Perturbing question B must leave question
    A's output bit-identical."""
    torch.manual_seed(0)
    net = IsolatedBranches(d_model=32, n_layers=2, n_heads=4).eval()
    h = torch.randn(2, 16, 32)
    mask = torch.ones(2, 16, dtype=torch.bool)
    q = torch.randn(2, 3, 32)

    q2 = q.clone()
    q2[:, 1] = torch.randn(2, 32)  # perturb only question index 1

    with torch.no_grad():
        a = net(q, h, mask)
        b = net(q2, h, mask)

    torch.testing.assert_close(a[:, 0], b[:, 0])  # untouched
    torch.testing.assert_close(a[:, 2], b[:, 2])  # untouched
    assert (a[:, 1] - b[:, 1]).abs().max().item() > 1e-4  # did change


def test_branches_do_attend_to_the_state():
    torch.manual_seed(0)
    net = IsolatedBranches(d_model=32, n_layers=1, n_heads=4).eval()
    mask = torch.ones(1, 16, dtype=torch.bool)
    q = torch.randn(1, 2, 32)
    with torch.no_grad():
        a = net(q, torch.randn(1, 16, 32), mask)
        b = net(q, torch.randn(1, 16, 32), mask)
    assert (a - b).abs().max().item() > 1e-4


def test_output_shape_matches_question_count():
    net = IsolatedBranches(d_model=32, n_layers=1, n_heads=4)
    out = net(torch.randn(4, 7, 32), torch.randn(4, 11, 32),
              torch.ones(4, 11, dtype=torch.bool))
    assert out.shape == (4, 7, 32)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_branches.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'prosodia.model.branches'`

- [ ] **Step 3: Implement**

```python
# src/prosodia/model/branches.py
"""Isolated question branches — the Jev-shaped core.

Each question cross-attends to the shared state; no self-attention runs across
the question axis. Isolation is structural, not a mask on a shared attention:
questions are folded into the batch dimension so they cannot interact at all.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn


class _BranchLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm_ff = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, q: Tensor, kv: Tensor, kv_pad: Tensor) -> Tensor:
        attn_out, _ = self.attn(self.norm_q(q), self.norm_kv(kv), self.norm_kv(kv),
                                key_padding_mask=kv_pad, need_weights=False)
        q = q + attn_out
        return q + self.ff(self.norm_ff(q))


class IsolatedBranches(nn.Module):
    def __init__(self, d_model: int = 256, n_layers: int = 2, n_heads: int = 4) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [_BranchLayer(d_model, n_heads) for _ in range(n_layers)]
        )

    def forward(self, q: Tensor, h_state: Tensor, state_mask: Tensor) -> Tensor:
        """q: (B, Q, D); h_state: (B, T, D); state_mask: (B, T) True where valid."""
        b, n_q, d = q.shape
        t = h_state.shape[1]

        # Fold questions into batch: each becomes an independent sequence of
        # length 1. Cross-question attention is then impossible by construction.
        qf = q.reshape(b * n_q, 1, d)
        kv = h_state.unsqueeze(1).expand(b, n_q, t, d).reshape(b * n_q, t, d)
        kv_pad = (~state_mask).unsqueeze(1).expand(b, n_q, t).reshape(b * n_q, t)

        for layer in self.layers:
            qf = layer(qf, kv, kv_pad)
        return qf.reshape(b, n_q, d)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_branches.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/prosodia/model/branches.py tests/test_branches.py
git commit -m "feat: isolated question branches with structural isolation"
```

---

### Task 10: Three heads with a strictly linear readout

**Files:**
- Create: `src/prosodia/model/heads.py`
- Test: `tests/test_heads.py`

**Interfaces:**
- Consumes: nothing
- Produces: `ReadoutHead(d_model)` with `.forward(branch_vec, option_vecs) -> logits`; `score_expectation(probs) -> Tensor`; `confidence(probs) -> Tensor`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_heads.py
import torch

from prosodia.model.heads import ReadoutHead, confidence, score_expectation


def test_readout_is_linear_in_the_branch_vector():
    """Spec §5: the readout must stay linear. Phase 2 interpretability asks
    which directions in h produce confidence, which only has an answer if the
    map from h to logits is linear. Exact additivity is that property."""
    torch.manual_seed(0)
    head = ReadoutHead(d_model=16).eval()
    opts = torch.randn(1, 4, 16)
    h1, h2 = torch.randn(1, 1, 16), torch.randn(1, 1, 16)
    with torch.no_grad():
        z1, z2 = head(h1, opts), head(h2, opts)
        zs = head(h1 + h2, opts)
    torch.testing.assert_close(zs, z1 + z2, rtol=1e-4, atol=1e-4)


def test_readout_handles_variable_option_counts():
    head = ReadoutHead(d_model=16)
    assert head(torch.randn(2, 1, 16), torch.randn(2, 3, 16)).shape == (2, 3)
    assert head(torch.randn(2, 1, 16), torch.randn(2, 7, 16)).shape == (2, 7)


def test_score_expectation_matches_the_api_docs_example():
    """docs.typesafe.ai: probabilities {0:0.05, 1:0.30, 2:0.65} -> score 1.6"""
    probs = torch.tensor([[0.05, 0.30, 0.65]])
    torch.testing.assert_close(score_expectation(probs),
                               torch.tensor([1.60]), rtol=1e-6, atol=1e-6)


def test_confidence_is_higher_for_a_peaked_distribution():
    peaked = torch.tensor([[0.9, 0.05, 0.05]])
    flat = torch.tensor([[0.34, 0.33, 0.33]])
    assert confidence(peaked).item() > confidence(flat).item()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_heads.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'prosodia.model.heads'`

- [ ] **Step 3: Implement**

```python
# src/prosodia/model/heads.py
"""Readout: z = Wh, strictly linear and bias-free, per spec §5.

Archer Hume's reverse-engineering of the model this project replicates
reported an affine readout, z = Wh + b. This implementation deliberately
drops the bias: a global bias is meaningless for a pointer head whose
option set changes between requests, and exact linearity from h to logits
is required by a later interpretability phase.

One pointer-style head serves all three primitives: logits are an inner
product between the projected branch vector and projected option embeddings,
so the head handles any number of candidates without retraining.

  Noul   -> two options ["false", "true"]; P(true) is probs[..., 1]
  Choice -> softmax over the supplied option set
  Score  -> softmax over ordered levels, read out as an expectation
"""
from __future__ import annotations

import torch
from torch import Tensor, nn


class ReadoutHead(nn.Module):
    def __init__(self, d_model: int = 256) -> None:
        super().__init__()
        # Both projections are bias-free: the map h -> logits must be exactly
        # linear (Phase 2 depends on it), and a global bias is meaningless for
        # a pointer head whose option set changes between requests.
        self.w_branch = nn.Linear(d_model, d_model, bias=False)
        self.w_option = nn.Linear(d_model, d_model, bias=False)
        self.scale = d_model ** -0.5

    def forward(self, branch_vec: Tensor, option_vecs: Tensor) -> Tensor:
        """branch_vec: (B, 1, D); option_vecs: (B, K, D) -> logits (B, K)."""
        h = self.w_branch(branch_vec)            # (B, 1, D)
        o = self.w_option(option_vecs)           # (B, K, D)
        return (h @ o.transpose(1, 2)).squeeze(1) * self.scale


def score_expectation(probs: Tensor) -> Tensor:
    """Probability-weighted level index: score = sum_i i * p(i)."""
    idx = torch.arange(probs.shape[-1], device=probs.device, dtype=probs.dtype)
    return (probs * idx).sum(-1)


def confidence(probs: Tensor) -> Tensor:
    """Normalized certainty in [0, 1]: 1 - H(p)/log(K)."""
    k = probs.shape[-1]
    if k < 2:
        return torch.ones(probs.shape[:-1], device=probs.device)
    entropy = -(probs.clamp_min(1e-12) * probs.clamp_min(1e-12).log()).sum(-1)
    return 1.0 - entropy / torch.log(torch.tensor(float(k), device=probs.device))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_heads.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/prosodia/model/heads.py tests/test_heads.py
git commit -m "feat: linear pointer readout for noul/choice/score"
```

---

### Task 11: Model assembly

**Correction (owner Decision 3, 2026-09-19):** the `_encode_state` code block below is the historical record of this task's original implementation and is now superseded on two points not reflected in the snippet. (1) A second `_encode_state` test, `test_muted_audio_gate_covers_the_c2_stat_tokens_too`, was added post-review to check the `audio_present` content gate against the C2 stat-token positions `StateEncoder` prepends; it was never synced into this doc. (2) The final-review fix this correction documents: `_encode_state`'s `torch.where(audio_present, h, self.audio_absent...)` neutralized the audio state's CONTENT when muted but left `mask` — built in `collate_batch` from each example's real cached-feature length — untouched, so a muted row's genuine audio DURATION still reached `IsolatedBranches`' `nn.MultiheadAttention` via `key_padding_mask`'s valid-position count (measured leak: up to 4.7e-3, larger than the 2.0e-3 content leak the C2 test guards against). `_encode_state` now also does `mask = mask & audio_present.view(-1, 1)` immediately after the content gate, so a muted row's audio positions (pooled frames and C2 stat tokens alike, since both live in the one `mask` tensor `StateEncoder` returns) are excluded from attention entirely rather than merely content-neutralized. This is required for the text-only baseline (Decision 2) to be genuinely modality-isolated, since success criterion #1 depends on it being a controlled comparison. See `src/prosodia/model/prosodia.py` for the authoritative current version and `tests/test_model.py::test_muted_audio_gate_also_isolates_the_real_audio_duration` for the regression test.

**Correction (I6, final-review fix, 2026-09-20):** the `forward` code block below is superseded — it is the historical, unbatched implementation, kept in the current source only as `_forward_looped_reference` (a test-only equivalence oracle). `collate_batch` builds a padded `(B, T, D)` batch and `_encode_state` runs it batched, but this original `forward` immediately un-batched that work: a Python loop called `self.branches` with `b=1` and `question_encoder.embed_texts` `B * (1 + n_questions)` times per batch. Profiling (`in_dim=1024, d_model=256, B=16, T=150`, this machine's MPS backend, `model.eval()` — MPS's `scaled_dot_product_attention` does not support the nonzero dropout `nn.TransformerEncoderLayer` defaults to, so timing used eval mode, which does not affect gradient flow) found the per-example loop was the dominant cost, but NOT primarily in `self.branches` as originally suspected: `torch.profiler` attributed ~62% of self CPU time to `aten::copy_`/`aten::to`, the repeated device transfer inside `QuestionEncoder.embed_texts` (one `.to(device, dtype)` and one `nn.Linear` per *call*, regardless of how many strings are passed, called up to 64 times per forward on this batch). The fix in `src/prosodia/model/prosodia.py`'s current `forward`: (1) collects every instruction/option string the batch needs into one list and calls `embed_texts` ONCE, indexing back into the result by stored spans; (2) groups examples by their exact ordered question-key tuple and runs `self.branches` once per group instead of once per example (covers the `if label is None: continue` path in `ProsodiaDataset`, which can give two examples different key sets, though MELD never exercises it); (3) sub-groups the readout by option COUNT within each group, since `permute_candidates`'s per-item RNG can give two examples in the same batch a different number of options for the identical question key — a naive `(B, Q, K, D)` reshape cannot represent that raggedness at all. Measured: forward 48.1ms → 11.5ms (~4.2x), forward+backward 122.9ms → 43.4ms (~2.8x) on the quoted config; see `tests/test_model.py::test_batched_forward_matches_looped_reference_on_a_uniform_batch` and `..._with_ragged_option_counts_and_keys` for the bit-for-bit (to float32 matmul-reassociation noise, measured ~5e-9, tolerance asserted at 1e-5) equivalence proof. See the task report for full profiling detail and fault-injection numbers.

**Caveat (F4, final-cleanup review, 2026-09-20):** the ~2.8x forward+backward figure above is an **eval-mode upper bound, not a measured training-mode speedup** — it was measured entirely under `.eval()`, forced by the MPS SDPA-dropout limitation noted above. This fix removes a fixed, dropout-independent per-forward overhead (the repeated `embed_texts` device transfers). In `.train()` mode, the untouched attention and backward cost grows (dropout is active, and MPS's `scaled_dot_product_attention` cannot use its fast path with nonzero dropout), so that same fixed saving becomes a smaller fraction of a larger total — the real training-mode speedup is plausibly less than 2.8x, by an unbounded margin. No training-mode estimate has been measured; do not budget machine time against 2.8x.

**Further correction (F1, final-cleanup review, 2026-09-20):** `tests/test_model.py::test_batched_forward_matches_looped_reference_with_ragged_option_counts_and_keys`'s fixture (`_ragged_batch`) was strengthened without changing the test count above. Its only multi-row (key, option-count) bucket ("sentiment", K=3, rows 0-2) previously carried byte-identical option text on every row, so a row-order PERMUTATION bug inside the grouped/sub-batched `self.readout` call (e.g. `branch_vecs` and `opt_vecs` built from differently-ordered row lists) was a semantic no-op there — confirmed by fault injection, that exact fault measured a diff of ~1e-9, invisible against the test's 1e-5 tolerance. `_ragged_batch` now gives each of those three rows distinct "sentiment" text at the same option count; the identical fault now measures ~7e-3, well above tolerance. See the task report for the full fault-injection numbers.

**Files:**
- Create: `src/prosodia/model/prosodia.py`
- Test: `tests/test_model.py`

**Interfaces:**
- Consumes: `StateEncoder` (7), `QuestionEncoder` (8), `IsolatedBranches` (9), `ReadoutHead` (10)
- Produces: `ProsodiaModel(in_dim, d_model=256, ...)` with `forward(batch) -> list[dict[str, Tensor]]` returning per-example, per-question logits

- [ ] **Step 1: Write the failing test**

```python
# tests/test_model.py
import copy

import torch

from prosodia.model.prosodia import ProsodiaModel


def _batch(n=2, t=24, d=8):
    qs = {"emotion": {"instructions": "Which emotion?",
                      "options": ["anger", "joy", "neutral"], "qtype": "choice"},
          "sentiment": {"instructions": "Rate sentiment.",
                        "options": ["negative", "neutral", "positive"],
                        "qtype": "score"}}
    return {
        "audio": torch.randn(n, t, d),
        "audio_mask": torch.ones(n, t, dtype=torch.bool),
        "audio_present": torch.ones(n, dtype=torch.bool),
        "context": ["Joey: hello"] * n,
        "context_present": torch.ones(n, dtype=torch.bool),
        "questions": [qs for _ in range(n)],
    }


def test_forward_returns_logits_per_question():
    model = ProsodiaModel(in_dim=8, d_model=32).eval()
    with torch.no_grad():
        out = model(_batch())
    assert len(out) == 2
    assert out[0]["emotion"].shape == (3,)
    assert out[0]["sentiment"].shape == (3,)


def test_dropping_audio_changes_the_prediction():
    # `muted` must be a deep copy of `full`, not a fresh `_batch()` call: two
    # independent calls draw different `torch.randn` audio, which confounds
    # the comparison -- a diff would then show up even with the audio_present
    # gate completely removed, since the audio *content* also differs. Holding
    # everything but the presence flag fixed isolates the gate as the only
    # possible source of the diff.
    torch.manual_seed(0)
    model = ProsodiaModel(in_dim=8, d_model=32).eval()
    full = _batch()
    muted = copy.deepcopy(full)
    muted["audio_present"] = torch.zeros(2, dtype=torch.bool)
    with torch.no_grad():
        a = model(full)[0]["emotion"]
        b = model(muted)[0]["emotion"]
    assert (a - b).abs().max().item() > 1e-5


def test_trainable_parameter_count_is_small():
    model = ProsodiaModel(in_dim=1024, d_model=256)
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # Measured trainable count at this config is ~3.65M. The range is kept
    # tight around that so a leak of the ~22.7M-parameter frozen
    # sentence-transformer (which would push the total to ~26.4M) cannot
    # slip through unnoticed -- see the direct freeze check below for the
    # primary guard against that specific regression.
    assert 1e6 < n < 6e6, f"{n/1e6:.1f}M trainable params outside expected range"


def test_frozen_sentence_transformer_has_no_trainable_params():
    # Directly guards the property the numeric range above can only proxy:
    # if the sentence-transformer's freeze ever broke, its ~22.7M parameters
    # would leak into the trainable set. That leak still lands well under the
    # numeric ceiling above (and would even fit under the original 40e6
    # ceiling), so only a direct check on the frozen submodule's parameters
    # can catch it.
    model = ProsodiaModel(in_dim=1024, d_model=256)
    frozen_params = list(model.question_encoder._st.parameters())
    assert frozen_params, "expected the frozen sentence-transformer to have parameters"
    assert not any(p.requires_grad for p in frozen_params)


def test_model_output_is_not_invariant_to_time_reversal():
    """C1 (2026-09-19 final review): prior to the positional-encoding fix,
    three compounding causes -- no positions on `StateEncoder`'s
    self-attention, `IsolatedBranches`' cross-attention being
    permutation-invariant over its keys, and `AttentionPool` swapping
    weights and values together within a window -- made the WHOLE ASSEMBLED
    MODEL exactly invariant to time order (measured max|logit diff| ~1e-9
    for a full reversal). This has to be a model-level test, not a
    pooling-level one: any strided reducer preserves the time axis, so
    comparing pooled SEQUENCES element-wise (Task 7's old
    `test_attention_pool_distinguishes_a_rise_from_a_fall`) passes
    regardless of whether the model as a whole is order-sensitive."""
    torch.manual_seed(0)
    model = ProsodiaModel(in_dim=8, d_model=32).eval()
    batch = _batch()
    reversed_batch = copy.deepcopy(batch)
    reversed_batch["audio"] = torch.flip(batch["audio"], dims=[1])
    with torch.no_grad():
        a = model(batch)[0]["emotion"]
        b = model(reversed_batch)[0]["emotion"]
    # 1e-5 rather than 1e-3: this is an untrained, randomly-initialized
    # model, so the effect size is small but still >1000x the ~1e-9 the
    # review measured for the genuinely time-order-invariant model, and
    # >>float32 eps (~1.2e-7).
    assert (a - b).abs().max().item() > 1e-5


def test_model_distinguishes_rising_from_falling_pitch_contour():
    """C1, the concrete manifestation the review measured directly: a rising
    and a falling ramp (identical means, opposite contour shape) produced
    bit-identical logits (max|diff| = 9.3e-10) before the positional
    encoding fix. Prosody is supra-segmental -- it lives in exactly this
    kind of contour."""
    torch.manual_seed(0)
    model = ProsodiaModel(in_dim=8, d_model=32).eval()
    t = 24
    rise = torch.linspace(0.0, 1.0, t).view(1, t, 1).expand(2, t, 8).contiguous()
    fall = torch.linspace(1.0, 0.0, t).view(1, t, 1).expand(2, t, 8).contiguous()
    batch_rise = _batch(n=2, t=t)
    batch_rise["audio"] = rise
    batch_fall = copy.deepcopy(batch_rise)
    batch_fall["audio"] = fall
    with torch.no_grad():
        a = model(batch_rise)[0]["emotion"]
        b = model(batch_fall)[0]["emotion"]
    assert (a - b).abs().max().item() > 1e-3


def test_model_distinguishes_utterance_level_from_identical_contour_shape():
    """C2: `speaker_relative_norm` z-normalizes within each utterance, which
    is correct for making "high pitch" speaker-relative but, uncorrected,
    also strips absolute LEVEL -- a loud/high-pitched utterance and a
    quiet/low one with the identical normalized contour shape produced
    EXACTLY identical output (max|diff| = 0.0, measured on
    `speaker_relative_norm` directly in the review) because nothing
    downstream ever saw the un-normalized mean/std. Mirrors the review's own
    numbers: F0 mean 210.0 Hz / RMS 0.80 vs F0 mean 105.0 Hz / RMS 0.10,
    same shape."""
    torch.manual_seed(0)
    model = ProsodiaModel(in_dim=8, d_model=32).eval()
    t = 24
    shape = torch.randn(t, 8)
    loud = (shape * 0.80 + 210.0).unsqueeze(0).expand(2, t, 8).contiguous()
    quiet = (shape * 0.10 + 105.0).unsqueeze(0).expand(2, t, 8).contiguous()
    batch_loud = _batch(n=2, t=t)
    batch_loud["audio"] = loud
    batch_quiet = copy.deepcopy(batch_loud)
    batch_quiet["audio"] = quiet
    with torch.no_grad():
        a = model(batch_loud)[0]["emotion"]
        b = model(batch_quiet)[0]["emotion"]
    # 1e-5 for the same reason as the time-reversal test above: small but
    # real, vs. exact 0.0 for the unfixed model.
    assert (a - b).abs().max().item() > 1e-5


def test_fully_masked_and_absent_state_keeps_a_valid_position_and_no_nan():
    # The mask's leading `torch.ones` column (see `_encode_state`) does two
    # jobs: it makes the context position attendable, and it guarantees at
    # least one state position is always valid -- the only reason a
    # fully-masked state row is unreachable. `nn.MultiheadAttention` can NaN
    # on an entirely masked key set, and `IsolatedBranches` has no guard
    # against it.
    #
    # This scenario needs BOTH `audio_mask` fully False *and* `context_present`
    # False to actually exercise that edge: with context still flagged
    # present, the context column stays valid on its own regardless of the
    # mask column's construction, so a fault that makes the column
    # conditional on `context_present` wouldn't show up unless context is
    # *also* absent. `ProsodiaDataset`'s modality dropout never drops both
    # modalities at once, but the model itself must not rely on that caller
    # discipline -- hence constructing the adversarial batch directly here.
    #
    # Two assertions, for two different reasons:
    #   1. The mask invariant is checked directly on `_encode_state`'s output.
    #      This is the one proven (by fault injection, see the task report)
    #      to actually fail if the ones-column is made conditional on
    #      `context_present`.
    #   2. The end-to-end no-NaN check documents the property the invariant
    #      exists to protect. It is kept even though, empirically, this
    #      torch release's `need_weights=False` fast path already zero-fills
    #      a fully-masked row on CPU/MPS instead of NaN'ing -- an internal
    #      implementation detail this code must not rely on, not a contract.
    model = ProsodiaModel(in_dim=8, d_model=32).eval()
    batch = _batch()
    batch["audio_mask"] = torch.zeros(2, 24, dtype=torch.bool)
    batch["audio_present"] = torch.zeros(2, dtype=torch.bool)
    batch["context_present"] = torch.zeros(2, dtype=torch.bool)

    with torch.no_grad():
        _, mask = model._encode_state(batch)
    assert mask.any(dim=-1).all(), "every state row must keep at least one valid position"

    with torch.no_grad():
        out = model(batch)
    for per_question in out:
        for logits in per_question.values():
            assert torch.isfinite(logits).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_model.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'prosodia.model.prosodia'`

- [ ] **Step 3: Implement**

```python
# src/prosodia/model/prosodia.py
from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from prosodia.model.branches import IsolatedBranches
from prosodia.model.heads import ReadoutHead
from prosodia.model.qencoder import QuestionEncoder
from prosodia.model.state import StateEncoder


class ProsodiaModel(nn.Module):
    """state (audio + context) -> per-question logits over supplied options."""

    def __init__(
        self, in_dim: int, d_model: int = 256, state_layers: int = 2,
        branch_layers: int = 2, n_heads: int = 4, stride: int = 2,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.state_encoder = StateEncoder(in_dim, d_model, state_layers, n_heads, stride)
        self.question_encoder = QuestionEncoder(d_model=d_model)
        self.branches = IsolatedBranches(d_model, branch_layers, n_heads)
        self.readout = ReadoutHead(d_model)
        self.audio_absent = nn.Parameter(torch.zeros(d_model))
        self.context_absent = nn.Parameter(torch.zeros(d_model))

    def _encode_state(self, batch: dict[str, Any]) -> tuple[Tensor, Tensor]:
        h, mask = self.state_encoder(batch["audio"], batch["audio_mask"])

        # Modality dropout: replace the whole audio state with a learned token.
        audio_present = batch["audio_present"].to(h.device).view(-1, 1, 1)
        h = torch.where(audio_present, h, self.audio_absent.view(1, 1, -1).expand_as(h))

        ctx_present = batch["context_present"].to(h.device)
        ctx_vecs = self.question_encoder.embed_texts(list(batch["context"]))
        ctx_vecs = torch.where(ctx_present.view(-1, 1), ctx_vecs,
                               self.context_absent.view(1, -1).expand_as(ctx_vecs))

        # Context joins the state as one extra position the branches attend to.
        h = torch.cat([ctx_vecs.unsqueeze(1), h], dim=1)
        mask = torch.cat([torch.ones(h.shape[0], 1, dtype=torch.bool, device=mask.device),
                          mask], dim=1)
        return h, mask

    def forward(self, batch: dict[str, Any]) -> list[dict[str, Tensor]]:
        h, mask = self._encode_state(batch)
        results: list[dict[str, Tensor]] = []

        for i, questions in enumerate(batch["questions"]):
            keys = list(questions)
            if not keys:
                results.append({})
                continue

            q_vecs = self.question_encoder.embed_texts(
                [questions[k]["instructions"] for k in keys]
            ).unsqueeze(0)                                        # (1, Q, D)
            branch_out = self.branches(q_vecs, h[i : i + 1], mask[i : i + 1])

            per_question: dict[str, Tensor] = {}
            for j, key in enumerate(keys):
                opts = questions[key]["options"]
                opt_vecs = self.question_encoder.embed_texts(opts).unsqueeze(0)
                logits = self.readout(branch_out[:, j : j + 1], opt_vecs)
                per_question[key] = logits.squeeze(0)
            results.append(per_question)

        return results
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_model.py -v`
Expected: 8 passed originally; 10 passed after the C2 stat-token gate test and Decision 3's audio-duration-mask-leak regression test were added; 12 passed after I6's batched-forward equivalence tests were added (see the Correction notes above)

- [ ] **Step 5: Commit**

```bash
git add src/prosodia/model/prosodia.py tests/test_model.py
git commit -m "feat: assemble Prosodia model with modality dropout"
```

---

### Task 12: Losses — CE, Brier, composite

**Files:**
- Create: `src/prosodia/train/__init__.py`
- Create: `src/prosodia/train/losses.py`
- Test: `tests/test_losses.py`

**Interfaces:**
- Consumes: nothing
- Produces: `cross_entropy_loss(logits, target)`; `brier_loss(logits, target)`; `composite_loss(logits, target, brier_weight=0.0)`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_losses.py
import pytest
import torch

from prosodia.train.losses import brier_loss, composite_loss, cross_entropy_loss


def test_brier_is_zero_for_a_perfect_confident_prediction():
    logits = torch.tensor([[50.0, -50.0, -50.0]])
    assert brier_loss(logits, torch.tensor([0])).item() < 1e-6


def test_brier_is_maximal_for_a_confident_wrong_prediction():
    logits = torch.tensor([[50.0, -50.0, -50.0]])
    assert brier_loss(logits, torch.tensor([1])).item() > 1.9  # -> 2.0


def test_brier_penalises_overconfidence_more_than_ce_at_the_margin():
    """A proper scoring rule is what makes probabilities honest (spec §6)."""
    target = torch.tensor([0])
    mild = torch.tensor([[1.0, 0.0, 0.0]])
    overconfident = torch.tensor([[20.0, 0.0, 0.0]])
    # both are correct; CE rewards the confident one far more than Brier does
    ce_gain = cross_entropy_loss(mild, target) - cross_entropy_loss(overconfident, target)
    br_gain = brier_loss(mild, target) - brier_loss(overconfident, target)
    assert ce_gain > br_gain


def test_composite_with_zero_weight_equals_cross_entropy():
    logits, target = torch.randn(4, 5), torch.randint(0, 5, (4,))
    torch.testing.assert_close(composite_loss(logits, target, brier_weight=0.0),
                               cross_entropy_loss(logits, target))


def test_composite_is_between_its_components():
    logits, target = torch.randn(8, 4), torch.randint(0, 4, (8,))
    ce = cross_entropy_loss(logits, target)
    br = brier_loss(logits, target)
    mix = composite_loss(logits, target, brier_weight=0.5)
    assert min(ce, br) <= mix <= max(ce, br)


def test_composite_raises_on_negative_brier_weight():
    """A negative brier_weight is a config typo, not a valid Arm A request.
    The old `<= 0.0` check silently routed it to pure cross-entropy -- same
    defect family as the headline Arm C bug: a config field quietly not
    meaning what its name says. Only exactly 0.0 should mean 'Arm A'."""
    logits, target = torch.randn(4, 3), torch.randint(0, 3, (4,))
    with pytest.raises(ValueError, match="brier_weight"):
        composite_loss(logits, target, brier_weight=-0.1)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_losses.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'prosodia.train'`

- [ ] **Step 3: Implement**

```python
# src/prosodia/train/losses.py
"""Training objectives for the loss ablation (spec §6).

  Arm A: brier_weight = 0.0            (cross-entropy only)
  Arm B: brier_weight > 0.0            (RLCD stand-in — proper scoring composite)
  Arm C: Arm A + post-hoc temperature scaling (Task 13)

If B is indistinguishable from C, calibration is a scalar and that IS the
answer to "where does calibration live".
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def cross_entropy_loss(logits: Tensor, target: Tensor) -> Tensor:
    return F.cross_entropy(logits, target)


def brier_loss(logits: Tensor, target: Tensor) -> Tensor:
    """Multiclass Brier score: mean squared error against the one-hot target.

    Range [0, 2]. Strictly proper, so it is minimized only by the true
    probabilities — unlike CE it does not reward unbounded confidence.
    """
    probs = torch.softmax(logits, dim=-1)
    onehot = F.one_hot(target, num_classes=logits.shape[-1]).to(probs.dtype)
    return ((probs - onehot) ** 2).sum(-1).mean()


def composite_loss(logits: Tensor, target: Tensor, brier_weight: float = 0.0) -> Tensor:
    if brier_weight < 0.0:
        # Same defect family as the headline Arm C bug: a config field that
        # quietly stops meaning what its name says. A negative weight (a
        # config typo) used to silently fall through to Arm A (pure CE)
        # instead of raising, making a broken ablation arm look like a
        # deliberate one.
        raise ValueError(f"brier_weight must be >= 0.0, got {brier_weight!r}")
    if brier_weight == 0.0:
        return cross_entropy_loss(logits, target)
    ce = cross_entropy_loss(logits, target)
    br = brier_loss(logits, target)
    return (1.0 - brier_weight) * ce + brier_weight * br
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_losses.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add src/prosodia/train/ tests/test_losses.py
git commit -m "feat: cross-entropy, Brier, and composite objectives"
```

---

### Task 13: Calibration metrics and temperature scaling

**Files:**
- Create: `src/prosodia/evaluation/__init__.py`
- Create: `src/prosodia/evaluation/metrics.py`
- Create: `src/prosodia/train/calibrate.py`
- Test: `tests/test_metrics.py`

**Interfaces:**
- Consumes: `assert_close_across_devices` (Task 1)
- Produces: `expected_calibration_error(probs, targets, n_bins=10)`; `brier_score(probs, targets)`; `negative_log_likelihood(probs, targets)`; `accuracy(probs, targets)`; `macro_f1(probs, targets)`; `coverage_curve(probs, targets)` -> `(thresholds, coverage, error_rate)`; `TemperatureScaler()` with `.fit(logits, targets)`, `.transform(logits)`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_metrics.py
import torch

from prosodia.device import get_device
from prosodia.evaluation.metrics import (
    coverage_curve, expected_calibration_error, brier_score, macro_f1,
)
from prosodia.train.calibrate import TemperatureScaler


def test_ece_is_zero_for_a_perfectly_calibrated_set():
    """80 predictions at p=0.8, exactly 80% correct -> ECE 0."""
    probs = torch.full((100, 2), 0.2)
    probs[:, 1] = 0.8
    targets = torch.zeros(100, dtype=torch.long)
    targets[:80] = 1  # the p=0.8 class is right exactly 80% of the time
    assert expected_calibration_error(probs, targets, n_bins=10).item() < 1e-6


def test_ece_matches_a_hand_computed_case():
    # all mass in one bin: stated confidence 0.9, observed accuracy 0.5
    probs = torch.tensor([[0.1, 0.9]] * 10)
    targets = torch.tensor([1] * 5 + [0] * 5)
    torch.testing.assert_close(
        expected_calibration_error(probs, targets, n_bins=10),
        torch.tensor(0.4), rtol=1e-5, atol=1e-5,
    )


def test_ece_does_not_let_over_and_under_confidence_cancel():
    """Two bins with equal-magnitude, opposite-sign (conf - accuracy) errors.

    Without abs() per bin, the signed errors would net to ~0; ECE must sum
    the *magnitudes*, so it should land near 0.1, not near 0.
    """
    # bin (0.8, 0.9]: confidence 0.9, all 10 correct -> conf - acc = -0.1 (underconfident)
    probs_under = torch.tensor([[0.1, 0.9]] * 10)
    targets_under = torch.tensor([1] * 10)
    # bin (0.5, 0.6]: confidence 0.6, 5/10 correct -> conf - acc = +0.1 (overconfident)
    probs_over = torch.tensor([[0.4, 0.6]] * 10)
    targets_over = torch.tensor([1] * 5 + [0] * 5)

    probs = torch.cat([probs_under, probs_over])
    targets = torch.cat([targets_under, targets_over])
    torch.testing.assert_close(
        expected_calibration_error(probs, targets, n_bins=10),
        torch.tensor(0.1), rtol=1e-5, atol=1e-5,
    )


def test_brier_score_bounds():
    probs = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    assert brier_score(probs, torch.tensor([0, 1])).item() < 1e-6
    assert brier_score(probs, torch.tensor([1, 0])).item() > 1.9


def test_macro_f1_is_zero_not_nan_for_an_absent_class():
    """Class 2 appears in neither predictions nor targets -> its F1 is 0/0.

    Constructed directly on get_device() (this box's default is MPS) rather
    than CPU: macro_f1's 0/0 guard used a CPU-only torch.tensor(0.0) that
    torch.stack could not combine with the accelerator-resident per-class
    scores, raising instead of returning a value. A CPU-only test cannot
    distinguish that fixed state from the broken one.
    """
    device = get_device()
    # 3-class problem; only classes 0 and 1 ever appear, both predicted perfectly.
    probs = torch.tensor(
        [[0.9, 0.1, 0.0], [0.1, 0.9, 0.0], [0.9, 0.1, 0.0], [0.1, 0.9, 0.0]],
        device=device,
    )
    targets = torch.tensor([0, 1, 0, 1], device=device)
    f1 = macro_f1(probs, targets)
    assert not torch.isnan(f1)
    torch.testing.assert_close(f1, torch.tensor(2.0 / 3.0, device=device), rtol=1e-4, atol=1e-4)


def test_coverage_curve_is_monotone_in_coverage():
    """Constructed on get_device() (this box's default is MPS), not CPU.

    coverage_curve's thresholds/coverage/error tensors were built without a
    shared device=, so on an accelerator it silently returned a mixed-device
    tuple: no crash on its own, only on the first attempt to combine the
    three (e.g. torch.stack, or plotting code that assumes one device). A
    CPU-only test can't see that, since every tensor defaults to CPU there.
    """
    device = get_device()
    torch.manual_seed(0)
    logits = torch.randn(500, 4, device=device)
    probs = torch.softmax(logits, -1)
    targets = torch.randint(0, 4, (500,), device=device)
    thresholds, coverage, error = coverage_curve(probs, targets)
    assert torch.all(coverage[1:] <= coverage[:-1] + 1e-6)  # higher t -> less coverage
    assert thresholds.shape == coverage.shape
    # the three returned tensors must live on one device -- a caller that
    # stacks or concatenates them, or a reliability-diagram plot that moves
    # one to numpy and not the others, would otherwise silently drop data.
    assert thresholds.device == coverage.device == error.device


def test_temperature_scaling_reduces_ece_on_overconfident_logits():
    """Constructed on get_device() (this box's default is MPS), not CPU.

    TemperatureScaler.fit's log_t was built without device=logits.device, so
    on an accelerator the LBFGS closure mixed a CPU log_t with accelerator
    logits/targets and crashed outright. A CPU-only test can't see that,
    since log_t's default CPU placement matches everything else there.
    """
    device = get_device()
    torch.manual_seed(0)
    targets = torch.randint(0, 3, (600,), device=device)
    logits = torch.randn(600, 3, device=device)
    logits[torch.arange(600, device=device), targets] += 1.0
    logits = logits * 4.0  # deliberately overconfident

    before = expected_calibration_error(torch.softmax(logits, -1), targets)
    scaler = TemperatureScaler().fit(logits, targets)
    after = expected_calibration_error(torch.softmax(scaler.transform(logits), -1), targets)
    assert after < before
    assert scaler.temperature.item() > 1.0  # softening, as expected
```

Note: `test_ece_does_not_let_over_and_under_confidence_cancel` was added beyond the
original brief during implementation — the two given hand-computed ECE cases both have
confidence > accuracy (positive signed error), so neither exercises the `.abs()` in the
per-bin term. Fault-injection during self-review (dropping `.abs()`) confirmed the
original two tests stayed green with the bug present; this test catches it (expects
0.1, buggy code produces exactly 0.0).

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_metrics.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'prosodia.evaluation'`

- [ ] **Step 3: Implement metrics**

```python
# src/prosodia/evaluation/metrics.py
"""Calibration metrics are first-class here, not an afterthought (spec §7).

All functions take PROBABILITIES and run in fp32. Accuracy answers "is it
right"; ECE answers "does it know when it is right", which is the quantity
this whole project is about.
"""
from __future__ import annotations

import torch
from torch import Tensor


def _fp32(x: Tensor) -> Tensor:
    return x.detach().to(torch.float32)


def accuracy(probs: Tensor, targets: Tensor) -> Tensor:
    return (_fp32(probs).argmax(-1) == targets).float().mean()


def macro_f1(probs: Tensor, targets: Tensor) -> Tensor:
    p = _fp32(probs)
    preds = p.argmax(-1)
    scores = []
    for c in range(probs.shape[-1]):
        tp = ((preds == c) & (targets == c)).sum().float()
        fp = ((preds == c) & (targets != c)).sum().float()
        fn = ((preds != c) & (targets == c)).sum().float()
        denom = 2 * tp + fp + fn
        scores.append(torch.tensor(0.0, device=p.device) if denom == 0 else 2 * tp / denom)
    return torch.stack(scores).mean()


def brier_score(probs: Tensor, targets: Tensor) -> Tensor:
    p = _fp32(probs)
    onehot = torch.zeros_like(p).scatter_(-1, targets.unsqueeze(-1), 1.0)
    return ((p - onehot) ** 2).sum(-1).mean()


def negative_log_likelihood(probs: Tensor, targets: Tensor) -> Tensor:
    p = _fp32(probs).clamp_min(1e-12)
    return -p.gather(-1, targets.unsqueeze(-1)).squeeze(-1).log().mean()


def expected_calibration_error(probs: Tensor, targets: Tensor, n_bins: int = 10) -> Tensor:
    """Top-label ECE: weighted mean |confidence - accuracy| across bins."""
    p = _fp32(probs)
    conf, pred = p.max(-1)
    correct = (pred == targets).float()

    edges = torch.linspace(0.0, 1.0, n_bins + 1, device=p.device)
    total = torch.tensor(0.0, device=p.device)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        in_bin = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        n = in_bin.sum()
        if n == 0:
            continue
        total = total + (n.float() / p.shape[0]) * \
            (conf[in_bin].mean() - correct[in_bin].mean()).abs()
    return total


def coverage_curve(
    probs: Tensor, targets: Tensor, n_points: int = 50
) -> tuple[Tensor, Tensor, Tensor]:
    """Accuracy-vs-coverage: at confidence threshold t, what fraction of traffic
    is automated and what error rate does it carry (spec §7)."""
    p = _fp32(probs)
    conf, pred = p.max(-1)
    correct = (pred == targets)

    thresholds = torch.linspace(0.0, conf.max().item(), n_points, device=p.device)
    coverage, error = [], []
    for t in thresholds:
        keep = conf >= t
        n = keep.sum()
        coverage.append(n.float() / p.shape[0])
        error.append(torch.tensor(0.0, device=p.device) if n == 0 else 1.0 - correct[keep].float().mean())
    return thresholds, torch.stack(coverage), torch.stack(error)
```

Note (post-review fix): `macro_f1`'s `torch.tensor(0.0)` and `coverage_curve`'s
`torch.linspace(...)` / `torch.tensor(0.0)` originally had no `device=`, so they stayed on
CPU while the rest of the computation ran on the accelerator. `macro_f1` crashed
(`torch.stack` does not cross-device promote: "Passed CPU tensor to MPS op"); `coverage_curve`
silently returned a mixed-device tuple (`thresholds` on CPU, `coverage`/`error` on the
accelerator) that broke on the first attempt to combine them. Both now take their device from
`p = _fp32(probs)`, mirroring `expected_calibration_error`'s existing pattern. Reproduced and
verified fixed on MPS directly (not just via CPU-only pytest) — see the Task 13 fix report.

- [ ] **Step 4: Implement temperature scaling**

```python
# src/prosodia/train/calibrate.py
"""Post-hoc temperature scaling — ablation Arm C (spec §6).

This is the control that decides the Phase 2 interpretability question. If
CE+Brier is no better calibrated than CE plus a single learned scalar, then
calibration lives in a temperature and there is nothing distributed to find.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


class TemperatureScaler:
    def __init__(self) -> None:
        self.temperature = torch.ones(1)

    def fit(self, logits: Tensor, targets: Tensor, max_iter: int = 100) -> "TemperatureScaler":
        logits = logits.detach().to(torch.float32)
        log_t = torch.zeros(1, device=logits.device, requires_grad=True)  # optimise log T to keep T > 0
        optimizer = torch.optim.LBFGS([log_t], lr=0.1, max_iter=max_iter)

        def closure():
            optimizer.zero_grad()
            loss = F.cross_entropy(logits / log_t.exp(), targets)
            loss.backward()
            return loss

        optimizer.step(closure)
        self.temperature = log_t.exp().detach()
        return self

    def transform(self, logits: Tensor) -> Tensor:
        return logits.detach().to(torch.float32) / self.temperature.to(logits.device)
```

Note (post-review fix): `log_t` originally had no `device=`, so it stayed on CPU while
`logits` (and, on the accelerator, `targets`) were on MPS/CUDA — the LBFGS closure's
`F.cross_entropy(logits / log_t.exp(), targets)` then crashed with "Expected all tensors to
be on the same device, but found at least two devices, mps:0 and cpu!" This is Arm C's fit
routine, so it could not run at all on the project's target hardware. Fixed by giving `log_t`
`device=logits.device`. Reproduced and verified fixed on MPS directly — see the Task 13 fix
report.

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_metrics.py -v`
Expected: 7 passed

- [ ] **Step 6: Add the cross-device guard to metrics and re-run**

```python
# append to tests/test_metrics.py
from prosodia.device import assert_close_across_devices


def test_metrics_agree_across_devices():
    """Spec §10: an MPS numerical artifact and a finding look identical in a plot."""
    torch.manual_seed(0)
    probs = torch.softmax(torch.randn(256, 5), -1)
    targets = torch.randint(0, 5, (256,))
    for fn in (brier_score, expected_calibration_error):
        assert_close_across_devices(fn, probs, targets)
```

Note: the original brief used `assert_close_across_devices(lambda p: fn(p, targets), probs)`.
On a machine with an accelerator (this repo's dev box has MPS), that fails for a reason
unrelated to calibration correctness: `targets` is captured in the lambda's closure, so
`assert_close_across_devices` never moves it to the accelerator device, and the op errors
with "Passed CPU tensor to MPS op" when it mixes an MPS `probs` with a CPU `targets`. Passing
`targets` as a real positional argument lets the helper move it correctly (its dtype-preserving,
device-only branch for non-floating tensors), which is what actually exercises the fp32
cross-device guarantee this test exists to check.

Run: `uv run pytest tests/test_metrics.py -v`
Expected: 8 passed

- [ ] **Step 7: Commit**

```bash
git add src/prosodia/evaluation/ src/prosodia/train/calibrate.py tests/test_metrics.py
git commit -m "feat: calibration metrics, coverage curves, temperature scaling"
```

---

### Task 14: Training loop with W&B and checkpointing

**Files:**
- Create: `src/prosodia/train/loop.py`
- Create: `src/prosodia/config.py`
- Test: `tests/test_loop.py`

**Interfaces:**
- Consumes: `ProsodiaModel` (11), `composite_loss` (12), metrics (13), `ProsodiaDataset`/`collate_batch` (6)
- Produces: `RunConfig` dataclass; `train_one_epoch(model, loader, optimizer, cfg, epoch, run=None) -> dict`; `evaluate(model, loader, run=None, epoch=None) -> dict`; `init_wandb(cfg)`; `save_checkpoint(path, model, optimizer, epoch, cfg)`; `load_checkpoint(path, model, optimizer) -> int`

> **Ruling (binding):** `ProsodiaDataset` seeds its per-item RNG from `(rng_seed, epoch, idx)` alone (Task 6), so *something* must call `dataset.set_epoch(epoch)` before each training pass or every epoch silently draws the identical augmentation/modality-dropout pattern. `train_one_epoch` owns this: `epoch` is a **required** argument (no default) and the first line of the function is `loader.dataset.set_epoch(epoch)`. `evaluate()` never calls `set_epoch` — eval loaders are built with `augment=False`, so there is nothing to advance.

**Correction (I5a, final-review fix, 2026-09-20):** the `save_checkpoint`/`load_checkpoint` code block below is superseded on one point — it is the historical, pre-fix version that saves `model.state_dict()` unfiltered. `question_encoder._st` (the frozen `all-MiniLM-L6-v2` sentence transformer, `requires_grad_(False)` in `qencoder.py`) is reconstructed from the HuggingFace hub by `QuestionEncoder.__init__` regardless of what any checkpoint restores, so saving it is pure waste: measured at the grid's config (`in_dim=1024, d_model=256`), it is 22.71M of the model's 26.37M state_dict elements (86.1%). The current `src/prosodia/train/loop.py` adds `_trainable_state_dict(model)` (`model.state_dict()` filtered to drop any key starting with `question_encoder._st.`) and has `save_checkpoint` use it instead. `load_checkpoint` correspondingly loads with `strict=False`, but does NOT simply swallow every missing/unexpected key: it inspects `load_state_dict`'s returned `missing_keys`/`unexpected_keys` and raises `RuntimeError` unless every missing key is one of the expected frozen-encoder ones — so a checkpoint that is missing a genuinely trainable key (corruption, or a future over-broad exclusion) still fails loudly instead of silently leaving part of the model at its random init. Net effect: a checkpoint (model + populated AdamW state) shrinks from ~135 MB to ~44 MB (measured, 67% reduction). See `tests/test_loop.py::test_save_checkpoint_excludes_the_frozen_sentence_transformer`, `::test_load_checkpoint_raises_when_a_trainable_key_is_missing`, `::test_load_checkpoint_raises_on_an_unexpected_key`, and the I5 fix report for fault-injection numbers.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_loop.py
import pytest
import torch
from torch.utils.data import DataLoader

from prosodia.config import RunConfig
from prosodia.data import ProsodiaDataset, collate_batch
from prosodia.device import get_device
from prosodia.features import FeatureCache
from prosodia.model.prosodia import ProsodiaModel
from prosodia.schema import Example, Label, LabelTier, QuestionSpec
from prosodia.train.loop import evaluate, load_checkpoint, save_checkpoint, train_one_epoch

SPECS = [QuestionSpec("emotion", "choice", "Which emotion?",
                      {"anger": None, "joy": None, "neutral": None})]


def _loader(tmp_path, n=8):
    cache = FeatureCache(tmp_path / "c")
    exs = []
    for i in range(n):
        cache.write(f"u{i}", torch.randn(16, 8))
        exs.append(Example(f"u{i}", "meld", "/x.wav", "ctx",
                           {"emotion": Label("joy", LabelTier.HUMAN)}, speaker="Joey"))
    ds = ProsodiaDataset(exs, SPECS, cache, augment=False, modality_dropout=0.0)
    return DataLoader(ds, batch_size=4, collate_fn=collate_batch)


def test_train_one_epoch_reduces_loss_on_a_memorisable_batch(tmp_path):
    torch.manual_seed(0)
    model = ProsodiaModel(in_dim=8, d_model=32)
    loader = _loader(tmp_path)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    cfg = RunConfig(name="t", brier_weight=0.0, encoder="wavlm")
    first = train_one_epoch(model, loader, opt, cfg, epoch=0)["loss"]
    for epoch in range(1, 9):
        last = train_one_epoch(model, loader, opt, cfg, epoch=epoch)["loss"]
    assert last < first


def test_evaluate_reports_calibration_metrics(tmp_path):
    model = ProsodiaModel(in_dim=8, d_model=32).eval()
    out = evaluate(model, _loader(tmp_path))
    for key in ("accuracy", "ece", "brier", "nll"):
        assert key in out["emotion"], f"missing {key}"


def test_checkpoint_roundtrip_restores_weights(tmp_path):
    # load_checkpoint places both model and optimizer state on get_device(),
    # so `model` is moved there too before comparison.
    device = get_device()
    model = ProsodiaModel(in_dim=8, d_model=32).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    cfg = RunConfig(name="t", brier_weight=0.3, encoder="wavlm")
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, model, opt, epoch=3, cfg=cfg)

    fresh = ProsodiaModel(in_dim=8, d_model=32)
    fresh_opt = torch.optim.AdamW(fresh.parameters(), lr=1e-3)
    assert load_checkpoint(path, fresh, fresh_opt) == 3
    for a, b in zip(model.state_dict().values(), fresh.state_dict().values()):
        torch.testing.assert_close(a, b)


def test_train_one_epoch_advances_the_dataset_epoch(tmp_path):
    """Regression guard for the ruling above: two consecutive epochs must
    draw different augmentation/dropout patterns, or train_one_epoch is not
    advancing the dataset's epoch."""
    ...  # see tests/test_loop.py for the full, exact test


def test_train_one_epoch_requires_an_explicit_epoch(tmp_path):
    """epoch has no default -- a caller cannot forget to decide what epoch
    it is."""
    model = ProsodiaModel(in_dim=8, d_model=32)
    loader = _loader(tmp_path)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    cfg = RunConfig(name="t", brier_weight=0.0, encoder="wavlm")
    with pytest.raises(TypeError):
        train_one_epoch(model, loader, opt, cfg)  # type: ignore[call-arg]
```

(`tests/test_loop.py` additionally covers: `evaluate()` never advances the epoch; `evaluate()` keeps logits/targets correctly aligned when option-count widths vary across rows, instead of filtering `logits` and slicing `targets` independently; `evaluate()` refuses to silently score a minority subset when the modal width covers less than half the rows; and checkpoint round-trip restores optimizer momentum, not just weights. See the file for all nine tests.)

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_loop.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'prosodia.config'`

- [ ] **Step 3: Implement the config**

```python
# src/prosodia/config.py
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class RunConfig:
    name: str
    encoder: str = "wavlm"          # wavlm | whisper | prosody
    brier_weight: float = 0.0       # 0.0 = Arm A, > 0 = Arm B
    temperature_scale: bool = False  # Arm C
    d_model: int = 256
    state_layers: int = 2
    branch_layers: int = 2
    n_heads: int = 4
    stride: int = 2
    lr: float = 3e-4
    weight_decay: float = 0.01
    epochs: int = 20
    batch_size: int = 16
    modality_dropout: float = 0.15
    seed: int = 0
    wandb_project: str = "prosodia"

    def as_dict(self) -> dict:
        return asdict(self)
```

- [ ] **Step 4: Implement the loop**

```python
# src/prosodia/train/loop.py
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from prosodia.config import RunConfig
from prosodia.device import get_device
from prosodia.evaluation.metrics import (
    accuracy, brier_score, expected_calibration_error, macro_f1,
    negative_log_likelihood,
)
from prosodia.train.losses import composite_loss


def _move(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out = dict(batch)
    for key in ("audio", "audio_mask", "audio_present", "context_present"):
        out[key] = batch[key].to(device)
    return out


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    cfg: RunConfig,
    epoch: int,
    run: Any = None,
) -> dict[str, float]:
    """`epoch` is mandatory -- see the ruling above. The first thing this
    function does is `loader.dataset.set_epoch(epoch)`."""
    loader.dataset.set_epoch(epoch)

    device = get_device()
    model.to(device).train()
    total, n = 0.0, 0

    for batch in loader:
        batch = _move(batch, device)
        outputs = model(batch)
        loss = torch.zeros((), device=device)
        count = 0
        for out, targets in zip(outputs, batch["targets"]):
            for key, logits in out.items():
                target = torch.tensor([targets[key]], device=device)
                loss = loss + composite_loss(logits.unsqueeze(0), target,
                                             cfg.brier_weight)
                count += 1
        if count == 0:
            continue
        loss = loss / count

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total += loss.item()
        n += 1

    metrics = {"loss": total / max(n, 1)}
    if run is not None:
        run.log({"train/loss": metrics["loss"], "epoch": epoch})
    return metrics


@torch.no_grad()
def evaluate(
    model: nn.Module, loader: DataLoader, run: Any = None, epoch: int | None = None,
) -> dict[str, dict[str, float]]:
    """Returns per-question metrics. Probabilities are computed in fp32 on
    CPU. Does NOT call `set_epoch` -- see the ruling above."""
    device = get_device()
    model.to(device).eval()
    logits_by_q: dict[str, list[torch.Tensor]] = defaultdict(list)
    targets_by_q: dict[str, list[int]] = defaultdict(list)

    for batch in loader:
        batch = _move(batch, device)
        for out, targets in zip(model(batch), batch["targets"]):
            for key, logits in out.items():
                logits_by_q[key].append(logits.detach().float().cpu())
                targets_by_q[key].append(targets[key])

    results: dict[str, dict[str, float]] = {}
    for key, rows in logits_by_q.items():
        targets_list = targets_by_q[key]
        widths = [r.shape[-1] for r in rows]
        modal_width, modal_count = Counter(widths).most_common(1)[0]

        if modal_count < len(rows):
            kept_frac = modal_count / len(rows)
            if kept_frac < 0.5:
                # Refuse rather than silently score a minority subset.
                raise ValueError(
                    f"evaluate(): question {key!r} has option-count widths "
                    f"{sorted(set(widths))} across {len(rows)} rows; the modal "
                    f"width {modal_width} covers only {kept_frac:.0%} of them."
                )
            # Filter logits and targets TOGETHER -- filtering only `rows`
            # and then slicing `targets_list` by position would silently
            # misalign the two lists.
            paired = [(r, t) for r, t in zip(rows, targets_list) if r.shape[-1] == modal_width]
            rows = [r for r, _ in paired]
            targets_list = [t for _, t in paired]

        logits = torch.stack(rows)
        targets = torch.tensor(targets_list)
        probs = torch.softmax(logits, dim=-1)
        results[key] = {
            "accuracy": accuracy(probs, targets).item(),
            "macro_f1": macro_f1(probs, targets).item(),
            "ece": expected_calibration_error(probs, targets).item(),
            "brier": brier_score(probs, targets).item(),
            "nll": negative_log_likelihood(probs, targets).item(),
        }
        if run is not None:
            log = {f"eval/{key}/{m}": v for m, v in results[key].items()}
            if epoch is not None:
                log["epoch"] = epoch
            run.log(log)
    return results


def init_wandb(cfg: RunConfig) -> Any:
    """Lazily imports and starts a W&B run. The import lives inside this
    function (never at module scope) so importing/testing this module never
    requires `wandb`, network access, or credentials."""
    import wandb

    return wandb.init(project=cfg.wandb_project, name=cfg.name, config=cfg.as_dict())


def save_checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer,
                    epoch: int, cfg: RunConfig) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "epoch": epoch, "config": cfg.as_dict()}, path)


def load_checkpoint(path: Path, model: nn.Module,
                    optimizer: torch.optim.Optimizer | None = None) -> int:
    """Loads onto get_device() and explicitly relocates optimizer state
    tensors there too -- Optimizer.load_state_dict does not reliably move
    its state to match the parameters' device, so without this a resumed
    run can pair accelerator-resident parameters with CPU-resident Adam
    moment buffers, silently, until the first post-resume `.step()`."""
    device = get_device()
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
    return int(ckpt["epoch"])
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_loop.py -v`
Expected: 9 passed originally; more added since (`evaluate()` width-mismatch/alignment tests, `return_logits` tests, epoch-advancement tests — see the file); 14 passed after I5a's checkpoint-filtering tests (`test_save_checkpoint_excludes_the_frozen_sentence_transformer` and its two fault-injection siblings) were added (final-review fix, 2026-09-20)

- [ ] **Step 6: Commit**

```bash
git add src/prosodia/config.py src/prosodia/train/loop.py tests/test_loop.py
git commit -m "feat: training loop, evaluation, checkpointing"
```

---

### Task 15: Baselines and the 12-arm ablation runner

**Correction (owner Decision 2, 2026-09-19):** the code blocks below are the historical record of this task's original implementation (9-arm grid; `run_arm` reading `cfg.temperature_scale`, per "Ruling 1") and are now superseded on two points that are NOT reflected in the snippets: (1) `ARMS` gained 3 `TextOnlyBaseline` arms, one per loss regime (12 total, not 9 — the encoder axis collapses for text-only since audio is absent for every example), wired into `_build_loaders` via a `mute_audio`-wrapping `collate_fn` when `cfg.text_only`; (2) `run_arm` now also computes and logs a `coverage_curve`-based W&B table/plot per question via `_log_coverage_curves`, closing the gap where `coverage_curve` and `TextOnlyBaseline`/`mute_audio` existed and were tested but had zero call sites. See `src/prosodia/evaluation/baselines.py` and `scripts/run_ablation.py` for the authoritative current version, and spec §6/§7.1 for the design rationale (one text-only arm per loss regime, not per encoder).

**Correction (I2, final-review fix, 2026-09-20):** the `__main__` block below is superseded on one point: it never checked whether the encoder caches the grid was about to use actually covered the same examples. `ProsodiaDataset`'s own per-arm coverage floor (Task 6's correction) protects any ONE arm from training on a badly incomplete cache, but two arms can each individually clear that floor while still training on *different* data (e.g. two 99%-covered caches missing a different 1%) — and the grid's whole point is comparing arms trained on the same data. The current `scripts/run_ablation.py` adds `assert_uniform_cache_coverage(examples, encoders, cache_root)`, called in `__main__` right after `splits` is built and before the per-arm loop, over the set of encoders the actual run will use (honoring `--only`) and the union of every split's examples. It fails loudly — differing per-encoder counts and a sample of missing uids — before any arm trains, rather than letting a partial cache silently shrink one arm's dataset relative to the others. See `tests/test_run_ablation.py::test_assert_uniform_cache_coverage_raises_when_caches_disagree` and its two sibling tests (not tied to a numbered Task step in this doc, per this file's existing convention for `test_run_ablation.py`).

**Correction (I9, final-review fix, 2026-09-20):** the `run_arm` code block below is superseded on its central point: it trained Arm C (`cfg.temperature_scale=True`) as a fully independent run of the same config as Arm A, only branching to `_calibrated_test_metrics` for the final scoring step. Their identity therefore rested on `torch.manual_seed(cfg.seed)` plus an identical op sequence giving bit-identical results across two SEPARATE trainings — likely, but never asserted, and the Arm B vs Arm C comparison this whole grid exists to produce has an expected effect size (~0.01 ECE) that ordinary training noise could fully absorb. The current `scripts/run_ablation.py` derives Arm C instead: `run_arm` dispatches `cfg.temperature_scale=True` straight to `run_derived_arm_c`, which never calls `train_one_epoch`. It resolves its companion Arm A's name via `evaluation/baselines.py`'s new `companion_arm_a_name(cfg)` — built structurally from `cfg.encoder`/`cfg.text_only`, never by string-editing `cfg.name`, so a text-only Arm C cannot resolve to an encoder arm's name (or vice versa) — locates that arm's most recent checkpoint via `_find_latest_checkpoint` (a glob over `ckpt_root/<arm_a_name>/epoch*.pt`, robust to Arm A having trained in a separate process/invocation), loads it with `train.loop.load_checkpoint`, and applies the unchanged `_calibrated_test_metrics` scoring step. Checkpoint route chosen over "keep Arm A's model in memory and derive C immediately after" for robustness: it survives `--only`-style arm filtering and mid-grid interruption, at the cost of depending on `ARMS`' A-before-C-per-group ordering (satisfied by construction — see `evaluation/baselines.py`) for an un-filtered full run. If Arm A's checkpoint is missing, `_find_latest_checkpoint` raises `FileNotFoundError` naming the missing arm and the searched path — a loud failure, never a silent retrain-from-scratch fallback, which would restore the exact defect this fixes. A derived arm still logs its own coverage curve and test metrics (`run_derived_arm_c` calls `_log_coverage_curves`/`run.log` exactly as the trained path does) — the shortcut is in training, not in reporting. See `tests/test_run_ablation.py::test_derived_arm_c_never_calls_train_one_epoch`, `::test_derived_arm_c_pre_temperature_predictions_are_bit_identical_to_arm_a`, `::test_run_derived_arm_c_fails_loudly_when_arm_a_checkpoint_is_missing`, and `tests/test_baselines.py`'s three `companion_arm_a_name` tests for the regression coverage.

**Further correction (F3, final-cleanup review, 2026-09-20):** the "satisfied by construction" ordering claim above was flagged by the I9 implementer as resting only on how `ARMS` happens to be written, with the `FileNotFoundError` above as the sole runtime guard — which only fires if someone actually runs a misordered grid. `evaluation/baselines.py` now adds `assert_derived_arms_follow_their_source(arms)`, a static structural check (a single pass building a name-to-index map) asserting every derived Arm C's `companion_arm_a_name` appears at an earlier index than the Arm C itself. It is called once, at import time, against the real `ARMS`, so a future reordering fails immediately on import rather than only surfacing as a `FileNotFoundError` the next time the grid actually runs. The runtime `FileNotFoundError` guard is unchanged and still needed — it catches a missing/deleted checkpoint, a failure mode this static check cannot see. See `tests/test_baselines.py::test_assert_derived_arms_follow_their_source_passes_for_the_real_arms` and its two fault-injection siblings.

**Correction (I5b, final-review fix, 2026-09-20) — explicit model selection.** Before this fix, `run_arm` saved one checkpoint per epoch (`epochN.pt`) and scored TEST from `model`'s state at loop-exit — i.e. whatever the LAST epoch happened to produce, never selected on anything. That is an arbitrary point on each arm's trajectory to make the Arm B vs Arm C comparison from, at an expected effect size (~0.01 ECE) that arbitrariness could fully absorb. `run_arm` now:
1. Computes `dev_stats = evaluate(model, loaders["dev"])` every epoch as before, but reduces it via the new `train.loop.dev_selection_score(dev_stats)` to one scalar: the unweighted mean, across question keys, of `train.loop.DEV_SELECTION_METRIC = "nll"` — a **named constant**, not a buried literal. NLL, not ECE, is the selection metric: see the spec §10 I5 correction for the full reasoning (ECE is binned and noisy on a small dev split; selecting on it and then reporting test ECE is a subtle circularity that NLL, a proper scoring rule already computed by `evaluate()`, avoids).
2. Writes `arm_dir / "last.pt"` (module constant `LAST_CKPT_NAME`) unconditionally every epoch, and `arm_dir / "best.pt"` (`BEST_CKPT_NAME`) only when that epoch's `dev_selection_score` improves on every prior epoch's.
3. After the loop, reloads `best.pt` into `model` (`load_checkpoint`) BEFORE calling `_plain_test_metrics` — so TEST is always scored from the dev-selected checkpoint, never from the loop's post-hoc in-memory state.

This also replaces the pre-I5b per-epoch checkpoint scheme entirely (no more `epochN.pt` files at all), which is most of I5a's disk-budget fix in practice: two fixed files per arm instead of up to 20.

**Arm C is tied to Arm A's selected checkpoint, not to an independent choice.** `run_derived_arm_c`'s `_find_best_checkpoint` (renamed from the pre-I5b `_find_latest_checkpoint`, which globbed for the highest-numbered `epochN.pt`) now reads the fixed `ckpt_root / arm_a_name / "best.pt"` path directly — the exact file Arm A's own `run_arm` call scored its reported TEST metrics from. There is no second decision to make or get wrong: Arm C structurally cannot load a different point on Arm A's trajectory than the one Arm A itself reports, which is the property I9's "same model plus a temperature" derivation depends on. `_find_best_checkpoint` still fails LOUDLY (`FileNotFoundError`, naming `arm_name` and the searched path) when that file is missing — unchanged from I9's guard, just against the new fixed filename.

See `tests/test_run_ablation.py::test_run_arm_scores_test_from_the_best_dev_epoch_not_the_last` — constructs a 3-epoch run where the best dev-selection score lands on epoch 1 (not epoch 2, the last), tags each epoch's model with an observable marker, and asserts TEST scoring used epoch 1's marker. The I5 fix report has the fault-injection numbers (this test fails, asserting `2.0 == 1.0`, when TEST scoring is reverted to using the loop's post-hoc `model` state directly).

**Files:**
- Create: `src/prosodia/evaluation/baselines.py`
- Create: `scripts/run_ablation.py`
- Test: `tests/test_baselines.py`

**Interfaces:**
- Consumes: everything above
- Produces: `TextOnlyBaseline` (same architecture, audio permanently absent); `jev_baseline(examples, specs, api_key) -> dict[uid, dict[key, probs]]`; `ARMS: list[RunConfig]`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_baselines.py
from prosodia.evaluation.baselines import ARMS, build_jev_request
from prosodia.schema import QuestionSpec


def test_ablation_grid_is_three_losses_by_three_encoders():
    assert len(ARMS) == 9
    assert {a.encoder for a in ARMS} == {"wavlm", "whisper", "prosody"}
    names = {(a.brier_weight > 0, a.temperature_scale) for a in ARMS}
    assert names == {(False, False), (True, False), (False, True)}


def test_every_arm_has_a_unique_name():
    assert len({a.name for a in ARMS}) == len(ARMS)


def test_jev_request_matches_the_documented_schema():
    specs = [
        QuestionSpec("emotion", "choice", "Which emotion?",
                     {"anger": None, "joy": None}),
        QuestionSpec("sentiment", "score", "Rate sentiment.",
                     ["negative", "neutral", "positive"]),
        QuestionSpec("is_negative", "noul", "Is it negative?", None),
    ]
    req = build_jev_request("Joey: You fell asleep!", specs)
    assert req["model"] == "jev-latest"
    assert req["questions"]["emotion"]["type"] == "choice"
    assert set(req["questions"]["emotion"]["criteria"]) == {"anger", "joy"}
    assert req["questions"]["sentiment"]["criteria"] == \
        ["negative", "neutral", "positive"]
    assert req["questions"]["is_negative"]["type"] == "noul"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_baselines.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'prosodia.evaluation.baselines'`

- [ ] **Step 3: Implement**

```python
# src/prosodia/evaluation/baselines.py
"""Two baselines doing different jobs (spec §7).

  TextOnlyBaseline — CONTROLLED. Identical architecture and training, audio
                     permanently absent. Isolates modality and nothing else.
  Jev via API      — PRACTICAL. The actual text-state System One model.
"""
from __future__ import annotations

from typing import Any, Sequence

from prosodia.config import RunConfig
from prosodia.schema import QuestionSpec

JEV_ENDPOINT = "https://api.typesafe.ai/v1/systemone"


def build_jev_request(state: str, specs: Sequence[QuestionSpec]) -> dict[str, Any]:
    """Serialize our question bank into the documented TypeSafe schema."""
    questions: dict[str, Any] = {}
    for spec in specs:
        q: dict[str, Any] = {"type": spec.qtype, "instructions": spec.instructions}
        if spec.qtype == "choice":
            q["criteria"] = {k: v for k, v in spec.criteria.items()}
        elif spec.qtype == "score":
            q["criteria"] = list(spec.criteria)
        elif spec.criteria:
            q["criteria"] = dict(spec.criteria)
        questions[spec.key] = q
    return {"state": state, "model": "jev-latest", "questions": questions}


def _arm_name(encoder: str, brier: float, temp: bool) -> str:
    loss = "C-temp" if temp else ("B-brier" if brier > 0 else "A-ce")
    return f"{encoder}__{loss}"


ARMS: list[RunConfig] = [
    RunConfig(name=_arm_name(enc, brier, temp), encoder=enc,
              brier_weight=brier, temperature_scale=temp)
    for enc in ("wavlm", "whisper", "prosody")
    for brier, temp in ((0.0, False), (0.5, False), (0.0, True))
]


class TextOnlyBaseline:
    """Wraps a RunConfig so audio is absent for every example.

    Implemented as modality forcing rather than a separate model, so the
    controlled comparison holds architecture, parameter count, optimizer and
    data order fixed — only the modality changes.
    """

    def __init__(self, cfg: RunConfig) -> None:
        self.cfg = cfg

    @staticmethod
    def mute_audio(batch: dict[str, Any]) -> dict[str, Any]:
        out = dict(batch)
        out["audio_present"] = batch["audio_present"].clone().fill_(False)
        return out
```

- [ ] **Step 4: Write the ablation runner**

```python
# scripts/run_ablation.py
"""Run the 9-arm grid. Resumes from checkpoints, logs every arm to W&B.

Arm C (`cfg.temperature_scale`) is not a separate training run: its
training is identical to Arm A (`brier_weight=0.0`). Only the post-training
scoring step differs -- see `_calibrated_test_metrics`, which fits a
`TemperatureScaler` on the DEV split's logits and applies it to the TEST
split's logits. Fitting on TEST instead would leak the test set into the
calibration step and invalidate the Arm B vs Arm C comparison this whole
grid exists to produce.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from prosodia.config import RunConfig
from prosodia.corpora.meld import MeldCorpus
from prosodia.data import ProsodiaDataset, collate_batch
from prosodia.evaluation.baselines import ARMS
from prosodia.evaluation.metrics import compute_metrics
from prosodia.features import FeatureCache
from prosodia.model.prosodia import ProsodiaModel
from prosodia.schema import assert_thesis_safe
from prosodia.train.calibrate import TemperatureScaler
from prosodia.train.loop import evaluate, save_checkpoint, train_one_epoch


def _calibrated_test_metrics(
    model: Any, dev_loader: DataLoader, test_loader: DataLoader,
) -> dict[str, dict[str, float]]:
    """Arm C: fit a per-question `TemperatureScaler` on the DEV split's
    logits/targets, then transform the TEST split's logits before scoring.

    `evaluate(..., return_logits=True)` hands back the exact per-question
    logits/targets it already collected and aligned, so this reuses that one
    corrected path instead of re-deriving it -- see its docstring for why a
    second, independent collection loop would be risky here.
    """
    _, dev_logits, dev_targets = evaluate(model, dev_loader, return_logits=True)
    _, test_logits, test_targets = evaluate(model, test_loader, return_logits=True)

    results: dict[str, dict[str, float]] = {}
    for key, logits in test_logits.items():
        scaler = TemperatureScaler().fit(dev_logits[key], dev_targets[key])
        scaled = scaler.transform(logits)
        probs = torch.softmax(scaled, dim=-1)
        results[key] = compute_metrics(probs, test_targets[key])
    return results


def run_arm(
    cfg: RunConfig,
    loaders: dict[str, DataLoader],
    in_dim: int,
    ckpt_root: Path,
    run: Any = None,
) -> dict[str, dict[str, float]]:
    """Trains one arm end-to-end and returns its test-set metrics.

    Reads `cfg.temperature_scale` (Arm C): training is identical to Arm A,
    and only the final scoring step branches, via `_calibrated_test_metrics`.
    """
    torch.manual_seed(cfg.seed)
    model = ProsodiaModel(in_dim=in_dim, d_model=cfg.d_model,
                          state_layers=cfg.state_layers,
                          branch_layers=cfg.branch_layers,
                          n_heads=cfg.n_heads, stride=cfg.stride)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                            weight_decay=cfg.weight_decay)

    for epoch in range(cfg.epochs):
        train_stats = train_one_epoch(model, loaders["train"], opt, cfg, epoch)
        dev_stats = evaluate(model, loaders["dev"])
        if run is not None:
            run.log({"epoch": epoch, **{f"train/{k}": v for k, v in train_stats.items()},
                     **{f"dev/{q}/{m}": v for q, mm in dev_stats.items()
                        for m, v in mm.items()}})
        save_checkpoint(ckpt_root / cfg.name / f"epoch{epoch}.pt",
                        model, opt, epoch, cfg)

    if cfg.temperature_scale:
        test_stats = _calibrated_test_metrics(model, loaders["dev"], loaders["test"])
    else:
        test_stats = evaluate(model, loaders["test"])

    if run is not None:
        run.log({f"test/{q}/{m}": v for q, mm in test_stats.items()
                 for m, v in mm.items()})
    return test_stats


def _build_loaders(
    splits: dict[str, list], specs, cache: FeatureCache, cfg: RunConfig,
) -> dict[str, DataLoader]:
    return {
        name: DataLoader(
            ProsodiaDataset(exs, specs, cache, rng_seed=cfg.seed,
                            augment=(name == "train"),
                            modality_dropout=cfg.modality_dropout if name == "train" else 0.0),
            batch_size=cfg.batch_size, shuffle=(name == "train"),
            collate_fn=collate_batch)
        for name, exs in splits.items()
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-root", type=Path, required=True)
    ap.add_argument("--cache-root", type=Path, required=True)
    ap.add_argument("--ckpt-root", type=Path, default=Path.home() / "prosodia-ckpt")
    ap.add_argument("--only", nargs="*", help="arm names to run; default all")
    args = ap.parse_args()

    corpus = MeldCorpus(args.corpus_root)
    specs = corpus.question_specs()
    keys = [s.key for s in specs]

    splits = {s: list(corpus.iter_examples(s)) for s in ("train", "dev", "test")}
    # Guardrail: no model-output labels may back a thesis-testing split.
    assert_thesis_safe(splits["test"], keys)

    import wandb  # lazy: keeps this script importable (e.g. by tests) offline

    for cfg in ARMS:
        if args.only and cfg.name not in args.only:
            continue
        cache = FeatureCache(args.cache_root / cfg.encoder)
        loaders = _build_loaders(splits, specs, cache, cfg)

        in_dim = next(iter(loaders["train"]))["audio"].shape[-1]

        run = wandb.init(project=cfg.wandb_project, name=cfg.name,
                         config=cfg.as_dict(), reinit=True)
        test_stats = run_arm(cfg, loaders, in_dim, ckpt_root=args.ckpt_root, run=run)
        print(cfg.name, test_stats)
        run.finish()
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_baselines.py -v`
Expected: 3 passed originally; 6 passed after Decision 2 added the text-only-arm and `mute_audio` tests; 9 passed after I9's `companion_arm_a_name` tests were added; 12 passed after F3's `assert_derived_arms_follow_their_source` ordering-guard tests were added (final-cleanup review, 2026-09-20 — see `tests/test_baselines.py`)

- [ ] **Step 6: Run the full test suite**

Run: `uv run pytest -v`
Expected: all tests pass. Record the count.

- [ ] **Step 7: Smoke-run one arm for two epochs**

```bash
WANDB_API_KEY=<key> uv run python scripts/run_ablation.py \
  --corpus-root ~/prosodia-data/meld --cache-root ~/prosodia-cache \
  --only wavlm__A-ce
```

Expected: W&B run appears, dev ECE and accuracy logged per epoch, checkpoints written.

- [ ] **Step 8: Commit**

```bash
git add src/prosodia/evaluation/baselines.py scripts/run_ablation.py tests/test_baselines.py
git commit -m "feat: controlled and Jev baselines, 12-arm ablation runner"
```

---

## Phase 1 Exit Criteria

- [ ] All tests pass (`uv run pytest`)
- [ ] MELD extracted, WavLM features cached, cache size recorded
- [ ] All 12 arms trained to completion, logged to W&B (9-arm encoder grid + 3-arm text-only baseline, one per loss regime — Decision 2)
- [ ] Test-set ECE, Brier, NLL, accuracy, macro-F1 and coverage curves recorded per arm, per question (coverage curves logged to W&B as a table/plot via `_log_coverage_curves`, not bare tensors)
- [ ] Each encoder arm's calibration is diffed against the same-loss-regime text-only baseline arm — the controlled comparison success criterion #1 depends on (modality isolation for that baseline covers both content and audio duration — Decision 3)
- [ ] MELD's speaker-shared splits are flagged (startup warning + W&B config field) rather than silently treated as evidence for an audio-improves-calibration claim (Decision 1)
- [ ] The Arm B vs Arm C comparison is resolved — is calibration distributed, or is it a scalar?

That last one determines whether Phase 2's interpretability hunt is worth running at all. If CE+Brier is indistinguishable from CE plus one learned temperature, the answer to "where does calibration live" is *in a temperature*, and Phase 2 narrows to confirming that rather than searching for structure.
