"""Arm C wiring tests for the 9-arm ablation runner (Task 15, Ruling 1).

The brief's `run_ablation.py` declared `cfg.temperature_scale` on
`RunConfig` and set it per-arm in `ARMS`, but its runner never read the
field: every "Arm C" run would have trained and scored identically to Arm A,
just logged under a different W&B name. These tests exercise the actual
post-hoc temperature-scaling wiring in `scripts/run_ablation.py`:

  - `test_calibrated_test_metrics_fits_on_dev_not_test` is a leak-detection
    test: it constructs DEV logits that are confidently WRONG (driving the
    fitted temperature well away from 1) and TEST logits that are
    confidently RIGHT. If the code fit on TEST instead of DEV, "calibrated"
    test NLL would come out indistinguishable from unscaled -- exactly what
    Ruling 1 forbids ("fitting on test would leak the test set into the
    calibration").
  - `test_run_arm_dispatches_to_calibrated_scoring_only_when_configured`
    checks the wiring itself: `run_arm` must call the calibrated path when
    `cfg.temperature_scale` is True and the plain path otherwise -- this is
    the literal defect (the field never being read at all).
  - `test_arm_with_temperature_scale_differs_from_the_same_arm_without_it`
    is the end-to-end version the brief's Ruling 1 explicitly asks for: two
    otherwise-identical arms, same seed, same data, differing only in
    `temperature_scale`, must not produce identical test metrics.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from prosodia.config import RunConfig
from prosodia.data import ProsodiaDataset, collate_batch
from prosodia.evaluation.metrics import negative_log_likelihood
from prosodia.features import FeatureCache
from prosodia.schema import Example, Label, LabelTier, QuestionSpec

_SPEC = importlib.util.spec_from_file_location(
    "run_ablation", Path(__file__).resolve().parents[1] / "scripts" / "run_ablation.py")
run_ablation = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(run_ablation)


SPECS = [QuestionSpec("emotion", "choice", "Which emotion?",
                      {"anger": None, "joy": None, "neutral": None})]


# --- shared stub plumbing (mirrors tests/test_loop.py's pattern) ----------

def _dummy_state_tensors(n):
    return {
        "audio": torch.zeros(n, 1),
        "audio_mask": torch.zeros(n, 1, dtype=torch.bool),
        "audio_present": torch.zeros(n, dtype=torch.bool),
        "context_present": torch.zeros(n, dtype=torch.bool),
    }


class _StubLoader:
    def __init__(self, batches):
        self._batches = batches

    def __iter__(self):
        return iter(self._batches)


class _RoutingStubModel:
    """Duck-typed stand-in that selects its canned output by inspecting
    which loader's batch it was actually called with (via `batch["uid"]`'s
    tag), not by call order.

    This distinction matters: a call-order-based stub (return outputs[0] on
    the first call, outputs[1] on the second) cannot tell a dev_loader/
    test_loader *argument swap* apart from correct code -- both still call
    evaluate() exactly twice, in some order, and a call-order stub would
    hand back the "right" outputs regardless of which physical loader was
    actually passed where. Routing by the batch's own identity closes that
    hole: if `_calibrated_test_metrics` ever fits on the wrong loader, this
    model sees the wrong batch and the test's downstream NLL assertion
    fails for the right reason.
    """

    def __init__(self, routes: dict[str, list[dict]]) -> None:
        self._routes = {tag: list(outs) for tag, outs in routes.items()}
        self._i = {tag: 0 for tag in routes}

    def to(self, device):
        return self

    def eval(self):
        return self

    def __call__(self, batch):
        tag = batch["uid"][0].rstrip("0123456789")
        out = self._routes[tag][self._i[tag]]
        self._i[tag] += 1
        return out


def _tagged_batch(tag, logits, targets):
    n = len(targets)
    batch = _dummy_state_tensors(n)
    batch["uid"] = [f"{tag}{i}" for i in range(n)]
    batch["targets"] = [{"q": t} for t in targets]
    return batch, [{"q": l} for l in logits]


def test_calibrated_test_metrics_fits_on_dev_not_test():
    # DEV: confidently WRONG (label 0, logits favor class 1) -- an
    # overconfident, badly miscalibrated split, which should pull the fitted
    # temperature well above 1.
    dev_logits = torch.tensor([
        [6.0, -6.0], [6.0, -6.0], [6.0, -6.0], [-6.0, 6.0],
    ])
    dev_targets = [0, 0, 0, 0]

    # TEST: confidently RIGHT and matches its own logits exactly -- already
    # well-calibrated w.r.t. itself, so unscaled test NLL is ~0.
    test_logits = torch.tensor([[6.0, -6.0]] * 4)
    test_targets = [0, 0, 0, 0]

    dev_batch, dev_outputs = _tagged_batch("dev", dev_logits, dev_targets)
    test_batch, test_outputs = _tagged_batch("test", test_logits, test_targets)

    model = _RoutingStubModel({"dev": [dev_outputs], "test": [test_outputs]})
    dev_loader = _StubLoader([dev_batch])
    test_loader = _StubLoader([test_batch])

    calibrated = run_ablation._calibrated_test_metrics(model, dev_loader, test_loader)

    unscaled_probs = torch.softmax(test_logits, dim=-1)
    unscaled_nll = negative_log_likelihood(unscaled_probs, torch.tensor(test_targets)).item()

    assert unscaled_nll < 0.01, "sanity check: test logits should be near-perfectly confident"
    # If the code (wrongly) fit on TEST instead of DEV, calibrated NLL would
    # land right back at ~unscaled_nll, since TEST is self-consistent.
    assert calibrated["q"]["nll"] > unscaled_nll + 0.1, (
        "calibrated test NLL should be visibly worse than unscaled -- the "
        "temperature was fit on DEV's confident-and-wrong logits, so "
        "applying it to TEST's confident-and-right logits should soften "
        "them noticeably. A near-zero result here means the scaler was fit "
        "on TEST (a leak) or never applied at all."
    )


def _split_examples_and_cache(tmp_path, n=8, seed=1234):
    """TRAIN is single-class ("joy" for every example, with context/audio
    carrying no real signal), so the model's cheapest fit is an overwhelming
    "always predict joy" bias -- it converges to extreme confidence. DEV and
    TEST use a different, mixed label pattern on independent examples, so
    that same "always joy" bias is confidently WRONG on roughly two-thirds
    of them: genuine miscalibration for a temperature fit on DEV to correct,
    rather than a degenerate 100%-confident-and-100%-correct case where a
    fitted temperature has nothing to do (T would stay ~1 either way, which
    would make an Arm A vs Arm C comparison look identical for the wrong
    reason -- not because Arm C's wiring is broken).

    Seeded independently of run_arm's own torch.manual_seed(cfg.seed) call,
    so the *input data* two arms train on is identical regardless of which
    arm runs first -- only cfg.temperature_scale is allowed to differ.
    """
    gen = torch.Generator().manual_seed(seed)
    cache = FeatureCache(tmp_path / "c")
    cycle = ["joy", "anger", "neutral"]

    def _make(prefix, label_fn):
        exs = []
        for i in range(n):
            uid = f"{prefix}{i}"
            cache.write(uid, torch.randn(16, 8, generator=gen))
            exs.append(Example(uid, "meld", "/x.wav", "ctx",
                               {"emotion": Label(label_fn(i), LabelTier.HUMAN)}, speaker="Joey"))
        return exs

    return {
        "train": _make("tr", lambda i: "joy"),
        "dev": _make("dv", lambda i: cycle[i % 3]),
        "test": _make("te", lambda i: cycle[i % 3]),
    }, cache


def _loaders(split_examples, cache, batch_size=4):
    def _one(name, augment, modality_dropout):
        ds = ProsodiaDataset(split_examples[name], SPECS, cache, rng_seed=0,
                             augment=augment, modality_dropout=modality_dropout)
        return DataLoader(ds, batch_size=batch_size, shuffle=augment, collate_fn=collate_batch)

    return {
        "train": _one("train", augment=True, modality_dropout=0.0),
        "dev": _one("dev", augment=False, modality_dropout=0.0),
        "test": _one("test", augment=False, modality_dropout=0.0),
    }


def _small_loaders(tmp_path, n=8):
    split_examples, cache = _split_examples_and_cache(tmp_path, n=n)
    return _loaders(split_examples, cache)


def test_run_arm_dispatches_to_calibrated_scoring_only_when_configured(tmp_path, monkeypatch):
    calibrated_calls = []
    plain_calls = []

    def fake_calibrated(model, dev_loader, test_loader, return_probs=False):
        calibrated_calls.append(1)
        stats = {"emotion": {"accuracy": 0.0, "macro_f1": 0.0, "ece": 0.0,
                             "brier": 0.0, "nll": 0.0}}
        if return_probs:
            probs = {"emotion": torch.tensor([[0.5, 0.5]])}
            targets = {"emotion": torch.tensor([0])}
            return stats, probs, targets
        return stats

    real_evaluate = run_ablation.evaluate

    def spy_evaluate(model, loader, *a, **k):
        if "return_logits" not in k:
            plain_calls.append(1)
        return real_evaluate(model, loader, *a, **k)

    monkeypatch.setattr(run_ablation, "_calibrated_test_metrics", fake_calibrated)
    monkeypatch.setattr(run_ablation, "evaluate", spy_evaluate)

    loaders_a = _small_loaders(tmp_path / "a")
    cfg_a = RunConfig(name="t-a", brier_weight=0.0, temperature_scale=False,
                      encoder="wavlm", d_model=16, epochs=1, batch_size=4)
    run_ablation.run_arm(cfg_a, loaders_a, in_dim=8, ckpt_root=tmp_path / "ckpt-a")
    assert len(calibrated_calls) == 0, "Arm A must not take the calibrated path"
    assert len(plain_calls) >= 1, "Arm A must score TEST with the plain evaluate() path"

    calibrated_calls.clear()
    plain_calls.clear()
    loaders_c = _small_loaders(tmp_path / "c")
    cfg_c = RunConfig(name="t-c", brier_weight=0.0, temperature_scale=True,
                      encoder="wavlm", d_model=16, epochs=1, batch_size=4)
    run_ablation.run_arm(cfg_c, loaders_c, in_dim=8, ckpt_root=tmp_path / "ckpt-c")
    assert len(calibrated_calls) == 1, (
        "Arm C (temperature_scale=True) must dispatch to _calibrated_test_metrics -- "
        "this is the exact defect Ruling 1 fixes: cfg.temperature_scale being read at all"
    )


def test_arm_with_temperature_scale_differs_from_the_same_arm_without_it(tmp_path):
    """End-to-end version of the same guard: two arms, identical in every
    way except `temperature_scale`, must not produce identical test metrics.
    Before this task, Arm C was bit-identical to Arm A under a different
    name -- this is the test that would have caught it."""
    kwargs = dict(brier_weight=0.0, encoder="wavlm", d_model=16,
                  epochs=6, lr=1e-2, batch_size=8, seed=0)

    # Same underlying examples/cache for both arms -- only cfg.temperature_scale
    # differs. run_arm's own torch.manual_seed(cfg.seed) call (same seed for
    # both) makes the rest of each run (model init, shuffling, training)
    # bit-for-bit identical, so any difference in the returned metrics can
    # only come from the post-training temperature-scaling branch.
    split_examples, cache = _split_examples_and_cache(tmp_path)

    loaders_a = _loaders(split_examples, cache, batch_size=8)
    cfg_a = RunConfig(name="arm-a", temperature_scale=False, **kwargs)
    stats_a = run_ablation.run_arm(cfg_a, loaders_a, in_dim=8, ckpt_root=tmp_path / "ckpt-a")

    loaders_c = _loaders(split_examples, cache, batch_size=8)
    cfg_c = RunConfig(name="arm-c", temperature_scale=True, **kwargs)
    stats_c = run_ablation.run_arm(cfg_c, loaders_c, in_dim=8, ckpt_root=tmp_path / "ckpt-c")

    assert stats_a != stats_c, (
        "an arm with temperature_scale=True produced identical test metrics "
        "to the same arm without it -- Arm C is not actually doing anything"
    )


# --- Decision 1: MELD speaker-leakage warning ------------------------------

class _NotMeld:
    """Stands in for a non-MELD corpus (e.g. IEMOCAP) -- the warning must
    not fire for it."""


def test_meld_speaker_leakage_warning_fires_for_meld_corpus():
    from prosodia.corpora.meld import MeldCorpus
    corpus = MeldCorpus(root=Path("/nonexistent"))  # construction does no I/O
    warning = run_ablation._meld_speaker_leakage_warning(corpus)
    assert warning is not None
    assert "speaker" in warning.lower()
    assert "pipeline validation" in warning.lower()


def test_meld_speaker_leakage_warning_is_silent_for_non_meld_corpus():
    """Fault this catches: a warning that fires unconditionally (or on the
    wrong type check) would wrongly flag IEMOCAP runs as speaker-leaked,
    undermining the very claim IEMOCAP is supposed to carry."""
    assert run_ablation._meld_speaker_leakage_warning(_NotMeld()) is None


def test_print_meld_warning_writes_to_stderr_only_for_meld(capsys):
    from prosodia.corpora.meld import MeldCorpus

    run_ablation._print_meld_warning_if_applicable(_NotMeld())
    assert capsys.readouterr().err == ""

    run_ablation._print_meld_warning_if_applicable(MeldCorpus(root=Path("/nonexistent")))
    captured = capsys.readouterr()
    assert "MELD SPEAKER-LEAKAGE WARNING" in captured.err
    assert "audio arm" in captured.err.lower() or "audio-improves-calibration" in captured.err.lower()


# --- Decision 2: coverage curves -------------------------------------------

class _RecordingRun:
    def __init__(self):
        self.logged: list[dict] = []

    def log(self, payload):
        self.logged.append(payload)


def test_log_coverage_curves_logs_one_plot_per_question():
    """Fault this catches: a coverage-curve call site that computes the
    tensors but never logs them (or logs them as bare tensors nobody can
    read later) would leave §7's 'practical artifact' invisible in W&B
    despite `coverage_curve` itself being fully tested."""
    run = _RecordingRun()
    probs_by_q = {
        "emotion": torch.softmax(torch.randn(20, 3), dim=-1),
        "sentiment": torch.softmax(torch.randn(20, 2), dim=-1),
    }
    targets_by_q = {
        "emotion": torch.randint(0, 3, (20,)),
        "sentiment": torch.randint(0, 2, (20,)),
    }
    run_ablation._log_coverage_curves(run, probs_by_q, targets_by_q)

    logged_keys = {k for payload in run.logged for k in payload}
    assert logged_keys == {"test/emotion/coverage_curve", "test/sentiment/coverage_curve"}
    # not bare tensors: each logged value is a wandb chart object
    for payload in run.logged:
        for value in payload.values():
            assert not isinstance(value, torch.Tensor)


def test_run_arm_logs_coverage_curves_when_run_is_given(tmp_path):
    """End-to-end: `run_arm` must call `_log_coverage_curves` when a `run`
    is supplied, for both the plain (Arm A) and calibrated (Arm C) paths --
    exit criteria promised a curve was produced, and before this wiring
    neither branch called it at all."""
    calls = []

    def fake_log_curves(run, probs, targets, prefix="test"):
        calls.append(prefix)

    import prosodia.train.loop  # noqa: F401  (ensure real evaluate is loaded)
    orig = run_ablation._log_coverage_curves
    run_ablation._log_coverage_curves = fake_log_curves
    try:
        split_examples, cache = _split_examples_and_cache(tmp_path, n=8)
        loaders = _loaders(split_examples, cache, batch_size=8)
        cfg = RunConfig(name="t", brier_weight=0.0, temperature_scale=False,
                        encoder="wavlm", d_model=16, epochs=1, batch_size=8)
        run = _RecordingRun()
        run_ablation.run_arm(cfg, loaders, in_dim=8, ckpt_root=tmp_path / "ckpt", run=run)
    finally:
        run_ablation._log_coverage_curves = orig

    assert calls == ["test"], "run_arm must log a coverage curve for its test-set result"


# --- Decision 2: text-only baseline wiring ---------------------------------

def test_build_loaders_forces_audio_absent_when_cfg_text_only(tmp_path):
    """Fault this catches: if `_build_loaders` forgot to wrap the collate_fn
    (or wrapped it with a no-op), a `text_only=True` arm would train with
    real audio -- silently invalidating the controlled comparison success
    criterion #1 depends on, exactly as if `TextOnlyBaseline` did not exist."""
    split_examples, cache = _split_examples_and_cache(tmp_path, n=8)
    cfg = RunConfig(name="text-only-t", encoder="wavlm", text_only=True,
                    modality_dropout=0.0, batch_size=8)
    loaders = run_ablation._build_loaders(split_examples, SPECS, cache, cfg)
    batch = next(iter(loaders["test"]))
    assert not batch["audio_present"].any(), (
        "text_only=True must force audio_present False for every example"
    )


def test_build_loaders_leaves_audio_present_alone_when_not_text_only(tmp_path):
    """Companion to the fault above: confirms the text-only wrapping is
    conditional, not a global change that would silently mute audio for
    every arm (the encoder ablation arms would be meaningless if so)."""
    split_examples, cache = _split_examples_and_cache(tmp_path, n=8)
    cfg = RunConfig(name="wavlm-t", encoder="wavlm", text_only=False,
                    modality_dropout=0.0, batch_size=8)
    loaders = run_ablation._build_loaders(split_examples, SPECS, cache, cfg)
    batch = next(iter(loaders["test"]))
    assert batch["audio_present"].all(), (
        "audio_present should be True for every example when modality_dropout=0 "
        "and text_only=False"
    )
