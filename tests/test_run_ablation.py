"""Arm C wiring tests for the 9-arm ablation runner (Task 15, Ruling 1; I9).

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
  - `test_run_arm_dispatches_to_derived_scoring_only_when_temperature_scale`
    checks the wiring itself: `run_arm` must dispatch Arm C
    (`temperature_scale=True`) to `run_derived_arm_c` -- never to
    `train_one_epoch` -- and everything else to the plain training path.
  - `test_arm_with_temperature_scale_differs_from_the_same_arm_without_it`
    is the end-to-end version the brief's Ruling 1 explicitly asks for: an
    Arm A run and its derived Arm C companion, same underlying trained
    model, must not produce identical (uncalibrated vs. calibrated) test
    metrics.

I9 (owner-approved final review fix, 2026-09-19): Arm C used to be an
INDEPENDENT retraining of Arm A's exact config, so its identity with Arm A
rested on `torch.manual_seed` plus an identical op sequence giving
bit-identical MPS results across two separate runs -- likely, but never
asserted, and the Arm B vs Arm C comparison this whole grid exists to
produce has an expected effect size (~0.01 ECE) that ordinary training
noise could fully absorb. Arm C is now DERIVED: `run_derived_arm_c` loads
its companion Arm A's saved checkpoint and applies only the post-hoc
temperature-scaling step, making the identity exact by construction. New
tests below cover this:

  - `test_derived_arm_c_never_calls_train_one_epoch` -- Arm C must not
    retrain at all, proven by making `train_one_epoch` explode if called.
  - `test_derived_arm_c_pre_temperature_predictions_are_bit_identical_to_arm_a`
    -- the property the old two-independent-trainings design only hoped
    for, now asserted directly.
  - `test_run_derived_arm_c_fails_loudly_when_arm_a_checkpoint_is_missing`
    -- Arm C run via `--only` without its Arm A companion having trained
    first must raise, never silently retrain from scratch.
  - `test_run_derived_arm_c_logs_its_own_coverage_curve` -- a derived arm
    is still a materially distinct scored run and needs its own W&B log.
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


def test_run_arm_dispatches_to_derived_scoring_only_when_temperature_scale(tmp_path, monkeypatch):
    """Arm A (temperature_scale=False) must take the plain training path and
    never touch `_calibrated_test_metrics`. Arm C (temperature_scale=True)
    must dispatch to the derived path -- which loads its companion Arm A's
    checkpoint and calls `_calibrated_test_metrics` -- and must NEVER call
    `train_one_epoch` (I9: Arm C is no longer an independent training run).
    """
    calibrated_calls = []

    def fake_calibrated(model, dev_loader, test_loader, return_probs=False):
        calibrated_calls.append(1)
        stats = {"emotion": {"accuracy": 0.0, "macro_f1": 0.0, "ece": 0.0,
                             "brier": 0.0, "nll": 0.0}}
        if return_probs:
            probs = {"emotion": torch.tensor([[0.5, 0.5]])}
            targets = {"emotion": torch.tensor([0])}
            return stats, probs, targets
        return stats

    monkeypatch.setattr(run_ablation, "_calibrated_test_metrics", fake_calibrated)

    ckpt_root = tmp_path / "ckpt"
    split_examples, cache = _split_examples_and_cache(tmp_path)

    loaders_a = _loaders(split_examples, cache)
    # Names follow the real ARMS convention (see `companion_arm_a_name`):
    # Arm C's derivation looks up its companion by NAME, not by object
    # identity, so the fixture must use the real naming scheme.
    cfg_a = RunConfig(name="wavlm__A-ce", brier_weight=0.0, temperature_scale=False,
                      encoder="wavlm", d_model=16, epochs=1, batch_size=4)
    run_ablation.run_arm(cfg_a, loaders_a, in_dim=8, ckpt_root=ckpt_root)
    assert len(calibrated_calls) == 0, "Arm A must not take the calibrated path"

    def _train_should_not_be_called(*a, **k):
        raise AssertionError(
            "Arm C must not call train_one_epoch -- it derives from Arm A's "
            "checkpoint (I9), it does not retrain"
        )
    monkeypatch.setattr(run_ablation, "train_one_epoch", _train_should_not_be_called)

    loaders_c = _loaders(split_examples, cache)
    cfg_c = RunConfig(name="wavlm__C-temp", brier_weight=0.0, temperature_scale=True,
                      encoder="wavlm", d_model=16, epochs=1, batch_size=4)
    run_ablation.run_arm(cfg_c, loaders_c, in_dim=8, ckpt_root=ckpt_root)
    assert len(calibrated_calls) == 1, (
        "Arm C (temperature_scale=True) must dispatch to _calibrated_test_metrics "
        "via its derived checkpoint path"
    )


def test_arm_with_temperature_scale_differs_from_the_same_arm_without_it(tmp_path):
    """End-to-end version of the same guard: Arm A and its derived Arm C
    companion (uncalibrated vs. calibrated scoring of the SAME trained
    weights) must not produce identical test metrics. Before I9, Arm C
    also independently retrained; this asserts derivation didn't collapse
    the comparison back to a no-op."""
    kwargs = dict(brier_weight=0.0, encoder="wavlm", d_model=16,
                  epochs=6, lr=1e-2, batch_size=8, seed=0)
    ckpt_root = tmp_path / "ckpt"

    # Same underlying examples/cache for both arms.
    split_examples, cache = _split_examples_and_cache(tmp_path)

    loaders_a = _loaders(split_examples, cache, batch_size=8)
    cfg_a = RunConfig(name="wavlm__A-ce", temperature_scale=False, **kwargs)
    stats_a = run_ablation.run_arm(cfg_a, loaders_a, in_dim=8, ckpt_root=ckpt_root)

    # Arm C shares ckpt_root with Arm A -- it derives from Arm A's just-saved
    # checkpoint rather than training its own model (I9).
    loaders_c = _loaders(split_examples, cache, batch_size=8)
    cfg_c = RunConfig(name="wavlm__C-temp", temperature_scale=True, **kwargs)
    stats_c = run_ablation.run_arm(cfg_c, loaders_c, in_dim=8, ckpt_root=ckpt_root)

    assert stats_a != stats_c, (
        "an arm with temperature_scale=True produced identical test metrics "
        "to the same arm without it -- Arm C is not actually doing anything"
    )


# --- I9: Arm C is DERIVED from Arm A's checkpoint, not independently trained

def test_derived_arm_c_never_calls_train_one_epoch(tmp_path, monkeypatch):
    """The literal I9 defect, most directly stated: Arm C must never train.
    Fault this catches: reverting `run_arm` to the pre-I9 design (train an
    "identical" model, then calibrate) -- this test would fail the instant
    `train_one_epoch` is invoked for Arm C."""
    kwargs = dict(brier_weight=0.0, encoder="wavlm", d_model=16,
                  epochs=2, lr=1e-2, batch_size=8, seed=0)
    ckpt_root = tmp_path / "ckpt"
    split_examples, cache = _split_examples_and_cache(tmp_path)

    loaders_a = _loaders(split_examples, cache, batch_size=8)
    cfg_a = RunConfig(name="wavlm__A-ce", temperature_scale=False, **kwargs)
    run_ablation.run_arm(cfg_a, loaders_a, in_dim=8, ckpt_root=ckpt_root)

    def _boom(*a, **k):
        raise AssertionError("Arm C called train_one_epoch -- I9 regression")
    monkeypatch.setattr(run_ablation, "train_one_epoch", _boom)

    loaders_c = _loaders(split_examples, cache, batch_size=8)
    cfg_c = RunConfig(name="wavlm__C-temp", temperature_scale=True, **kwargs)
    run_ablation.run_arm(cfg_c, loaders_c, in_dim=8, ckpt_root=ckpt_root)  # must not raise


def test_derived_arm_c_pre_temperature_predictions_are_bit_identical_to_arm_a(tmp_path, monkeypatch):
    """I9's core guarantee, asserted directly: a derived Arm C's
    PRE-temperature-scaling logits on TEST must match Arm A's own to
    within hardware floating-point noise -- the property the old "two
    independent trainings" design only hoped would hold via
    torch.manual_seed + deterministic ops, and never actually checked.

    Forces `get_device()` to CPU for this test (`prosodia.train.loop`'s
    bound reference specifically) to get out from under MPS-specific
    non-determinism. Even so -- confirmed empirically while writing this
    test -- two SEPARATELY CONSTRUCTED `nn.Module` instances holding a
    bit-identical `state_dict` (verified directly: `torch.equal` on every
    parameter tensor) still produce forward outputs differing by ~1e-8 on
    this machine's Accelerate/BLAS backend, purely from floating-point
    non-associativity across distinct memory allocations -- NOT from
    anything Arm C's derivation does wrong. This is exactly the
    hardware-noise-vs-real-finding problem
    `prosodia.device.assert_close_across_devices` exists to guard against
    elsewhere in this codebase, which is also tolerance-based rather than
    `torch.equal`. So this test uses `atol=1e-5` -- ~2-3 orders of
    magnitude above the observed noise floor, and ~3-4 orders of magnitude
    below the divergence an actually-independent retraining produces (see
    the fault-injection note in the task report) -- rather than exact
    bitwise equality, which this platform cannot deliver across model
    objects regardless of correctness.

    Spies on `run_ablation.evaluate`'s `return_logits=True` calls against
    each arm's own TEST loader (both `_plain_test_metrics` for Arm A and
    `_calibrated_test_metrics` for Arm C call `evaluate(..., test_loader,
    return_logits=True)` as their first step, before any temperature
    scaling is applied) and compares the captured logits.

    Fault this catches: any Arm C path that independently constructs or
    trains a new model (even with the same seed/config) instead of loading
    Arm A's actual saved weights -- an independent retraining diverges by
    orders of magnitude more than hardware float noise (see the task
    report's fault-injection numbers), so this still fails hard if I9
    regresses.
    """
    monkeypatch.setattr("prosodia.train.loop.get_device", lambda: torch.device("cpu"))

    kwargs = dict(brier_weight=0.0, encoder="wavlm", d_model=16,
                  epochs=3, lr=1e-2, batch_size=8, seed=0)
    ckpt_root = tmp_path / "ckpt"
    split_examples, cache = _split_examples_and_cache(tmp_path)

    loaders_a = _loaders(split_examples, cache, batch_size=8)
    loaders_c = _loaders(split_examples, cache, batch_size=8)

    captured: dict[str, torch.Tensor] = {}
    real_evaluate = run_ablation.evaluate

    def spying_evaluate(model, loader, *a, **k):
        result = real_evaluate(model, loader, *a, **k)
        if k.get("return_logits") and loader is loaders_c["test"]:
            captured["arm_c"] = result[1]["emotion"]
        elif k.get("return_logits") and loader is loaders_a["test"]:
            captured["arm_a"] = result[1]["emotion"]
        return result

    monkeypatch.setattr(run_ablation, "evaluate", spying_evaluate)

    cfg_a = RunConfig(name="wavlm__A-ce", temperature_scale=False, **kwargs)
    run_ablation.run_arm(cfg_a, loaders_a, in_dim=8, ckpt_root=ckpt_root)

    cfg_c = RunConfig(name="wavlm__C-temp", temperature_scale=True, **kwargs)
    run_ablation.run_arm(cfg_c, loaders_c, in_dim=8, ckpt_root=ckpt_root)

    assert "arm_a" in captured and "arm_c" in captured
    assert torch.allclose(captured["arm_a"], captured["arm_c"], atol=1e-5, rtol=0), (
        "derived Arm C's pre-temperature TEST logits do not match Arm A's "
        f"to within hardware float noise (max diff "
        f"{(captured['arm_a'] - captured['arm_c']).abs().max().item():.2e}) "
        "-- Arm C must load Arm A's exact saved weights"
    )


def test_run_derived_arm_c_fails_loudly_when_arm_a_checkpoint_is_missing(tmp_path):
    """I9's other required guarantee: if Arm C is invoked (e.g. via
    `--only`) without its Arm A companion having trained first under the
    same --ckpt-root, this must raise loudly and immediately -- never
    silently fall back to training Arm C from scratch, which would restore
    the exact non-identity defect I9 fixes."""
    split_examples, cache = _split_examples_and_cache(tmp_path)
    loaders_c = _loaders(split_examples, cache, batch_size=8)
    cfg_c = RunConfig(name="wavlm__C-temp", brier_weight=0.0, temperature_scale=True,
                      encoder="wavlm", d_model=16, epochs=1, batch_size=8)
    empty_ckpt_root = tmp_path / "ckpt-empty"

    try:
        run_ablation.run_arm(cfg_c, loaders_c, in_dim=8, ckpt_root=empty_ckpt_root)
        assert False, "expected a FileNotFoundError when Arm A's checkpoint is missing"
    except FileNotFoundError as e:
        assert "wavlm__A-ce" in str(e)


def test_run_derived_arm_c_logs_its_own_coverage_curve(tmp_path):
    """A derived arm is still a materially distinct SCORED run (calibrated
    metrics differ from Arm A's uncalibrated ones) and must get its own
    logged W&B run -- I9's derivation shortcut is a training optimization,
    not a reporting one. Fault this catches: a derived-arm path that skips
    `_log_coverage_curves` because "it's just Arm A again"."""
    calls = []

    def fake_log_curves(run, probs, targets, prefix="test"):
        calls.append(prefix)

    kwargs = dict(brier_weight=0.0, encoder="wavlm", d_model=16,
                  epochs=1, batch_size=8, seed=0)
    ckpt_root = tmp_path / "ckpt"
    split_examples, cache = _split_examples_and_cache(tmp_path)

    loaders_a = _loaders(split_examples, cache, batch_size=8)
    cfg_a = RunConfig(name="wavlm__A-ce", temperature_scale=False, **kwargs)
    run_ablation.run_arm(cfg_a, loaders_a, in_dim=8, ckpt_root=ckpt_root)

    orig = run_ablation._log_coverage_curves
    run_ablation._log_coverage_curves = fake_log_curves
    try:
        loaders_c = _loaders(split_examples, cache, batch_size=8)
        cfg_c = RunConfig(name="wavlm__C-temp", temperature_scale=True, **kwargs)
        run = _RecordingRun()
        run_ablation.run_arm(cfg_c, loaders_c, in_dim=8, ckpt_root=ckpt_root, run=run)
    finally:
        run_ablation._log_coverage_curves = orig

    assert calls == ["test"], "derived Arm C must still log its own coverage curve"


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


# --- I2, cross-arm half: assert_uniform_cache_coverage ---------------------

def _examples(n):
    return [
        Example(f"u{i}", "meld", "/x.wav", "ctx",
               {"emotion": Label("joy", LabelTier.HUMAN)}, speaker="Joey")
        for i in range(n)
    ]


def _cache_with(tmp_path, name, uids):
    cache = FeatureCache(tmp_path / name)
    for uid in uids:
        cache.write(uid, torch.zeros(4, 8))
    return cache


def test_assert_uniform_cache_coverage_passes_when_caches_agree(tmp_path):
    """Fault this catches: a version of the check that always raises (or
    always passes) regardless of actual agreement -- the healthy case must
    not be blocked."""
    exs = _examples(10)
    uids = [e.uid for e in exs]
    _cache_with(tmp_path, "wavlm", uids)
    _cache_with(tmp_path, "whisper", uids)
    _cache_with(tmp_path, "prosody", uids)
    # Must not raise.
    run_ablation.assert_uniform_cache_coverage(
        exs, ["wavlm", "whisper", "prosody"], tmp_path)


def test_assert_uniform_cache_coverage_raises_when_caches_disagree(tmp_path):
    """I2's core scenario: a half-finished extraction for one encoder arm
    (here `whisper` is missing 3 of 10 uids that `wavlm` and `prosody` both
    have) must fail loudly, before any arm trains -- not silently give that
    arm a smaller, different training set. This is the check that protects
    the ablation grid's COMPARISON, distinct from (and complementary to)
    `ProsodiaDataset`'s own per-arm coverage floor: each cache here could
    individually clear that floor (7/10 = 70% is the extreme case chosen to
    make the fault obvious, but even two 99%-covered caches missing a
    *different* 1% would trip this and not that).

    Fault this catches: the pre-fix state of the world -- nothing in
    `run_ablation.py` ever compared caches across arms at all, so this
    scenario trained silently. Confirmed by fault injection (see the task
    report): with the check's body replaced by `return`, this test fails
    with `Failed: DID NOT RAISE ValueError`.
    """
    exs = _examples(10)
    uids = [e.uid for e in exs]
    _cache_with(tmp_path, "wavlm", uids)
    _cache_with(tmp_path, "whisper", uids[:7])  # missing u7, u8, u9
    _cache_with(tmp_path, "prosody", uids)

    try:
        run_ablation.assert_uniform_cache_coverage(
            exs, ["wavlm", "whisper", "prosody"], tmp_path)
        assert False, "expected a ValueError for disagreeing cache coverage"
    except ValueError as e:
        msg = str(e)
        assert "whisper" in msg
        assert "u7" in msg or "u8" in msg or "u9" in msg


def test_assert_uniform_cache_coverage_is_a_noop_for_a_single_encoder(tmp_path):
    """A single-encoder run (e.g. `--only` restricted to one arm's name)
    has nothing to be inconsistent WITH; the check must not require a
    second cache to exist at all.

    NOTE (F2 review): this test passes identically whether the
    `len(encoders) <= 1` guard exists or not -- a one-element coverage dict
    trivially self-compares (`reference = coverage[encoders[0]]` IS the
    only element, so `all(cov == reference for cov in coverage.values())`
    is true regardless). It provides zero regression protection for the
    line it is named after. The guard's actual load-bearing case is
    `encoders == []`, which WOULD raise `IndexError` on
    `coverage[encoders[0]]` without it -- see the dedicated test below."""
    exs = _examples(5)
    _cache_with(tmp_path, "wavlm", [e.uid for e in exs][:2])  # badly incomplete
    # Must not raise: only one encoder is in play, so there is no
    # cross-arm comparison to make. (ProsodiaDataset's own per-arm floor,
    # not this check, is what would catch this cache being bad on its own.)
    run_ablation.assert_uniform_cache_coverage(exs, ["wavlm"], tmp_path)


# --- F2: --only matching zero arms must fail loudly, not run nothing ------

def test_resolve_arms_to_run_returns_all_arms_when_only_is_not_given():
    assert run_ablation._resolve_arms_to_run(None) == list(run_ablation.ARMS)
    # argparse's `nargs="*"` also produces `[]` when `--only` is passed with
    # no values -- same "run everything" meaning as not passing it at all.
    assert run_ablation._resolve_arms_to_run([]) == list(run_ablation.ARMS)


def test_resolve_arms_to_run_returns_only_the_matching_arms():
    result = run_ablation._resolve_arms_to_run(["wavlm__A-ce", "text_only__B-brier"])
    assert {cfg.name for cfg in result} == {"wavlm__A-ce", "text_only__B-brier"}


def test_resolve_arms_to_run_fails_loudly_when_only_matches_no_arm():
    """F2's core fix: a `--only` value matching no arm name (almost always a
    typo) must raise, listing the available arm names, instead of silently
    producing an empty arms_to_run. Before this fix, that empty list sailed
    through `assert_uniform_cache_coverage` (trivially true for zero
    encoders) and the per-arm loop (iterates zero times) -- a clean exit
    having done nothing.

    Fault this catches: reverting to the bare list comprehension
    `[cfg for cfg in ARMS if not only or cfg.name in only]` with no
    following check -- confirmed by fault injection (see the task report):
    with the guard removed, this typo'd name returns `[]` instead of
    raising, and the message assertions below have nothing to check.
    """
    typo = "wavlm__A-c"  # missing the trailing 'e' of the real "wavlm__A-ce"
    try:
        run_ablation._resolve_arms_to_run([typo])
        assert False, "expected a ValueError for a --only value matching no arm"
    except ValueError as e:
        msg = str(e)
        assert typo in msg
        # every real arm name must be listed so the typo is easy to spot
        for cfg in run_ablation.ARMS:
            assert cfg.name in msg


def test_assert_uniform_cache_coverage_is_a_noop_for_zero_encoders(tmp_path):
    """F2: the guard's actual load-bearing case, not exercised by the
    single-encoder test above. `encoders == []` arises whenever `--only`
    matches no arm name at all (see `_resolve_arms_to_run` below, which now
    fails loudly before this is ever reached in practice) or, prior to that
    fix, from an empty `--only` intersection more generally.

    Without the `len(encoders) <= 1` guard, `reference = coverage[encoders[0]]`
    raises `IndexError: list index out of range` on an empty `encoders` --
    confirmed by fault injection (see the task report): commenting out the
    guard turns this test's clean pass into
    `IndexError: list index out of range` immediately.

    No caches are created at all here -- irrelevant, since zero encoders
    means the function must return before ever touching a cache.
    """
    exs = _examples(3)
    run_ablation.assert_uniform_cache_coverage(exs, [], tmp_path)
