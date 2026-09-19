import pytest
import torch
from torch.utils.data import DataLoader

from prosodia.config import RunConfig
from prosodia.data import ProsodiaDataset, collate_batch
from prosodia.device import get_device
from prosodia.evaluation.metrics import accuracy
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
    # `load_checkpoint` places both model and optimizer state on
    # `get_device()` (see its docstring: this is what prevents accelerator
    # params from ending up paired with CPU-resident optimizer buffers), so
    # `model` is moved there too before comparison -- otherwise this
    # assertion would fail on a device mismatch alone on any MPS/CUDA
    # machine, independent of whether the weights actually round-tripped.
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


def test_save_checkpoint_excludes_the_frozen_sentence_transformer(tmp_path):
    """I5a: `question_encoder._st` is the frozen `all-MiniLM-L6-v2` sentence
    transformer -- never trained (`requires_grad_(False)`, see
    `model/qencoder.py`) and reconstructed from the HuggingFace hub by
    `QuestionEncoder.__init__` on every fresh model instantiation,
    regardless of what any checkpoint does or doesn't restore. Measured at
    the grid's actual config (in_dim=1024, d_model=256) it is 22.71M of the
    model's 26.37M state_dict elements (86.1%) with zero information
    content once excluded. `save_checkpoint`'s on-disk "model" dict must
    contain no `question_encoder._st.*` key at all.

    Fault this catches: reverting `save_checkpoint` to plain
    `model.state_dict()` -- every `_st` parameter/buffer name would
    reappear and this test fails immediately (verified: reverting locally
    makes `frozen_keys` non-empty and the size assertion below fail).
    """
    model = ProsodiaModel(in_dim=8, d_model=32)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    cfg = RunConfig(name="t", brier_weight=0.0, encoder="wavlm")
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, model, opt, epoch=0, cfg=cfg)

    raw = torch.load(path, map_location="cpu")
    frozen_keys = [k for k in raw["model"] if k.startswith("question_encoder._st.")]
    assert frozen_keys == [], f"checkpoint still contains frozen encoder keys: {frozen_keys}"
    # The trainable projection living right next to the frozen encoder in
    # the same submodule must still be saved -- this excludes only `_st`,
    # not `question_encoder` wholesale.
    assert "question_encoder.project.weight" in raw["model"]

    full_elements = sum(v.numel() for v in model.state_dict().values())
    saved_elements = sum(v.numel() for v in raw["model"].values())
    assert saved_elements < full_elements * 0.2, (
        f"saved {saved_elements} elements out of {full_elements} in the "
        "full state_dict -- expected the frozen sentence-transformer (the "
        "large majority) to be excluded"
    )


def test_load_checkpoint_raises_when_a_trainable_key_is_missing(tmp_path):
    """`load_checkpoint` now loads with `strict=False` so the EXPECTED
    absence of `question_encoder._st.*` (I5a) doesn't raise -- but that
    must not degrade into "swallow any missing key". A checkpoint missing
    a genuinely TRAINABLE key (stale save, corruption, or a future bug
    that over-excludes) must still fail loudly rather than silently
    leaving part of the model uninitialized.

    Fault this catches: `load_state_dict(ckpt["model"], strict=False)`
    with no missing/unexpected-key check at all -- that version would
    silently leave `readout` at its fresh random init here instead of
    raising (confirmed red: removing the `bad_missing`/`unexpected` check
    in `load_checkpoint` makes this test fail to raise).
    """
    model = ProsodiaModel(in_dim=8, d_model=32)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    cfg = RunConfig(name="t", brier_weight=0.0, encoder="wavlm")
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, model, opt, epoch=0, cfg=cfg)

    raw = torch.load(path, map_location="cpu")
    victim = next(k for k in raw["model"] if k.startswith("readout."))
    del raw["model"][victim]
    torch.save(raw, path)

    fresh = ProsodiaModel(in_dim=8, d_model=32)
    with pytest.raises(RuntimeError, match="TRAINABLE"):
        load_checkpoint(path, fresh)


def test_load_checkpoint_raises_on_an_unexpected_key(tmp_path):
    """The other half of the same guard: a checkpoint carrying a key the
    model doesn't recognize at all (e.g. loaded against the wrong model
    shape/architecture) must also fail loudly, not be silently ignored."""
    model = ProsodiaModel(in_dim=8, d_model=32)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    cfg = RunConfig(name="t", brier_weight=0.0, encoder="wavlm")
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, model, opt, epoch=0, cfg=cfg)

    raw = torch.load(path, map_location="cpu")
    raw["model"]["totally.bogus.key"] = torch.zeros(1)
    torch.save(raw, path)

    fresh = ProsodiaModel(in_dim=8, d_model=32)
    with pytest.raises(RuntimeError, match="TRAINABLE"):
        load_checkpoint(path, fresh)


def test_checkpoint_roundtrip_restores_optimizer_state(tmp_path):
    """save/load must restore optimizer momentum, not just weights -- resuming
    a run with a cold optimizer would silently distort the first few
    post-resume gradient steps (Adam's running moments reset to zero)."""
    model = ProsodiaModel(in_dim=8, d_model=32)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loader = _loader(tmp_path)
    cfg = RunConfig(name="t", brier_weight=0.0, encoder="wavlm")
    train_one_epoch(model, loader, opt, cfg, epoch=0)  # populate optimizer state

    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, model, opt, epoch=1, cfg=cfg)

    fresh = ProsodiaModel(in_dim=8, d_model=32)
    fresh_opt = torch.optim.AdamW(fresh.parameters(), lr=1e-3)
    load_checkpoint(path, fresh, fresh_opt)

    assert len(fresh_opt.state) > 0
    compared = 0
    for p, fp in zip(model.parameters(), fresh.parameters()):
        # Params that received no gradient this step (e.g. the audio/context
        # "absent" tokens, when modality dropout never fires) never get an
        # optimizer state entry at all -- skip those rather than KeyError.
        if p not in opt.state or "exp_avg" not in opt.state[p]:
            continue
        orig_step = opt.state[p]["exp_avg"]
        fresh_step = fresh_opt.state[fp]["exp_avg"]
        torch.testing.assert_close(orig_step, fresh_step)
        compared += 1
    assert compared > 0, "no parameters had optimizer state to compare"


# --- epoch-advancement regression (see task-14 ruling) ------------------
#
# ProsodiaDataset seeds its per-item RNG from (rng_seed, epoch, idx) alone
# (Task 6). If train_one_epoch never advances the epoch, every pass over the
# data draws the identical augmentation/modality-dropout pattern -- a silent
# failure: loss still descends, metrics still look plausible.

def _augmenting_loader(tmp_path, n=8):
    cache = FeatureCache(tmp_path / "c")
    exs = []
    for i in range(n):
        cache.write(f"u{i}", torch.randn(16, 8))
        exs.append(Example(f"u{i}", "meld", "/x.wav", "ctx",
                           {"emotion": Label("joy", LabelTier.HUMAN)}, speaker="Joey"))
    ds = ProsodiaDataset(exs, SPECS, cache, augment=True, modality_dropout=0.5)

    seen: list[list[tuple]] = []

    def recording_collate(items):
        seen.append([
            (dict(it["questions"]), it["audio_present"], it["context_present"])
            for it in items
        ])
        return collate_batch(items)

    loader = DataLoader(ds, batch_size=n, shuffle=False, collate_fn=recording_collate)
    return loader, seen


def test_train_one_epoch_advances_the_dataset_epoch(tmp_path):
    model = ProsodiaModel(in_dim=8, d_model=32)
    loader, seen = _augmenting_loader(tmp_path)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    cfg = RunConfig(name="t", brier_weight=0.0, encoder="wavlm")

    train_one_epoch(model, loader, opt, cfg, epoch=0)
    train_one_epoch(model, loader, opt, cfg, epoch=1)

    assert len(seen) == 2
    assert seen[0] != seen[1], (
        "train_one_epoch did not advance the dataset epoch -- two epochs "
        "drew the identical augmentation/dropout pattern"
    )


def test_train_one_epoch_requires_an_explicit_epoch(tmp_path):
    """epoch has no default: a caller cannot forget to decide what epoch it
    is. This is the enforcement mechanism for the ruling above."""
    model = ProsodiaModel(in_dim=8, d_model=32)
    loader = _loader(tmp_path)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    cfg = RunConfig(name="t", brier_weight=0.0, encoder="wavlm")
    with pytest.raises(TypeError):
        train_one_epoch(model, loader, opt, cfg)  # type: ignore[call-arg]


def test_evaluate_does_not_advance_the_dataset_epoch(tmp_path):
    """Eval loaders are built with augment=False; evaluate() must not call
    set_epoch -- doing so would be meaningless at best and would make eval
    depend on how many times it has previously been called."""
    loader = _loader(tmp_path)
    model = ProsodiaModel(in_dim=8, d_model=32).eval()

    epoch_before = loader.dataset._epoch
    evaluate(model, loader)
    evaluate(model, loader)
    assert loader.dataset._epoch == epoch_before


# --- evaluate() width-mismatch alignment ---------------------------------
#
# When augmentation is on, permute_candidates can subsample the option set,
# so different examples can produce different numbers of options for the
# same question. evaluate() must filter mismatched-width rows out of BOTH
# logits and targets together -- filtering logits but then slicing targets
# by position silently misaligns the two lists.

class _StubModel:
    """Duck-typed stand-in for ProsodiaModel: evaluate() only calls
    .to(device), .eval(), and model(batch)."""

    def __init__(self, batch_outputs):
        self._batch_outputs = list(batch_outputs)
        self._i = 0

    def to(self, device):
        return self

    def eval(self):
        return self

    def __call__(self, batch):
        out = self._batch_outputs[self._i]
        self._i += 1
        return out


class _StubLoader:
    def __init__(self, batches):
        self._batches = batches

    def __iter__(self):
        return iter(self._batches)


def _dummy_state_tensors(n):
    return {
        "audio": torch.zeros(n, 1),
        "audio_mask": torch.zeros(n, 1, dtype=torch.bool),
        "audio_present": torch.zeros(n, dtype=torch.bool),
        "context_present": torch.zeros(n, dtype=torch.bool),
    }


def test_evaluate_keeps_targets_aligned_when_filtering_mismatched_widths(tmp_path):
    # row0: width 3, target 2 (correct answer at index 2)
    # row1: width 3, target 0
    # row2: width 5, target 1  <- outlier width, must be dropped
    # row3: width 3, target 2
    logits = [
        torch.tensor([-5.0, -5.0, 5.0]),   # argmax=2
        torch.tensor([5.0, -5.0, -5.0]),   # argmax=0
        torch.zeros(5),                    # outlier width, irrelevant values
        torch.tensor([-5.0, -5.0, 5.0]),   # argmax=2
    ]
    targets = [2, 0, 1, 2]

    batch = _dummy_state_tensors(4)
    batch["targets"] = [{"q": t} for t in targets]
    batch_outputs = [{"q": l} for l in logits]

    loader = _StubLoader([batch])
    model = _StubModel([batch_outputs])

    out = evaluate(model, loader)
    # Correctly aligned: kept rows are 0, 1, 3 with targets [2, 0, 2], and
    # every kept row's argmax matches its (correctly aligned) target.
    assert out["q"]["accuracy"] == 1.0


def test_evaluate_refuses_to_silently_discard_a_majority_of_rows(tmp_path):
    widths = [3, 3, 4, 4, 5]
    logits = [torch.zeros(w) for w in widths]
    targets = [0, 0, 0, 0, 0]

    batch = _dummy_state_tensors(5)
    batch["targets"] = [{"q": t} for t in targets]
    batch_outputs = [{"q": l} for l in logits]

    loader = _StubLoader([batch])
    model = _StubModel([batch_outputs])

    with pytest.raises(ValueError):
        evaluate(model, loader)


# --- evaluate(return_logits=True) — Task 15 / Arm C -----------------------
#
# Arm C needs raw per-question logits and targets to fit and apply
# TemperatureScaler. This must reuse evaluate()'s own aligned rows/targets
# (the ones fixed in the width-mismatch test above) rather than a second,
# independently-filtered collection loop -- a second loop is exactly how
# the target/logit misalignment bug could resurface.

def test_evaluate_return_logits_matches_the_aligned_rows_used_for_metrics(tmp_path):
    # Same fixture as test_evaluate_keeps_targets_aligned_when_filtering_
    # mismatched_widths: row2 (width 5) must be dropped, and the survivors'
    # targets must line up with their own logits, not a positional slice.
    logits = [
        torch.tensor([-5.0, -5.0, 5.0]),   # argmax=2, target=2 -> correct
        torch.tensor([5.0, -5.0, -5.0]),   # argmax=0, target=0 -> correct
        torch.zeros(5),                    # outlier width, dropped
        torch.tensor([-5.0, -5.0, 5.0]),   # argmax=2, target=2 -> correct
    ]
    targets = [2, 0, 1, 2]

    batch = _dummy_state_tensors(4)
    batch["targets"] = [{"q": t} for t in targets]
    batch_outputs = [{"q": l} for l in logits]

    loader = _StubLoader([batch])
    model = _StubModel([batch_outputs])

    results, out_logits, out_targets = evaluate(model, loader, return_logits=True)

    assert results["q"]["accuracy"] == 1.0
    assert out_logits["q"].shape == (3, 3)
    assert torch.equal(out_targets["q"], torch.tensor([2, 0, 2]))
    # The returned logits/targets must be the SAME pair evaluate() scored --
    # recomputing metrics from them independently must match exactly.
    probs = torch.softmax(out_logits["q"], dim=-1)
    assert accuracy(probs, out_targets["q"]).item() == results["q"]["accuracy"]


def test_evaluate_default_call_is_unaffected_by_the_new_parameter(tmp_path):
    """Existing callers (train_one_epoch's dev-eval, save/log call sites)
    must keep getting a plain metrics dict back with no code changes."""
    model = ProsodiaModel(in_dim=8, d_model=32).eval()
    out = evaluate(model, _loader(tmp_path))
    assert isinstance(out, dict)
    assert isinstance(out["emotion"], dict)
    assert "accuracy" in out["emotion"]
