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
    """C1: prior to the positional-encoding fix, three compounding causes —
    no positions on `StateEncoder`'s self-attention, `IsolatedBranches`'
    cross-attention being permutation-invariant over its keys, and
    `AttentionPool` swapping weights and values together within a window —
    made the WHOLE ASSEMBLED MODEL exactly invariant to time order. Verified
    on this exact test before the fix: max|logit diff| for a time-reversed
    input was 1.4e-09 (see the final report for the fault-injection re-run).

    This has to be a model-level test, not a pooling-level one: any strided
    reducer preserves the time axis, so comparing pooled SEQUENCES
    element-wise (the old `test_attention_pool_distinguishes_a_rise_from_a_
    fall` in `test_pooling.py`) passes regardless of whether the model as a
    whole is order-sensitive. Only comparing the model's own output on
    time-reversed input actually exercises the invariant that matters.
    """
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


def test_muted_audio_gate_covers_the_c2_stat_tokens_too():
    """The C2 fix (`utterance_statistics`) added a second path from raw audio
    into the state: per-channel mean/std computed BEFORE speaker-relative
    normalization strips them, projected and prepended by `StateEncoder` as
    two "stat token" positions ahead of the pooled sequence. That path did
    not exist when the `audio_present` gate in `_encode_state`
    (`torch.where(audio_present, h, self.audio_absent...)`) was written, so
    it needed re-checking that the gate still covers it -- `StateEncoder`
    returns ALL positions (stat tokens included) as one `h` tensor, and the
    gate replaces that whole tensor per-example when `audio_present` is
    False, so it does.

    This matters specifically for Decision 2's text-only baseline
    (`TextOnlyBaseline.mute_audio`, `scripts/run_ablation.py`): it is the
    ONLY controlled comparison this phase has. If audio ever leaked past
    this gate, the "text-only" arm would silently become a second audio arm,
    and the headline claim ("audio improves calibration over a controlled
    text-only baseline") would be comparing audio against audio -- nothing
    else in the suite would notice, since every other model test uses
    `audio_present=True` for its audio-content assertions.

    Asserts EXACT equality (not just "small"): with `audio_present=False`,
    `_encode_state`'s output no longer depends on `batch["audio"]`'s content
    at all, so a loud/quiet pair (differs only in level) and a rise/fall
    pair (differs only in contour) must produce bit-identical logits, not
    merely close ones.
    """
    torch.manual_seed(0)
    model = ProsodiaModel(in_dim=8, d_model=32).eval()
    t = 24

    shape = torch.randn(t, 8)
    loud = (shape * 0.80 + 210.0).unsqueeze(0).expand(2, t, 8).contiguous()
    quiet = (shape * 0.10 + 105.0).unsqueeze(0).expand(2, t, 8).contiguous()
    rise = torch.linspace(0.0, 1.0, t).view(1, t, 1).expand(2, t, 8).contiguous()
    fall = torch.linspace(1.0, 0.0, t).view(1, t, 1).expand(2, t, 8).contiguous()

    def _muted_batch(audio):
        batch = _batch(n=2, t=t)
        batch["audio"] = audio
        batch["audio_present"] = torch.zeros(2, dtype=torch.bool)
        return batch

    with torch.no_grad():
        loud_out = model(_muted_batch(loud))[0]["emotion"]
        quiet_out = model(_muted_batch(quiet))[0]["emotion"]
        rise_out = model(_muted_batch(rise))[0]["emotion"]
        fall_out = model(_muted_batch(fall))[0]["emotion"]

    loud_vs_quiet = (loud_out - quiet_out).abs().max().item()
    rise_vs_fall = (rise_out - fall_out).abs().max().item()
    assert loud_vs_quiet == 0.0, f"audio_present=False leaked level info: diff={loud_vs_quiet:.3e}"
    assert rise_vs_fall == 0.0, f"audio_present=False leaked contour info: diff={rise_vs_fall:.3e}"


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
