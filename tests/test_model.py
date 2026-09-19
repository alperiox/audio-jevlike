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
