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
# actually order-sensitive. It passed even after the reviewer swapped in a
# within-window mean, ~1000x past its own threshold. C1's real invariant --
# whether the ASSEMBLED MODEL'S OUTPUT changes under time reversal -- can
# only be tested at the model level; see
# `test_model.py::test_model_output_is_not_invariant_to_time_reversal` and
# `test_model.py::test_model_distinguishes_rising_from_falling_pitch_contour`.


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
