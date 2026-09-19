# tests/test_pooling.py
import torch

from prosodia.model.pooling import AttentionPool, speaker_relative_norm
from prosodia.model.state import StateEncoder


def _ramp(n, lo, hi):
    return torch.linspace(lo, hi, n).unsqueeze(-1).repeat(1, 4).unsqueeze(0)


def test_attention_pool_distinguishes_a_rise_from_a_fall():
    """Spec §11 trap 3: mean pooling destroys contours. A rising and a falling
    ramp have IDENTICAL means; pooled representations must still differ."""
    rise, fall = _ramp(32, 0.0, 1.0), _ramp(32, 1.0, 0.0)
    assert torch.allclose(rise.mean(1), fall.mean(1), atol=1e-6)  # means match

    pool = AttentionPool(dim=4, stride=2).eval()
    # Fix the scorer instead of trusting random init: if the sampled weights
    # happened to sum near zero the attention would be uniform (i.e. a mean)
    # and the test would flake rather than fail honestly.
    with torch.no_grad():
        pool.score.weight.fill_(1.0)
        pool.score.bias.zero_()
    mask = torch.ones(1, 32, dtype=torch.bool)
    with torch.no_grad():
        pr, _ = pool(rise, mask)
        pf, _ = pool(fall, mask)
    assert (pr - pf).abs().max().item() > 1e-3


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
    test fails if that warning returns."""
    enc = StateEncoder(in_dim=8, d_model=16, n_layers=1, n_heads=2, stride=2).eval()
    audio = torch.randn(2, 20, 8)
    mask = torch.ones(2, 20, dtype=torch.bool)
    mask[1, 12:] = False  # second example is shorter: only 12 valid frames
    with torch.no_grad():
        h, h_mask = enc(audio, mask)
    assert h.shape == (2, 10, 16)
    assert h_mask.shape == (2, 10)
    assert h_mask[0].all()
    assert h_mask[1].sum().item() == 6  # 12 valid frames / stride 2
    assert not torch.isnan(h).any()
