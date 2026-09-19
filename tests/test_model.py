# tests/test_model.py
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
    torch.manual_seed(0)
    model = ProsodiaModel(in_dim=8, d_model=32).eval()
    full, muted = _batch(), _batch()
    muted["audio_present"] = torch.zeros(2, dtype=torch.bool)
    with torch.no_grad():
        a = model(full)[0]["emotion"]
        b = model(muted)[0]["emotion"]
    assert (a - b).abs().max().item() > 1e-5


def test_trainable_parameter_count_is_small():
    model = ProsodiaModel(in_dim=1024, d_model=256)
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert 1e6 < n < 40e6, f"{n/1e6:.1f}M trainable params outside expected range"
