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
