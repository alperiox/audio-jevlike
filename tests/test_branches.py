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
