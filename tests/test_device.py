import torch
from prosodia.device import get_device, assert_close_across_devices


def test_get_device_returns_a_real_device():
    dev = get_device()
    assert dev.type in {"mps", "cuda", "cpu"}
    torch.zeros(4, device=dev)  # must actually allocate


def test_assert_close_across_devices_passes_for_stable_op():
    def softmax_entropy(x):
        p = torch.softmax(x, dim=-1)
        return -(p * p.log()).sum(-1)

    x = torch.randn(8, 16, dtype=torch.float32)
    assert_close_across_devices(softmax_entropy, x)


def test_assert_close_across_devices_raises_on_divergence():
    def bad(x):
        # deliberately device-dependent
        return x.sum() + (0.0 if x.device.type == "cpu" else 1.0)

    x = torch.randn(4, dtype=torch.float32)
    dev = get_device()
    if dev.type == "cpu":
        return  # nothing to compare against
    try:
        assert_close_across_devices(bad, x)
    except AssertionError:
        return
    raise AssertionError("expected divergence to be caught")


def test_assert_close_across_devices_moves_keyword_tensor_args():
    dev = get_device()
    if dev.type == "cpu":
        return  # nothing to compare against

    def add(x, *, y):
        # requires x and y on the same device
        return x + y

    x = torch.randn(4, dtype=torch.float32)
    y = torch.randn(4, dtype=torch.float32)
    assert_close_across_devices(add, x, y=y)


def test_assert_close_across_devices_preserves_non_float_dtype():
    dev = get_device()
    if dev.type == "cpu":
        return  # nothing to compare against

    def check_dtypes(x, mask, idx):
        assert mask.dtype == torch.bool, mask.dtype
        assert idx.dtype == torch.int64, idx.dtype
        return x[mask][: idx.numel()].sum()

    x = torch.randn(6, dtype=torch.float32)
    mask = torch.tensor([True, False, True, False, True, True])
    idx = torch.tensor([0, 1], dtype=torch.int64)
    assert_close_across_devices(check_dtypes, x, mask, idx)
