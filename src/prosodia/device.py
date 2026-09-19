"""Device selection and the numerical-fidelity guard.

Prosodia claims small effects (ECE deltas ~0.01, entropy shifts from single
ablations). Those are exactly the magnitudes an MPS backend inconsistency can
manufacture. Measurement code runs fp32 and is checked against CPU.
"""
from __future__ import annotations

import torch


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def assert_close_across_devices(fn, *args, atol: float = 1e-4, **kwargs) -> None:
    """Run `fn` on the accelerator and on CPU in fp32; assert agreement.

    No-op when the accelerator *is* CPU. Use on every metric and every
    interpretability measurement.
    """
    dev = get_device()
    if dev.type == "cpu":
        return

    def _to(obj, d):
        if isinstance(obj, torch.Tensor):
            return obj.to(device=d, dtype=torch.float32)
        return obj

    acc = fn(*[_to(a, dev) for a in args], **kwargs)
    cpu = fn(*[_to(a, "cpu") for a in args], **kwargs)
    acc_t = acc.detach().to("cpu", torch.float32)
    cpu_t = cpu.detach().to(torch.float32)
    max_diff = (acc_t - cpu_t).abs().max().item()
    if max_diff > atol:
        raise AssertionError(
            f"{getattr(fn, '__name__', fn)} diverges across devices: "
            f"max|{dev.type} - cpu| = {max_diff:.3e} > atol={atol:.3e}"
        )
