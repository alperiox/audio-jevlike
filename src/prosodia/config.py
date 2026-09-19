from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class RunConfig:
    name: str
    encoder: str = "wavlm"          # wavlm | whisper | prosody
    brier_weight: float = 0.0       # 0.0 = Arm A, > 0 = Arm B
    temperature_scale: bool = False  # Arm C
    d_model: int = 256
    state_layers: int = 2
    branch_layers: int = 2
    n_heads: int = 4
    stride: int = 2
    lr: float = 3e-4
    weight_decay: float = 0.01
    epochs: int = 20
    batch_size: int = 16
    modality_dropout: float = 0.15
    seed: int = 0
    wandb_project: str = "prosodia"

    def as_dict(self) -> dict:
        return asdict(self)
