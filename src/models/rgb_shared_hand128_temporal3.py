from __future__ import annotations

from typing import Any

import torch
from torch import nn


class Model(nn.Module):
    """Shared 128-d projection over six ordered hand/temporal RGB tokens."""

    def __init__(self, *, nodes: int, classes: int, config: dict[str, Any]) -> None:
        super().__init__()
        del nodes, config
        self.shared = nn.Sequential(
            nn.Linear(576, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.3),
        )
        self.classifier = nn.Linear(6 * 128, classes)

    def forward(self, features: torch.Tensor, token_present: torch.Tensor) -> torch.Tensor:
        batch = features.shape[0]
        tokens = features.reshape(batch, 6, 576)
        encoded = self.shared(tokens)
        encoded = encoded * token_present.reshape(batch, 6, 1).to(encoded.dtype)
        return self.classifier(encoded.reshape(batch, 6 * 128))
