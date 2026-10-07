from __future__ import annotations

from typing import Any

import torch
from torch import nn


class Model(nn.Module):
    """Shared projection over ordered left/right global RGB embeddings."""

    def __init__(self, *, nodes: int, classes: int, config: dict[str, Any]) -> None:
        super().__init__()
        del nodes, config
        self.shared = nn.Sequential(
            nn.Linear(576, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.3),
        )
        self.classifier = nn.Linear(256, classes)

    def forward(self, features: torch.Tensor, frame_mask: torch.Tensor) -> torch.Tensor:
        del frame_mask
        hands = features.reshape(features.shape[0], 2, 576)
        encoded = self.shared(hands)
        return self.classifier(encoded.reshape(features.shape[0], -1))
