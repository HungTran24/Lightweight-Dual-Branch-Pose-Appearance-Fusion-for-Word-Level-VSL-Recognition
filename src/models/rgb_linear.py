from __future__ import annotations

from typing import Any

import torch
from torch import nn


class Model(nn.Module):
    """Linear classifier over flattened pooled left/right RGB embeddings."""

    def __init__(self, *, nodes: int, classes: int, config: dict[str, Any]) -> None:
        super().__init__()
        del nodes, config
        self.classifier = nn.Linear(1152, classes)

    def forward(self, features: torch.Tensor, frame_mask: torch.Tensor) -> torch.Tensor:
        del frame_mask
        return self.classifier(features.reshape(features.shape[0], -1))
