from __future__ import annotations

from typing import Any

import torch
from torch import nn


class Model(nn.Module):
    """Shared token projection over six hand/temporal-bin RGB embeddings."""

    def __init__(self, *, nodes: int, classes: int, config: dict[str, Any]) -> None:
        super().__init__()
        del nodes, config
        self.norm = nn.LayerNorm(576, elementwise_affine=False)
        self.projection = nn.Linear(576, 64)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(6 * 64, classes)

    def forward(self, features: torch.Tensor, token_present: torch.Tensor) -> torch.Tensor:
        batch = features.shape[0]
        tokens = features.reshape(batch, 6, 576)
        tokens = self.dropout(self.activation(self.projection(self.norm(tokens))))
        tokens = tokens * token_present.reshape(batch, 6, 1).to(tokens.dtype)
        return self.classifier(tokens.reshape(batch, 6 * 64))
