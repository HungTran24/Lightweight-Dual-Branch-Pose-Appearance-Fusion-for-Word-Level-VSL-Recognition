from __future__ import annotations

from typing import Any

import torch
from torch import nn


class Model(nn.Module):
    """BiLSTM over per-frame concatenated left/right hand MobileNet embeddings."""

    def __init__(self, *, nodes: int = 1, classes: int = 400, config: dict[str, Any] | None = None) -> None:
        super().__init__()
        del nodes
        cfg = config or {}
        input_dim = int(cfg.get("input_dim", 1152))
        proj_dim = int(cfg.get("projection_dim", 256))
        hidden_size = int(cfg.get("hidden_size", 128))
        dropout = float(cfg.get("dropout", 0.0))

        self.projection = nn.Sequential(
            nn.Linear(input_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )
        self.lstm = nn.LSTM(
            input_size=proj_dim,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=True,
        )
        self.classifier = nn.Linear(hidden_size * 2, classes)

    def forward(self, features: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        # features: [batch, T, 2, 576] or [batch, T, 1152]
        if features.dim() == 4:
            batch, frames, hands, dim = features.shape
            x = features.reshape(batch, frames, hands * dim)
        else:
            x = features

        # valid_mask: [batch, T]
        if valid_mask.dim() == 1:
            valid_mask = valid_mask.unsqueeze(0)

        projected = self.projection(x)
        out, _ = self.lstm(projected)

        mask = valid_mask.unsqueeze(-1).to(out.dtype)
        pooled = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return self.classifier(pooled)
