from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


class BiLSTM(nn.Module):
    """One-layer bidirectional LSTM over flattened keypoint coordinates."""

    def __init__(self, *, nodes: int, classes: int, config: dict[str, Any]) -> None:
        super().__init__()
        self.input_channels = int(config["input_channels"])
        features = nodes * self.input_channels
        hidden = int(config.get("hidden_size", 128))
        self.input_norm = nn.BatchNorm1d(features)
        self.lstm = nn.LSTM(
            input_size=features,
            hidden_size=hidden,
            batch_first=True,
            bidirectional=True,
        )
        self.classifier = nn.Linear(hidden * 2, classes)

    def forward(self, x: torch.Tensor, valid_frame_mask: torch.Tensor) -> torch.Tensor:
        batch, frames, nodes, _ = x.shape
        x = x[:, :, :, : self.input_channels].reshape(
            batch, frames, nodes * self.input_channels
        )
        x = self.input_norm(x.transpose(1, 2)).transpose(1, 2)
        lengths = valid_frame_mask.sum(dim=1).clamp_min(1).cpu()
        packed = pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )
        _, (hidden, _) = self.lstm(packed)
        return self.classifier(torch.cat((hidden[-2], hidden[-1]), dim=1))


Model = BiLSTM
