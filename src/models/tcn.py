from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional


class _Block(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, kernel: int, dilation: int, dropout: float) -> None:
        super().__init__()
        padding = (kernel - 1) * dilation
        self.padding = padding
        self.conv1 = nn.Conv1d(input_channels, output_channels, kernel, dilation=dilation)
        self.conv2 = nn.Conv1d(output_channels, output_channels, kernel, dilation=dilation)
        self.norm1 = nn.BatchNorm1d(output_channels)
        self.norm2 = nn.BatchNorm1d(output_channels)
        self.dropout = nn.Dropout(dropout)
        self.skip = nn.Conv1d(input_channels, output_channels, 1) if input_channels != output_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        x = functional.pad(x, (self.padding, 0))
        x = self.dropout(torch.relu(self.norm1(self.conv1(x))))
        x = functional.pad(x, (self.padding, 0))
        x = self.dropout(torch.relu(self.norm2(self.conv2(x))))
        return torch.relu(x + residual)


class TCN(nn.Module):
    def __init__(self, *, nodes: int, classes: int, config: dict[str, Any]) -> None:
        super().__init__()
        channels = tuple(config["channels"])
        dilations = tuple(config["dilations"])
        input_channels = int(config["input_channels"])
        self.input_channels = input_channels
        features = nodes * input_channels
        self.input_norm = nn.BatchNorm1d(features)
        self.input_projection = nn.Conv1d(features, channels[0], 1)
        blocks: list[nn.Module] = []
        current = channels[0]
        for output, dilation in zip(channels, dilations, strict=True):
            blocks.append(_Block(current, output, int(config["kernel_size"]), int(dilation), float(config["dropout"])))
            current = output
        self.blocks = nn.Sequential(*blocks)
        self.classifier = nn.Linear(current, classes)

    def forward(self, x: torch.Tensor, valid_frame_mask: torch.Tensor) -> torch.Tensor:
        batch, frames, nodes, _ = x.shape
        x = x[:, :, :, : self.input_channels].reshape(batch, frames, nodes * self.input_channels)
        features = self.input_projection(self.input_norm(x.transpose(1, 2)))
        features = self.blocks(features)
        mask = valid_frame_mask.unsqueeze(1).to(features.dtype)
        pooled = (features * mask).sum(dim=-1) / mask.sum(dim=-1).clamp_min(1)
        return self.classifier(pooled)


Model = TCN
