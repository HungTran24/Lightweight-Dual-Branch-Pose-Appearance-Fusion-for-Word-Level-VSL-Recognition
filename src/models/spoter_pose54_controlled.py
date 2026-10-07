"""SPOTER Pose54 controlled adaptation under the project training protocol.

Architecture adapted from matyasbohacek/spoter@0f909bf9 (Apache-2.0).
It preserves the published value-only decoder change, but uses the project's
fixed-length Pose54 features and shared runner rather than Apple Vision inputs.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


class _ValueOnlyDecoderLayer(nn.Module):
    """Official SPOTER decoder layer without redundant class-query self-attention."""

    def __init__(self, *, dropout: float) -> None:
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(108, 9, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(108, 2048)
        self.linear2 = nn.Linear(2048, 108)
        self.norm1 = nn.LayerNorm(108)
        self.norm2 = nn.LayerNorm(108)
        self.norm3 = nn.LayerNorm(108)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        memory_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        target = self.norm1(tgt + self.dropout1(tgt))
        cross = self.multihead_attn(
            target,
            memory,
            memory,
            key_padding_mask=memory_key_padding_mask,
            need_weights=False,
        )[0]
        target = self.norm2(target + self.dropout2(cross))
        feedforward = self.linear2(self.dropout(torch.relu(self.linear1(target))))
        return self.norm3(target + self.dropout3(feedforward))


class SPOTERPose54Controlled(nn.Module):
    """Published SPOTER topology adapted to project Pose54 xy frame tokens."""

    def __init__(self, *, nodes: int, classes: int, config: dict[str, Any]) -> None:
        super().__init__()
        if nodes != 54:
            raise ValueError(f"SPOTER Pose54 requires 54 nodes, got {nodes}")
        self.max_frames = int(config.get("max_frames", 60))
        if self.max_frames <= 0:
            raise ValueError("max_frames must be positive")
        dropout = float(config.get("dropout", 0.1))
        # Official SPOTER adds one learned positional tensor to its input token.
        self.pos = nn.Parameter(torch.rand(1, 1, 108))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=108,
            nhead=9,
            dim_feedforward=2048,
            dropout=dropout,
            activation="relu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        self.decoder = nn.ModuleList(
            _ValueOnlyDecoderLayer(dropout=dropout) for _ in range(6)
        )
        self.class_query = nn.Parameter(torch.rand(1, 1, 108))
        self.classifier = nn.Linear(108, classes)

    def forward(self, x: torch.Tensor, valid_frame_mask: torch.Tensor) -> torch.Tensor:
        batch, frames, nodes, channels = x.shape
        if nodes != 54 or channels < 2:
            raise ValueError(f"Expected [B,T,54,2+] input, got {tuple(x.shape)}")
        if frames > self.max_frames:
            raise ValueError(f"SPOTER supports at most {self.max_frames} frames")
        padding = ~valid_frame_mask.to(dtype=torch.bool)
        safe_padding = padding.clone()
        all_padding = safe_padding.all(dim=1)
        safe_padding[all_padding, 0] = False
        pose = x[..., :2].reshape(batch, frames, 108)
        memory = self.encoder(
            pose + self.pos,
            src_key_padding_mask=safe_padding,
        )
        query = self.class_query.expand(batch, -1, -1)
        decoded = query
        for layer in self.decoder:
            decoded = layer(decoded, memory, memory_key_padding_mask=safe_padding)
        return self.classifier(decoded[:, 0])


Model = SPOTERPose54Controlled
