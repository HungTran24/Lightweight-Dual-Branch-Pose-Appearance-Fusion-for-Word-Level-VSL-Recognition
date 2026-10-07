"""Project SPOTER-style classifier adaptation for the 54-node pose view."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


class SPOTER(nn.Module):
    """Minimal SPOTER-style temporal encoder-decoder classifier.

    This is a project adaptation using PyTorch's TransformerDecoder, not a full
    reproduction of the official SPOTER custom decoder.
    """

    def __init__(self, *, nodes: int, classes: int, config: dict[str, Any]) -> None:
        super().__init__()
        if nodes != 54:
            raise ValueError(f"SPOTER requires 54 nodes, got {nodes}")
        self.input_projection = nn.Linear(54 * 2, 108)
        self.frame_embedding = nn.Embedding(int(config.get("max_frames", 30)), 108)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=108,
            nhead=9,
            dim_feedforward=2048,
            dropout=float(config.get("dropout", 0.1)),
            batch_first=True,
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=108,
            nhead=9,
            dim_feedforward=2048,
            dropout=float(config.get("dropout", 0.1)),
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
        self.class_query = nn.Parameter(torch.zeros(1, 1, 108))
        self.classifier = nn.Linear(108, classes)
        nn.init.normal_(self.class_query, std=0.02)

    def forward(self, x: torch.Tensor, valid_frame_mask: torch.Tensor) -> torch.Tensor:
        batch, frames, nodes, channels = x.shape
        if nodes != 54 or channels < 2:
            raise ValueError(f"Expected [B,T,54,3+] input, got {tuple(x.shape)}")
        if frames > self.frame_embedding.num_embeddings:
            raise ValueError(
                f"SPOTER supports at most {self.frame_embedding.num_embeddings} frames"
            )
        padding = ~valid_frame_mask.to(dtype=torch.bool)
        # Keep one zero frame visible for all-padding examples to avoid attention NaNs.
        safe_padding = padding.clone()
        all_padding = safe_padding.all(dim=1)
        safe_padding[all_padding, 0] = False
        frame_ids = torch.arange(frames, device=x.device)
        encoded_input = self.input_projection(x[..., :2].reshape(batch, frames, -1))
        encoded_input = encoded_input + self.frame_embedding(frame_ids)[None]
        memory = self.encoder(encoded_input, src_key_padding_mask=safe_padding)
        query = self.class_query.expand(batch, -1, -1)
        decoded = self.decoder(query, memory, memory_key_padding_mask=safe_padding)
        return self.classifier(decoded[:, 0])


Model = SPOTER
