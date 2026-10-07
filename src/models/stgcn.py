"""Canonical ST-GCN classifier on anatomical pose graphs."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional


def build_spatial_adjacency(
    node_count: int,
    edges: tuple[tuple[int, int], ...],
    center_nodes: tuple[int, ...] = (0, 1),
) -> np.ndarray:
    """Build 3-partition adjacency (self, inward, outward) from anatomy edges."""
    neighbors: list[list[int]] = [[] for _ in range(node_count)]
    full = np.eye(node_count, dtype=np.float32)
    for source, target in edges:
        neighbors[source].append(target)
        neighbors[target].append(source)
        full[source, target] = 1.0
        full[target, source] = 1.0

    distance = np.full(node_count, np.inf, dtype=np.float32)
    frontier = list(center_nodes)
    distance[frontier] = 0.0
    while frontier:
        source = frontier.pop(0)
        for target in neighbors[source]:
            if np.isinf(distance[target]):
                distance[target] = distance[source] + 1.0
                frontier.append(target)

    column_degree = full.sum(axis=0)
    normalized = full @ np.diag(column_degree**-1.0)
    partitions = np.zeros((3, node_count, node_count), dtype=np.float32)
    partitions[0] = np.diag(np.diag(normalized))
    for source, target in edges:
        for input_node, output_node in ((source, target), (target, source)):
            partition = 1 if distance[input_node] <= distance[output_node] else 2
            partitions[partition, input_node, output_node] = normalized[input_node, output_node]
    return partitions


class _GraphConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, partitions: int) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.partitions = partitions
        self.projection = nn.Conv2d(in_channels, out_channels * partitions, kernel_size=1)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        batch, _, frames, nodes = x.shape
        projected = self.projection(x).reshape(
            batch, self.partitions, self.out_channels, frames, nodes
        )
        return torch.einsum("nkctv,kvw->nctw", projected, adjacency).contiguous()


class _Block(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        partitions: int,
        temporal_kernel: int,
        stride: int,
        dropout: float,
        residual: bool = True,
    ) -> None:
        super().__init__()
        padding = (temporal_kernel - 1) // 2
        self.gcn = _GraphConv(in_channels, out_channels, partitions)
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=(temporal_kernel, 1),
                stride=(stride, 1),
                padding=(padding, 0),
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout),
        )
        if not residual:
            self.residual = None
        elif in_channels == out_channels and stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        residual = 0 if self.residual is None else self.residual(x)
        return self.relu(self.tcn(self.gcn(x, adjacency)) + residual)


class STGCN(nn.Module):
    def __init__(self, *, nodes: int, classes: int, config: dict[str, Any]) -> None:
        super().__init__()
        from src.pose_views.pose59 import POSE59_CENTER_NODES, POSE59_EDGES
        from src.rtmw.pose42 import POSE42_EDGES
        from src.rtmw.pose53 import POSE53_EDGES

        graphs = {
            42: (POSE42_EDGES, (0, 21)),
            53: (POSE53_EDGES, (0, 1)),
            59: (POSE59_EDGES, POSE59_CENTER_NODES),
        }
        graph = graphs.get(nodes)
        if graph is None:
            raise ValueError(f"No default edges for nodes={nodes}")
        edges, center_nodes = graph
        self.register_buffer(
            "adjacency",
            torch.from_numpy(build_spatial_adjacency(nodes, edges, center_nodes)),
        )
        self.node_count = nodes
        input_channels = int(config.get("input_channels", 2))
        self.input_channels = input_channels
        dropout = float(config.get("dropout", 0.5))
        self.input_norm = nn.BatchNorm1d(input_channels * nodes)

        specifications = (
            (input_channels, 64, 1, False),
            (64, 64, 1, True),
            (64, 64, 1, True),
            (64, 64, 1, True),
            (64, 128, 2, True),
            (128, 128, 1, True),
            (128, 128, 1, True),
            (128, 256, 2, True),
            (256, 256, 1, True),
            (256, 256, 1, True),
        )
        partitions = 3
        self.blocks = nn.ModuleList(
            _Block(
                block_in,
                block_out,
                partitions=partitions,
                temporal_kernel=int(config.get("temporal_kernel", 9)),
                stride=stride,
                dropout=0.0 if index == 0 else dropout,
                residual=residual,
            )
            for index, (block_in, block_out, stride, residual) in enumerate(specifications)
        )
        self.edge_importance = nn.ParameterList(
            nn.Parameter(torch.ones(3, nodes, nodes)) for _ in self.blocks
        )
        self.classifier = nn.Conv2d(256, classes, kernel_size=1)
        self.apply(self._initialize_weights)

    @staticmethod
    def _initialize_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Conv1d, nn.Conv2d)):
            nn.init.kaiming_normal_(module.weight, mode="fan_out")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, 0.0, 0.01)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor, valid_frame_mask: torch.Tensor) -> torch.Tensor:
        batch, frames, nodes, _ = x.shape
        graph = x[:, :, :, : self.input_channels].permute(0, 3, 1, 2).contiguous()
        mask = valid_frame_mask.ne(0).to(dtype=graph.dtype)
        graph = graph * mask[:, None, :, None]
        normalized = self.input_norm(
            graph.permute(0, 3, 1, 2).reshape(batch, nodes * self.input_channels, frames)
        )
        hidden = normalized.reshape(batch, nodes, self.input_channels, frames).permute(0, 2, 3, 1)
        adjacency = self.adjacency.to(dtype=hidden.dtype)
        for block, importance in zip(self.blocks, self.edge_importance, strict=True):
            hidden = block(hidden, adjacency * importance)
        pooled_mask = functional.interpolate(
            mask[:, None], size=hidden.shape[2], mode="nearest"
        )[:, :, :, None]
        pooled = (hidden * pooled_mask).sum(dim=2) / pooled_mask.sum(dim=2).clamp_min(1.0)
        return self.classifier(pooled.mean(dim=2)[:, :, None, None]).flatten(1)


Model = STGCN
