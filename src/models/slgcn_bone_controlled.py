"""SL-GCN Pose27 bone-stream controlled adaptation.

Adapted from jackyjsy/CVPR21Chal-SLR@de6c53a6 (CC0-1.0), specifically
SL-GCN graph/sign_27.py and model/decouple_gcn_attn.py. The project uses
Pose27 xy bone vectors and fixed Uniform-30 features, so this is not the
full four-stream SAM-SLR system or a VSL400 reproduction.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional

NODE_COUNT = 27
# Exact sign_27 graph from the official implementation, converted to 0-based nodes.
INWARD = (
    (0, 1), (0, 2), (1, 3), (3, 5), (2, 4), (4, 6),
    (7, 8), (7, 9), (7, 11), (7, 13), (7, 15),
    (9, 10), (11, 12), (13, 14), (15, 16),
    (17, 18), (17, 19), (17, 21), (17, 23), (17, 25),
    (19, 20), (21, 22), (23, 24), (25, 26),
    (5, 7), (6, 17),
)
BONE_PAIRS = INWARD


def _adjacency() -> np.ndarray:
    identity = np.eye(NODE_COUNT, dtype=np.float32)
    inward = np.zeros((NODE_COUNT, NODE_COUNT), dtype=np.float32)
    for source, target in INWARD:
        inward[target, source] = 1.0
    outward = inward.T
    for matrix in (inward, outward):
        degree = matrix.sum(axis=0)
        nonzero = degree > 0
        matrix[:, nonzero] /= degree[nonzero]
    return np.stack((identity, inward, outward), axis=0)


class _DropSkeleton(nn.Module):
    def __init__(self, *, nodes: int) -> None:
        super().__init__()
        self.nodes = nodes

    def forward(self, x: torch.Tensor, keep_prob: float, adjacency: torch.Tensor) -> torch.Tensor:
        if not self.training or keep_prob >= 1.0:
            return x
        importance = x.abs().mean(dim=(1, 2))
        importance = importance / importance.sum().clamp_min(1e-12) * importance.numel()
        gamma = (1.0 - keep_prob) / 2.92
        seed = torch.bernoulli((importance * gamma).clamp(max=1.0))
        mask = (seed @ adjacency).gt(1e-3).to(dtype=x.dtype).view(len(x), 1, 1, self.nodes)
        keep = 1.0 - mask
        return x * keep * (keep.numel() / keep.sum().clamp_min(1.0))


class _DropTemporal(nn.Module):
    def __init__(self, *, block_size: int) -> None:
        super().__init__()
        self.block_size = block_size

    def forward(self, x: torch.Tensor, keep_prob: float) -> torch.Tensor:
        if not self.training or keep_prob >= 1.0:
            return x
        batch, channels, frames, nodes = x.shape
        importance = x.abs().mean(dim=(1, 3))
        importance = importance / importance.sum().clamp_min(1e-12) * importance.numel()
        gamma = (1.0 - keep_prob) / self.block_size
        seeds = torch.bernoulli((importance[:, None] * gamma).clamp(max=1.0))
        expanded = functional.max_pool1d(
            seeds, kernel_size=self.block_size, stride=1, padding=self.block_size // 2
        )
        keep = 1.0 - expanded.repeat(1, channels * nodes, 1)
        flattened = x.permute(0, 1, 3, 2).reshape(batch, channels * nodes, frames)
        return (
            flattened * keep * (keep.numel() / keep.sum().clamp_min(1.0))
        ).reshape(batch, channels, nodes, frames).permute(0, 1, 3, 2)


class _DecoupledGCN(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, adjacency: np.ndarray, groups: int) -> None:
        super().__init__()
        if out_channels % groups:
            raise ValueError("SL-GCN out_channels must divide groups")
        self.out_channels = out_channels
        self.groups = groups
        self.subsets = 3
        self.nodes = adjacency.shape[-1]
        base = torch.tensor(adjacency[:, None], dtype=torch.float32).repeat(1, groups, 1, 1)
        self.decoupled_adjacency = nn.Parameter(base)
        self.linear_weight = nn.Parameter(torch.empty(in_channels, out_channels * self.subsets))
        self.linear_bias = nn.Parameter(torch.full((1, out_channels * self.subsets, 1, 1), 1e-6))
        nn.init.normal_(self.linear_weight, 0.0, math.sqrt(0.5 / (out_channels * self.subsets)))
        self.bn0 = nn.BatchNorm2d(out_channels * self.subsets)
        self.bn = nn.BatchNorm2d(out_channels)
        self.down = (
            nn.Sequential(nn.Conv2d(in_channels, out_channels, 1), nn.BatchNorm2d(out_channels))
            if in_channels != out_channels
            else nn.Identity()
        )
        self.relu = nn.ReLU()
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.constant_(self.bn.weight, 1e-6)

    def _normalize(self, adjacency: torch.Tensor) -> torch.Tensor:
        # Official normalize_digraph divides each column by its incoming degree.
        degree = adjacency.sum(dim=2, keepdim=True).clamp_min(1e-3)
        return adjacency / degree

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        learned = self.decoupled_adjacency.repeat_interleave(self.out_channels // self.groups, dim=1)
        learned = torch.cat([self._normalize(learned[index : index + 1]) for index in range(3)])
        projected = torch.einsum("nctv,cd->ndtv", x, self.linear_weight) + self.linear_bias
        projected = self.bn0(projected)
        batch, _, frames, nodes = projected.shape
        projected = projected.view(batch, self.subsets, self.out_channels, frames, nodes)
        convolved = torch.einsum("nkctv,kcvw->nctw", projected, learned)
        return self.relu(self.bn(convolved) + self.down(x))


class _Unit(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        adjacency: np.ndarray,
        *,
        groups: int,
        block_size: int,
        stride: int = 1,
        residual: bool = True,
        attention: bool = True,
    ) -> None:
        super().__init__()
        self.gcn = _DecoupledGCN(in_channels, out_channels, adjacency, groups)
        temporal_conv = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=(9, 1),
            stride=(stride, 1),
            padding=(4, 0),
        )
        temporal_norm = nn.BatchNorm2d(out_channels)
        self.tcn = nn.Sequential(temporal_conv, temporal_norm)
        nn.init.kaiming_normal_(temporal_conv.weight, mode="fan_out")
        nn.init.ones_(temporal_norm.weight)
        nn.init.zeros_(temporal_norm.bias)
        self.residual = (
            nn.Identity()
            if residual and in_channels == out_channels and stride == 1
            else nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, stride=(stride, 1)), nn.BatchNorm2d(out_channels))
            if residual
            else None
        )
        self.tcn_skeleton_drop = _DropSkeleton(nodes=NODE_COUNT)
        self.tcn_temporal_drop = _DropTemporal(block_size=block_size)
        self.skip_skeleton_drop = _DropSkeleton(nodes=NODE_COUNT)
        self.skip_temporal_drop = _DropTemporal(block_size=block_size)
        self.register_buffer("graph", torch.tensor(adjacency.sum(axis=0), dtype=torch.float32))
        self.attention = attention
        if attention:
            spatial_kernel = NODE_COUNT
            self.spatial_attention = nn.Conv1d(out_channels, 1, spatial_kernel, padding=(spatial_kernel - 1) // 2)
            self.temporal_attention = nn.Conv1d(out_channels, 1, 9, padding=4)
            self.channel_reduce = nn.Linear(out_channels, out_channels // 2)
            self.channel_expand = nn.Linear(out_channels // 2, out_channels)
            nn.init.xavier_normal_(self.spatial_attention.weight)
            assert self.spatial_attention.bias is not None
            nn.init.zeros_(self.spatial_attention.bias)
            nn.init.zeros_(self.temporal_attention.weight)
            assert self.temporal_attention.bias is not None
            nn.init.zeros_(self.temporal_attention.bias)
            nn.init.kaiming_normal_(self.channel_reduce.weight)
            nn.init.zeros_(self.channel_reduce.bias)
            nn.init.zeros_(self.channel_expand.weight)
            nn.init.zeros_(self.channel_expand.bias)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor, keep_prob: float) -> torch.Tensor:
        hidden = self.gcn(x)
        if self.attention:
            spatial = torch.sigmoid(self.spatial_attention(hidden.mean(dim=2))).unsqueeze(2)
            hidden = hidden + hidden * spatial
            temporal = torch.sigmoid(self.temporal_attention(hidden.mean(dim=3))).unsqueeze(3)
            hidden = hidden + hidden * temporal
            channel = hidden.mean(dim=(2, 3))
            channel = torch.sigmoid(self.channel_expand(self.relu(self.channel_reduce(channel))))
            hidden = hidden + hidden * channel[:, :, None, None]
        hidden = self.tcn(hidden)
        hidden = self.tcn_temporal_drop(
            self.tcn_skeleton_drop(hidden, keep_prob, self.graph), keep_prob
        )
        if self.residual is None:
            residual: torch.Tensor | int = 0
        else:
            residual = self.skip_temporal_drop(
                self.skip_skeleton_drop(self.residual(x), keep_prob, self.graph), keep_prob
            )
        return self.relu(hidden + residual)


class SLGCNBoneControlled(nn.Module):
    """Official SL-GCN 27-node bone-stream architecture on project Pose27 xy."""

    def __init__(self, *, nodes: int, classes: int, config: dict[str, Any]) -> None:
        super().__init__()
        if nodes != NODE_COUNT:
            raise ValueError(f"SL-GCN bone stream requires {NODE_COUNT} nodes, got {nodes}")
        if int(config.get("input_channels", 2)) != 2:
            raise ValueError("SL-GCN controlled adaptation requires input_channels=2")
        adjacency = _adjacency()
        groups = int(config.get("groups", 16))
        block_size = int(config.get("block_size", 41))
        self.keep_rate = float(config.get("keep_rate", 0.9))
        self.input_norm = nn.BatchNorm1d(NODE_COUNT * 2)
        specs = ((2, 64, 1, False), (64, 64, 1, True), (64, 64, 1, True), (64, 64, 1, True), (64, 128, 2, True), (128, 128, 1, True), (128, 128, 1, True), (128, 256, 2, True), (256, 256, 1, True), (256, 256, 1, True))
        self.blocks = nn.ModuleList(
            _Unit(
                in_channels,
                out_channels,
                adjacency,
                groups=groups,
                block_size=block_size,
                stride=stride,
                residual=residual,
            )
            for in_channels, out_channels, stride, residual in specs
        )
        self.classifier = nn.Linear(256, classes)
        nn.init.normal_(self.classifier.weight, 0.0, math.sqrt(2.0 / classes))
        nn.init.zeros_(self.classifier.bias)
        nn.init.ones_(self.input_norm.weight)
        nn.init.zeros_(self.input_norm.bias)

    def forward(self, x: torch.Tensor, valid_frame_mask: torch.Tensor) -> torch.Tensor:
        batch, frames, nodes, channels = x.shape
        if nodes != NODE_COUNT or channels < 2:
            raise ValueError(f"Expected [B,T,{NODE_COUNT},2+] input, got {tuple(x.shape)}")
        mask = valid_frame_mask.to(dtype=x.dtype)
        joints = x[..., :2] * mask[:, :, None, None]
        bones = torch.zeros_like(joints)
        for parent, child in BONE_PAIRS:
            bones[:, :, child] = joints[:, :, child] - joints[:, :, parent]
        normalized = self.input_norm(bones.permute(0, 2, 3, 1).reshape(batch, NODE_COUNT * 2, frames))
        hidden = normalized.reshape(batch, NODE_COUNT, 2, frames).permute(0, 2, 3, 1)
        keep_prob = self.keep_rate if self.training else 1.0
        for index, block in enumerate(self.blocks):
            block_keep_rate = keep_prob if index >= 6 else 1.0
            hidden = block(hidden, block_keep_rate)
        pooled_mask = functional.interpolate(mask[:, None], size=hidden.shape[2], mode="nearest")
        pooled = (hidden * pooled_mask[:, :, :, None]).sum(dim=2) / pooled_mask.sum(dim=2).clamp_min(1.0)[:, :, None]
        return self.classifier(pooled.mean(dim=2))


Model = SLGCNBoneControlled
