"""Tests for ST-GCN model on canonical Pose42, Pose53, and Pose59 graphs."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.models.stgcn import STGCN, build_spatial_adjacency
from src.models.tcn import TCN
from src.pose_views.pose59 import POSE59_CENTER_NODES, POSE59_EDGES
from src.rtmw.pose42 import POSE42_EDGES, POSE42_NODE_COUNT
from src.rtmw.pose53 import POSE53_EDGES, POSE53_NODE_COUNT


def test_build_spatial_adjacency_pose42_partitions() -> None:
    partitions = build_spatial_adjacency(
        node_count=POSE42_NODE_COUNT,
        edges=POSE42_EDGES,
        center_nodes=(0, 21),
    )
    assert partitions.shape == (3, 42, 42)
    # Partition 0: Self-connection (diagonal non-empty)
    assert np.all(np.diag(partitions[0]) > 0.0)
    # Partitions 1 and 2: Inward and outward edges are non-empty
    assert np.count_nonzero(partitions[1]) > 0
    assert np.count_nonzero(partitions[2]) > 0
    # Every edge in POSE42_EDGES must be represented
    for src, tgt in POSE42_EDGES:
        assert partitions[:, src, tgt].sum() > 0.0
        assert partitions[:, tgt, src].sum() > 0.0


def test_build_spatial_adjacency_pose53_partitions() -> None:
    partitions = build_spatial_adjacency(
        node_count=POSE53_NODE_COUNT,
        edges=POSE53_EDGES,
        center_nodes=(0, 1),
    )
    assert partitions.shape == (3, 53, 53)
    assert np.all(np.diag(partitions[0]) > 0.0)
    assert np.count_nonzero(partitions[1]) > 0
    assert np.count_nonzero(partitions[2]) > 0
    for src, tgt in POSE53_EDGES:
        assert partitions[:, src, tgt].sum() > 0.0
        assert partitions[:, tgt, src].sum() > 0.0


def test_build_spatial_adjacency_pose59_topology() -> None:
    assert len(POSE59_EDGES) == 61
    assert len(set(POSE59_EDGES)) == 61
    assert all(0 <= node < 59 for edge in POSE59_EDGES for node in edge)
    assert (9, 17) in POSE59_EDGES
    assert (10, 38) in POSE59_EDGES

    neighbors = {node: set() for node in range(59)}
    for source, target in POSE59_EDGES:
        neighbors[source].add(target)
        neighbors[target].add(source)
    visited = set(POSE59_CENTER_NODES)
    queue = list(POSE59_CENTER_NODES)
    while queue:
        node = queue.pop()
        for neighbor in neighbors[node] - visited:
            visited.add(neighbor)
            queue.append(neighbor)
    assert visited == set(range(59))

    partitions = build_spatial_adjacency(
        node_count=59,
        edges=POSE59_EDGES,
        center_nodes=POSE59_CENTER_NODES,
    )
    assert partitions.shape == (3, 59, 59)
    assert np.isfinite(partitions).all()


@pytest.mark.parametrize("nodes", [42, 53, 59])
@pytest.mark.parametrize("frames", [16, 24, 30, 60])
def test_stgcn_forward_nodes_and_frames(nodes: int, frames: int) -> None:
    batch_size = 2
    num_classes = 10
    config = {"input_channels": 2, "dropout": 0.1, "temporal_kernel": 5}
    model = STGCN(nodes=nodes, classes=num_classes, config=config)
    model.eval()

    x = torch.randn(batch_size, frames, nodes, 3)
    valid_frame_mask = torch.ones(batch_size, frames, dtype=torch.uint8)

    with torch.no_grad():
        out = model(x, valid_frame_mask)

    assert out.shape == (batch_size, num_classes)
    assert torch.isfinite(out).all()


def test_tcn_forward_uniform60() -> None:
    model = TCN(
        nodes=53,
        classes=10,
        config={
            "input_channels": 2,
            "channels": [16, 16, 16],
            "dilations": [1, 2, 4],
            "kernel_size": 5,
            "dropout": 0.1,
        },
    )
    model.eval()
    with torch.no_grad():
        output = model(torch.randn(2, 60, 53, 3), torch.ones(2, 60, dtype=torch.uint8))
    assert output.shape == (2, 10)
    assert torch.isfinite(output).all()


@pytest.mark.parametrize("nodes", [42, 53, 59])
def test_stgcn_forward_empty_and_partial_mask(nodes: int) -> None:
    batch_size = 2
    frames = 16
    num_classes = 5
    model = STGCN(nodes=nodes, classes=num_classes, config={"input_channels": 2})
    model.eval()

    x = torch.randn(batch_size, frames, nodes, 3)

    # Empty mask (all zeros - no valid frames)
    empty_mask = torch.zeros(batch_size, frames, dtype=torch.uint8)
    with torch.no_grad():
        out_empty = model(x, empty_mask)
    assert out_empty.shape == (batch_size, num_classes)
    assert torch.isfinite(out_empty).all()

    # Partial mask
    partial_mask = torch.ones(batch_size, frames, dtype=torch.uint8)
    partial_mask[:, frames // 2 :] = 0
    with torch.no_grad():
        out_partial = model(x, partial_mask)
    assert out_partial.shape == (batch_size, num_classes)
    assert torch.isfinite(out_partial).all()


def test_stgcn_unsupported_node_count() -> None:
    with pytest.raises(ValueError, match="No default edges for nodes=99"):
        STGCN(nodes=99, classes=10, config={})
