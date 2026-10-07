"""Tests for Pose42, Pose53 view definitions and pipeline build_split."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.pose_views import pose42, pose53
from src.pose_views.pipeline import _load_view, build_split
from src.rtmw.diagnostics import (
    COCO_WHOLEBODY_LEFT_HAND_SLICE,
    COCO_WHOLEBODY_RIGHT_HAND_SLICE,
)
from src.rtmw.pose48 import POSE48_KEYPOINT_INDICES
from src.rtmw.schema import RawRTMWShard


def _create_synthetic_shard(video_ids: tuple[str, ...] = ("000100", "000101")) -> RawRTMWShard:
    """Create a minimal synthetic RawRTMWShard with valid schema."""
    video_count = len(video_ids)
    frames_per_video = 3
    total_frames = video_count * frames_per_video

    video_frame_offsets = np.arange(0, total_frames + 1, frames_per_video, dtype=np.int64)
    frame_person_offsets = np.arange(0, total_frames + 1, dtype=np.int64)  # 1 person per frame

    keypoints = np.zeros((total_frames, 133, 3), dtype=np.float32)
    for i in range(total_frames):
        keypoints[i, :, 0] = np.linspace(10.0, 500.0, 133)
        keypoints[i, :, 1] = np.linspace(20.0, 600.0, 133)
        keypoints[i, :, 2] = 0.9

    return RawRTMWShard(
        video_id=np.array(list(video_ids), dtype="U6"),
        video_frame_offsets=video_frame_offsets,
        frame_idx=np.tile(np.arange(frames_per_video, dtype=np.int32), video_count),
        timestamp_ms=np.tile(np.arange(frames_per_video, dtype=np.int64) * 33, video_count),
        frame_width=np.full(video_count, 1080, dtype=np.int32),
        frame_height=np.full(video_count, 1080, dtype=np.int32),
        frame_person_offsets=frame_person_offsets,
        person_bbox_xyxy=np.tile(
            np.array([[50.0, 50.0, 800.0, 900.0]], dtype=np.float32),
            (total_frames, 1),
        ),
        person_score=np.full(total_frames, 0.95, dtype=np.float32),
        keypoints=keypoints,
    )


def test_pose42_view_definition() -> None:
    assert pose42.VIEW_NAME == "pose42"
    assert pose42.NODE_COUNT == 42
    expected_indices = np.array(
        [
            *range(COCO_WHOLEBODY_LEFT_HAND_SLICE.start, COCO_WHOLEBODY_LEFT_HAND_SLICE.stop),
            *range(COCO_WHOLEBODY_RIGHT_HAND_SLICE.start, COCO_WHOLEBODY_RIGHT_HAND_SLICE.stop),
        ],
        dtype=np.int32,
    )
    np.testing.assert_array_equal(pose42.KEYPOINT_INDICES, expected_indices)
    assert len(pose42.EDGES) == 40
    assert callable(pose42.BUILD_VIEW)


def test_pose53_view_definition() -> None:
    assert pose53.VIEW_NAME == "pose53"
    assert pose53.NODE_COUNT == 53
    expected_indices = np.array([*POSE48_KEYPOINT_INDICES, 0, 1, 2, 3, 4], dtype=np.int32)
    np.testing.assert_array_equal(pose53.KEYPOINT_INDICES, expected_indices)
    assert len(pose53.EDGES) == 53  # 47 Pose48 edges + 6 face edges
    assert callable(pose53.BUILD_VIEW)


def test_load_view() -> None:
    assert _load_view("pose42") is pose42
    assert _load_view("pose53") is pose53
    with pytest.raises(ValueError, match="Unsupported view"):
        _load_view("pose99")


@pytest.mark.parametrize("view_module,node_count", [(pose42, 42), (pose53, 53)])
def test_build_split_shapes_and_types(view_module: object, node_count: int) -> None:
    shard = _create_synthetic_shard(("000100", "000101"))
    records = [
        {"video_id": "000100", "signer_id": "001", "class_id": "10"},
        {"video_id": "000101", "signer_id": "002", "class_id": "25"},
    ]
    shards = {
        "000100": shard,
        "000101": shard,
    }

    frames = 16
    threshold = 0.1
    split = build_split(records, shards, view_module, frames=frames, threshold=threshold)

    assert split["X"].shape == (2, frames, node_count, 3)
    assert split["X"].dtype == np.float32
    assert split["y"].shape == (2,)
    assert split["y"].dtype == np.int64
    np.testing.assert_array_equal(split["y"], np.array([10, 25], dtype=np.int64))

    assert split["video_id"].shape == (2,)
    np.testing.assert_array_equal(split["video_id"], np.array(["000100", "000101"]))

    assert split["signer_id"].shape == (2,)
    np.testing.assert_array_equal(split["signer_id"], np.array(["001", "002"]))

    assert split["frame_indices"].shape == (2, frames)
    assert split["frame_indices"].dtype == np.int32

    assert split["valid_frame_mask"].shape == (2, frames)
    assert split["valid_frame_mask"].dtype == np.uint8

    assert split["keypoint_valid_mask"].shape == (2, frames, node_count)
    assert split["keypoint_valid_mask"].dtype == np.uint8


def test_build_split_from_npz_path(tmp_path: Path) -> None:
    shard = _create_synthetic_shard(("000100",))
    shard_path = tmp_path / "shard_000100.npz"
    shard.save_npz(shard_path)

    records = [{"video_id": "000100", "signer_id": "001", "class_id": "5"}]
    shards = {"000100": shard_path}

    split = build_split(records, shards, pose42, frames=8, threshold=0.0)
    assert split["X"].shape == (1, 8, 42, 3)
    assert split["y"].tolist() == [5]
    assert split["video_id"].tolist() == ["000100"]
    assert split["signer_id"].tolist() == ["001"]
