"""Canonical all-frame Pose59 layout and validator."""

from __future__ import annotations

import numpy as np

POSE59_KEYPOINTS = 59
POSE59_CENTER_NODES = (5, 6)


def _hand_edges(offset: int) -> tuple[tuple[int, int], ...]:
    chains = (
        (0, 1, 2, 3, 4),
        (0, 5, 6, 7, 8),
        (0, 9, 10, 11, 12),
        (0, 13, 14, 15, 16),
        (0, 17, 18, 19, 20),
    )
    return tuple(
        (offset + source, offset + target)
        for chain in chains
        for source, target in zip(chain, chain[1:], strict=False)
    )


POSE59_EDGES = (
    # COCO-17 person topology.
    (15, 13),
    (13, 11),
    (16, 14),
    (14, 12),
    (11, 12),
    (5, 11),
    (6, 12),
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (1, 2),
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (3, 5),
    (4, 6),
    # Left/right hand chains and person-wrist bridges.
    *_hand_edges(17),
    *_hand_edges(38),
    (9, 17),
    (10, 38),
)


POSE59_FIELDS = frozenset(
    {
        "X",
        "keypoint_valid_mask",
        "object_confidence",
        "frame_idx",
        "video_id",
        "split",
        "frame_width",
        "frame_height",
        "video_frame_offsets",
    }
)


def validate_pose59_schema(arrays: dict[str, np.ndarray]) -> None:
    if set(arrays) != POSE59_FIELDS:
        raise ValueError(f"Pose59 fields mismatch: {sorted(set(arrays) ^ POSE59_FIELDS)}")
    frames = len(arrays["X"])
    for name, shape, dtype in (
        ("X", (frames, POSE59_KEYPOINTS, 3), np.float32),
        ("keypoint_valid_mask", (frames, POSE59_KEYPOINTS), np.uint8),
        ("object_confidence", (frames, 3), np.float32),
        ("frame_idx", (frames,), np.int32),
    ):
        if arrays[name].shape != shape or arrays[name].dtype != np.dtype(dtype):
            raise ValueError(f"{name} must have shape {shape} and dtype {np.dtype(dtype)}")
    videos = len(arrays["video_id"])
    for name, dtype in (("video_id", "U6"), ("split", "U5")):
        if arrays[name].shape != (videos,) or arrays[name].dtype != np.dtype(dtype):
            raise ValueError(f"{name} must have shape ({videos},) and dtype {dtype}")
    for name in ("frame_width", "frame_height"):
        if arrays[name].shape != (videos,) or arrays[name].dtype != np.int32:
            raise ValueError(f"{name} must have shape ({videos},) and dtype int32")
    offsets = arrays["video_frame_offsets"]
    if (
        offsets.shape != (videos + 1,)
        or offsets.dtype != np.int64
        or offsets[0] != 0
        or offsets[-1] != frames
        or np.any(np.diff(offsets) <= 0)
    ):
        raise ValueError("video_frame_offsets must be strict positive prefix sums")
    if (
        not set(map(str, arrays["split"])).issubset({"train", "val", "test"})
        or np.any(arrays["frame_width"] <= 0)
        or np.any(arrays["frame_height"] <= 0)
    ):
        raise ValueError("Pose59 metadata is invalid")
    if (
        not np.isfinite(arrays["X"]).all()
        or not np.isfinite(arrays["object_confidence"]).all()
        or np.any(arrays["X"] < 0)
        or np.any(arrays["X"] > 1)
        or np.any(arrays["object_confidence"] < 0)
        or np.any(arrays["object_confidence"] > 1)
    ):
        raise ValueError("Pose59 values must be finite and normalized")
    if not np.isin(arrays["keypoint_valid_mask"], (0, 1)).all():
        raise ValueError("Invalid keypoint mask")
    for start, stop in zip(offsets[:-1], offsets[1:], strict=True):
        frame_idx = arrays["frame_idx"][start:stop]
        if not np.array_equal(frame_idx, np.arange(stop - start, dtype=np.int32)):
            raise ValueError("frame_idx must restart at zero for every video")
        invalid = arrays["keypoint_valid_mask"][start:stop] == 0
        if np.any(arrays["X"][start:stop][invalid] != 0):
            raise ValueError("Invalid keypoints must be zero-filled")
