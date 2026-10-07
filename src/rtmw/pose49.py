"""Build 49-keypoint anatomical upper-body, hands, and nose feature view from raw RTMW shards."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from src.rtmw.pose48 import (
    POSE48_CHANNELS,
    POSE48_EDGES,
    POSE48_KEYPOINT_INDICES,
    FrameSampler,
    uniform_sample_frame_positions,
)
from src.rtmw.schema import RawRTMWShard

DEFAULT_POSE49_SEQUENCE_LENGTH: int = 30
POSE49_NODE_COUNT: int = 49
POSE49_CHANNELS: int = POSE48_CHANNELS  # 3 (x, y, score)

# Pose49 adds exactly 1 keypoint: Nose (COCO-WholeBody 0)
POSE49_NOSE_INDEX: int = 0

# Layout:
# 0..47: Exactly identical to Pose48 (6 body + 21 left hand + 21 right hand)
# 48: nose (COCO 0)
POSE49_KEYPOINT_INDICES: np.ndarray = np.array(
    [
        *POSE48_KEYPOINT_INDICES,
        POSE49_NOSE_INDEX,
    ],
    dtype=np.int32,
)

# Pose49 Graph Topology for ST-GCN:
# Preserves all 47 Pose48 edges.
# Adds 2 edges connecting nose to shoulders:
# - nose (node 48) to left_shoulder (node 0)
# - nose (node 48) to right_shoulder (node 1)
POSE49_EDGES: tuple[tuple[int, int], ...] = (
    *POSE48_EDGES,
    (48, 0),  # nose - left_shoulder
    (48, 1),  # nose - right_shoulder
)


class RTMWPose49Error(ValueError):
    """Raised when a pose49 view cannot be built from a raw RTMW shard."""


@dataclass(frozen=True)
class RTMWPose49View:
    """Derived 49-keypoint anatomical feature view from raw RTMW shard."""

    X: np.ndarray
    valid_frame_mask: np.ndarray
    keypoint_valid_mask: np.ndarray
    frame_indices: np.ndarray
    person_score: np.ndarray


def build_rtmw_pose49_view(
    shard: RawRTMWShard,
    video_id: str,
    *,
    sequence_length: int = DEFAULT_POSE49_SEQUENCE_LENGTH,
    sampler: FrameSampler | Callable[[int, int], np.ndarray] = uniform_sample_frame_positions,
    keypoint_score_threshold: float = 0.0,
    normalization_width: int | None = None,
    normalization_height: int | None = None,
) -> RTMWPose49View:
    """Extract derived 49-keypoint anatomical view from a raw RTMW shard.

    The primary person is selected by highest person_score per frame.
    Coordinates are normalized by frame_width/frame_height and clipped to [0, 1]
    only in the derived view; raw keypoints are preserved unmodified.
    Raw keypoint scores are preserved without clipping in the 3rd channel.
    """
    shard.validate()
    if sequence_length <= 0:
        raise RTMWPose49Error(f"sequence_length must be positive, got {sequence_length}")
    if keypoint_score_threshold < 0.0:
        raise RTMWPose49Error(
            f"keypoint_score_threshold must be non-negative, got {keypoint_score_threshold}"
        )

    video_matches = np.flatnonzero(shard.video_id.astype(str) == str(video_id))
    if video_matches.size != 1:
        raise RTMWPose49Error(f"video_id must identify exactly one shard video: {video_id}")

    video_position = int(video_matches[0])
    frame_start = int(shard.video_frame_offsets[video_position])
    frame_stop = int(shard.video_frame_offsets[video_position + 1])
    frame_count = frame_stop - frame_start

    width = float(
        normalization_width
        if normalization_width is not None
        else shard.frame_width[video_position]
    )
    height = float(
        normalization_height
        if normalization_height is not None
        else shard.frame_height[video_position]
    )
    if width <= 0.0 or height <= 0.0:
        raise RTMWPose49Error(
            f"normalization dimensions must be positive, got width={width}, height={height}"
        )

    sampled_positions = np.asarray(sampler(frame_count, sequence_length), dtype=np.int32)
    if sampled_positions.shape != (sequence_length,):
        raise RTMWPose49Error(
            f"sampler returned shape {sampled_positions.shape}, expected ({sequence_length},)"
        )

    feature_x = np.zeros((sequence_length, POSE49_NODE_COUNT, POSE49_CHANNELS), dtype=np.float32)
    valid_frame_mask = np.zeros(sequence_length, dtype=np.uint8)
    keypoint_valid_mask = np.zeros((sequence_length, POSE49_NODE_COUNT), dtype=np.uint8)
    frame_indices = np.full(sequence_length, -1, dtype=np.int32)
    person_score = np.zeros(sequence_length, dtype=np.float32)

    for output_pos, sampled_pos in enumerate(sampled_positions):
        if sampled_pos < 0:
            continue
        if sampled_pos >= frame_count:
            raise RTMWPose49Error(
                f"sampled position {sampled_pos} out of range for frame_count {frame_count}"
            )

        frame_pos = frame_start + int(sampled_pos)
        valid_frame_mask[output_pos] = 1
        frame_indices[output_pos] = shard.frame_idx[frame_pos]

        person_start = int(shard.frame_person_offsets[frame_pos])
        person_stop = int(shard.frame_person_offsets[frame_pos + 1])
        if person_start == person_stop:
            # Zero-filled missing person frame
            continue

        primary_person = person_start + int(
            np.argmax(shard.person_score[person_start:person_stop])
        )
        p_score = float(shard.person_score[primary_person])
        person_score[output_pos] = p_score

        # Extract 49 keypoints: body (6), left hand (21), right hand (21), nose (1)
        raw_kpts = shard.keypoints[primary_person, POSE49_KEYPOINT_INDICES]

        # Normalized xy clipped to [0, 1] in derived view
        x_norm = np.clip(raw_kpts[:, 0] / width, 0.0, 1.0).astype(np.float32)
        y_norm = np.clip(raw_kpts[:, 1] / height, 0.0, 1.0).astype(np.float32)
        scores = raw_kpts[:, 2].astype(np.float32)

        feature_x[output_pos, :, 0] = x_norm
        feature_x[output_pos, :, 1] = y_norm
        feature_x[output_pos, :, 2] = scores

        if keypoint_score_threshold > 0.0:
            keypoint_valid_mask[output_pos] = (scores >= keypoint_score_threshold).astype(np.uint8)
        else:
            keypoint_valid_mask[output_pos] = (scores > 0.0).astype(np.uint8)

    return RTMWPose49View(
        X=feature_x,
        valid_frame_mask=valid_frame_mask,
        keypoint_valid_mask=keypoint_valid_mask,
        frame_indices=frame_indices,
        person_score=person_score,
    )
