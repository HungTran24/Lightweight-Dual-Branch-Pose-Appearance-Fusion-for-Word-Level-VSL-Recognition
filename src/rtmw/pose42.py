"""Build 42-keypoint anatomical hands-only feature view from raw RTMW shards."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from src.rtmw.diagnostics import (
    COCO_WHOLEBODY_LEFT_HAND_SLICE,
    COCO_WHOLEBODY_RIGHT_HAND_SLICE,
)
from src.rtmw.pose48 import (
    POSE48_CHANNELS,
    FrameSampler,
    _hand_edges,
    uniform_sample_frame_positions,
)
from src.rtmw.schema import RawRTMWShard

DEFAULT_POSE42_SEQUENCE_LENGTH: int = 30
POSE42_NODE_COUNT: int = 42
POSE42_CHANNELS: int = POSE48_CHANNELS  # 3 (x, y, score)

POSE42_LEFT_HAND_SLICE: slice = COCO_WHOLEBODY_LEFT_HAND_SLICE
POSE42_RIGHT_HAND_SLICE: slice = COCO_WHOLEBODY_RIGHT_HAND_SLICE

# Layout:
# 0..20: 21 left-hand points (COCO-WholeBody 91:112)
# 21..41: 21 right-hand points (COCO-WholeBody 112:133)
POSE42_KEYPOINT_INDICES: np.ndarray = np.array(
    [
        *range(POSE42_LEFT_HAND_SLICE.start, POSE42_LEFT_HAND_SLICE.stop),
        *range(POSE42_RIGHT_HAND_SLICE.start, POSE42_RIGHT_HAND_SLICE.stop),
    ],
    dtype=np.int32,
)

# Pose42 Graph Topology for ST-GCN:
# Left hand (nodes 0..20) with root at node 0
# Right hand (nodes 21..41) with root at node 21
POSE42_EDGES: tuple[tuple[int, int], ...] = (
    *_hand_edges(0),
    *_hand_edges(21),
)


class RTMWPose42Error(ValueError):
    """Raised when a pose42 view cannot be built from a raw RTMW shard."""


@dataclass(frozen=True)
class RTMWPose42View:
    """Derived 42-keypoint anatomical hands feature view from raw RTMW shard."""

    X: np.ndarray
    valid_frame_mask: np.ndarray
    keypoint_valid_mask: np.ndarray
    frame_indices: np.ndarray
    person_score: np.ndarray


def build_rtmw_pose42_view(
    shard: RawRTMWShard,
    video_id: str,
    *,
    sequence_length: int = DEFAULT_POSE42_SEQUENCE_LENGTH,
    sampler: FrameSampler | Callable[[int, int], np.ndarray] = uniform_sample_frame_positions,
    keypoint_score_threshold: float = 0.0,
    normalization_width: int | None = None,
    normalization_height: int | None = None,
) -> RTMWPose42View:
    """Extract derived 42-keypoint hands view from a raw RTMW shard.

    The primary person is selected by highest person_score per frame.
    Coordinates are normalized by frame_width/frame_height and clipped to [0, 1]
    only in the derived view; raw keypoints are preserved unmodified.
    Raw keypoint scores are preserved without clipping in the 3rd channel.
    """
    shard.validate()
    if sequence_length <= 0:
        raise RTMWPose42Error(f"sequence_length must be positive, got {sequence_length}")
    if keypoint_score_threshold < 0.0:
        raise RTMWPose42Error(
            f"keypoint_score_threshold must be non-negative, got {keypoint_score_threshold}"
        )

    video_matches = np.flatnonzero(shard.video_id.astype(str) == str(video_id))
    if video_matches.size != 1:
        raise RTMWPose42Error(f"video_id must identify exactly one shard video: {video_id}")

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
        raise RTMWPose42Error(
            f"normalization dimensions must be positive, got width={width}, height={height}"
        )

    sampled_positions = np.asarray(sampler(frame_count, sequence_length), dtype=np.int32)
    if sampled_positions.shape != (sequence_length,):
        raise RTMWPose42Error(
            f"sampler returned shape {sampled_positions.shape}, expected ({sequence_length},)"
        )

    feature_x = np.zeros((sequence_length, POSE42_NODE_COUNT, POSE42_CHANNELS), dtype=np.float32)
    valid_frame_mask = np.zeros(sequence_length, dtype=np.uint8)
    keypoint_valid_mask = np.zeros((sequence_length, POSE42_NODE_COUNT), dtype=np.uint8)
    frame_indices = np.full(sequence_length, -1, dtype=np.int32)
    person_score = np.zeros(sequence_length, dtype=np.float32)

    for output_pos, sampled_pos in enumerate(sampled_positions):
        if sampled_pos < 0:
            continue
        if sampled_pos >= frame_count:
            raise RTMWPose42Error(
                f"sampled position {sampled_pos} out of range for frame_count {frame_count}"
            )

        frame_pos = frame_start + int(sampled_pos)
        valid_frame_mask[output_pos] = 1
        frame_indices[output_pos] = shard.frame_idx[frame_pos]

        person_start = int(shard.frame_person_offsets[frame_pos])
        person_stop = int(shard.frame_person_offsets[frame_pos + 1])
        if person_start == person_stop:
            continue

        primary_person = person_start + int(
            np.argmax(shard.person_score[person_start:person_stop])
        )
        person_score[output_pos] = float(shard.person_score[primary_person])

        raw_kpts = shard.keypoints[primary_person, POSE42_KEYPOINT_INDICES]

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

    return RTMWPose42View(
        X=feature_x,
        valid_frame_mask=valid_frame_mask,
        keypoint_valid_mask=keypoint_valid_mask,
        frame_indices=frame_indices,
        person_score=person_score,
    )
