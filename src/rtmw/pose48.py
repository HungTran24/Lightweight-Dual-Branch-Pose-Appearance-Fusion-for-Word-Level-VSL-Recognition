"""Build 48-keypoint anatomical upper-body and hands feature view from raw RTMW shards."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from src.rtmw.diagnostics import (
    COCO_WHOLEBODY_LEFT_HAND_SLICE,
    COCO_WHOLEBODY_RIGHT_HAND_SLICE,
)
from src.rtmw.schema import RawRTMWShard

DEFAULT_POSE48_SEQUENCE_LENGTH: int = 30
POSE48_NODE_COUNT: int = 48
POSE48_CHANNELS: int = 3

# Fixed anatomical order:
# 1. 6 upper-body points: COCO-WholeBody indices [5, 6, 7, 8, 9, 10]
#    (left_shoulder, right_shoulder, left_elbow, right_elbow, left_wrist, right_wrist)
# 2. 21 left-hand points: COCO-WholeBody slice 91:112
# 3. 21 right-hand points: COCO-WholeBody slice 112:133
POSE48_BODY_INDICES: tuple[int, ...] = (5, 6, 7, 8, 9, 10)
POSE48_LEFT_HAND_SLICE: slice = COCO_WHOLEBODY_LEFT_HAND_SLICE
POSE48_RIGHT_HAND_SLICE: slice = COCO_WHOLEBODY_RIGHT_HAND_SLICE

POSE48_KEYPOINT_INDICES: np.ndarray = np.array(
    [
        *POSE48_BODY_INDICES,
        *range(POSE48_LEFT_HAND_SLICE.start, POSE48_LEFT_HAND_SLICE.stop),
        *range(POSE48_RIGHT_HAND_SLICE.start, POSE48_RIGHT_HAND_SLICE.stop),
    ],
    dtype=np.int32,
)


def _hand_edges(offset: int) -> tuple[tuple[int, int], ...]:
    """Return wrist-to-finger-chain edges for one 21-node hand."""
    chains = (
        (0, 1, 2, 3, 4),
        (0, 5, 6, 7, 8),
        (0, 9, 10, 11, 12),
        (0, 13, 14, 15, 16),
        (0, 17, 18, 19, 20),
    )
    return tuple(
        (offset + a, offset + b)
        for chain in chains
        for a, b in zip(chain, chain[1:], strict=False)
    )


# Edges for a future ST-GCN view. Nodes 6 and 27 are hand wrists.
POSE48_EDGES: tuple[tuple[int, int], ...] = (
    (0, 2),
    (2, 4),
    (4, 6),
    (1, 3),
    (3, 5),
    (5, 27),
    (0, 1),
    *_hand_edges(6),
    *_hand_edges(27),
)


class RTMWPose48Error(ValueError):
    """Raised when a pose48 view cannot be built from a raw RTMW shard."""


@dataclass(frozen=True)
class RTMWPose48View:
    """Derived 48-keypoint anatomical feature view from raw RTMW shard."""

    X: np.ndarray
    valid_frame_mask: np.ndarray
    keypoint_valid_mask: np.ndarray
    frame_indices: np.ndarray
    person_score: np.ndarray


class FrameSampler(Protocol):
    """Protocol for temporal frame sampling strategies."""

    def __call__(self, frame_count: int, sequence_length: int) -> np.ndarray:
        """Return an array of sampled frame positions of shape (sequence_length,).

        Args:
            frame_count: Total available frames in the video segment.
            sequence_length: Target number of frames to sample.

        Returns:
            1D array of integer frame positions (0-indexed) or -1 for padding slots.
        """
        ...


def uniform_sample_frame_positions(
    frame_count: int,
    sequence_length: int = DEFAULT_POSE48_SEQUENCE_LENGTH,
) -> np.ndarray:
    """Uniform linear frame sampling matching standard VSL temporal downsampling."""
    if sequence_length <= 0:
        raise RTMWPose48Error(f"sequence_length must be positive, got {sequence_length}")
    if frame_count <= 0:
        return np.full(sequence_length, -1, dtype=np.int32)
    if frame_count >= sequence_length:
        return np.linspace(0, frame_count - 1, sequence_length).round().astype(np.int32)

    positions = np.full(sequence_length, -1, dtype=np.int32)
    positions[:frame_count] = np.arange(frame_count, dtype=np.int32)
    return positions


@dataclass(frozen=True)
class UniformSampler:
    """Callable uniform frame sampler."""

    def __call__(self, frame_count: int, sequence_length: int) -> np.ndarray:
        return uniform_sample_frame_positions(frame_count, sequence_length)


def build_rtmw_pose48_view(
    shard: RawRTMWShard,
    video_id: str,
    *,
    sequence_length: int = DEFAULT_POSE48_SEQUENCE_LENGTH,
    sampler: FrameSampler | Callable[[int, int], np.ndarray] = uniform_sample_frame_positions,
    keypoint_score_threshold: float = 0.0,
    normalization_width: int | None = None,
    normalization_height: int | None = None,
) -> RTMWPose48View:
    """Extract derived 48-keypoint anatomical view from a raw RTMW shard.

    The primary person is selected by highest person_score per frame.
    Coordinates are normalized by frame_width/frame_height and clipped to [0, 1]
    only in the derived view; raw keypoints are preserved unmodified.
    Raw keypoint scores are preserved without clipping in the 3rd channel.
    """
    shard.validate()
    if sequence_length <= 0:
        raise RTMWPose48Error(f"sequence_length must be positive, got {sequence_length}")
    if keypoint_score_threshold < 0.0:
        raise RTMWPose48Error(
            f"keypoint_score_threshold must be non-negative, got {keypoint_score_threshold}"
        )

    video_matches = np.flatnonzero(shard.video_id.astype(str) == str(video_id))
    if video_matches.size != 1:
        raise RTMWPose48Error(f"video_id must identify exactly one shard video: {video_id}")

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
        raise RTMWPose48Error(
            f"normalization dimensions must be positive, got width={width}, height={height}"
        )

    sampled_positions = np.asarray(sampler(frame_count, sequence_length), dtype=np.int32)
    if sampled_positions.shape != (sequence_length,):
        raise RTMWPose48Error(
            f"sampler returned shape {sampled_positions.shape}, expected ({sequence_length},)"
        )

    feature_x = np.zeros((sequence_length, POSE48_NODE_COUNT, POSE48_CHANNELS), dtype=np.float32)
    valid_frame_mask = np.zeros(sequence_length, dtype=np.uint8)
    keypoint_valid_mask = np.zeros((sequence_length, POSE48_NODE_COUNT), dtype=np.uint8)
    frame_indices = np.full(sequence_length, -1, dtype=np.int32)
    person_score = np.zeros(sequence_length, dtype=np.float32)

    for output_pos, sampled_pos in enumerate(sampled_positions):
        if sampled_pos < 0:
            continue
        if sampled_pos >= frame_count:
            raise RTMWPose48Error(
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

        # Extract 48 keypoints in fixed anatomical order: body (6), left hand (21), right hand (21)
        raw_kpts = shard.keypoints[primary_person, POSE48_KEYPOINT_INDICES]

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

    return RTMWPose48View(
        X=feature_x,
        valid_frame_mask=valid_frame_mask,
        keypoint_valid_mask=keypoint_valid_mask,
        frame_indices=frame_indices,
        person_score=person_score,
    )
