"""RTMW quality and kinematic diagnostics module.

This module provides lightweight extraction diagnostics without modifying raw
prediction tensors or altering person instance selection/packing:
  - Left / right hand availability rates evaluated via raw uncalibrated score threshold.
    Note: Raw keypoint scores are uncalibrated model outputs, not calibrated confidence probabilities.
  - Normalized frame-to-frame hand center displacement (heuristic velocity normalized by inter-shoulder scale).
  - Second-order temporal hand jitter (heuristic step-to-step velocity change).
  - Out-of-frame keypoint coordinate detection rate.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

# COCO-WholeBody (133 keypoints) index mappings
COCO_WHOLEBODY_LEFT_SHOULDER_IDX: int = 5
COCO_WHOLEBODY_RIGHT_SHOULDER_IDX: int = 6
COCO_WHOLEBODY_LEFT_HAND_SLICE: slice = slice(91, 112)  # 21 keypoints [91..111]
COCO_WHOLEBODY_RIGHT_HAND_SLICE: slice = slice(112, 133)  # 21 keypoints [112..132]


@dataclass(frozen=True)
class HandFrameState:
    """Per-frame diagnostic state for a single person instance."""

    has_person: bool
    left_hand_available: bool
    right_hand_available: bool
    left_hand_score: float
    right_hand_score: float
    left_hand_center: tuple[float, float] | None
    right_hand_center: tuple[float, float] | None
    norm_scale: float
    out_of_frame_kpt_count: int
    total_kpt_count: int


@dataclass(frozen=True)
class RTMWShardDiagnostics:
    """Aggregated quality and kinematic diagnostics across processed videos."""

    total_frames_evaluated: int
    left_hand_availability_rate: float
    right_hand_availability_rate: float
    any_hand_availability_rate: float
    both_hands_availability_rate: float
    left_hand_mean_displacement: float
    right_hand_mean_displacement: float
    left_hand_mean_jitter: float
    right_hand_mean_jitter: float
    out_of_frame_keypoint_rate: float

    def to_metrics_dict(self) -> dict[str, float]:
        """Return formatted metrics dictionary suitable for shard telemetry."""
        return {
            "left_hand_availability_rate": round(self.left_hand_availability_rate, 4),
            "right_hand_availability_rate": round(self.right_hand_availability_rate, 4),
            "any_hand_availability_rate": round(self.any_hand_availability_rate, 4),
            "both_hands_availability_rate": round(self.both_hands_availability_rate, 4),
            "left_hand_mean_displacement": round(self.left_hand_mean_displacement, 4),
            "right_hand_mean_displacement": round(self.right_hand_mean_displacement, 4),
            "left_hand_mean_jitter": round(self.left_hand_mean_jitter, 4),
            "right_hand_mean_jitter": round(self.right_hand_mean_jitter, 4),
            "out_of_frame_keypoint_rate": round(self.out_of_frame_keypoint_rate, 4),
        }


def extract_frame_hand_state(
    keypoints_3d: np.ndarray | None,
    frame_width: int,
    frame_height: int,
    score_thr: float = 0.3,
) -> HandFrameState:
    """Extract hand availability, hand center, scale, and out-of-frame count for a single frame.

    Args:
        keypoints_3d: Keypoint array for the primary person instance, shape (133, 3).
                      If no person exists, pass an empty array of shape (0, 133, 3) or None.
        frame_width: Video frame width in pixels.
        frame_height: Video frame height in pixels.
        score_thr: Confidence threshold to consider a hand detected/available.

    Returns:
        HandFrameState instance.
    """
    if keypoints_3d is None or keypoints_3d.size == 0:
        return HandFrameState(
            has_person=False,
            left_hand_available=False,
            right_hand_available=False,
            left_hand_score=0.0,
            right_hand_score=0.0,
            left_hand_center=None,
            right_hand_center=None,
            norm_scale=float(max(frame_width, frame_height, 1)),
            out_of_frame_kpt_count=0,
            total_kpt_count=0,
        )

    kpts = keypoints_3d[0] if keypoints_3d.ndim == 3 else keypoints_3d

    # Normalization scale: inter-shoulder distance if both shoulders confident, else frame diagonal
    norm_scale = float(max(frame_width, frame_height, 1))
    ls_score = float(kpts[COCO_WHOLEBODY_LEFT_SHOULDER_IDX, 2])
    rs_score = float(kpts[COCO_WHOLEBODY_RIGHT_SHOULDER_IDX, 2])
    if ls_score >= score_thr and rs_score >= score_thr:
        ls_xy = kpts[COCO_WHOLEBODY_LEFT_SHOULDER_IDX, :2]
        rs_xy = kpts[COCO_WHOLEBODY_RIGHT_SHOULDER_IDX, :2]
        shoulder_dist = float(np.linalg.norm(ls_xy - rs_xy))
        if shoulder_dist > 5.0:
            norm_scale = shoulder_dist

    # Left hand keypoints
    lh_kpts = kpts[COCO_WHOLEBODY_LEFT_HAND_SLICE]
    lh_mean_score = float(np.mean(lh_kpts[:, 2]))
    lh_available = bool(lh_mean_score >= score_thr)
    lh_center: tuple[float, float] | None = None
    if lh_available:
        mean_xy = np.mean(lh_kpts[:, :2], axis=0)
        lh_center = (float(mean_xy[0]), float(mean_xy[1]))

    # Right hand keypoints
    rh_kpts = kpts[COCO_WHOLEBODY_RIGHT_HAND_SLICE]
    rh_mean_score = float(np.mean(rh_kpts[:, 2]))
    rh_available = bool(rh_mean_score >= score_thr)
    rh_center: tuple[float, float] | None = None
    if rh_available:
        mean_xy = np.mean(rh_kpts[:, :2], axis=0)
        rh_center = (float(mean_xy[0]), float(mean_xy[1]))

    # Out of frame keypoint coordinates count
    xs = kpts[:, 0]
    ys = kpts[:, 1]
    out_of_frame = (
        (xs < 0.0) | (xs >= float(frame_width)) | (ys < 0.0) | (ys >= float(frame_height))
    )
    out_count = int(np.sum(out_of_frame))

    return HandFrameState(
        has_person=True,
        left_hand_available=lh_available,
        right_hand_available=rh_available,
        left_hand_score=lh_mean_score,
        right_hand_score=rh_mean_score,
        left_hand_center=lh_center,
        right_hand_center=rh_center,
        norm_scale=norm_scale,
        out_of_frame_kpt_count=out_count,
        total_kpt_count=len(kpts),
    )


def compute_shard_hand_diagnostics(
    video_frame_states: Sequence[Sequence[HandFrameState]],
) -> RTMWShardDiagnostics:
    """Compute aggregated hand availability, displacement, jitter, and out-of-frame metrics across videos.

    Args:
        video_frame_states: Sequence of per-video frame states.

    Returns:
        RTMWShardDiagnostics object.
    """
    total_frames = sum(len(v) for v in video_frame_states)
    if total_frames == 0:
        return RTMWShardDiagnostics(
            total_frames_evaluated=0,
            left_hand_availability_rate=0.0,
            right_hand_availability_rate=0.0,
            any_hand_availability_rate=0.0,
            both_hands_availability_rate=0.0,
            left_hand_mean_displacement=0.0,
            right_hand_mean_displacement=0.0,
            left_hand_mean_jitter=0.0,
            right_hand_mean_jitter=0.0,
            out_of_frame_keypoint_rate=0.0,
        )

    lh_avail_count = 0
    rh_avail_count = 0
    any_avail_count = 0
    both_avail_count = 0
    total_out_kpts = 0
    total_evaluated_kpts = 0

    all_lh_displacements: list[float] = []
    all_rh_displacements: list[float] = []
    all_lh_jitters: list[float] = []
    all_rh_jitters: list[float] = []

    for video_states in video_frame_states:
        video_lh_displacements: list[float] = []
        video_rh_displacements: list[float] = []

        for idx, state in enumerate(video_states):
            if state.left_hand_available:
                lh_avail_count += 1
            if state.right_hand_available:
                rh_avail_count += 1
            if state.left_hand_available or state.right_hand_available:
                any_avail_count += 1
            if state.left_hand_available and state.right_hand_available:
                both_avail_count += 1

            total_out_kpts += state.out_of_frame_kpt_count
            total_evaluated_kpts += state.total_kpt_count

            # Calculate consecutive frame-to-frame displacement if both consecutive frames have hand center
            if idx > 0:
                prev_state = video_states[idx - 1]

                # Left hand displacement
                if state.left_hand_center is not None and prev_state.left_hand_center is not None:
                    scale = (state.norm_scale + prev_state.norm_scale) / 2.0
                    dx = state.left_hand_center[0] - prev_state.left_hand_center[0]
                    dy = state.left_hand_center[1] - prev_state.left_hand_center[1]
                    disp = float(np.hypot(dx, dy) / scale)
                    video_lh_displacements.append(disp)
                    all_lh_displacements.append(disp)

                # Right hand displacement
                if state.right_hand_center is not None and prev_state.right_hand_center is not None:
                    scale = (state.norm_scale + prev_state.norm_scale) / 2.0
                    dx = state.right_hand_center[0] - prev_state.right_hand_center[0]
                    dy = state.right_hand_center[1] - prev_state.right_hand_center[1]
                    disp = float(np.hypot(dx, dy) / scale)
                    video_rh_displacements.append(disp)
                    all_rh_displacements.append(disp)

        # Jitter: second-order difference (absolute change in step-to-step velocity)
        for i in range(len(video_lh_displacements) - 1):
            jitter = abs(video_lh_displacements[i + 1] - video_lh_displacements[i])
            all_lh_jitters.append(jitter)

        for i in range(len(video_rh_displacements) - 1):
            jitter = abs(video_rh_displacements[i + 1] - video_rh_displacements[i])
            all_rh_jitters.append(jitter)

    lh_rate = lh_avail_count / total_frames
    rh_rate = rh_avail_count / total_frames
    any_rate = any_avail_count / total_frames
    both_rate = both_avail_count / total_frames

    lh_mean_disp = float(np.mean(all_lh_displacements)) if all_lh_displacements else 0.0
    rh_mean_disp = float(np.mean(all_rh_displacements)) if all_rh_displacements else 0.0
    lh_mean_jit = float(np.mean(all_lh_jitters)) if all_lh_jitters else 0.0
    rh_mean_jit = float(np.mean(all_rh_jitters)) if all_rh_jitters else 0.0

    out_rate = total_out_kpts / total_evaluated_kpts if total_evaluated_kpts > 0 else 0.0

    return RTMWShardDiagnostics(
        total_frames_evaluated=total_frames,
        left_hand_availability_rate=lh_rate,
        right_hand_availability_rate=rh_rate,
        any_hand_availability_rate=any_rate,
        both_hands_availability_rate=both_rate,
        left_hand_mean_displacement=lh_mean_disp,
        right_hand_mean_displacement=rh_mean_disp,
        left_hand_mean_jitter=lh_mean_jit,
        right_hand_mean_jitter=rh_mean_jit,
        out_of_frame_keypoint_rate=out_rate,
    )
