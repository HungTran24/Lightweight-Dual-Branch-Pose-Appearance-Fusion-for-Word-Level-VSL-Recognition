"""Pose53 view definition: keypoint indices and node count."""

from __future__ import annotations

from src.rtmw.pose53 import (
    POSE53_EDGES,
    POSE53_KEYPOINT_INDICES,
    POSE53_NODE_COUNT,
    build_rtmw_pose53_view,
)

VIEW_NAME = "pose53"
NODE_COUNT = POSE53_NODE_COUNT
KEYPOINT_INDICES = POSE53_KEYPOINT_INDICES
EDGES = POSE53_EDGES
BUILD_VIEW = build_rtmw_pose53_view
