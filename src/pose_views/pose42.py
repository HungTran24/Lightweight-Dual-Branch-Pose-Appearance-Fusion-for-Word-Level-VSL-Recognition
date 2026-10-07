"""Pose42 view definition: keypoint indices and node count."""

from __future__ import annotations

from src.rtmw.pose42 import (
    POSE42_EDGES,
    POSE42_KEYPOINT_INDICES,
    POSE42_NODE_COUNT,
    build_rtmw_pose42_view,
)

VIEW_NAME = "pose42"
NODE_COUNT = POSE42_NODE_COUNT
KEYPOINT_INDICES = POSE42_KEYPOINT_INDICES
EDGES = POSE42_EDGES
BUILD_VIEW = build_rtmw_pose42_view
