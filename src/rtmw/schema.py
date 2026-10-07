"""Packed raw RTMW shard schema and artifact identity contract."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

EXPECTED_WHOLEBODY_KEYPOINTS = 133
RAW_RTMW_SCHEMA_VERSION = "rtmw-raw-v1"


class RawRTMWSchemaError(ValueError):
    """Raised when a packed RTMW shard violates the raw artifact contract."""


@dataclass(frozen=True)
class RTMWArtifactIdentityConfig:
    """Inference inputs that determine the meaning of a raw RTMW artifact."""

    dataset_repo: str
    dataset_revision: str
    manifest_sha256: str
    detector_config: str
    detector_checkpoint_url: str
    detector_checkpoint_sha256: str
    pose_config: str
    pose_checkpoint_url: str
    pose_checkpoint_sha256: str
    bbox_score_threshold: float
    keypoint_score_threshold: float
    precision: str
    source_git_commit: str
    schema_version: str = RAW_RTMW_SCHEMA_VERSION

    def validate(self) -> None:
        """Validate identity inputs before hashing them."""
        nonempty_fields = {
            "dataset_repo": self.dataset_repo,
            "dataset_revision": self.dataset_revision,
            "manifest_sha256": self.manifest_sha256,
            "detector_config": self.detector_config,
            "detector_checkpoint_url": self.detector_checkpoint_url,
            "detector_checkpoint_sha256": self.detector_checkpoint_sha256,
            "pose_config": self.pose_config,
            "pose_checkpoint_url": self.pose_checkpoint_url,
            "pose_checkpoint_sha256": self.pose_checkpoint_sha256,
            "precision": self.precision,
            "source_git_commit": self.source_git_commit,
            "schema_version": self.schema_version,
        }
        missing = [name for name, value in nonempty_fields.items() if not value.strip()]
        if missing:
            raise RawRTMWSchemaError(f"Artifact identity fields must be non-empty: {missing}")
        for name, value in {
            "manifest_sha256": self.manifest_sha256,
            "detector_checkpoint_sha256": self.detector_checkpoint_sha256,
            "pose_checkpoint_sha256": self.pose_checkpoint_sha256,
        }.items():
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                raise RawRTMWSchemaError(f"{name} must be a lowercase SHA256 hex digest")
        for name, value in {
            "bbox_score_threshold": self.bbox_score_threshold,
            "keypoint_score_threshold": self.keypoint_score_threshold,
        }.items():
            if not 0.0 <= value <= 1.0:
                raise RawRTMWSchemaError(f"{name} must be in [0.0, 1.0], got {value}")

    def canonical_json(self) -> str:
        """Return a stable representation used for artifact identity hashing."""
        self.validate()
        return json.dumps(asdict(self), ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    @property
    def config_sha256(self) -> str:
        """Return the SHA256 hash of all raw prediction semantics."""
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    @property
    def artifact_id(self) -> str:
        """Return a concise stable identifier for receipts and artifact paths."""
        return f"{self.schema_version}-{self.config_sha256[:16]}"


@dataclass(frozen=True)
class RawRTMWShard:
    """Packed ragged raw RTMW output for one deterministic video shard.

    Offsets are exclusive prefix sums. `video_frame_offsets` maps videos to
    frames, and `frame_person_offsets` maps frames to person instances.
    `keypoints[..., 2]` holds RTMW's raw keypoint score, not a normalized
    confidence probability.
    """

    video_id: np.ndarray
    video_frame_offsets: np.ndarray
    frame_idx: np.ndarray
    timestamp_ms: np.ndarray
    frame_width: np.ndarray
    frame_height: np.ndarray
    frame_person_offsets: np.ndarray
    person_bbox_xyxy: np.ndarray
    person_score: np.ndarray
    keypoints: np.ndarray

    def validate(self) -> None:
        """Validate dtypes, shapes, offsets, and numeric invariants."""
        _validate_video_ids(self.video_id)
        video_count = self.video_id.shape[0]
        if video_count == 0:
            raise RawRTMWSchemaError("A raw shard must contain at least one video")
        _validate_array("video_frame_offsets", self.video_frame_offsets, np.int64, (video_count + 1,))
        _validate_array("frame_width", self.frame_width, np.int32, (video_count,))
        _validate_array("frame_height", self.frame_height, np.int32, (video_count,))
        if np.any(self.frame_width <= 0) or np.any(self.frame_height <= 0):
            raise RawRTMWSchemaError("frame_width and frame_height must be positive")

        frame_count = _validate_offsets("video_frame_offsets", self.video_frame_offsets)
        _validate_array("frame_idx", self.frame_idx, np.int32, (frame_count,))
        _validate_array("timestamp_ms", self.timestamp_ms, np.int64, (frame_count,))
        _validate_array("frame_person_offsets", self.frame_person_offsets, np.int64, (frame_count + 1,))
        if np.any(self.frame_idx < 0) or np.any(self.timestamp_ms < 0):
            raise RawRTMWSchemaError("frame_idx and timestamp_ms must be non-negative")

        person_count = _validate_offsets("frame_person_offsets", self.frame_person_offsets)
        _validate_array("person_bbox_xyxy", self.person_bbox_xyxy, np.float32, (person_count, 4))
        _validate_array("person_score", self.person_score, np.float32, (person_count,))
        _validate_array(
            "keypoints",
            self.keypoints,
            np.float32,
            (person_count, EXPECTED_WHOLEBODY_KEYPOINTS, 3),
        )
        for name, array in {
            "person_bbox_xyxy": self.person_bbox_xyxy,
            "person_score": self.person_score,
            "keypoints": self.keypoints,
        }.items():
            if not np.isfinite(array).all():
                raise RawRTMWSchemaError(f"{name} contains NaN or Inf")
        if np.any(self.person_score < 0.0) or np.any(self.person_score > 1.0):
            raise RawRTMWSchemaError("person_score must be within [0.0, 1.0]")
        if np.any(self.keypoints[:, :, 2] < 0.0):
            raise RawRTMWSchemaError("raw keypoint scores must be non-negative")

    def arrays(self) -> dict[str, np.ndarray]:
        """Return the safe numeric/string arrays that form a shard payload."""
        self.validate()
        return {
            "video_id": self.video_id,
            "video_frame_offsets": self.video_frame_offsets,
            "frame_idx": self.frame_idx,
            "timestamp_ms": self.timestamp_ms,
            "frame_width": self.frame_width,
            "frame_height": self.frame_height,
            "frame_person_offsets": self.frame_person_offsets,
            "person_bbox_xyxy": self.person_bbox_xyxy,
            "person_score": self.person_score,
            "keypoints": self.keypoints,
        }

    def save_npz(self, path: Path | str) -> None:
        """Serialize a validated shard without object arrays or pickle."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(destination, **self.arrays())  # type: ignore[reportArgumentType]

    @classmethod
    def load_npz(cls, path: Path | str) -> RawRTMWShard:
        """Load and validate a shard serialized by :meth:`save_npz`."""
        source = Path(path)
        with np.load(source, allow_pickle=False) as payload:
            required = {
                "video_id",
                "video_frame_offsets",
                "frame_idx",
                "timestamp_ms",
                "frame_width",
                "frame_height",
                "frame_person_offsets",
                "person_bbox_xyxy",
                "person_score",
                "keypoints",
            }
            missing = required.difference(payload.files)
            unexpected = set(payload.files).difference(required)
            if missing or unexpected:
                raise RawRTMWSchemaError(
                    f"Shard arrays must match schema exactly; missing={sorted(missing)}, "
                    f"unexpected={sorted(unexpected)}"
                )
            shard = cls(**{name: payload[name] for name in required})
        shard.validate()
        return shard


def _validate_video_ids(video_id: np.ndarray) -> None:
    if video_id.ndim != 1 or video_id.dtype.kind not in {"U", "S"}:
        raise RawRTMWSchemaError("video_id must be a one-dimensional fixed-width string array")
    normalized = [str(value) for value in video_id.tolist()]
    if any(len(value) != 6 or not value.isdigit() for value in normalized):
        raise RawRTMWSchemaError("video_id values must preserve six-digit numeric IDs")
    if len(set(normalized)) != len(normalized):
        raise RawRTMWSchemaError("video_id values must be unique within a shard")


def _validate_array(name: str, array: np.ndarray, dtype: Any, shape: tuple[int, ...]) -> None:
    if not isinstance(array, np.ndarray):
        raise RawRTMWSchemaError(f"{name} must be a NumPy array")
    if array.dtype != np.dtype(dtype):
        raise RawRTMWSchemaError(f"{name} must use dtype {np.dtype(dtype)}, got {array.dtype}")
    if array.shape != shape:
        raise RawRTMWSchemaError(f"{name} must have shape {shape}, got {array.shape}")


def _validate_offsets(name: str, offsets: np.ndarray) -> int:
    if offsets[0] != 0:
        raise RawRTMWSchemaError(f"{name}[0] must equal 0")
    if np.any(offsets[1:] < offsets[:-1]):
        raise RawRTMWSchemaError(f"{name} must be non-decreasing")
    return int(offsets[-1])
