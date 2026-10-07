"""Native-resolution RGB hand crop encoding and Parquet output for E021."""

from __future__ import annotations

import math
import os
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ..pose_views.pose59 import validate_pose59_schema

LETTERBOX_PAD_VALUE = (114, 114, 114)
PARQUET_SCHEMA = pa.schema(
    [
        pa.field("video_id", pa.string(), nullable=False),
        pa.field("split", pa.string(), nullable=False),
        pa.field("frame_idx", pa.int32(), nullable=False),
        pa.field("frame_width", pa.int32(), nullable=False),
        pa.field("frame_height", pa.int32(), nullable=False),
        pa.field("left_jpeg", pa.binary(), nullable=True),
        pa.field("right_jpeg", pa.binary(), nullable=True),
        pa.field("left_valid", pa.uint8(), nullable=False),
        pa.field("right_valid", pa.uint8(), nullable=False),
        pa.field("left_bbox", pa.list_(pa.float32(), 4), nullable=False),
        pa.field("right_bbox", pa.list_(pa.float32(), 4), nullable=False),
        pa.field("left_confidence", pa.float32(), nullable=False),
        pa.field("right_confidence", pa.float32(), nullable=False),
        pa.field("left_roi", pa.list_(pa.int32(), 4), nullable=False),
        pa.field("right_roi", pa.list_(pa.int32(), 4), nullable=False),
        pa.field("left_padding", pa.list_(pa.int32(), 4), nullable=False),
        pa.field("right_padding", pa.list_(pa.int32(), 4), nullable=False),
    ]
)


def inverse_letterbox_boxes(
    boxes: np.ndarray,
    *,
    scale: float,
    pad_x: int,
    pad_y: int,
    width: int,
    height: int,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Project letterboxed bounding boxes back to original frame pixel space float32."""
    result = np.asarray(boxes, dtype=np.float32).copy()
    if result.size == 0:
        return result
    result[..., 0] = np.clip((result[..., 0] - pad_x) / scale, 0.0, float(width))
    result[..., 1] = np.clip((result[..., 1] - pad_y) / scale, 0.0, float(height))
    result[..., 2] = np.clip((result[..., 2] - pad_x) / scale, 0.0, float(width))
    result[..., 3] = np.clip((result[..., 3] - pad_y) / scale, 0.0, float(height))
    if valid_mask is not None:
        mask = np.asarray(valid_mask, dtype=bool)
        result[~mask] = 0.0
    else:
        invalid = (result[..., 2] <= result[..., 0]) | (result[..., 3] <= result[..., 1])
        result[invalid] = 0.0
    return result


def square_crop_rgb(
    image: np.ndarray,
    bbox: np.ndarray | list[float],
    keypoints: np.ndarray,
    *,
    margin: float = 1.3,
    quality: int = 95,
) -> tuple[bytes, np.ndarray, np.ndarray]:
    """Crop a padded centered square around bbox and valid keypoint envelope."""
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must have shape (H, W, 3) RGB")
    box = np.asarray(bbox, dtype=np.float32).reshape(2, 2)
    pts = np.asarray(keypoints, dtype=np.float32)
    valid = (pts[..., 2] > 0) & np.isfinite(pts[..., :2]).all(axis=-1)
    envelope = np.concatenate((box, pts[valid, :2]), axis=0) if np.any(valid) else box
    x0, y0 = envelope.min(axis=0)
    x1, y1 = envelope.max(axis=0)
    side = max(1, int(math.ceil(float(margin) * max(float(x1 - x0), float(y1 - y0)))))
    left = int(math.floor((x0 + x1 - side) / 2.0))
    top = int(math.floor((y0 + y1 - side) / 2.0))
    roi = np.array([left, top, left + side, top + side], dtype=np.int32)
    height, width = image.shape[:2]
    padding = np.array(
        [max(0, -left), max(0, -top), max(0, roi[2] - width), max(0, roi[3] - height)],
        dtype=np.int32,
    )
    crop = np.full((side, side, 3), LETTERBOX_PAD_VALUE, dtype=np.uint8)
    ix0, iy0, ix1, iy1 = max(0, left), max(0, top), min(width, roi[2]), min(height, roi[3])
    if ix1 > ix0 and iy1 > iy0:
        crop[iy0 - top : iy1 - top, ix0 - left : ix1 - left] = image[iy0:iy1, ix0:ix1]
    ok, encoded = cv2.imencode(
        ".jpg", cv2.cvtColor(crop, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    )
    if not ok:
        raise RuntimeError("OpenCV JPEG encoding failed")
    return encoded.tobytes(), roi, padding


def crop_hand_rgb(
    image: np.ndarray | None,
    bbox: np.ndarray | list[float],
    keypoints: np.ndarray,
    is_valid: bool | int,
    *,
    margin: float = 1.3,
    quality: int = 95,
) -> tuple[bytes | None, np.ndarray, np.ndarray]:
    """Return null JPEG and zero geometry when a hand has no valid positive-area box."""
    if not is_valid or image is None:
        return None, np.zeros(4, dtype=np.int32), np.zeros(4, dtype=np.int32)
    box = np.asarray(bbox, dtype=np.float32).reshape(-1)
    if box.shape != (4,) or not np.isfinite(box).all() or box[2] <= box[0] or box[3] <= box[1]:
        return None, np.zeros(4, dtype=np.int32), np.zeros(4, dtype=np.int32)
    return square_crop_rgb(image, bbox, keypoints, margin=margin, quality=quality)


def write_crop_shard(
    rows: list[dict[str, Any]],
    output_root: Path,
    part: int,
    index: int,
) -> dict[str, Any]:
    """Atomically write one bounded crop row buffer with fixed nullable JPEG columns."""
    name = f"part_{part:02d}_crops_{index:03d}.parquet"
    path = output_root / name
    temp = output_root / f"{name}.tmp"
    pq.write_table(pa.Table.from_pylist(rows, schema=PARQUET_SCHEMA), temp)
    os.replace(temp, path)
    return {"name": name, "frames": len(rows)}


def validate_extended_schema(
    arrays: dict[str, np.ndarray], validate_base: Any = validate_pose59_schema
) -> None:
    """Validate E021 native-box NPZ fields around the unchanged Pose59 schema."""
    base = {k: v for k, v in arrays.items() if k not in {"object_bbox_xyxy", "object_valid_mask"}}
    validate_base(base)
    frames = len(arrays["X"])
    boxes, valid = arrays["object_bbox_xyxy"], arrays["object_valid_mask"]
    if boxes.shape != (frames, 3, 4) or boxes.dtype != np.float32:
        raise ValueError("object_bbox_xyxy must have shape (frames, 3, 4) and dtype float32")
    if valid.shape != (frames, 3) or valid.dtype != np.uint8 or not np.isin(valid, (0, 1)).all():
        raise ValueError(
            "object_valid_mask must have shape (frames, 3), dtype uint8, and values 0 or 1"
        )
    positive = (boxes[..., 2] > boxes[..., 0]) & (boxes[..., 3] > boxes[..., 1])
    if (
        not np.isfinite(boxes).all()
        or np.any(boxes[valid == 0] != 0)
        or np.any(positive[valid == 1] == 0)
    ):
        raise ValueError(
            "Object boxes must be finite, positive-area when valid, and zero otherwise"
        )


validate_hand_rgb_schema = validate_extended_schema


def validate_crop_shards(
    marker: dict[str, Any],
    output_root: Path,
    part: int,
    expected_frames: int,
    sha256_fn: Any,
    arrays: dict[str, np.ndarray] | None = None,
) -> None:
    """Validate the exact E021 crop shard set, schema, rows, hashes, and marker counts."""
    shards = marker.get("shards")
    if not isinstance(shards, list) or not shards:
        raise RuntimeError("Incomplete crop shard marker")
    expected = {str(item.get("name")) for item in shards}
    actual = {path.name for path in output_root.glob(f"part_{part:02d}_crops_*.parquet")}
    if expected != actual:
        raise RuntimeError("Crop shard set mismatch on resume")
    totals = {"left": 0, "right": 0, "rows": 0}
    for item in shards:
        path = output_root / str(item["name"])
        if not path.is_file() or sha256_fn(path) != item.get("sha256"):
            raise RuntimeError(f"Corrupt or missing parquet shard on resume: {path}")
        table = pq.read_table(path)
        if table.schema != PARQUET_SCHEMA or len(table) != int(item.get("frames", -1)):
            raise RuntimeError(f"Invalid parquet shard schema or row count: {path}")
        totals["rows"] += len(table)
        for side in ("left", "right"):
            valid = table.column(f"{side}_valid").to_numpy()
            is_null = table.column(f"{side}_jpeg").is_null().to_numpy()
            if not np.array_equal(is_null, valid == 0):
                raise RuntimeError(f"Crop JPEG null alignment mismatch: {path}")
            totals[side] += int(np.sum(valid))
    if totals["rows"] != expected_frames:
        raise RuntimeError("Aggregate crop shard rows mismatch")
    missing = marker.get("missing_counts", {})
    crops = marker.get("crop_counts", {})
    if crops != {"left": totals["left"], "right": totals["right"]}:
        raise RuntimeError("Crop count marker mismatch")
    if (
        missing.get("left") != expected_frames - totals["left"]
        or missing.get("right") != expected_frames - totals["right"]
    ):
        raise RuntimeError("Missing count marker mismatch")
    if arrays is not None and missing.get("person") != int(
        np.sum(arrays["object_valid_mask"][:, 0] == 0)
    ):
        raise RuntimeError("Person missing count marker mismatch")


class HandRgbSink:
    """Buffers, crops, and flushes Parquet shards for E021 hand extraction."""

    def __init__(
        self, output_root: Path, part: int, config: dict[str, Any], sha256_fn: Any
    ) -> None:
        self.output_root = output_root
        self.part = part
        self.sha256_fn = sha256_fn
        crop_cfg = config.get("crops", {"crop_shard_frames": 4096, "margin": 1.3, "quality": 95})
        self.shard_limit = int(crop_cfg.get("crop_shard_frames", 4096))
        self.margin = float(crop_cfg.get("margin", 1.3))
        self.quality = int(crop_cfg.get("quality", 95))
        self.rows: list[dict[str, Any]] = []
        self.shards: list[dict[str, Any]] = []
        self.crop_time: float = 0.0
        self.write_time: float = 0.0
        for stale in output_root.glob(f"part_{part:02d}_crops_*.parquet"):
            stale.unlink()

    def append_frame(
        self,
        image: np.ndarray,
        video_id: str,
        split: str,
        frame_idx: int,
        width: int,
        height: int,
        inv_kpts: np.ndarray,
        boxes: np.ndarray,
        confidence: np.ndarray,
        valid: np.ndarray,
    ) -> None:
        t0 = time.perf_counter()
        crops: dict[str, tuple[bytes | None, np.ndarray, np.ndarray]] = {}
        for slot, side, hand_kpts in ((1, "left", inv_kpts[17:38]), (2, "right", inv_kpts[38:])):
            crops[side] = crop_hand_rgb(
                image,
                boxes[slot],
                hand_kpts * np.array([width, height, 1.0], np.float32),
                valid[slot],
                margin=self.margin,
                quality=self.quality,
            )
        left, left_roi, left_pad = crops["left"]
        right, right_roi, right_pad = crops["right"]
        self.crop_time += time.perf_counter() - t0
        self.rows.append(
            {
                "video_id": video_id,
                "split": split,
                "frame_idx": frame_idx,
                "frame_width": width,
                "frame_height": height,
                "left_jpeg": left,
                "right_jpeg": right,
                "left_valid": int(valid[1]),
                "right_valid": int(valid[2]),
                "left_bbox": boxes[1].tolist(),
                "right_bbox": boxes[2].tolist(),
                "left_confidence": float(confidence[1]),
                "right_confidence": float(confidence[2]),
                "left_roi": left_roi.tolist(),
                "right_roi": right_roi.tolist(),
                "left_padding": left_pad.tolist(),
                "right_padding": right_pad.tolist(),
            }
        )
        while len(self.rows) >= self.shard_limit:
            self._flush_slice(self.rows[: self.shard_limit])
            del self.rows[: self.shard_limit]

    def flush(self) -> None:
        if self.rows:
            self._flush_slice(self.rows)
            self.rows.clear()

    def _flush_slice(self, rows: list[dict[str, Any]]) -> None:
        t0 = time.perf_counter()
        info = write_crop_shard(rows, self.output_root, self.part, len(self.shards))
        path = self.output_root / info["name"]
        info.update({"sha256": self.sha256_fn(path), "bytes": path.stat().st_size})
        self.shards.append(info)
        self.write_time += time.perf_counter() - t0

    def marker_fields(self, arrays: dict[str, np.ndarray]) -> dict[str, Any]:
        return {
            "shards": self.shards,
            "crop_counts": {
                "left": int(np.sum(arrays["object_valid_mask"][:, 1])),
                "right": int(np.sum(arrays["object_valid_mask"][:, 2])),
            },
            "missing_counts": {
                "person": int(np.sum(arrays["object_valid_mask"][:, 0] == 0)),
                "left": int(np.sum(arrays["object_valid_mask"][:, 1] == 0)),
                "right": int(np.sum(arrays["object_valid_mask"][:, 2] == 0)),
            },
        }
