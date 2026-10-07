"""Geometric transforms and target storage for YOLO dual-head training."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

LETTERBOX_PAD_VALUE = (114, 114, 114)


def _letterbox(frame: np.ndarray, image_size: int) -> tuple[np.ndarray, float, int, int]:
    height, width = frame.shape[:2]
    scale = min(image_size / width, image_size / height)
    resized_width, resized_height = round(width * scale), round(height * scale)
    resized = cv2.resize(frame, (resized_width, resized_height))
    left = round((image_size - resized_width) / 2 - 0.1)
    top = round((image_size - resized_height) / 2 - 0.1)
    image = cv2.copyMakeBorder(
        resized,
        top,
        image_size - resized_height - top,
        left,
        image_size - resized_width - left,
        cv2.BORDER_CONSTANT,
        value=LETTERBOX_PAD_VALUE,
    )
    return image, scale, left, top


def _clip_box(box: np.ndarray, width: int, height: int) -> list[float] | None:
    if box.shape != (4,) or not np.isfinite(box).all():
        return None
    x0, y0, x1, y1 = map(float, box)
    x0, x1 = sorted((max(0.0, min(width, x0)), max(0.0, min(width, x1))))
    y0, y1 = sorted((max(0.0, min(height, y0)), max(0.0, min(height, y1))))
    return [x0, y0, x1, y1] if x1 > x0 and y1 > y0 else None


def _visible(points: np.ndarray, width: int, height: int, threshold: float) -> np.ndarray:
    return (
        np.isfinite(points).all(axis=1)
        & (points[:, 2] >= threshold)
        & (points[:, 0] >= 0)
        & (points[:, 0] < width)
        & (points[:, 1] >= 0)
        & (points[:, 1] < height)
    )


def _hand_box(
    points: np.ndarray,
    width: int,
    height: int,
    threshold: float,
) -> list[float] | None:
    valid = _visible(points, width, height, threshold)
    xy = points[valid, :2]
    if not len(xy):
        return None
    return _clip_box(np.array([*xy.min(axis=0), *xy.max(axis=0)]), width, height)


def _labels(
    shard: Any,
    frame_position: int,
    width: int,
    height: int,
    threshold: float,
    pose_indices: np.ndarray | slice = slice(0, 17),
    hand_slices: tuple[slice, slice] = (slice(91, 112), slice(112, 133)),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    boxes = np.zeros((3, 4), dtype=np.float32)
    classes = np.full(3, -1, dtype=np.int8)
    person_keypoints = np.zeros((17, 3), dtype=np.float32)
    hand_keypoints = np.zeros((2, 21, 3), dtype=np.float32)
    start = int(shard.frame_person_offsets[frame_position])
    stop = int(shard.frame_person_offsets[frame_position + 1])
    if start == stop:
        return boxes, classes, person_keypoints, hand_keypoints
    person = start + int(np.argmax(shard.person_score[start:stop]))
    person_box = _clip_box(shard.person_bbox_xyxy[person], width, height)
    if person_box is None:
        return boxes, classes, person_keypoints, hand_keypoints
    boxes[0], classes[0] = person_box, 0
    points = shard.keypoints[person, pose_indices]
    visible = _visible(points, width, height, threshold)
    person_keypoints[visible, :2] = points[visible, :2]
    person_keypoints[visible, 2] = 2.0
    for hand_idx, hand_slice in enumerate(hand_slices):
        raw_hand_pts = shard.keypoints[person, hand_slice]
        hand = _hand_box(raw_hand_pts, width, height, threshold)
        if hand is not None:
            slot = hand_idx + 1
            boxes[slot], classes[slot] = hand, slot
            hand_vis = _visible(raw_hand_pts, width, height, threshold)
            hand_keypoints[hand_idx, hand_vis, :2] = raw_hand_pts[hand_vis, :2]
            hand_keypoints[hand_idx, hand_vis, 2] = 2.0
    return boxes, classes, person_keypoints, hand_keypoints


def apply_augmentation(
    image: np.ndarray,
    boxes: np.ndarray,
    classes: np.ndarray,
    person_keypoints: np.ndarray,
    hand_keypoints: np.ndarray,
    *,
    image_size: int,
    seed: int,
    epoch: int,
    sample_index: int,
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Apply synchronized HSV and Affine transform (without horizontal flip) to image and targets."""
    sample_seed = (int(seed) * 1_000_003 + int(epoch) * 65_537 + int(sample_index)) & 0xFFFFFFFF
    rng = np.random.default_rng(sample_seed)

    # 1. HSV color jitter
    h_gain = float(config.get("hsv_h", 0.01))
    s_gain = float(config.get("hsv_s", 0.30))
    v_gain = float(config.get("hsv_v", 0.20))
    if h_gain > 0 or s_gain > 0 or v_gain > 0:
        r_h = rng.uniform(-h_gain, h_gain)
        r_s = rng.uniform(max(0.0, 1.0 - s_gain), 1.0 + s_gain)
        r_v = rng.uniform(max(0.0, 1.0 - v_gain), 1.0 + v_gain)
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[..., 0] = (hsv[..., 0] + r_h * 180.0) % 180.0
        hsv[..., 1] = np.clip(hsv[..., 1] * r_s, 0.0, 255.0)
        hsv[..., 2] = np.clip(hsv[..., 2] * r_v, 0.0, 255.0)
        image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    # 2. Affine transform (Scale + Translation around center)
    translate = float(config.get("translate", 0.10))
    scale_gain = float(config.get("scale", 0.20))
    if translate > 0 or scale_gain > 0:
        scale = float(rng.uniform(1.0 - scale_gain, 1.0 + scale_gain)) if scale_gain > 0 else 1.0
        tx = float(rng.uniform(-translate, translate) * image_size) if translate > 0 else 0.0
        ty = float(rng.uniform(-translate, translate) * image_size) if translate > 0 else 0.0
        cx = image_size / 2.0
        cy = image_size / 2.0

        m02 = (1.0 - scale) * cx + tx
        m12 = (1.0 - scale) * cy + ty
        matrix = np.array([[scale, 0.0, m02], [0.0, scale, m12]], dtype=np.float32)
        image = cv2.warpAffine(
            image,
            matrix,
            (image_size, image_size),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=LETTERBOX_PAD_VALUE,
        )

        boxes = boxes.copy()
        classes = classes.copy()
        person_keypoints = person_keypoints.copy()
        hand_keypoints = hand_keypoints.copy()

        # Person bbox and keypoints (slot 0)
        if classes[0] == 0:
            p_box = boxes[0]
            tx0 = scale * p_box[0] + m02
            ty0 = scale * p_box[1] + m12
            tx1 = scale * p_box[2] + m02
            ty1 = scale * p_box[3] + m12
            clipped = _clip_box(np.array([tx0, ty0, tx1, ty1]), image_size, image_size)
            if clipped is not None and (clipped[2] - clipped[0]) >= 2.0 and (clipped[3] - clipped[1]) >= 2.0:
                boxes[0] = clipped
                for k in range(17):
                    if person_keypoints[k, 2] > 0:
                        px = scale * person_keypoints[k, 0] + m02
                        py = scale * person_keypoints[k, 1] + m12
                        if 0.0 <= px < image_size and 0.0 <= py < image_size:
                            person_keypoints[k, 0] = px
                            person_keypoints[k, 1] = py
                        else:
                            person_keypoints[k] = 0.0
            else:
                boxes[0] = 0.0
                classes[0] = -1
                person_keypoints[:] = 0.0

        # Hand bboxes and keypoints (slot 1: left hand, slot 2: right hand)
        for slot in (1, 2):
            hand_idx = slot - 1
            if classes[slot] >= 1:
                h_box = boxes[slot]
                tx0 = scale * h_box[0] + m02
                ty0 = scale * h_box[1] + m12
                tx1 = scale * h_box[2] + m02
                ty1 = scale * h_box[3] + m12
                clipped = _clip_box(np.array([tx0, ty0, tx1, ty1]), image_size, image_size)
                if clipped is not None and (clipped[2] - clipped[0]) >= 2.0 and (clipped[3] - clipped[1]) >= 2.0:
                    boxes[slot] = clipped
                    for k in range(21):
                        if hand_keypoints[hand_idx, k, 2] > 0:
                            hx = scale * hand_keypoints[hand_idx, k, 0] + m02
                            hy = scale * hand_keypoints[hand_idx, k, 1] + m12
                            if 0.0 <= hx < image_size and 0.0 <= hy < image_size:
                                hand_keypoints[hand_idx, k, 0] = hx
                                hand_keypoints[hand_idx, k, 1] = hy
                            else:
                                hand_keypoints[hand_idx, k] = 0.0
                else:
                    boxes[slot] = 0.0
                    classes[slot] = -1
                    hand_keypoints[hand_idx] = 0.0

    return image, boxes, classes, person_keypoints, hand_keypoints


def _open_arrays(
    output: Path,
    selected: dict[str, list[dict[str, Any]]],
    frames: int,
) -> dict[str, np.memmap]:
    arrays: dict[str, np.memmap] = {}
    specs = {
        "boxes": ((3, 4), "float32"),
        "classes": ((3,), "int8"),
        "person_keypoints": ((17, 3), "float32"),
        "hand_keypoints": ((2, 21, 3), "float32"),
    }
    for split, rows in selected.items():
        (output / split / "images").mkdir(parents=True, exist_ok=True)
        count = len(rows) * frames
        for name, (tail, dtype) in specs.items():
            path = output / f"{split}_{name}.npy"
            mode = "r+" if path.is_file() else "w+"
            expected_shape = (count, *tail)
            array = np.lib.format.open_memmap(
                path,
                mode=mode,
                dtype=dtype,
                shape=expected_shape,
            )
            if array.shape != expected_shape or array.dtype != np.dtype(dtype):
                raise RuntimeError(f"Invalid target array: {path}")
            arrays[f"{split}_{name}"] = array
    return arrays
