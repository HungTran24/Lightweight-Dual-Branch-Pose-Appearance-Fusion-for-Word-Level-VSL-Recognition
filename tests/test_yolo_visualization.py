"""Unit tests for YOLO validation overlay visualization."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from src.yolo.visualization import (
    COCO_17_EDGES,
    HAND_21_EDGES,
    save_validation_overlay,
)


def test_edges_definition() -> None:
    assert len(COCO_17_EDGES) == 16
    assert len(HAND_21_EDGES) == 20


def test_save_validation_overlay_empty(tmp_path: Path) -> None:
    img = np.zeros((128, 128, 3), dtype=np.uint8)
    out_file = tmp_path / "empty_overlay.jpg"
    canvas = save_validation_overlay(
        out_file,
        img,
        person_target=None,
        hands_target=None,
        person_prediction_rows=None,
        hand_prediction_rows=None,
    )
    assert out_file.exists()
    assert canvas.shape == (128, 128, 3)
    assert canvas.dtype == np.uint8


def test_save_validation_overlay_tensor_chw_and_float(tmp_path: Path) -> None:
    # CHW float [0, 1] tensor
    img_chw = torch.ones((3, 64, 64), dtype=torch.float32) * 0.5
    out_file = tmp_path / "float_chw.jpg"
    canvas = save_validation_overlay(
        out_file,
        img_chw,
        person_target={"boxes": [[10, 10, 50, 50]]},
        person_prediction_rows=None,
    )
    assert out_file.exists()
    assert canvas.shape == (64, 64, 3)
    # Check that bounding box altered some pixels
    assert np.any(canvas != int(0.5 * 255))


def test_save_validation_overlay_targets_and_predictions(tmp_path: Path) -> None:
    # 256x256 RGB image
    img = np.zeros((256, 256, 3), dtype=np.uint8)

    # Person target: 1 bbox + 17 keypoints
    person_kpts = np.zeros((17, 3), dtype=np.float32)
    person_kpts[0] = [100, 50, 1.0]  # nose
    person_kpts[5] = [80, 80, 1.0]   # left shoulder
    person_kpts[6] = [120, 80, 1.0]  # right shoulder
    person_target = {
        "boxes": np.array([[60, 40, 140, 200]], dtype=np.float32),
        "keypoints": person_kpts[None, ...],
    }

    # Hands target: 2 hands (left hand cls 0, right hand cls 1)
    hand_kpts = np.zeros((2, 21, 3), dtype=np.float32)
    # left hand wrist and thumb
    hand_kpts[0, 0] = [40, 120, 1.0]
    hand_kpts[0, 1] = [45, 115, 1.0]
    # right hand wrist
    hand_kpts[1, 0] = [150, 120, 1.0]
    hands_target = {
        "boxes": np.array([[30, 110, 50, 140], [140, 110, 160, 140]], dtype=np.float32),
        "classes": np.array([0, 1], dtype=np.int64),
        "keypoints": hand_kpts,
    }

    # Person prediction row format: [x1, y1, x2, y2, conf, cls, kpt0_x, kpt0_y, kpt0_conf, ...]
    person_pred_row = np.zeros(6 + 17 * 3, dtype=np.float32)
    person_pred_row[:6] = [62, 42, 138, 198, 0.95, 0]
    person_pred_row[6:9] = [100, 52, 0.9]  # predicted nose
    person_pred_row[6 + 5 * 3 : 6 + 5 * 3 + 3] = [82, 82, 0.9]
    person_pred_row[6 + 6 * 3 : 6 + 6 * 3 + 3] = [118, 82, 0.9]

    # Hand prediction rows: 2 detections (left hand, right hand)
    hand_pred_rows = np.zeros((2, 6 + 21 * 3), dtype=np.float32)
    hand_pred_rows[0, :6] = [32, 112, 48, 138, 0.88, 0]  # left hand (cls 0)
    hand_pred_rows[0, 6:9] = [41, 121, 0.9]
    hand_pred_rows[1, :6] = [142, 112, 158, 138, 0.91, 1]  # right hand (cls 1)
    hand_pred_rows[1, 6:9] = [151, 121, 0.9]

    out_file = tmp_path / "full_overlay.png"
    canvas = save_validation_overlay(
        out_file,
        torch.from_numpy(img),
        person_target=person_target,
        hands_target=hands_target,
        person_prediction_rows=torch.from_numpy(person_pred_row[None, :]),
        hand_prediction_rows=torch.from_numpy(hand_pred_rows),
    )

    assert out_file.exists()
    assert canvas.shape == (256, 256, 3)
    assert np.any(canvas > 0)


def test_confidence_threshold_filtering(tmp_path: Path) -> None:
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    # Low confidence detection (0.1 < default 0.25)
    pred_row = np.zeros((1, 6 + 17 * 3), dtype=np.float32)
    pred_row[0, :6] = [10, 10, 30, 30, 0.1, 0]

    canvas_filtered = save_validation_overlay(
        None,
        img,
        person_prediction_rows=pred_row,
        confidence_threshold=0.25,
    )
    # Filtered out -> canvas should stay completely black (0)
    assert np.all(canvas_filtered == 0)

    canvas_included = save_validation_overlay(
        None,
        img,
        person_prediction_rows=pred_row,
        confidence_threshold=0.05,
    )
    # Included -> canvas should have drawn bbox
    assert np.any(canvas_included > 0)


def test_invalid_image_raises() -> None:
    with pytest.raises(ValueError, match="Image cannot be empty"):
        save_validation_overlay(None, np.array([]))
