from __future__ import annotations

import numpy as np
import pytest
import torch

from src.yolo.metrics import DualHeadMetrics, _match, validate_dual_head


def _sample(hand_class: int = 0) -> tuple[list[dict], list[dict]]:
    person_points = np.zeros((17, 3), dtype=float)
    person_points[:, :2] = [20, 20]
    person_points[:, 2] = 2
    hand_points = np.zeros((21, 3), dtype=float)
    hand_points[:, :2] = [45, 45]
    hand_points[:, 2] = 2
    prediction = {
        "person": {
            "boxes": [[10, 10, 30, 30]],
            "conf": [0.9],
            "cls": [0],
            "keypoints": person_points[None],
        },
        "hands": {
            "boxes": [[40, 40, 50, 50]],
            "conf": [0.8],
            "cls": [hand_class],
            "keypoints": hand_points[None],
        },
    }
    target = {
        "person": {"boxes": [[10, 10, 30, 30]], "keypoints": person_points[None]},
        "hands": {
            "boxes": [[40, 40, 50, 50]],
            "classes": [hand_class],
            "keypoints": hand_points[None],
        },
    }
    return [prediction], [target]


def test_perfect_person_and_hand_pose() -> None:
    result = validate_dual_head(*_sample())
    assert result["person_bbox"]["map50_95"] > 0.99
    assert result["person_pose17_oks"]["map50_95"] > 0.99
    assert result["hand_bbox"]["map50_95"] > 0.99
    assert result["hand_pose21_oks"]["map50_95"] > 0.99


def test_no_predictions_is_supported() -> None:
    _, targets = _sample()
    result = validate_dual_head([{"person": [], "hands": []}], targets)
    assert result["person_bbox"]["map50"] == 0.0
    assert result["person_pose17_oks"]["map50"] == 0.0
    assert result["hand_bbox"]["map50"] == 0.0
    assert result["hand_pose21_oks"]["map50"] == 0.0


def test_hand_metrics_keep_per_class_indices() -> None:
    predictions, targets = _sample(hand_class=1)
    result = validate_dual_head(predictions, targets)
    assert result["hand_bbox"]["ap_class_index"] == [1]
    assert result["hand_pose21_oks"]["right_hand_map50_95"] > 0.99


def test_hand_targets_align_boxes_classes_and_keypoints() -> None:
    hand_points = np.zeros((2, 21, 3), dtype=float)
    hand_points[0, :, :] = [45, 45, 2]
    target = {
        "person": {
            "boxes": [[10, 10, 30, 30]],
            "classes": [0],
            "keypoints": np.zeros((1, 17, 3)),
        },
        "hands": {
            "boxes": [[40, 40, 50, 50]],
            "classes": [0],
            "keypoints": hand_points[:1],
        },
    }
    prediction = {
        "person": [],
        "hands": {
            "boxes": [[40, 40, 50, 50]],
            "conf": [0.9],
            "cls": [0],
            "keypoints": hand_points[:1],
        },
    }
    result = validate_dual_head([prediction], [target])
    assert result["hand_pose21_oks"]["map50"] > 0.99


def test_pixel_coordinates_are_not_normalized() -> None:
    result = validate_dual_head(*_sample())
    assert result["person_bbox"]["map50"] > 0.99


def test_person_area_oks_uses_same_predictions_and_is_less_strict() -> None:
    predictions, targets = _sample()
    predictions[0]["hands"]["keypoints"] = predictions[0]["hands"]["keypoints"].copy()
    predictions[0]["hands"]["keypoints"][:, :, 0] += 2.0
    targets[0]["person"]["boxes"] = [[0, 0, 100, 100]]

    metrics = DualHeadMetrics(compare_hand_oks_areas=True)
    metrics.update(predictions[0], targets[0], 0)
    result = metrics.compute()

    assert result["hand_pose21_oks"] == result["hand_pose21_oks_hand_area"]
    assert (
        result["hand_pose21_oks_person_area"]["map50_95"]
        > result["hand_pose21_oks_hand_area"]["map50_95"]
    )
    assert result["hand_area_over_person_area"]["median"] == pytest.approx(0.01)


def test_incremental_updates_and_greedy_deduplication() -> None:
    predictions, targets = _sample()
    metrics = DualHeadMetrics()
    metrics.update(predictions[0], targets[0], 0)
    assert metrics.compute()["person_bbox"]["map50"] > 0.99
    overlaps = torch.tensor([[0.9, 0.8], [0.85, 0.1]])
    matched = _match(torch.zeros(2, dtype=torch.long), torch.zeros(2, dtype=torch.long), overlaps)
    assert matched[:, 0].sum().item() == 1
    assert matched[0, 0]
