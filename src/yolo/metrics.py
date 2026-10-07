"""Reusable validation metrics for the YOLO dual-head model."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
import torch
from ultralytics.utils.metrics import DetMetrics, PoseMetrics, box_iou, kpt_iou

IOU_THRESHOLDS = torch.arange(0.50, 1.00, 0.05)
PERSON_KEYPOINTS = 17
HAND_KEYPOINTS = 21
PERSON_SIGMAS = [0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072, 0.062, 0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089]
# COCO-WholeBody hand sigmas from the MMPose dataset metainfo.
HAND_SIGMAS = [0.029, 0.022, 0.035, 0.037, 0.047, 0.026, 0.025, 0.024, 0.035, 0.018, 0.024, 0.022, 0.026, 0.017, 0.021, 0.021, 0.032, 0.020, 0.019, 0.022, 0.031]
OKS_AREA_FACTOR = 0.53


def _tensor(value: Any, *, shape: tuple[int, ...] | None = None) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        result = value.detach().to(device="cpu", dtype=torch.float32)
    else:
        result = torch.as_tensor(value, dtype=torch.float32, device="cpu")
    if shape is not None and result.numel() == 0:
        return torch.zeros(shape, dtype=torch.float32)
    return result.reshape(shape) if shape is not None and result.numel() == int(np.prod(shape)) else result


def _rows(value: Any, *, keypoints: bool = False, keypoint_count: int = PERSON_KEYPOINTS) -> tuple[torch.Tensor, ...]:
    if isinstance(value, Mapping):
        boxes = _tensor(value.get("boxes", []), shape=(0, 4))
        confidence = _tensor(value.get("confidence", value.get("conf", [])), shape=(0,))
        classes = _tensor(value.get("classes", value.get("cls", [])), shape=(0,)).long()
        if keypoints:
            points = value.get("keypoints")
            pose = _tensor(points, shape=(len(boxes), keypoint_count, 3)) if points is not None else torch.zeros((len(boxes), keypoint_count, 3))
            if pose.ndim == 3 and pose.shape[1] != keypoint_count:
                pose = pose[:, :keypoint_count]
            return boxes, confidence, classes, pose
        return boxes, confidence, classes
    rows = _tensor(value, shape=(0, 6))
    if rows.ndim != 2 or rows.shape[1] < 6:
        raise ValueError("decoded rows must have shape (N, >=6): xyxy, confidence, class")
    points = rows[:, 6:].reshape(-1, keypoint_count, 3) if keypoints and rows.shape[1] > 6 and rows[:, 6:].shape[1] == keypoint_count * 3 else torch.zeros((len(rows), keypoint_count, 3))
    result = (rows[:, :4], rows[:, 4], rows[:, 5].long())
    return (*result, points) if keypoints else result


def _target(target: Mapping[str, Any], name: str, keypoint_count: int = PERSON_KEYPOINTS) -> tuple[torch.Tensor, ...]:
    value = target.get(name)
    if not isinstance(value, Mapping):
        raise ValueError(f"target must contain a '{name}' mapping")

    boxes = _tensor(value.get("boxes", []), shape=(0, 4))
    classes = _tensor(value.get("classes", value.get("cls", [])), shape=(0,)).long()
    points_value = value.get("keypoints", value.get("person_keypoints", value.get("hand_keypoints", [])))
    points = _tensor(points_value, shape=(len(boxes), keypoint_count, 3))
    if points.ndim == 2:
        points = points.unsqueeze(0)
    if len(points) != len(boxes):
        padded = torch.zeros((len(boxes), keypoint_count, 3), dtype=torch.float32)
        padded[: min(len(points), len(boxes))] = points[: len(boxes), :keypoint_count]
        points = padded
    elif points.shape[1] != keypoint_count:
        points = points[:, :keypoint_count]

    if name == "person":
        return boxes, points
    if len(classes) != len(boxes):
        raise ValueError(f"target '{name}' classes and boxes must have equal lengths")
    return boxes, classes, points


def _match(pred_classes: torch.Tensor, true_classes: torch.Tensor, overlaps: torch.Tensor) -> torch.Tensor:
    """Greedy one-to-one matching equivalent to BaseValidator.match_predictions."""
    correct = torch.zeros((len(pred_classes), 10), dtype=torch.bool)
    overlaps = overlaps * (true_classes[:, None] == pred_classes)
    for column, threshold in enumerate(IOU_THRESHOLDS):
        matches = (overlaps >= threshold).nonzero(as_tuple=False)
        if len(matches) > 1:
            scores = overlaps[matches[:, 0], matches[:, 1]]
            matches = matches[scores.argsort(descending=True)]
            used_labels = torch.zeros(len(true_classes), dtype=torch.bool)
            used_predictions = torch.zeros(len(pred_classes), dtype=torch.bool)
            kept: list[torch.Tensor] = []
            for label_index, prediction_index in matches:
                label_index, prediction_index = int(label_index), int(prediction_index)
                if not used_labels[label_index] and not used_predictions[prediction_index]:
                    used_labels[label_index] = used_predictions[prediction_index] = True
                    kept.append(torch.tensor([label_index, prediction_index]))
            matches = torch.stack(kept) if kept else matches[:0]
        if len(matches):
            correct[matches[:, 1], column] = True
    return correct


def _stats(predicted: tuple[torch.Tensor, ...], target_boxes: torch.Tensor, target_classes: torch.Tensor) -> tuple[np.ndarray, ...]:
    pred_boxes, confidence, pred_classes = predicted[:3]
    correct = torch.zeros((len(pred_boxes), 10), dtype=torch.bool)
    if len(pred_boxes) and len(target_boxes):
        correct = _match(pred_classes, target_classes, box_iou(target_boxes, pred_boxes))
    return correct.numpy(), confidence.numpy(), pred_classes.numpy(), target_classes.numpy()


def _result(
    metrics: DetMetrics | PoseMetrics,
    *,
    pose: bool = False,
    class_names: Mapping[int, str] | None = None,
) -> dict[str, Any]:
    metric = metrics.pose if pose else metrics.box
    res = {
        "map50": float(metric.map50),
        "map50_95": float(metric.map),
        "per_class_map50": [float(x) for x in np.asarray(metric.ap50)],
        "per_class_map50_95": [float(x) for x in np.asarray(metric.ap)],
        "ap_class_index": [int(x) for x in metric.ap_class_index],
    }
    if class_names is not None:
        for idx, cls_id in enumerate(res["ap_class_index"]):
            name = class_names.get(cls_id, f"class_{cls_id}")
            if idx < len(res["per_class_map50"]):
                res[f"{name}_map50"] = res["per_class_map50"][idx]
            if idx < len(res["per_class_map50_95"]):
                res[f"{name}_map50_95"] = res["per_class_map50_95"][idx]
    return res


class DualHeadMetrics:
    """Incremental validation accumulator for decoded dual-head predictions."""

    def __init__(self, *, hand_classes: int = 2, compare_hand_oks_areas: bool = False) -> None:
        self.hand_names: dict[int, str] = {
            i: str(name) for i, name in enumerate(("left_hand", "right_hand")[:hand_classes])
        }
        self.person = PoseMetrics(names={0: "person"})
        self.hands = PoseMetrics(names=self.hand_names)
        self.hands_person_area = (
            PoseMetrics(names=self.hand_names) if compare_hand_oks_areas else None
        )
        self.hand_to_person_area_ratios: list[float] = []

    def update(self, prediction: Mapping[str, Any], target: Mapping[str, Any], image_index: int) -> None:
        person_pred = _rows(prediction.get("person", []), keypoints=True, keypoint_count=PERSON_KEYPOINTS)
        hand_pred = _rows(prediction.get("hands", []), keypoints=True, keypoint_count=HAND_KEYPOINTS)
        person_boxes, person_points = _target(target, "person", PERSON_KEYPOINTS)
        hand_boxes, target_hand_classes, hand_points = _target(target, "hands", HAND_KEYPOINTS)
        person_classes = torch.zeros(len(person_boxes), dtype=torch.long)
        person_stat = _stats(person_pred, person_boxes, person_classes)
        hand_stat = _stats(hand_pred, hand_boxes, target_hand_classes)
        person_area = (
            (person_boxes[:, 2] - person_boxes[:, 0])
            * (person_boxes[:, 3] - person_boxes[:, 1])
            * OKS_AREA_FACTOR
        )
        hand_area = (
            (hand_boxes[:, 2] - hand_boxes[:, 0])
            * (hand_boxes[:, 3] - hand_boxes[:, 1])
            * OKS_AREA_FACTOR
        )
        person_tp = torch.zeros((len(person_pred[0]), 10), dtype=torch.bool)
        hand_tp = torch.zeros((len(hand_pred[0]), 10), dtype=torch.bool)
        if len(person_pred[0]) and len(person_boxes):
            person_tp = _match(person_pred[2], person_classes, kpt_iou(person_points, person_pred[3], area=person_area, sigma=PERSON_SIGMAS))
        if len(hand_pred[0]) and len(hand_boxes):
            hand_tp = _match(hand_pred[2], target_hand_classes, kpt_iou(hand_points, hand_pred[3], area=hand_area, sigma=HAND_SIGMAS))
        hand_person_area_tp = None
        if self.hands_person_area is not None:
            if len(person_area) != 1 and len(hand_boxes):
                raise ValueError("Person-area hand OKS requires exactly one person target per image")
            paired_person_area = person_area.expand(len(hand_boxes))
            if len(hand_boxes):
                self.hand_to_person_area_ratios.extend(
                    (hand_area / paired_person_area).tolist()
                )
            hand_person_area_tp = torch.zeros((len(hand_pred[0]), 10), dtype=torch.bool)
            if len(hand_pred[0]) and len(hand_boxes):
                hand_person_area_tp = _match(
                    hand_pred[2],
                    target_hand_classes,
                    kpt_iou(
                        hand_points,
                        hand_pred[3],
                        area=paired_person_area,
                        sigma=HAND_SIGMAS,
                    ),
                )
        self.person.update_stats(
            {
                "tp": person_stat[0],
                "tp_p": person_tp.numpy(),
                "conf": person_stat[1],
                "target_cls": person_stat[3],
                "pred_cls": person_stat[2],
                "target_img": np.full(len(person_stat[3]), image_index, dtype=np.int64),
                "im_name": str(image_index),
            }
        )
        hand_stats = {
            "tp": hand_stat[0],
            "tp_p": hand_tp.numpy(),
            "conf": hand_stat[1],
            "target_cls": hand_stat[3],
            "pred_cls": hand_stat[2],
            "target_img": np.full(len(hand_stat[3]), image_index, dtype=np.int64),
            "im_name": str(image_index),
        }
        self.hands.update_stats(hand_stats)
        if self.hands_person_area is not None and hand_person_area_tp is not None:
            self.hands_person_area.update_stats(
                {**hand_stats, "tp_p": hand_person_area_tp.numpy()}
            )

    def compute(self) -> dict[str, Any]:
        self.person.process()
        self.hands.process()
        person_bbox = _result(self.person)
        person_pose = _result(self.person, pose=True)
        hand_bbox = _result(self.hands)
        hand_pose = _result(self.hands, pose=True, class_names=self.hand_names)
        result = {
            "person_bbox": person_bbox,
            "person_pose17_oks": person_pose,
            "hand_bbox": hand_bbox,
            "hand_pose21_oks": hand_pose,
        }
        if self.hands_person_area is not None:
            self.hands_person_area.process()
            ratios = np.asarray(self.hand_to_person_area_ratios, dtype=np.float64)
            result["hand_pose21_oks_hand_area"] = hand_pose
            result["hand_pose21_oks_person_area"] = _result(
                self.hands_person_area,
                pose=True,
                class_names=self.hand_names,
            )
            result["hand_area_over_person_area"] = {
                "count": int(ratios.size),
                "mean": float(ratios.mean()) if ratios.size else 0.0,
                "median": float(np.median(ratios)) if ratios.size else 0.0,
            }
        return result


def validate_dual_head(predictions: Iterable[Mapping[str, Any]], targets: Iterable[Mapping[str, Any]], *, hand_classes: int = 2) -> dict[str, Any]:
    """Convenience wrapper over :class:`DualHeadMetrics` for finite iterables."""
    metrics = DualHeadMetrics(hand_classes=hand_classes)
    for image_index, (prediction, target) in enumerate(zip(predictions, targets, strict=True)):
        metrics.update(prediction, target, image_index)
    return metrics.compute()
