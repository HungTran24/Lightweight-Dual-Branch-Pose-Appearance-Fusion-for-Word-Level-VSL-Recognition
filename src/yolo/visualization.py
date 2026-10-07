"""Overlay visualization for YOLO dual-head targets and predictions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

COCO_17_EDGES: tuple[tuple[int, int], ...] = (
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
)

HAND_21_EDGES: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
)

COLORS: dict[tuple[str, bool], tuple[int, int, int]] = {
    ("person", True): (0, 200, 0),
    ("left_hand", True): (230, 180, 0),
    ("right_hand", True): (0, 220, 200),
    ("person", False): (0, 100, 255),
    ("left_hand", False): (180, 0, 220),
    ("right_hand", False): (255, 50, 100),
}


def _to_numpy(val: Any) -> np.ndarray | None:
    if val is None:
        return None
    if hasattr(val, "detach"):
        val = val.detach().cpu().numpy()
    arr = np.asarray(val)
    return arr if arr.size > 0 else None


def _prepare_image(image: Any) -> np.ndarray:
    arr = _to_numpy(image)
    if arr is None:
        raise ValueError("Image cannot be empty")
    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[2] not in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    elif arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)

    if np.issubdtype(arr.dtype, np.floating):
        if float(arr.max()) <= 1.0 + 1e-5:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    elif arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)

    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR) if arr.shape[2] == 3 else arr


def _extract_items(
    data: Any,
    default_label: str,
    kpt_count: int,
    *,
    is_target: bool,
    conf_thresh: float = 0.25,
) -> list[dict[str, Any]]:
    if data is None:
        return []
    boxes, scores, classes, keypoints = None, None, None, None
    if isinstance(data, dict):
        boxes = _to_numpy(data.get("boxes", data.get("bbox")))
        scores = _to_numpy(data.get("confidence", data.get("conf", data.get("scores"))))
        classes = _to_numpy(data.get("classes", data.get("cls")))
        keypoints = _to_numpy(data.get("keypoints", data.get("person_keypoints", data.get("hand_keypoints"))))
    else:
        arr = _to_numpy(data)
        if arr is None:
            return []
        if arr.ndim == 1:
            if len(arr) >= 6:
                arr = arr[None, :]
            elif len(arr) == 4:
                boxes = arr[None, :]
        if arr.ndim == 2:
            if arr.shape[1] >= 6:
                boxes, scores, classes = arr[:, :4], arr[:, 4], arr[:, 5].astype(int)
                if arr.shape[1] > 6:
                    kpt_vals = arr[:, 6:]
                    dim = 3 if kpt_vals.shape[1] >= kpt_count * 3 else 2
                    keypoints = kpt_vals[:, : kpt_count * dim].reshape(-1, kpt_count, dim)
            elif arr.shape[1] == 4:
                boxes = arr
            elif arr.shape[0] == kpt_count and arr.shape[1] in (2, 3):
                keypoints = arr[None, ...]
        elif arr.ndim == 3 and arr.shape[1] == kpt_count:
            keypoints = arr

    if keypoints is not None and keypoints.ndim == 2 and keypoints.shape[0] == kpt_count:
        keypoints = keypoints[None, ...]
    if boxes is None and keypoints is None:
        return []

    count = len(boxes) if boxes is not None else len(keypoints)  # type: ignore[arg-type]
    items = []
    for i in range(count):
        box = boxes[i].tolist() if boxes is not None and i < len(boxes) else None
        score = float(scores[i]) if scores is not None and i < len(scores) else None
        cls_id = int(classes[i]) if classes is not None and i < len(classes) else 0

        if not is_target and score is not None and score < conf_thresh:
            continue

        label = ("left_hand" if cls_id == 0 else "right_hand") if default_label.startswith("hand") else default_label
        kpts = None
        if keypoints is not None and i < len(keypoints):
            kpts = keypoints[i]
            if kpts.ndim == 3 and len(kpts) <= 2:
                kpts = kpts[cls_id if cls_id < len(kpts) else 0]

        items.append({"box": box, "label": label, "score": score, "keypoints": kpts, "is_target": is_target})
    return items


def _draw_box(canvas: np.ndarray, box: list[float], label: str, score: float | None, color: tuple[int, int, int], is_target: bool) -> None:
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 1 if is_target else 2)
    text = f"{label} [GT]" if is_target else (f"{label} {score:.2f}" if score is not None else label)
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
    text_y = max(y1, th + baseline + 2)
    cv2.rectangle(canvas, (x1, text_y - th - 2), (x1 + tw + 2, text_y + baseline), color, -1)
    text_color = (0, 0, 0) if sum(color) > 380 else (255, 255, 255)
    cv2.putText(canvas, text, (x1 + 1, text_y - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1, cv2.LINE_AA)


def _draw_kpts(canvas: np.ndarray, kpts: np.ndarray, edges: tuple[tuple[int, int], ...], color: tuple[int, int, int], is_target: bool) -> None:
    kpts_arr = np.asarray(kpts)
    if kpts_arr.ndim != 2:
        return
    valid = [(pt[0] > 0 or pt[1] > 0) and (len(pt) <= 2 or pt[2] > 0.05) for pt in kpts_arr]
    for u, v in edges:
        if u < len(kpts_arr) and v < len(kpts_arr) and valid[u] and valid[v]:
            p1 = (int(round(kpts_arr[u][0])), int(round(kpts_arr[u][1])))
            p2 = (int(round(kpts_arr[v][0])), int(round(kpts_arr[v][1])))
            cv2.line(canvas, p1, p2, color, 1 if is_target else 2, lineType=cv2.LINE_AA)
    for i, pt in enumerate(kpts_arr):
        if valid[i]:
            cv2.circle(canvas, (int(round(pt[0])), int(round(pt[1]))), 2 if is_target else 3, color, -1, lineType=cv2.LINE_AA)


def save_validation_overlay(
    path: str | Path | None,
    image: Any,
    person_target: Any = None,
    hands_target: Any = None,
    person_prediction_rows: Any = None,
    hand_prediction_rows: Any = None,
    *,
    confidence_threshold: float = 0.25,
) -> np.ndarray:
    """Render target vs prediction overlays on a letterboxed image and save to path."""
    canvas = _prepare_image(image).copy()
    items = [
        *_extract_items(person_target, "person", 17, is_target=True),
        *_extract_items(hands_target, "hands", 21, is_target=True),
        *_extract_items(person_prediction_rows, "person", 17, is_target=False, conf_thresh=confidence_threshold),
        *_extract_items(hand_prediction_rows, "hands", 21, is_target=False, conf_thresh=confidence_threshold),
    ]

    for is_tgt in (True, False):
        for it in items:
            if it["is_target"] != is_tgt:
                continue
            color = COLORS.get((it["label"], it["is_target"]), (0, 255, 0) if it["is_target"] else (0, 0, 255))
            if it["box"] is not None:
                _draw_box(canvas, it["box"], it["label"], it["score"], color, it["is_target"])
            if it["keypoints"] is not None:
                edges = COCO_17_EDGES if it["label"] == "person" else HAND_21_EDGES
                _draw_kpts(canvas, it["keypoints"], edges, color, it["is_target"])

    if path is not None:
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(out_path), canvas):
            raise RuntimeError(f"Cannot write overlay: {out_path}")

    return canvas
