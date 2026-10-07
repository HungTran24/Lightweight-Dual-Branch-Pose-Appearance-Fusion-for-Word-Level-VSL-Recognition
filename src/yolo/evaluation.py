"""Reusable validation and full evaluation for YOLO dual-head models."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from ..models.yolo_dual_head import decode
from .dataset import make_targets
from .metrics import DualHeadMetrics
from .training import forward_loss
from .visualization import save_validation_overlay


def evaluate_yolo(
    model: torch.nn.Module,
    pose_loss: Any,
    hand_loss: Any,
    data_loader: DataLoader[dict[str, torch.Tensor]],
    *,
    config: dict[str, Any],
    device: torch.device | str = "cuda",
    amp: bool = False,
    overlay_dir: Path | None = None,
) -> dict[str, Any]:
    """Run full validation over DataLoader, computing loss and DualHeadMetrics."""
    target_device = torch.device(device) if isinstance(device, str) else device
    model.eval()

    eval_cfg = config.get("evaluation", {})
    compare_hand_oks_areas = bool(eval_cfg.get("compare_hand_oks_areas", False))
    metrics = DualHeadMetrics(
        hand_classes=2,
        compare_hand_oks_areas=compare_hand_oks_areas,
    )
    nms_conf = float(eval_cfg.get("metric_nms_confidence", eval_cfg.get("nms_confidence", 0.001)))
    nms_iou = float(eval_cfg.get("metric_nms_iou", eval_cfg.get("nms_iou", 0.7)))
    max_det = int(eval_cfg.get("max_detections", 300))
    image_size = int(config["model"]["image_size"])

    if target_device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(target_device)

    total_loss_sum = 0.0
    loss_components: dict[str, float] = {}
    total_images = 0
    image_index = 0
    overlay_limit = int(eval_cfg.get("overlay_samples", 0))
    overlay_paths: list[str] = []
    start_time = time.perf_counter()

    with torch.inference_mode():
        for batch in data_loader:
            pose, hands, _ = make_targets(batch, device=target_device, image_size=image_size)
            output, loss, batch_components = forward_loss(
                model, pose_loss, hand_loss, pose, hands, amp=amp
            )
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("Non-finite validation loss encountered")

            batch_len = len(batch["image"])
            total_loss_sum += float(loss.item())
            for name, value in batch_components.items():
                loss_components[name] = loss_components.get(name, 0.0) + value * batch_len
            total_images += batch_len

            preds = decode(
                output,
                confidence=nms_conf,
                iou=nms_iou,
                max_detections=max_det,
            )

            boxes = batch["boxes"]
            classes = batch["classes"]
            person_keypoints = batch["person_keypoints"]
            hand_keypoints = batch["hand_keypoints"]
            for i in range(batch_len):
                person_valid = classes[i, :1] == 0
                hand_valid = classes[i, 1:] >= 1
                target_i = {
                    "person": {
                        "boxes": boxes[i, :1][person_valid],
                        "classes": classes[i, :1][person_valid],
                        "keypoints": person_keypoints[i : i + 1][person_valid],
                    },
                    "hands": {
                        "boxes": boxes[i, 1:][hand_valid],
                        "classes": classes[i, 1:][hand_valid] - 1,
                        "keypoints": hand_keypoints[i][hand_valid],
                    },
                }
                metrics.update(preds[i], target_i, image_index)
                if overlay_dir is not None and len(overlay_paths) < overlay_limit:
                    overlay_path = overlay_dir / f"val_{image_index:06d}.jpg"
                    save_validation_overlay(
                        overlay_path,
                        batch["image"][i],
                        person_target=target_i["person"],
                        hands_target=target_i["hands"],
                        person_prediction_rows=preds[i]["person"],
                        hand_prediction_rows=preds[i]["hands"],
                        confidence_threshold=float(eval_cfg.get("nms_confidence", 0.25)),
                    )
                    overlay_paths.append(str(overlay_path.relative_to(overlay_dir.parent)))
                image_index += 1

    if target_device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(target_device)
        peak_alloc = int(torch.cuda.max_memory_allocated(target_device))
        peak_res = int(torch.cuda.max_memory_reserved(target_device))
    else:
        peak_alloc = 0
        peak_res = 0

    duration_s = time.perf_counter() - start_time
    throughput = total_images / max(duration_s, 1e-9)
    metric_results = metrics.compute()

    result = {
        "validation_loss_per_image": total_loss_sum / max(total_images, 1),
        "loss_components": {name: value / max(total_images, 1) for name, value in loss_components.items()},
        "total_images": total_images,
        "duration_seconds": duration_s,
        "end_to_end_images_per_second": throughput,
        "images_per_second": throughput,
        "peak_allocated_vram_bytes": peak_alloc,
        "peak_reserved_vram_bytes": peak_res,
        "person_bbox": metric_results["person_bbox"],
        "person_pose17_oks": metric_results["person_pose17_oks"],
        "hand_bbox": metric_results["hand_bbox"],
        "hand_pose21_oks": metric_results["hand_pose21_oks"],
        "overlays": overlay_paths,
    }
    if compare_hand_oks_areas:
        result.update(
            {
                "hand_pose21_oks_hand_area": metric_results["hand_pose21_oks_hand_area"],
                "hand_pose21_oks_person_area": metric_results["hand_pose21_oks_person_area"],
                "hand_area_over_person_area": metric_results["hand_area_over_person_area"],
            }
        )
    return result
