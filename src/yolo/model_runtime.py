"""Reusable YOLO dual-head runtime model initialization and loss binding."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import ultralytics

from ..models.yolo_dual_head import build_losses, build_model, sync_heads
from .checkpoint import load_checkpoint


def initialize_model(
    config: dict[str, Any],
    device: torch.device,
    checkpoint_path: Path | None = None,
) -> tuple[Any, Any, Any, dict[str, Any]]:
    """Build dual-head model, sync heads, optionally load checkpoint, and build losses."""
    model_cfg = config["model"]
    if checkpoint_path is not None and "architecture" in model_cfg:
        model_source = str(model_cfg["architecture"])
    else:
        model_source = str(model_cfg["name"])

    person_keypoints = int(model_cfg["person_keypoints"])
    hand_keypoints = int(model_cfg["hand_keypoints"])
    model, audit = build_model(
        torch,
        ultralytics,
        model_name=model_source,
        person_keypoints=person_keypoints,
        hand_keypoints=hand_keypoints,
    )
    model = model.to(device)
    sync_heads(model, torch)
    if checkpoint_path is not None:
        load_checkpoint(checkpoint_path, device=device, model=model, strict=True)
    pose_loss, hand_loss = build_losses(model, torch)
    return model, pose_loss, hand_loss, audit
