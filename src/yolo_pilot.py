"""Compatibility shim for legacy YOLO pilot imports."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .models.yolo_dual_head import build_losses, build_model, decode, sync_heads
from .yolo.dataset import YoloDataset, make_targets
from .yolo.experiment import run_materialized, run_yolo


def run(
    data_root: Path,
    output_root: Path,
    config: dict[str, Any],
    *,
    device: str | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    """Compatibility wrapper executing YOLO pipeline with explicit data and output paths."""
    return run_materialized(
        config,
        data_root=data_root,
        output_root=output_root,
        device=device,
        resume=resume,
    )


__all__ = [
    "YoloDataset",
    "build_losses",
    "build_model",
    "decode",
    "make_targets",
    "run",
    "run_materialized",
    "run_yolo",
    "sync_heads",
]
