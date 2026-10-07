"""YOLO dual-head reusable pipeline, evaluation, and experiment orchestration."""

from __future__ import annotations

from .checkpoint import (
    download_checkpoint,
    load_checkpoint,
    save_checkpoint,
    upload_checkpoint,
)
from .dataset import YoloDataset, epoch_indices, loader, make_targets
from .evaluation import evaluate_yolo
from .experiment import run_materialized, run_yolo
from .fit import fit_yolo
from .materialize import materialize
from .probe import probe_batch_capacity
from .selection import canonical_selection
from .training import (
    batch_loss,
    build_yolo_optimizer,
    build_yolo_scheduler,
    train_step,
    validation_loss,
    yolo_lr_factor,
)

__all__ = [
    "YoloDataset",
    "batch_loss",
    "build_yolo_optimizer",
    "build_yolo_scheduler",
    "canonical_selection",
    "download_checkpoint",
    "epoch_indices",
    "evaluate_yolo",
    "fit_yolo",
    "load_checkpoint",
    "loader",
    "make_targets",
    "materialize",
    "probe_batch_capacity",
    "run_materialized",
    "run_yolo",
    "save_checkpoint",
    "train_step",
    "upload_checkpoint",
    "validation_loss",
    "yolo_lr_factor",
]
