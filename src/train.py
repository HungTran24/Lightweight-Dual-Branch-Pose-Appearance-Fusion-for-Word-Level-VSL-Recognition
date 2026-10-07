from __future__ import annotations

import math
import random
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from .checkpoint import save_best, save_last
from .evaluate import evaluate
from .metrics import new_train_metric


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def build_optimizer_and_scheduler(
    model: torch.nn.Module, train_config: dict[str, Any]
) -> tuple[torch.optim.Optimizer, Any]:
    name = train_config["optimizer"]
    lr = float(train_config["learning_rate"])
    weight_decay = float(train_config["weight_decay"])
    if name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {name}")
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )
    return optimizer, scheduler


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    *,
    num_classes: int,
    device: torch.device,
    grad_clip_norm: float,
) -> dict[str, float]:
    model.train()
    top1 = new_train_metric(num_classes, device)
    loss_total = 0.0
    sample_total = 0
    for features, labels, frame_mask in loader:
        features = features.to(device)
        labels = labels.to(device)
        frame_mask = frame_mask.to(device)
        optimizer.zero_grad()
        logits = model(features, frame_mask)
        loss = torch.nn.functional.cross_entropy(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()
        top1.update(logits.detach(), labels)
        loss_total += float(loss.item()) * len(labels)
        sample_total += len(labels)
    return {
        "loss": loss_total / max(sample_total, 1),
        "top1": float(top1.compute().detach().cpu().item()),
    }


def fit(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    *,
    epochs: int,
    min_epochs: int = 0,
    patience: int = 15,
    num_classes: int,
    device: torch.device,
    grad_clip_norm: float,
    output_dir: Path,
    data_revision: str,
    on_epoch_end: Callable[[int, dict[str, float], dict[str, float], float, float], None]
    | None = None,
    resume_state: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], int]:
    """Train model over epochs with validation, early stopping, and atomic checkpointing."""
    best: dict[str, Any] = {"epoch": 0, "top1": -1.0, "loss": float("inf")}
    bad_epochs = 0
    start_epoch = 1
    epochs_trained = 0

    if resume_state is not None:
        best = resume_state["best"]
        bad_epochs = int(resume_state["bad_epochs"])
        epochs_trained = int(resume_state["epoch"])
        start_epoch = epochs_trained + 1

    for epoch in range(start_epoch, epochs + 1):
        started = time.perf_counter()
        epochs_trained = epoch
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            num_classes=num_classes,
            device=device,
            grad_clip_norm=grad_clip_norm,
        )
        val_metrics, _, _ = evaluate(
            model,
            val_loader,
            num_classes=num_classes,
            device=device,
        )
        scheduler.step(val_metrics["loss"])
        elapsed = time.perf_counter() - started
        lr = float(optimizer.param_groups[0]["lr"])

        improved = val_metrics["top1"] > best["top1"] or (
            math.isclose(val_metrics["top1"], best["top1"])
            and val_metrics["loss"] < best["loss"]
        )
        if improved:
            best = {"epoch": epoch, "top1": val_metrics["top1"], "loss": val_metrics["loss"]}
            bad_epochs = 0
            save_best(output_dir / "best.pt", model, epoch=epoch, metrics=val_metrics)
        else:
            bad_epochs += 1

        save_last(
            output_dir / "last.pt",
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best": best,
                "bad_epochs": bad_epochs,
                "data_revision": data_revision,
            },
        )

        if on_epoch_end is not None:
            on_epoch_end(epoch, train_metrics, val_metrics, lr, elapsed)

        if epoch >= min_epochs and patience > 0 and bad_epochs >= patience:
            break

    return best, epochs_trained
