"""Multi-epoch training loop and checkpoint cadence for YOLO dual-head model."""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

from ..integrations.wandb import log_metrics
from .checkpoint import load_checkpoint, save_checkpoint, upload_checkpoint
from .dataset import YoloDataset, epoch_indices, loader, make_targets
from .training import train_step

_VALIDATION_GROUPS = (
    "person_bbox",
    "person_pose17_oks",
    "hand_bbox",
    "hand_pose21_oks",
)


def fit_yolo(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    pose_loss: Any,
    hand_loss: Any,
    *,
    data_root: Path,
    output_root: Path,
    config: dict[str, Any],
    target_device: torch.device,
    seed: int,
    resume: bool = False,
    scheduler: Any | None = None,
    validate: Callable[[torch.nn.Module], dict[str, Any]] | None = None,
    wandb_run: Any | None = None,
) -> dict[str, Any]:
    """Train, validate per epoch, early-stop, and checkpoint a YOLO dual-head model."""
    data_cfg = config["data"]
    frames = int(data_cfg["frames_per_video"])
    train_images = int(data_cfg["train_videos"]) * frames
    image_size = int(config["model"]["image_size"])
    train_cfg = config["train"]
    epochs = int(train_cfg.get("epochs", 1))
    batch_size = int(train_cfg["batch_size"])
    workers = int(train_cfg.get("workers", 8))
    amp = bool(train_cfg.get("amp", True))
    patience = int(train_cfg.get("patience", 0))

    materialization = json.loads((data_root / "materialization.json").read_text(encoding="utf-8"))
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    identity = f"{materialization['identity']}:{config_hash}"

    checkpoint_file = output_root / "last.pt"
    best_checkpoint = output_root / "best.pt"
    epoch_batches = math.ceil(train_images / batch_size)
    start_epoch = 0
    start_sample = 0
    loss_sum = 0.0
    previous_seconds = 0.0
    component_sums: dict[str, float] = {}
    best_metric = float("-inf")
    best_epoch = -1
    epochs_without_improvement = 0
    completed_epoch = -1
    stopped_early = False

    if resume and checkpoint_file.is_file():
        state = load_checkpoint(
            checkpoint_file,
            device=target_device,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            scheduler=scheduler,
            strict=True,
            expected_identity=identity,
        )
        if scheduler is not None and state.get("scheduler") is None:
            raise RuntimeError("E005 resume checkpoint is missing scheduler state")
        start_epoch = int(state.get("epoch", 0))
        start_sample = int(state.get("next_sample", 0))
        loss_sum = float(state.get("loss_sum", 0.0))
        component_sums = {
            str(name): float(value)
            for name, value in state.get("component_sums", {}).items()
        }
        previous_seconds = float(state.get("train_seconds", 0.0))
        best_metric = float(state.get("best_metric", float("-inf")))
        best_epoch = int(state.get("best_epoch", -1))
        epochs_without_improvement = int(state.get("epochs_without_improvement", 0))
        completed_epoch = int(state.get("completed_epoch", -1))
        if start_sample >= train_images:
            start_epoch += 1
            start_sample = 0
        if scheduler is not None:
            expected_step = start_epoch * epoch_batches + math.ceil(start_sample / batch_size)
            if int(scheduler.last_epoch) != expected_step:
                raise RuntimeError("Checkpoint scheduler state does not match global sample position")
        if patience > 0 and epochs_without_improvement >= patience and start_sample == 0 and completed_epoch >= 0:
            stopped_early = True
            start_epoch = epochs

    model.train()
    if target_device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(target_device)
    started = time.perf_counter()
    upload_seconds = 0.0
    batches = start_epoch * epoch_batches + math.ceil(start_sample / batch_size)
    total_processed = start_epoch * train_images + start_sample
    validation_history: list[dict[str, Any]] = []

    log_every = int(train_cfg.get("log_every_batches", 100))
    print_every = int(train_cfg.get("print_every_batches", 1000))
    checkpoint_every = int(train_cfg.get("checkpoint_every_batches", 2000))
    shuffle = bool(train_cfg.get("shuffle", True))

    def checkpoint_state(
        epoch: int,
        next_sample: int,
        train_seconds: float,
        role: str,
    ) -> dict[str, Any]:
        return {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "epoch": epoch,
            "next_sample": next_sample,
            "loss_sum": loss_sum,
            "component_sums": component_sums,
            "train_seconds": train_seconds,
            "identity": identity,
            "checkpoint_role": role,
            "best_metric": best_metric,
            "best_epoch": best_epoch,
            "epochs_without_improvement": epochs_without_improvement,
            "completed_epoch": completed_epoch,
        }

    def log_validation(epoch: int, metrics: dict[str, Any]) -> None:
        values: dict[str, float] = {
            "train/epoch": float(epoch + 1),
            "train/lr_shared_person": float(optimizer.param_groups[0]["lr"]),
            "train/lr_hands": float(
                optimizer.param_groups[1]["lr"]
                if len(optimizer.param_groups) > 1
                else optimizer.param_groups[0]["lr"]
            ),
            "early_stop/best_metric": best_metric,
            "early_stop/best_epoch": float(best_epoch),
            "early_stop/epochs_without_improvement": float(epochs_without_improvement),
        }
        if "validation_loss_per_image" in metrics:
            values["val/loss_per_image"] = float(metrics["validation_loss_per_image"])
        for comp_name, comp_value in metrics.get("loss_components", {}).items():
            values[f"val/{comp_name}"] = float(comp_value)
        for group_name in _VALIDATION_GROUPS:
            group = metrics.get(group_name)
            if not isinstance(group, dict):
                continue
            for metric_name in ("map50", "map50_95"):
                metric_value = group.get(metric_name)
                if isinstance(metric_value, (int, float)):
                    values[f"val/{group_name}_{metric_name}"] = float(metric_value)
        log_metrics(wandb_run, values, step=batches)

    aug_cfg = train_cfg.get("augmentation")
    aug_dict = (
        aug_cfg
        if isinstance(aug_cfg, dict)
        else ({"enabled": True} if aug_cfg is True else None)
    )

    for epoch in range(start_epoch, epochs):
        sample_offset = start_sample if epoch == start_epoch else 0
        train_indices = epoch_indices(
            train_images,
            seed=seed,
            epoch=epoch,
            start_sample=sample_offset,
            shuffle=shuffle,
        )
        train_dataset = YoloDataset(
            data_root,
            "train",
            images=train_images,
            frames_per_video=frames,
            image_size=image_size,
            indices=train_indices,
            augmentation=aug_dict,
            seed=seed,
            epoch=epoch,
        )
        train_loader = loader(train_dataset, batch_size=batch_size, workers=workers)

        for batch in train_loader:
            pose, hands, _ = make_targets(batch, device=target_device, image_size=image_size)
            loss, batch_components = train_step(
                model,
                optimizer,
                scaler,
                pose_loss,
                hand_loss,
                pose,
                hands,
                amp=amp,
                sample_index=total_processed,
                return_components=True,
            )
            if scheduler is not None:
                scheduler.step()
            b_len = len(batch["image"])
            sample_offset += b_len
            total_processed += b_len
            batches += 1
            loss_sum += float(loss.detach().item())
            for name, value in batch_components.items():
                component_sums[name] = component_sums.get(name, 0.0) + value * b_len

            elapsed = previous_seconds + time.perf_counter() - started - upload_seconds
            throughput = (total_processed - (start_epoch * train_images + start_sample)) / max(
                elapsed - previous_seconds, 1e-9
            )

            if batches % log_every == 0:
                shared_lr = float(optimizer.param_groups[0]["lr"])
                hands_lr = float(
                    optimizer.param_groups[1]["lr"] if len(optimizer.param_groups) > 1 else shared_lr
                )
                log_metrics(
                    wandb_run,
                    {
                        "train/loss_per_image": loss_sum / max(total_processed, 1),
                        "train/images_per_second": throughput,
                        "train/images_processed": float(total_processed),
                        "train/epoch": epoch + (sample_offset / train_images),
                        "train/learning_rate": shared_lr,
                        "train/lr_shared_person": shared_lr,
                        "train/lr_hands": hands_lr,
                        **{f"train/{name}": value for name, value in batch_components.items()},
                    },
                    batches,
                )

            if batches % print_every == 0:
                print(
                    json.dumps(
                        {
                            "phase": "train",
                            "epoch": epoch,
                            "batch": batches,
                            "samples": sample_offset,
                            "total_samples": train_images,
                            "images_per_second": round(throughput, 2),
                        }
                    ),
                    flush=True,
                )

            if batches % checkpoint_every == 0:
                save_checkpoint(
                    checkpoint_file,
                    state=checkpoint_state(epoch, sample_offset, elapsed, "last"),
                )

        improved = False
        epoch_elapsed = previous_seconds + time.perf_counter() - started - upload_seconds
        validation: dict[str, Any] | None = validate(model) if validate is not None else None
        if validation is not None:
            metric = float(validation["hand_pose21_oks"]["map50_95"])
            if not math.isfinite(metric):
                raise RuntimeError("Primary validation metric is not finite")
            validation_history.append(
                {
                    "epoch": epoch,
                    "validation_loss_per_image": validation.get("validation_loss_per_image"),
                    "hand_pose21_oks_map50_95": metric,
                }
            )
            improved = metric > best_metric
            if improved:
                best_metric = metric
                best_epoch = epoch
                epochs_without_improvement = 0
                save_checkpoint(
                    best_checkpoint,
                    state=checkpoint_state(epoch, train_images, epoch_elapsed, "best"),
                )
            else:
                epochs_without_improvement += 1
            log_validation(epoch, validation)

        completed_epoch = epoch
        save_checkpoint(
            checkpoint_file,
            state=checkpoint_state(epoch, sample_offset, epoch_elapsed, "last"),
        )
        up_start = time.perf_counter()
        upload_checkpoint(config, output_root, batch=batches, include_best=improved)
        upload_seconds += time.perf_counter() - up_start
        model.train()

        if validation is not None and patience > 0 and epochs_without_improvement >= patience:
            stopped_early = True
            break

    train_seconds = previous_seconds + time.perf_counter() - started - upload_seconds
    final_epoch = completed_epoch if completed_epoch >= 0 else epochs - 1
    save_checkpoint(
        checkpoint_file,
        state=checkpoint_state(final_epoch, train_images, train_seconds, "last"),
    )

    if target_device.type == "cuda" and torch.cuda.is_available():
        train_peak = {
            "allocated_bytes": int(torch.cuda.max_memory_allocated(target_device)),
            "reserved_bytes": int(torch.cuda.max_memory_reserved(target_device)),
        }
    else:
        train_peak = {"allocated_bytes": 0, "reserved_bytes": 0}

    shared_lr = float(optimizer.param_groups[0]["lr"])
    hands_lr = float(
        optimizer.param_groups[1]["lr"]
        if len(optimizer.param_groups) > 1
        else optimizer.param_groups[0]["lr"]
    )
    return {
        "seconds": train_seconds,
        "images_per_second": total_processed / max(train_seconds, 1e-9),
        "loss_per_image": loss_sum / max(total_processed, 1),
        "loss_components": {
            name: value / max(total_processed, 1)
            for name, value in component_sums.items()
        },
        "peak_vram": train_peak,
        "learning_rates": {"shared_person": shared_lr, "hands": hands_lr},
        "checkpoint_file": checkpoint_file,
        "best_checkpoint": best_checkpoint,
        "batches": batches,
        "last_epoch": final_epoch,
        "epochs_completed": completed_epoch + 1 if completed_epoch >= 0 else 0,
        "best_metric": best_metric,
        "best_epoch": best_epoch,
        "epochs_without_improvement": epochs_without_improvement,
        "stopped_early": stopped_early,
        "validation_history": validation_history,
    }
