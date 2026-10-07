"""Reusable end-to-end YOLO experiment orchestration (probe and train operations)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import ultralytics

from ..artifacts import sha256
from ..integrations.wandb import finish_run, log_metrics, start_run
from ..models.yolo_dual_head import build_losses
from .checkpoint import download_checkpoint, upload_checkpoint
from .dataset import YoloDataset, epoch_indices, loader
from .evaluation import evaluate_yolo
from .fit import fit_yolo
from .materialize import materialize
from .model_runtime import initialize_model
from .probe import probe_batch_capacity
from .training import build_yolo_optimizer, build_yolo_scheduler


def _run_probe(
    config: dict[str, Any],
    data_root: Path,
    output_root: Path,
    target_device: torch.device,
    seed: int,
) -> dict[str, Any]:
    """Execute operation: probe (pinned checkpoint download, capacity probe, reload, full validation)."""
    if target_device.type != "cuda":
        raise RuntimeError("YOLO capacity probe requires CUDA")

    ckpt_cfg = config.get("checkpoint")
    if not isinstance(ckpt_cfg, dict):
        raise ValueError("Probe operation requires a checkpoint configuration in YAML")

    checkpoint_path = download_checkpoint(
        repo_id=str(ckpt_cfg["repo_id"]),
        revision=str(ckpt_cfg["revision"]),
        path_in_repo=str(ckpt_cfg["path"]),
        cache_dir=output_root.parent / "cache",
    )

    data_cfg = config["data"]
    frames = int(data_cfg["frames_per_video"])
    train_images = int(data_cfg["train_videos"]) * frames
    val_images = int(data_cfg["val_videos"]) * frames
    image_size = int(config["model"]["image_size"])
    amp = bool(config.get("train", {}).get("amp", True))

    probe_cfg = config.get("probe", {})
    warmup_steps = int(probe_cfg.get("warmup_steps", 10))
    measured_steps = int(probe_cfg.get("measured_steps", 100))
    probe_batch_size = int(config.get("train", {}).get("batch_size", 64))
    workers = int(config.get("train", {}).get("workers", 8))

    env_info = {
        "device": torch.cuda.get_device_name(target_device),
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
    }
    wandb_run = start_run(config, env_info)
    probe_summary: dict[str, Any] | None = None
    probe_error: str | None = None

    model, pose_loss, hand_loss, _ = initialize_model(config, target_device, checkpoint_path)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.get("train", {}).get("learning_rate", 1e-4)),
        weight_decay=float(config.get("train", {}).get("weight_decay", 0.0)),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    train_dataset = YoloDataset(
        data_root,
        "train",
        images=train_images,
        frames_per_video=frames,
        image_size=image_size,
        indices=epoch_indices(train_images, seed=seed, epoch=0),
    )
    try:
        probe_summary = probe_batch_capacity(
            model,
            optimizer,
            scaler,
            pose_loss,
            hand_loss,
            train_dataset,
            batch_size=probe_batch_size,
            image_size=image_size,
            warmup_steps=warmup_steps,
            measured_steps=measured_steps,
            workers=workers,
            amp=amp,
            device=target_device,
        )
    except torch.OutOfMemoryError as exc:
        probe_error = str(exc)
    finally:
        del model, optimizer, scaler, pose_loss, hand_loss
        torch.cuda.empty_cache()

    # Discard and reload pristine checkpoint for validation
    val_model, val_pose_loss, val_hand_loss, model_audit = initialize_model(
        config, target_device, checkpoint_path
    )
    val_batch_size = (
        8
        if probe_error is not None
        else int(config.get("evaluation", {}).get("batch_size", probe_batch_size))
    )
    val_dataset = YoloDataset(
        data_root, "val", images=val_images, frames_per_video=frames, image_size=image_size
    )
    val_loader = loader(val_dataset, batch_size=val_batch_size, workers=workers)

    val_metrics = evaluate_yolo(
        val_model,
        val_pose_loss,
        val_hand_loss,
        val_loader,
        config=config,
        device=target_device,
        amp=amp,
        overlay_dir=output_root / "overlays",
    )

    summary: dict[str, Any] = {
        "status": "completed" if probe_error is None else "completed_with_probe_error",
        "operation": "probe",
        "environment": env_info,
        "settings": config,
        "checkpoint": {
            "source": ckpt_cfg,
            "local_path": str(checkpoint_path),
            "strict_reload": True,
        },
        "model_audit": {
            **model_audit,
            "parameters": sum(p.numel() for p in val_model.parameters()),
            "trainable_parameters": sum(
                p.numel() for p in val_model.parameters() if p.requires_grad
            ),
        },
        "probe": probe_summary if probe_summary is not None else {"error": probe_error},
        "validation": val_metrics,
    }

    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if probe_summary is not None:
        log_metrics(
            wandb_run,
            {
                "probe/compute_images_per_second": probe_summary["compute_images_per_second"],
                "probe/end_to_end_images_per_second": probe_summary["end_to_end_images_per_second"],
                "probe/peak_allocated_vram_mb": probe_summary["peak_allocated_vram_mb"],
                "probe/compute_step_p50_ms": probe_summary["compute_step_p50_ms"],
            },
            step=1,
        )

    log_metrics(
        wandb_run,
        {
            "val/loss_per_image": val_metrics["validation_loss_per_image"],
            "val/end_to_end_images_per_second": val_metrics["end_to_end_images_per_second"],
            **{f"val/{name}": value for name, value in val_metrics["loss_components"].items()},
            "val/person_bbox_map50": val_metrics["person_bbox"]["map50"],
            "val/person_bbox_map50_95": val_metrics["person_bbox"]["map50_95"],
            "val/person_pose17_oks_map50": val_metrics["person_pose17_oks"]["map50"],
            "val/person_pose17_oks_map50_95": val_metrics["person_pose17_oks"]["map50_95"],
            "val/hand_bbox_map50": val_metrics["hand_bbox"]["map50"],
            "val/hand_bbox_map50_95": val_metrics["hand_bbox"]["map50_95"],
            "val/hand_pose21_oks_map50": val_metrics["hand_pose21_oks"]["map50"],
            "val/hand_pose21_oks_map50_95": val_metrics["hand_pose21_oks"]["map50_95"],
        },
        step=2,
    )
    finish_run(wandb_run, summary)
    return summary


def _run_evaluate(
    config: dict[str, Any],
    data_root: Path,
    output_root: Path,
    target_device: torch.device,
) -> dict[str, Any]:
    """Evaluate a pinned checkpoint without training or changing checkpoint selection."""
    ckpt_cfg = config["checkpoint"]
    checkpoint_path = download_checkpoint(
        repo_id=str(ckpt_cfg["repo_id"]),
        revision=str(ckpt_cfg["revision"]),
        path_in_repo=str(ckpt_cfg["path"]),
        cache_dir=output_root.parent / "cache",
    )
    model, pose_loss, hand_loss, model_audit = initialize_model(
        config, target_device, checkpoint_path
    )
    data_cfg = config["data"]
    frames = int(data_cfg["frames_per_video"])
    val_images = int(data_cfg["val_videos"]) * frames
    train_cfg = config["train"]
    val_dataset = YoloDataset(
        data_root,
        "val",
        images=val_images,
        frames_per_video=frames,
        image_size=int(config["model"]["image_size"]),
    )
    val_loader = loader(
        val_dataset,
        batch_size=int(config["evaluation"]["batch_size"]),
        workers=int(train_cfg.get("workers", 8)),
    )
    env_info = {
        "device": torch.cuda.get_device_name(target_device)
        if target_device.type == "cuda"
        else "cpu",
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
    }
    wandb_run = start_run(config, env_info)
    val_metrics = evaluate_yolo(
        model,
        pose_loss,
        hand_loss,
        val_loader,
        config=config,
        device=target_device,
        amp=bool(train_cfg.get("amp", True)),
    )
    summary = {
        "status": "completed",
        "operation": "evaluate",
        "environment": env_info,
        "settings": config,
        "checkpoint": {
            "source": ckpt_cfg,
            "sha256": sha256(checkpoint_path),
            "strict_reload": True,
        },
        "metric_protocol": {
            "prediction_passes": 1,
            "hand_area": "tight hand bbox area * 0.53",
            "person_area": "person bbox area * 0.53",
            "person_area_note": "COCO-WholeBody-style approximation, not official segmentation area",
            "same_predictions": True,
        },
        "model_audit": {
            **model_audit,
            "parameters": sum(p.numel() for p in model.parameters()),
        },
        "validation": val_metrics,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log_metrics(
        wandb_run,
        {
            "val/hand_pose21_oks_hand_area_map50": val_metrics["hand_pose21_oks_hand_area"][
                "map50"
            ],
            "val/hand_pose21_oks_hand_area_map50_95": val_metrics["hand_pose21_oks_hand_area"][
                "map50_95"
            ],
            "val/hand_pose21_oks_person_area_map50": val_metrics["hand_pose21_oks_person_area"][
                "map50"
            ],
            "val/hand_pose21_oks_person_area_map50_95": val_metrics["hand_pose21_oks_person_area"][
                "map50_95"
            ],
        },
        step=0,
    )
    finish_run(wandb_run, summary)
    return summary


def _run_train(
    config: dict[str, Any],
    data_root: Path,
    output_root: Path,
    target_device: torch.device,
    seed: int,
    *,
    resume: bool = False,
) -> dict[str, Any]:
    """Execute operation: train (multi-epoch training loop, resume, strict reload, and validation)."""
    env_info = {
        "device": torch.cuda.get_device_name(target_device)
        if target_device.type == "cuda"
        else "cpu",
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
    }
    wandb_run = start_run(config, env_info)

    resume_checkpoint = (
        output_root / "last.pt" if resume and (output_root / "last.pt").is_file() else None
    )
    model, pose_loss, hand_loss, _ = initialize_model(config, target_device, resume_checkpoint)
    train_config = config["train"]
    if bool(train_config.get("discriminative_lr", False)):
        optimizer = build_yolo_optimizer(model, train_config)
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(train_config.get("learning_rate", 1e-4)),
            weight_decay=float(train_config.get("weight_decay", 0.0)),
        )
    train_images = int(config["data"]["train_videos"]) * int(config["data"]["frames_per_video"])
    steps_per_epoch = (train_images + int(train_config["batch_size"]) - 1) // int(
        train_config["batch_size"]
    )
    scheduler = (
        build_yolo_scheduler(optimizer, train_config, steps_per_epoch=steps_per_epoch)
        if train_config.get("lr_schedule") == "warmup_cosine"
        else None
    )
    scaler = torch.amp.GradScaler(
        target_device.type,
        enabled=bool(train_config.get("amp", True)) and target_device.type == "cuda",
    )

    output_root.mkdir(parents=True, exist_ok=True)
    data_cfg = config["data"]
    frames = int(data_cfg["frames_per_video"])
    train_images = int(data_cfg["train_videos"]) * frames
    val_images = int(data_cfg["val_videos"]) * frames
    val_dataset = YoloDataset(
        data_root,
        "val",
        images=val_images,
        frames_per_video=frames,
        image_size=int(config["model"]["image_size"]),
    )
    val_batch_size = int(
        config.get("evaluation", {}).get("batch_size", config["train"]["batch_size"])
    )
    val_loader = loader(
        val_dataset, batch_size=val_batch_size, workers=int(config["train"].get("workers", 8))
    )

    def validate_epoch(eval_model: torch.nn.Module) -> dict[str, Any]:
        eval_pose_loss, eval_hand_loss = build_losses(eval_model, torch)
        return evaluate_yolo(
            eval_model,
            eval_pose_loss,
            eval_hand_loss,
            val_loader,
            config=config,
            device=target_device,
            amp=bool(config["train"].get("amp", True)),
        )

    fit_summary = fit_yolo(
        model,
        optimizer,
        scaler,
        pose_loss,
        hand_loss,
        data_root=data_root,
        output_root=output_root,
        config=config,
        target_device=target_device,
        seed=seed,
        resume=resume,
        scheduler=scheduler,
        validate=validate_epoch,
        wandb_run=wandb_run,
    )

    del model, optimizer, scaler, scheduler
    if target_device.type == "cuda":
        torch.cuda.empty_cache()

    # Strict reload of the best checkpoint (or last checkpoint if not improved) before final validation
    best_file = fit_summary["best_checkpoint"]
    checkpoint_file = best_file if best_file.is_file() else fit_summary["checkpoint_file"]
    val_model, val_pose_loss, val_hand_loss, val_audit = initialize_model(
        config, target_device, checkpoint_file
    )

    val_metrics = evaluate_yolo(
        val_model,
        val_pose_loss,
        val_hand_loss,
        val_loader,
        config=config,
        device=target_device,
        amp=bool(config["train"].get("amp", True)),
        overlay_dir=output_root / "overlays",
    )

    summary = {
        "status": "completed",
        "operation": "train",
        "environment": env_info,
        "settings": config,
        "images": {"train": train_images, "val": val_images},
        "train": {
            "seconds": fit_summary["seconds"],
            "images_per_second": fit_summary["images_per_second"],
            "loss_per_image": fit_summary["loss_per_image"],
            "initialization": "fresh",
            "initialization_source": config["model"]["name"],
            "completed_prior_epochs": 0,
            "total_epochs": int(train_config.get("epochs", 1)),
            "patience": int(train_config.get("patience", 0)),
            "epochs_completed": fit_summary["epochs_completed"],
            "stopped_early": fit_summary["stopped_early"],
            "best_metric": fit_summary["best_metric"],
            "best_epoch": fit_summary["best_epoch"],
            "epochs_without_improvement": fit_summary["epochs_without_improvement"],
            "loss_components": fit_summary["loss_components"],
            "learning_rates": fit_summary["learning_rates"],
            "scheduler": train_config.get("lr_schedule"),
            "peak_vram": fit_summary["peak_vram"],
            "validation_history": fit_summary["validation_history"],
        },
        "validation": val_metrics,
        "model_audit": {
            **val_audit,
            "parameters": sum(p.numel() for p in val_model.parameters()),
            "trainable_parameters": sum(
                p.numel() for p in val_model.parameters() if p.requires_grad
            ),
        },
        "checkpoint": {
            "path": str(checkpoint_file),
            "best_path": str(fit_summary["best_checkpoint"]),
            "last_path": str(fit_summary["checkpoint_file"]),
            "bytes": checkpoint_file.stat().st_size,
            "strict_reload": True,
        },
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    upload_checkpoint(config, output_root, batch=fit_summary["batches"], final=True)
    log_metrics(
        wandb_run,
        {
            "train/lr_shared_person": fit_summary["learning_rates"]["shared_person"],
            "train/lr_hands": fit_summary["learning_rates"]["hands"],
            "val/loss_per_image": val_metrics["validation_loss_per_image"],
            "val/end_to_end_images_per_second": val_metrics["end_to_end_images_per_second"],
            **{f"val/{name}": value for name, value in val_metrics["loss_components"].items()},
            "val/person_bbox_map50": val_metrics["person_bbox"]["map50"],
            "val/person_bbox_map50_95": val_metrics["person_bbox"]["map50_95"],
            "val/person_pose17_oks_map50": val_metrics["person_pose17_oks"]["map50"],
            "val/person_pose17_oks_map50_95": val_metrics["person_pose17_oks"]["map50_95"],
            "val/hand_bbox_map50": val_metrics["hand_bbox"]["map50"],
            "val/hand_bbox_map50_95": val_metrics["hand_bbox"]["map50_95"],
            "val/hand_pose21_oks_map50": val_metrics["hand_pose21_oks"]["map50"],
            "val/hand_pose21_oks_map50_95": val_metrics["hand_pose21_oks"]["map50_95"],
        },
        step=fit_summary["batches"],
    )
    finish_run(wandb_run, summary)
    return summary


def _validate_runtime_schema(config: dict[str, Any]) -> None:
    expected_version = str(config["model"].get("ultralytics_version", ""))
    if expected_version and ultralytics.__version__ != expected_version:
        raise RuntimeError(
            f"Expected ultralytics {expected_version}, got {ultralytics.__version__}"
        )
    if (
        int(config["model"].get("person_keypoints", 0)) != 17
        or int(config["model"].get("hand_keypoints", 0)) != 21
    ):
        raise ValueError("Expected person_keypoints 17 and hand_keypoints 21")


def run_materialized(
    config: dict[str, Any],
    data_root: Path,
    output_root: Path,
    *,
    device: str | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    """Execute YOLO experiment from pre-materialized data root according to YAML operation."""
    _validate_runtime_schema(config)

    seed = int(config["experiment"]["seed"])
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    target_device = (
        torch.device(device)
        if device is not None
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    operation = config.get("experiment", {}).get("operation", "train")
    if operation == "probe":
        return _run_probe(config, data_root, output_root, target_device, seed)
    if operation == "evaluate":
        return _run_evaluate(config, data_root, output_root, target_device)
    if operation == "train":
        return _run_train(config, data_root, output_root, target_device, seed, resume=resume)
    raise ValueError(f"Unsupported experiment.operation: {operation}")


def run_yolo(
    config: dict[str, Any],
    project_root: Path,
    device: str | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    """Materialize canonical data and execute YOLO experiment."""
    _validate_runtime_schema(config)
    project_root = project_root.resolve()
    output_cfg = config["output"]
    root_path = project_root / output_cfg["root"]
    run_root = root_path / output_cfg.get("run_dir", "run")
    if run_root.is_dir() and any(run_root.iterdir()) and not resume:
        raise FileExistsError(
            f"Output directory '{run_root}' exists and is not empty; pass --resume or choose a new run directory"
        )

    if config.get("experiment", {}).get("operation") in {"extract", "extract_hand_rgb"}:
        from .extractor import extract_pose59

        return extract_pose59(config, project_root, run_root, device=device)

    data_root = root_path / output_cfg.get("data_dir", "data")
    materialize(data_root, project_root / "cache", project_root, config)
    return run_materialized(
        config, data_root=data_root, output_root=run_root, device=device, resume=resume
    )
