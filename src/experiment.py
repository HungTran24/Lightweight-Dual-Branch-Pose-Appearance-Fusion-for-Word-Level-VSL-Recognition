from __future__ import annotations

import csv
import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch
from dotenv import load_dotenv

from .artifacts import upload_with_retry
from .checkpoint import load_best, load_last
from .config import output_path
from .data import FeatureSplit, load_split, make_loader, select_view
from .e026_rgb_features import atomic_npz
from .e026_rgb_fusion import _validate_scores
from .evaluate import evaluate, evaluate_with_scores
from .integrations.huggingface import download_file, ensure_repo
from .integrations.wandb import finish_run, log_epoch, start_run
from .models import build_model
from .redflag_diagnostic import build_clean_signer_partition, score_metrics
from .train import build_optimizer_and_scheduler, fit, set_seed


def _environment(device: torch.device, input_shape: tuple[int, ...]) -> dict[str, Any]:
    environment: dict[str, Any] = {
        "device": str(device),
        "input_shape": list(input_shape),
        "precision": "float32",
        "torch": torch.__version__,
        "cuda": torch.version.cuda if torch.cuda.is_available() else None,
    }
    if device.type == "cuda":
        environment["gpu"] = torch.cuda.get_device_name(device)
        environment["vram_total_bytes"] = torch.cuda.get_device_properties(device).total_memory
    return environment


def _write_predictions(
    path: Path,
    split: FeatureSplit,
    labels: list[int],
    predictions: list[int],
    confidence: list[float],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["video_id", "true_class_id", "pred_class_id", "confidence", "correct"])
        for video_id, true, pred, score in zip(
            split.video_id, labels, predictions, confidence, strict=True
        ):
            writer.writerow([video_id, true, pred, score, true == pred])


def run_test_evaluate(
    config: dict[str, Any], project_root: Path, *, device: str | None = None
) -> dict[str, Any]:
    """Evaluate one frozen checkpoint on the locked test split."""
    load_dotenv(project_root / ".env")
    target = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    data = config["data"]
    evaluation = config["evaluation"]
    checkpoint = config["checkpoint"]
    cache_dir = project_root / "cache" / "hf"
    test_split = select_view(load_split(data, "test", cache_dir), config["feature"]["view"])
    output_dir = output_path(config, project_root)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Test evaluation already exists: {output_dir}")

    checkpoint_path = (
        project_root / checkpoint["local_path"]
        if "local_path" in checkpoint
        else download_file(
            repo_id=checkpoint["repo_id"],
            repo_type="model",
            revision=checkpoint["revision"],
            filename=checkpoint["path"],
            cache_dir=cache_dir,
        )
    )
    model = build_model(
        config["model"], nodes=test_split.X.shape[2], num_classes=int(data["num_classes"])
    ).to(target)
    checkpoint_state = load_best(checkpoint_path, model, target)
    metrics, predictions, confidence, labels, probabilities = evaluate_with_scores(
        model,
        make_loader(test_split, int(evaluation["batch_size"]), shuffle=False),
        num_classes=int(data["num_classes"]),
        device=target,
        collect_scores=bool(evaluation.get("export_scores", False)),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    if evaluation.get("export_scores", False):
        scores = probabilities.numpy().astype("float32", copy=False)
        if scores.shape[1] == 400:
            _validate_scores(test_split.video_id, test_split.y, scores)
        elif scores.shape != (len(test_split.video_id), int(data["num_classes"])):
            raise ValueError("Score archive shape does not match test split or class count")
        atomic_npz(
            output_dir / "scores.npz",
            video_id=test_split.video_id,
            true_class_id=np.array(labels, dtype=np.int64),
            probabilities=scores,
        )
    _write_predictions(
        output_dir / "predictions.csv",
        test_split,
        [int(value) for value in test_split.y],
        predictions,
        confidence,
    )
    summary = {
        "status": "ok",
        "experiment_id": config["experiment"]["id"],
        "operation": "test_evaluate",
        "seed": config["experiment"]["seed"],
        "source_experiment": checkpoint["source_experiment"],
        "checkpoint": {
            **(
                {"local_path": checkpoint["local_path"]}
                if "local_path" in checkpoint
                else {
                    "repo_id": checkpoint["repo_id"],
                    "revision": checkpoint["revision"],
                    "path": checkpoint["path"],
                }
            ),
            "epoch": checkpoint_state["epoch"],
        },
        "model": config["model"]["name"],
        "feature_view": config["feature"]["view"],
        "data_revision": data.get("revision", data.get("split_path", "local")),
        "artifact_path": data.get("artifact_path"),
        "split": "test",
        "environment": _environment(target, (1, *test_split.X.shape[1:])),
        "test_top1": metrics["top1"],
        "test_top5": metrics["top5"],
        "test_macro_f1": metrics["macro_f1"],
        "test_loss": metrics["loss"],
        "metrics": metrics,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _upload_hf(
    config: dict[str, Any], output_dir: Path, *, epoch: int, final: bool = False
) -> None:
    settings = config.get("integrations", {}).get("hf_checkpoints", {})
    if not isinstance(settings, dict) or not settings.get("enabled", False):
        return
    cadence = int(settings.get("upload_every_epochs", 5))
    if cadence <= 0:
        raise ValueError("integrations.hf_checkpoints.upload_every_epochs must be positive")
    if not final and epoch % cadence:
        return
    repo_id = settings["repo_id"]
    ensure_repo(repo_id=repo_id, repo_type="model", private=bool(settings.get("private", True)))
    prefix = f"runs/{config['experiment']['id']}_s{config['experiment']['seed']}"
    files = {
        f"{prefix}/last.pt": output_dir / "last.pt",
        f"{prefix}/best.pt": output_dir / "best.pt",
    }
    if final:
        files[f"{prefix}/summary.json"] = output_dir / "summary.json"
    try:
        upload_with_retry(
            repo_id=repo_id,
            repo_type="model",
            files=files,
            commit_message=f"checkpoint: {prefix} epoch {epoch}",
        )
    except Exception as exc:
        if final:
            raise
        warnings.warn(
            f"Periodic HF checkpoint upload failed at epoch {epoch} ({exc}); continuing training.",
            stacklevel=2,
        )


def run_experiment(
    config: dict[str, Any],
    project_root: Path,
    *,
    device: str | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    load_dotenv(project_root / ".env")
    set_seed(int(config["experiment"]["seed"]))
    target = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    data_config = config["data"]
    num_classes = int(data_config["num_classes"])
    cache_dir = project_root / "cache" / "hf"

    train_split = select_view(
        load_split(data_config, "train", cache_dir), config["feature"]["view"]
    )
    val_split = select_view(load_split(data_config, "val", cache_dir), config["feature"]["view"])
    if partition := data_config.get("signer_partition"):
        train_split, val_split = build_clean_signer_partition(
            {"train": train_split, "val": val_split}, partition, project_root
        )

    if config["experiment"]["mode"] == "smoke":
        subset = data_config.get("subset", {"train": 64, "val": 32})
        train_split = train_split.take(int(subset["train"]))
        val_split = val_split.take(int(subset["val"]))

    output_dir = output_path(config, project_root)
    last_checkpoint = output_dir / "last.pt"
    if not resume and last_checkpoint.exists():
        raise FileExistsError(f"Run already exists: {output_dir}. Use --resume to continue it.")

    nodes = train_split.X.shape[2]
    model = build_model(config["model"], nodes=nodes, num_classes=num_classes).to(target)
    optimizer, scheduler = build_optimizer_and_scheduler(model, config["train"])
    train_loader = make_loader(train_split, config["train"]["batch_size"], shuffle=True)
    val_loader = make_loader(val_split, config["train"]["batch_size"], shuffle=False)

    environment = _environment(target, (1, *train_split.X.shape[1:]))
    tracker = start_run(config, environment)

    data_revision = data_config.get("revision", data_config.get("split_path", "local"))
    resume_state = None
    if resume and last_checkpoint.exists():
        resume_state = load_last(last_checkpoint, model, optimizer, scheduler, target)
        if resume_state["data_revision"] != data_revision:
            raise ValueError("Checkpoint data revision does not match the experiment config")

    def on_epoch_end(
        epoch: int,
        train_metrics: dict[str, float],
        val_metrics: dict[str, float],
        lr: float,
        epoch_seconds: float,
    ) -> None:
        log_epoch(
            tracker,
            {
                **{f"train/{key}": value for key, value in train_metrics.items()},
                **{
                    f"validation/{key}": value
                    for key, value in val_metrics.items()
                    if key != "samples"
                },
                "learning_rate": lr,
                "epoch_seconds": epoch_seconds,
            },
            epoch,
        )
        _upload_hf(config, output_dir, epoch=epoch)

    best, epochs_trained = fit(
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        epochs=int(config["train"]["epochs"]),
        min_epochs=int(config["train"].get("min_epochs", 0)),
        patience=int(config["train"].get("patience", 0)),
        num_classes=num_classes,
        device=target,
        grad_clip_norm=float(config["train"]["grad_clip_norm"]),
        output_dir=output_dir,
        data_revision=data_revision,
        on_epoch_end=on_epoch_end,
        resume_state=resume_state,
    )

    load_best(output_dir / "best.pt", model, target)
    per_signer_metrics: list[dict[str, Any]] = []
    if partition:
        eval_metrics, predictions, confidence, labels, probabilities = evaluate_with_scores(
            model, val_loader, num_classes=num_classes, device=target
        )
        scores = probabilities.numpy().astype(np.float32, copy=False)
        _validate_scores(val_split.video_id, np.array(labels, dtype=np.int64), scores)
        atomic_npz(
            output_dir / "scores.npz",
            video_id=val_split.video_id,
            true_class_id=np.array(labels, dtype=np.int64),
            probabilities=scores,
        )
        for signer_id in sorted(set(map(str, val_split.signer_id))):
            indices = np.flatnonzero(val_split.signer_id == signer_id)
            per_signer_metrics.append(
                {"signer_id": signer_id, "samples": int(len(indices)), **score_metrics(scores[indices], val_split.y[indices])}
            )
    else:
        eval_metrics, predictions, confidence = evaluate(
            model, val_loader, num_classes=num_classes, device=target
        )
    _write_predictions(
        output_dir / "predictions.csv",
        val_split,
        [int(v) for v in val_split.y],
        predictions,
        confidence,
    )

    summary = {
        "status": "ok",
        "experiment_id": config["experiment"]["id"],
        "seed": config["experiment"]["seed"],
        "model": config["model"]["name"],
        "feature_view": config["feature"]["view"],
        "data_revision": data_revision,
        "environment": environment,
        "epochs_trained": epochs_trained,
        "best_epoch": best["epoch"],
        "best_validation_top1": eval_metrics["top1"],
        "best_validation_top5": eval_metrics["top5"],
        "best_validation_macro_f1": eval_metrics["macro_f1"],
        "best_validation_loss": eval_metrics["loss"],
        "metrics": eval_metrics,
        "per_signer_metrics": per_signer_metrics,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _upload_hf(config, output_dir, epoch=epochs_trained, final=True)
    finish_run(tracker, summary)
    return summary
