"""E038 per-frame RGB BiLSTM training and fusion evaluation."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .artifacts import atomic_json, sha256
from .checkpoint import load_best
from .e026_audit import load_dev_manifests
from .e026_rgb_features import atomic_npz
from .e026_rgb_fusion import _validate_scores, load_scores
from .e027_log_fusion import weighted_log_probabilities
from .evaluate import evaluate_with_scores
from .integrations.wandb import finish_run, log_epoch, start_run
from .metrics import as_floats, new_evaluation_metrics
from .models import build_model
from .train import build_optimizer_and_scheduler, fit, set_seed


@dataclass(frozen=True)
class FrameRGBSplit:
    features: np.ndarray        # [N, T, 2, 576]
    labels: np.ndarray          # [N]
    video_id: np.ndarray        # [N]
    valid_frame_mask: np.ndarray  # [N, T]


def load_frame_rgb_split(
    config: dict[str, Any], project_root: Path, split: str
) -> FrameRGBSplit:
    """Load per-frame RGB features [N, T, 2, 576] and join with clean labels."""
    if split not in {"train", "val", "test"}:
        raise ValueError(f"Unsupported split: {split}")
    records = load_dev_manifests(config, project_root, splits=(split,))[split]
    label_by_id = {row.video_id: int(row.class_id) for row in records}
    data = config["rgb_data"]
    split_path = project_root / data["root"] / f"{split}.npz"
    with np.load(split_path, allow_pickle=False) as loaded:
        video_id = np.array([str(v).zfill(6) for v in loaded["video_id"]], dtype="U6")
        features = loaded["X_rgb"]
        valid_mask = loaded["valid_frame_mask"]
    expected_ids = [row.video_id for row in records]
    if video_id.tolist() != expected_ids or set(video_id) != set(label_by_id):
        raise ValueError(f"{split} IDs must match clean manifest order")
    labels = np.array([label_by_id[v] for v in video_id], dtype=np.int64)
    return FrameRGBSplit(
        features=np.ascontiguousarray(features, dtype=np.float32),
        labels=labels,
        video_id=video_id,
        valid_frame_mask=np.ascontiguousarray(valid_mask, dtype=np.float32),
    )


def _frame_loader(split: FrameRGBSplit, batch_size: int, *, shuffle: bool) -> DataLoader:
    # Concatenate 2 hands: [N, T, 2, 576] -> [N, T, 1152]
    n, t, h, d = split.features.shape
    flat_feat = split.features.reshape(n, t, h * d)
    return DataLoader(
        TensorDataset(
            torch.from_numpy(flat_feat),
            torch.from_numpy(split.labels),
            torch.from_numpy(split.valid_frame_mask),
        ),
        batch_size=batch_size,
        shuffle=shuffle,
    )


def run_e038_rgb_bilstm_train(
    config: dict[str, Any], project_root: Path, *, device: str | None = None
) -> dict[str, Any]:
    """Train per-frame RGB BiLSTM and export validation scores."""
    seed = int(config["experiment"]["seed"])
    set_seed(seed)
    target = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    train_split = load_frame_rgb_split(config, project_root, "train")
    val_split = load_frame_rgb_split(config, project_root, "val")
    output_dir = project_root / config["output"]["root"]
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Run already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(config["model"], nodes=1, num_classes=400).to(target)
    optimizer, scheduler = build_optimizer_and_scheduler(model, config["train"])
    batch_size = int(config["train"]["batch_size"])
    train_loader = _frame_loader(train_split, batch_size, shuffle=True)
    val_loader = _frame_loader(val_split, batch_size, shuffle=False)

    environment = {
        "device": str(target),
        "precision": "float32",
        "torch": torch.__version__,
    }
    tracker = start_run(config, environment) if "integrations" in config else None

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
                **{f"train/{k}": v for k, v in train_metrics.items()},
                **{f"validation/{k}": v for k, v in val_metrics.items() if k != "samples"},
                "learning_rate": lr,
                "epoch_seconds": epoch_seconds,
            },
            epoch,
        )

    best, epochs = fit(
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        epochs=int(config["train"]["epochs"]),
        min_epochs=int(config["train"]["min_epochs"]),
        patience=int(config["train"]["patience"]),
        num_classes=400,
        device=target,
        grad_clip_norm=float(config["train"]["grad_clip_norm"]),
        output_dir=output_dir,
        data_revision="e038_bilstm",
        on_epoch_end=on_epoch_end,
    )
    load_best(output_dir / "best.pt", model, target)
    metrics, predictions, confidence, labels, probabilities = evaluate_with_scores(
        model, val_loader, num_classes=400, device=target
    )
    probabilities_np = probabilities.numpy().astype(np.float32, copy=False)
    _validate_scores(val_split.video_id, val_split.labels, probabilities_np)
    atomic_npz(
        output_dir / "scores.npz",
        video_id=val_split.video_id,
        true_class_id=np.array(labels, dtype=np.int64),
        probabilities=probabilities_np,
    )
    with (output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["video_id", "true_class_id", "pred_class_id", "confidence"])
        writer.writerows(zip(val_split.video_id, labels, predictions, confidence, strict=True))
    summary = {
        "status": "ok",
        "experiment_id": config["experiment"]["id"],
        "seed": seed,
        "parameters": sum(p.numel() for p in model.parameters()),
        "epochs_trained": epochs,
        "best_epoch": best["epoch"],
        "metrics": metrics,
    }
    atomic_json(output_dir / "summary.json", summary)
    finish_run(tracker, summary)
    return summary


def _score(probabilities: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    metric = new_evaluation_metrics(400, torch.device("cpu"))
    metric.update(torch.from_numpy(probabilities), torch.from_numpy(labels))
    return as_floats(metric.compute())


def _join_with_pose(
    rgb_scores: tuple[np.ndarray, np.ndarray, np.ndarray],
    pose_scores: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rgb_ids, labels, rgb = rgb_scores
    pose_ids, pose_labels, pose = pose_scores
    pose_map = {str(video_id): i for i, video_id in enumerate(pose_ids)}
    if set(map(str, rgb_ids)) != set(pose_map):
        raise ValueError("RGB and Pose score video IDs differ")
    order = np.array([pose_map[str(video_id)] for video_id in rgb_ids])
    if not np.array_equal(labels, pose_labels[order]):
        raise ValueError("RGB and Pose score labels differ")
    return labels, rgb, pose[order]


def run_e038_fixed_fusion(config: dict[str, Any], project_root: Path) -> dict[str, Any]:
    """Evaluate fixed-beta (0.5) log-probability fusion with Pose TCN U30."""
    experiment = config["experiment"]
    seed = int(experiment["seed"])
    candidate = load_scores(project_root / config["candidate_scores"])
    pose = load_scores(project_root / config["pose_scores"])
    labels, candidate_rgb, candidate_pose = _join_with_pose(candidate, pose)

    epsilon = 1e-12
    beta = float(config.get("fusion", {}).get("beta", 0.5))
    candidate_fused = weighted_log_probabilities(candidate_pose, candidate_rgb, beta, epsilon)
    candidate_metrics = _score(candidate_fused, labels)
    rgb_metrics = _score(candidate_rgb, labels)
    pose_metrics = _score(candidate_pose, labels)

    summary = {
        "status": "ok",
        "experiment_id": experiment["id"],
        "seed": seed,
        "beta": beta,
        "rgb_metrics": rgb_metrics,
        "pose_metrics": pose_metrics,
        "fused_metrics": candidate_metrics,
        "delta_vs_pose": {
            k: candidate_metrics[k] - pose_metrics[k] for k in ("top1", "top5", "macro_f1")
        },
    }
    output_dir = project_root / config["output"]["root"]
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(output_dir / "summary.json", summary)
    return summary


def run_e038_test_evaluate(
    config: dict[str, Any], project_root: Path, *, device: str | None = None
) -> dict[str, Any]:
    """Evaluate locked E038 checkpoints once on the frozen test split."""
    target = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    beta = float(config["fusion"]["beta"])
    if beta != 0.5:
        raise ValueError("E038 frozen test requires beta=0.5")
    test_split = load_frame_rgb_split(config, project_root, "test")
    loader = _frame_loader(test_split, int(config["batch_size"]), shuffle=False)
    output_root = project_root / config["output"]["root"]
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Test output already exists: {output_root}")
    results: list[dict[str, Any]] = []
    for seed in (42, 43, 44):
        model = build_model(config["model"], nodes=1, num_classes=400).to(target)
        checkpoint_path = project_root / config["checkpoints"][str(seed)]
        checkpoint = load_best(checkpoint_path, model, target)
        metrics, predictions, confidence, labels, probabilities = evaluate_with_scores(
            model, loader, num_classes=400, device=target
        )
        rgb_probs = probabilities.numpy().astype(np.float32, copy=False)
        _validate_scores(test_split.video_id, test_split.labels, rgb_probs)
        pose_scores = load_scores(project_root / config["pose_scores"][str(seed)])
        labels_joined, rgb_probs, pose_probs = _join_with_pose(
            (test_split.video_id, test_split.labels, rgb_probs), pose_scores
        )
        fused_probs = weighted_log_probabilities(
            pose_probs, rgb_probs, beta, float(config["fusion"]["epsilon"])
        )
        run_dir = output_root / f"seed{seed}"
        atomic_npz(
            run_dir / "rgb_scores.npz",
            video_id=test_split.video_id,
            true_class_id=labels_joined,
            probabilities=rgb_probs,
        )
        atomic_npz(
            run_dir / "fused_scores.npz",
            video_id=test_split.video_id,
            true_class_id=labels_joined,
            probabilities=fused_probs,
        )
        fused_metrics = _score(fused_probs, labels_joined)
        results.append(
            {
                "seed": seed,
                "rgb_metrics": metrics,
                "fused_metrics": fused_metrics,
                "checkpoint": {
                    "path": str(checkpoint_path.relative_to(project_root)),
                    "sha256": sha256(checkpoint_path),
                    "epoch": checkpoint["epoch"],
                },
                "pose_scores": config["pose_scores"][str(seed)],
            }
        )
    aggregate = {
        key: {
            metric: {
                "mean": float(np.mean([row[key][metric] for row in results])),
                "sample_sd": float(np.std([row[key][metric] for row in results], ddof=1)),
            }
            for metric in ("top1", "top5", "macro_f1")
        }
        for key in ("rgb_metrics", "fused_metrics")
    }
    summary = {
        "status": "ok",
        "experiment_id": config["experiment"]["id"],
        "test_manifest": config["manifests"]["test"],
        "test_videos": len(test_split.video_id),
        "fusion": {"beta": beta, "epsilon": float(config["fusion"]["epsilon"])},
        "device": str(target),
        "branches": results,
        "aggregate": aggregate,
    }
    atomic_json(output_root / "summary.json", summary)
    return summary
