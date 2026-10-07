"""E028 temporal RGB training and fixed-beta validation comparison."""

from __future__ import annotations

import csv
import json
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
class TemporalRGBSplit:
    features: np.ndarray
    labels: np.ndarray
    video_id: np.ndarray
    token_present: np.ndarray


def load_temporal_rgb_split(
    config: dict[str, Any], project_root: Path, split: str
) -> TemporalRGBSplit:
    """Load E028 temporal RGB and join clean labels in exact manifest order."""
    if split not in {"train", "val", "test"}:
        raise ValueError(f"Unsupported temporal RGB split: {split}")
    records = load_dev_manifests(config, project_root, splits=(split,))[split]
    label_by_id = {row.video_id: int(row.class_id) for row in records}
    data = config["rgb_data"]
    receipt_path = project_root / data["receipt"]
    if sha256(receipt_path) != data["receipt_sha256"]:
        raise ValueError("E028 receipt checksum mismatch")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("extraction_identity") != data.get("extraction_identity"):
        raise ValueError("E028 extraction identity mismatch")
    sampling_slots = int(data.get("sampling_slots", 30))
    if int(receipt.get("sampling_slots", 30)) != sampling_slots:
        raise ValueError("Temporal RGB sampling_slots mismatch")
    split_path = project_root / data["root"] / f"{split}.npz"
    if sha256(split_path) != receipt["files"][split]["sha256"]:
        raise ValueError(f"E028 {split} archive checksum mismatch")
    with np.load(split_path, allow_pickle=False) as loaded:
        video_id = np.array([str(v).zfill(6) for v in loaded["video_id"]], dtype="U6")
        features = loaded["X_rgb"]
        counts = loaded["hand_valid_count"]
    expected_ids = [row.video_id for row in records]
    if video_id.tolist() != expected_ids or set(video_id) != set(label_by_id):
        raise ValueError(f"E028 {split} IDs must exactly match clean manifest order")
    if features.shape != (len(records), 2, 3, 576) or features.dtype != np.float16:
        raise ValueError("E028 X_rgb must be float16 [N,2,3,576]")
    if (
        counts.shape != (len(records), 2, 3)
        or counts.dtype != np.uint8
        or np.any(counts > sampling_slots // 3)
        or not np.isfinite(features).all()
        or np.any(features[counts == 0] != 0)
    ):
        raise ValueError("E028 counts or feature validity contract failed")
    labels = np.array([label_by_id[v] for v in video_id], dtype=np.int64)
    return TemporalRGBSplit(
        features=np.ascontiguousarray(features, dtype=np.float32),
        labels=labels,
        video_id=video_id,
        token_present=np.ascontiguousarray((counts > 0).reshape(len(records), 6), dtype=np.uint8),
    )


def _loader(split: TemporalRGBSplit, batch_size: int, *, shuffle: bool) -> DataLoader:
    return DataLoader(
        TensorDataset(
            torch.from_numpy(split.features),
            torch.from_numpy(split.labels),
            torch.from_numpy(split.token_present),
        ),
        batch_size=batch_size,
        shuffle=shuffle,
    )


def run_e028_rgb_train(
    config: dict[str, Any], project_root: Path, *, device: str | None = None
) -> dict[str, Any]:
    """Train one temporal RGB seed with the shared classifier trainer."""
    seed = int(config["experiment"]["seed"])
    set_seed(seed)
    target = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    train_split = load_temporal_rgb_split(config, project_root, "train")
    val_split = load_temporal_rgb_split(config, project_root, "val")
    output_dir = project_root / config["output"]["root"]
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Run already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(config["model"], nodes=1, num_classes=400).to(target)
    optimizer, scheduler = build_optimizer_and_scheduler(model, config["train"])
    batch_size = int(config["train"]["batch_size"])
    val_loader = _loader(val_split, batch_size, shuffle=False)
    environment = {
        "device": str(target),
        "input_shape": [1, *train_split.features.shape[1:]],
        "precision": "float32",
        "torch": torch.__version__,
        "cuda": torch.version.cuda if torch.cuda.is_available() else None,
    }
    if target.type == "cuda":
        environment["gpu"] = torch.cuda.get_device_name(target)
        environment["vram_total_bytes"] = torch.cuda.get_device_properties(
            target
        ).total_memory
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

    best, epochs = fit(
        model,
        _loader(train_split, batch_size, shuffle=True),
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
        data_revision=sha256(project_root / config["rgb_data"]["receipt"]),
        on_epoch_end=on_epoch_end,
    )
    load_best(output_dir / "best.pt", model, target)
    metrics, predictions, confidence, labels, probabilities = evaluate_with_scores(
        model, val_loader, num_classes=400, device=target
    )
    if labels != val_split.labels.tolist():
        raise ValueError("Evaluation label order differs from E028 validation split")
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


def run_e028_fixed_fusion(config: dict[str, Any], project_root: Path) -> dict[str, Any]:
    """Compare E028 candidate against E026 RGB at the fixed E027 beta 0.5."""
    experiment = config["experiment"]
    seed = int(experiment["seed"])
    candidate = load_scores(project_root / config["candidate_scores"])
    baseline = load_scores(project_root / config["baseline_rgb_scores"])
    pose = load_scores(project_root / config["pose_scores"])
    candidate_ids = candidate[0]
    baseline_map = {str(video_id): i for i, video_id in enumerate(baseline[0])}
    if set(map(str, candidate_ids)) != set(baseline_map):
        raise ValueError("Candidate and baseline score video IDs differ")
    baseline_order = np.array([baseline_map[str(video_id)] for video_id in candidate_ids])
    baseline = (baseline[0][baseline_order], baseline[1][baseline_order], baseline[2][baseline_order])
    labels, candidate_rgb, candidate_pose = _join_with_pose(candidate, pose)
    baseline_labels, baseline_rgb, baseline_pose = _join_with_pose(baseline, pose)
    if not np.array_equal(labels, baseline_labels):
        raise ValueError("Candidate and baseline labels differ")

    epsilon = 1e-12
    beta = 0.5
    candidate_fused = weighted_log_probabilities(candidate_pose, candidate_rgb, beta, epsilon)
    baseline_fused = weighted_log_probabilities(baseline_pose, baseline_rgb, beta, epsilon)
    candidate_metrics = _score(candidate_fused, labels)
    baseline_metrics = _score(baseline_fused, labels)
    deltas = {name: candidate_metrics[name] - baseline_metrics[name] for name in ("top1", "top5", "macro_f1")}
    gate = deltas["top1"] >= 0.005 and deltas["top5"] >= 0 and deltas["macro_f1"] >= 0
    continuation = (
        "eligible_for_manual_seed_43_44_authorization" if gate else "stop_after_seed_42"
    ) if seed == 42 else "diagnostic_only"
    summary = {
        "status": "ok",
        "experiment_id": experiment["id"],
        "seed": seed,
        "formula": "log(max(p_pose, eps)) + 0.5 * log(max(p_rgb, eps))",
        "epsilon": epsilon,
        "beta": beta,
        "candidate_rgb_metrics": _score(candidate_rgb, labels),
        "baseline_rgb_metrics": _score(baseline_rgb, labels),
        "candidate_fused_metrics": candidate_metrics,
        "baseline_fused_metrics": baseline_metrics,
        "deltas": deltas,
        "gate_passed": gate,
        "continuation": continuation,
    }
    output_dir = project_root / config["output"]["root"]
    atomic_json(output_dir / "summary.json", summary)
    return summary
