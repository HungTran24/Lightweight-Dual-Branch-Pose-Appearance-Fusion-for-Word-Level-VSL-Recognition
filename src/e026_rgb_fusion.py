"""E026 RGB-U30 linear training and frozen Pose probability fusion."""

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
from .data import load_split, make_loader, select_view
from .e026_audit import load_dev_manifests
from .e026_rgb_features import atomic_npz
from .evaluate import evaluate_with_scores
from .integrations.huggingface import download_file
from .integrations.wandb import finish_run, log_epoch, start_run
from .metrics import as_floats, new_evaluation_metrics
from .models import build_model
from .train import build_optimizer_and_scheduler, fit, set_seed


@dataclass(frozen=True)
class RGBPooledSplit:
    """One pooled RGB split joined to clean labels by video ID."""

    features: np.ndarray
    labels: np.ndarray
    video_id: np.ndarray


def load_rgb_pooled_split(
    config: dict[str, Any], project_root: Path, split: str
) -> RGBPooledSplit:
    """Load pooled RGB view and perform an exact keyed join to clean manifest labels."""
    if split not in {"train", "val", "test"}:
        raise ValueError(f"Unsupported RGB split: {split}")
    view = config["rgb_data"].get("view", "u30")
    if view not in {"u24", "u30"}:
        raise ValueError("rgb_data.view must be u24 or u30")
    records = load_dev_manifests(config, project_root, splits=(split,))[split]
    label_by_id = {r.video_id: int(r.class_id) for r in records}
    if len(label_by_id) != len(records):
        raise ValueError(f"Duplicate video IDs in clean {split} manifest")

    rgb_cfg = config["rgb_data"]
    receipt_path = project_root / rgb_cfg["receipt"]
    if sha256(receipt_path) != rgb_cfg["receipt_sha256"]:
        raise ValueError("RGB extraction receipt checksum mismatch")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("extraction_identity") != rgb_cfg["extraction_identity"]:
        raise ValueError("RGB extraction identity mismatch")

    root = project_root / rgb_cfg["root"]
    split_path = root / f"{split}.npz"
    expected_sha256 = receipt["final_files"][view][split]["sha256"]
    if sha256(split_path) != expected_sha256:
        raise ValueError(f"RGB {split} feature checksum mismatch")
    with np.load(split_path, allow_pickle=False) as loaded:
        vids = np.array([str(v).zfill(6) for v in loaded["video_id"]], dtype="U6")
        x_rgb = loaded["X_rgb"]
    expected = [r.video_id for r in records]
    if vids.tolist() != expected or set(vids.tolist()) != set(label_by_id):
        raise ValueError(f"RGB {split} video IDs must exactly match clean manifest order")
    if x_rgb.shape != (len(vids), 2, 576) or x_rgb.dtype != np.float16:
        raise ValueError(f"RGB {split} X_rgb must be float16 [N,2,576]")
    if not np.isfinite(x_rgb).all():
        raise ValueError(f"RGB {split} X_rgb contains non-finite values")

    features = np.ascontiguousarray(x_rgb.reshape(len(vids), 1152), dtype=np.float32)
    labels = np.array([label_by_id[v] for v in vids], dtype=np.int64)
    return RGBPooledSplit(features=features, labels=labels, video_id=vids)


def _rgb_loader(split: RGBPooledSplit, batch_size: int, *, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(split.features),
        torch.from_numpy(split.labels),
        torch.ones((len(split.labels), 1), dtype=torch.uint8),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def _write_predictions(
    path: Path,
    split: RGBPooledSplit,
    predictions: list[int],
    confidence: list[float],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["video_id", "true_class_id", "pred_class_id", "confidence", "correct"])
        for vid, true, pred, score in zip(
            split.video_id, split.labels, predictions, confidence, strict=True
        ):
            writer.writerow([vid, int(true), pred, score, int(true) == pred])


def _validate_scores(video_id: np.ndarray, labels: np.ndarray, probabilities: np.ndarray) -> None:
    if len(video_id) != len({str(v) for v in video_id}):
        raise ValueError("Duplicate video IDs in scores")
    if labels.shape != (len(video_id),) or labels.dtype != np.int64:
        raise ValueError("true_class_id must be int64 [N]")
    if probabilities.shape != (len(video_id), 400) or probabilities.dtype != np.float32:
        raise ValueError("probabilities must be float32 [N,400]")
    if (
        not np.isfinite(probabilities).all()
        or np.any(probabilities < 0.0)
        or np.any(probabilities > 1.0)
        or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5)
    ):
        raise ValueError("Probabilities must be finite, in [0,1], and sum to one")


def _write_scores(path: Path, split: RGBPooledSplit, probabilities: np.ndarray) -> None:
    _validate_scores(split.video_id, split.labels, probabilities)
    atomic_npz(
        path,
        video_id=split.video_id,
        true_class_id=split.labels,
        probabilities=probabilities,
    )


def run_e026_rgb_train(
    config: dict[str, Any], project_root: Path, *, device: str | None = None
) -> dict[str, Any]:
    """Train one global RGB seed and export validation scores."""
    seed = int(config["experiment"]["seed"])
    set_seed(seed)
    target = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    train_split = load_rgb_pooled_split(config, project_root, "train")
    val_split = load_rgb_pooled_split(config, project_root, "val")
    output_dir = project_root / config["output"]["root"]
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Run already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(config["model"], nodes=1, num_classes=400).to(target)
    optimizer, scheduler = build_optimizer_and_scheduler(model, config["train"])
    batch_size = int(config["train"]["batch_size"])
    train_loader = _rgb_loader(train_split, batch_size, shuffle=True)
    val_loader = _rgb_loader(val_split, batch_size, shuffle=False)
    receipt_path = project_root / config["rgb_data"]["receipt"]
    data_revision = sha256(receipt_path)
    environment = {
        "device": str(target),
        "input_shape": [1, *train_split.features.shape[1:]],
        "precision": "float32",
        "torch": torch.__version__,
        "cuda": torch.version.cuda if torch.cuda.is_available() else None,
    }
    if target.type == "cuda":
        environment["gpu"] = torch.cuda.get_device_name(target)
        environment["vram_total_bytes"] = torch.cuda.get_device_properties(target).total_memory
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

    best, epochs_trained = fit(
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
        data_revision=data_revision,
        on_epoch_end=on_epoch_end,
    )
    load_best(output_dir / "best.pt", model, target)
    metrics, predictions, confidence, labels, probabilities = evaluate_with_scores(
        model, val_loader, num_classes=400, device=target
    )
    if labels != val_split.labels.tolist():
        raise ValueError("Evaluation label order differs from validation RGB split")
    probabilities_np = probabilities.numpy().astype(np.float32, copy=False)
    _write_predictions(output_dir / "predictions.csv", val_split, predictions, confidence)
    _write_scores(output_dir / "scores.npz", val_split, probabilities_np)

    summary = {
        "status": "ok",
        "experiment_id": config["experiment"]["id"],
        "seed": seed,
        "model": config["model"]["name"],
        "rgb_view": config["rgb_data"].get("view", "u30"),
        "parameters": sum(p.numel() for p in model.parameters()),
        "epochs_trained": epochs_trained,
        "best_epoch": best["epoch"],
        "metrics": metrics,
        "rgb_receipt_sha256": data_revision,
    }
    atomic_json(output_dir / "summary.json", summary)
    finish_run(tracker, summary)
    return summary


def load_scores(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load and validate one score archive."""
    with np.load(path, allow_pickle=False) as loaded:
        video_id = np.array([str(v).zfill(6) for v in loaded["video_id"]], dtype="U6")
        labels = np.asarray(loaded["true_class_id"])
        probabilities = np.asarray(loaded["probabilities"])
    _validate_scores(video_id, labels, probabilities)
    return video_id, labels, probabilities


def _score_probabilities(probabilities: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    metric = new_evaluation_metrics(400, torch.device("cpu"))
    metric.update(torch.from_numpy(probabilities), torch.from_numpy(labels))
    return as_floats(metric.compute())


def fuse_probability_scores(
    rgb_scores: tuple[np.ndarray, np.ndarray, np.ndarray],
    pose_scores: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> tuple[list[dict[str, float]], float, np.ndarray, dict[str, float], dict[str, float]]:
    """Join scores by ID, evaluate α curve, and select by Top-1, Macro-F1, then larger α."""
    rgb_ids, rgb_labels, rgb_probs = rgb_scores
    pose_ids, pose_labels, pose_probs = pose_scores
    pose_by_id = {str(v): i for i, v in enumerate(pose_ids)}
    if set(map(str, rgb_ids)) != set(pose_by_id):
        raise ValueError("RGB and Pose score video IDs differ")
    order = np.array([pose_by_id[str(v)] for v in rgb_ids], dtype=np.int64)
    pose_labels = pose_labels[order]
    pose_probs = pose_probs[order]
    if not np.array_equal(rgb_labels, pose_labels):
        raise ValueError("RGB and Pose true_class_id differ")

    curve: list[dict[str, float]] = []
    fused_by_alpha: dict[float, np.ndarray] = {}
    for step in range(11):
        alpha = step / 10.0
        fused = (alpha * pose_probs + (1.0 - alpha) * rgb_probs).astype(np.float32)
        metrics = _score_probabilities(fused, rgb_labels)
        curve.append({"alpha": alpha, **metrics})
        fused_by_alpha[alpha] = fused
    selected = max(curve, key=lambda row: (row["top1"], row["macro_f1"], row["alpha"]))
    alpha = selected["alpha"]
    return curve, alpha, fused_by_alpha[alpha], _score_probabilities(rgb_probs, rgb_labels), _score_probabilities(pose_probs, rgb_labels)


def aggregate_alpha_curves(
    curves: list[list[dict[str, float]]],
) -> tuple[list[dict[str, float]], dict[str, float]]:
    """Average three seed curves and select one shared α with the locked tie-break."""
    expected_alpha = np.arange(11, dtype=np.float64) / 10.0
    if len(curves) != 3:
        raise ValueError("Shared-alpha selection requires exactly three seed curves")
    for curve in curves:
        alpha = np.array([row["alpha"] for row in curve], dtype=np.float64)
        if len(curve) != 11 or not np.array_equal(alpha, expected_alpha):
            raise ValueError("Each seed curve must use α grid 0.0..1.0 in steps of 0.1")

    mean_curve = [
        {
            "alpha": float(expected_alpha[i]),
            **{
                f"mean_{metric}": float(np.mean([curve[i][metric] for curve in curves]))
                for metric in ("top1", "top5", "macro_f1")
            },
        }
        for i in range(11)
    ]
    selected = max(
        mean_curve,
        key=lambda row: (row["mean_top1"], row["mean_macro_f1"], row["alpha"]),
    )
    return mean_curve, selected


def run_e026_fusion_summary(config: dict[str, Any], project_root: Path) -> dict[str, Any]:
    """Select one shared α and summarize RGB, Pose, and fused validation metrics."""
    curves: list[list[dict[str, float]]] = []
    run_summaries: list[dict[str, Any]] = []
    seeds = [int(item["seed"]) for item in config["fusion_runs"]]
    for item in config["fusion_runs"]:
        run_root = project_root / item["root"]
        seed = int(item["seed"])
        run_summary = json.loads((run_root / "summary.json").read_text(encoding="utf-8"))
        expected_rgb_scores = f"artifacts/runs/e026_rgb_linear_u30_s{seed}/scores.npz"
        if (
            run_summary.get("experiment_id") != f"e026_fusion_u30_s{seed}"
            or run_summary.get("checkpoint", {}).get("path")
            != f"runs/e008_tcn_pose53_uniform30_s{seed}/best.pt"
            or ("seed" in run_summary and run_summary["seed"] != seed)
            or (
                "rgb_scores" in run_summary
                and run_summary["rgb_scores"] != expected_rgb_scores
            )
        ):
            raise ValueError(f"Fusion summary source mismatch for seed {seed}")
        run_summaries.append(run_summary)
        with (run_root / "alpha_curve.csv").open(newline="", encoding="utf-8") as handle:
            curves.append(
                [
                    {name: float(value) for name, value in row.items()}
                    for row in csv.DictReader(handle)
                ]
            )

    mean_curve, selected = aggregate_alpha_curves(curves)
    output_dir = project_root / config["output"]["root"]
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "mean_alpha_curve.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["alpha", "mean_top1", "mean_top5", "mean_macro_f1"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(mean_curve)

    selected_alpha = selected["alpha"]
    seed_results: dict[str, dict[str, Any]] = {}
    for seed, curve, run_summary in zip(seeds, curves, run_summaries, strict=True):
        fused = next(row for row in curve if row["alpha"] == selected_alpha)
        rgb = run_summary["rgb_metrics"]
        pose = run_summary["pose_metrics"]
        seed_results[str(seed)] = {
            "rgb": {name: rgb[name] for name in ("top1", "top5", "macro_f1")},
            "pose": {name: pose[name] for name in ("top1", "top5", "macro_f1")},
            "fused": fused,
            "top1_delta_vs_pose": fused["top1"] - pose["top1"],
            "macro_f1_delta_vs_pose": fused["macro_f1"] - pose["macro_f1"],
        }

    means = {
        source: {
            metric: float(np.mean([seed_results[str(seed)][source][metric] for seed in seeds]))
            for metric in ("top1", "top5", "macro_f1")
        }
        for source in ("rgb", "pose", "fused")
    }
    means["deltas_vs_pose"] = {
        metric: float(np.mean([seed_results[str(seed)][metric] for seed in seeds]))
        for metric in ("top1_delta_vs_pose", "macro_f1_delta_vs_pose")
    }
    summary = {
        "status": "ok",
        "experiment_id": config["experiment"]["id"],
        "seeds": seeds,
        "selected_alpha": selected_alpha,
        "selected_mean_metrics": selected,
        "per_seed": seed_results,
        "means": means,
    }
    atomic_json(output_dir / "summary.json", summary)
    return summary


def continuation_gate(selected: dict[str, float], pose: dict[str, float]) -> bool:
    """Apply the locked continuation gate to one seed's validation metrics."""
    return (
        selected["top1"] - pose["top1"] >= 0.005
        and selected["macro_f1"] >= pose["macro_f1"]
    )


def continuation_status(seed: int, gate_passed: bool) -> str:
    """Keep seed-42 gate semantics while marking later seeds diagnostic-only."""
    if seed != 42:
        return "diagnostic_only"
    return "eligible_for_manual_seed_43_44_authorization" if gate_passed else "stop_after_seed_42"


def run_e026_fusion(
    config: dict[str, Any], project_root: Path, *, device: str | None = None
) -> dict[str, Any]:
    """Export frozen E008 validation probabilities and fuse them with RGB scores."""
    target = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    output_dir = project_root / config["output"]["root"]
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Fusion output already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = project_root / "cache" / "hf"

    pose_cfg = config["pose"]
    pose_split = select_view(load_split(pose_cfg["data"], "val", cache_dir), "pose53")
    model = build_model(pose_cfg["model"], nodes=pose_split.X.shape[2], num_classes=400).to(target)
    checkpoint_cfg = pose_cfg["checkpoint"]
    checkpoint_path = download_file(
        repo_id=checkpoint_cfg["repo_id"],
        repo_type="model",
        revision=checkpoint_cfg["revision"],
        filename=checkpoint_cfg["path"],
        cache_dir=cache_dir,
    )
    checkpoint = load_best(checkpoint_path, model, target)
    if int(checkpoint["epoch"]) != int(checkpoint_cfg["expected_epoch"]):
        raise ValueError("Frozen Pose checkpoint epoch mismatch")
    _, _, _, pose_labels, pose_probabilities = evaluate_with_scores(
        model,
        make_loader(pose_split, int(config["evaluation"]["batch_size"]), shuffle=False),
        num_classes=400,
        device=target,
    )
    pose_scores_path = output_dir / "pose_scores.npz"
    pose_split_ids = np.array([str(v).zfill(6) for v in pose_split.video_id], dtype="U6")
    pose_labels_np = np.array(pose_labels, dtype=np.int64)
    pose_probs_np = pose_probabilities.numpy().astype(np.float32, copy=False)
    _validate_scores(pose_split_ids, pose_labels_np, pose_probs_np)
    atomic_npz(
        pose_scores_path,
        video_id=pose_split_ids,
        true_class_id=pose_labels_np,
        probabilities=pose_probs_np,
    )

    rgb_scores_path = project_root / config["rgb_scores"]
    rgb_scores = load_scores(rgb_scores_path)
    pose_scores = load_scores(pose_scores_path)
    curve, selected_alpha, fused, rgb_metrics, pose_metrics = fuse_probability_scores(
        rgb_scores, pose_scores
    )
    selected = next(row for row in curve if row["alpha"] == selected_alpha)
    gate_passed = continuation_gate(selected, pose_metrics)

    with (output_dir / "alpha_curve.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["alpha", "top1", "top5", "macro_f1"])
        writer.writeheader()
        writer.writerows(curve)
    rgb_ids, rgb_labels, _ = rgb_scores
    _validate_scores(rgb_ids, rgb_labels, fused)
    atomic_npz(
        output_dir / "fused_scores.npz",
        video_id=rgb_ids,
        true_class_id=rgb_labels,
        probabilities=fused,
    )
    seed = int(config["experiment"]["seed"])
    continuation = continuation_status(seed, gate_passed)
    summary = {
        "status": "ok",
        "experiment_id": config["experiment"]["id"],
        "seed": seed,
        "rgb_scores": config["rgb_scores"],
        "selected_alpha": selected_alpha,
        "selected_metrics": {k: selected[k] for k in ("top1", "top5", "macro_f1")},
        "rgb_metrics": rgb_metrics,
        "pose_metrics": pose_metrics,
        "gate_top1_delta": selected["top1"] - pose_metrics["top1"],
        "gate_passed": gate_passed,
        "continuation": continuation,
        "checkpoint": {**checkpoint_cfg, "epoch": int(checkpoint["epoch"])},
    }
    atomic_json(output_dir / "summary.json", summary)
    return summary
