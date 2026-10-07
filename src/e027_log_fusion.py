"""E027 shared-beta weighted log-probability fusion on persisted validation scores."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .artifacts import atomic_json, sha256
from .e026_rgb_fusion import load_scores
from .metrics import as_floats, new_evaluation_metrics

METRICS = ("top1", "top5", "macro_f1")


def _metrics(scores: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    metric = new_evaluation_metrics(400, torch.device("cpu"))
    metric.update(torch.from_numpy(scores), torch.from_numpy(labels))
    return as_floats(metric.compute())


def weighted_log_probabilities(
    pose: np.ndarray, rgb: np.ndarray, beta: float, epsilon: float
) -> np.ndarray:
    """Return stable normalized probabilities from the weighted log-probability score."""
    log_scores = np.log(np.maximum(pose, epsilon)) + beta * np.log(np.maximum(rgb, epsilon))
    log_scores -= log_scores.max(axis=1, keepdims=True)
    probabilities = np.exp(log_scores)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return probabilities.astype(np.float32)


def evaluate_log_fusion(
    seed_scores: list[
        tuple[int, tuple[np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]]
    ],
    *,
    epsilon: float,
) -> tuple[list[dict[str, float]], dict[str, float], dict[str, dict[str, Any]]]:
    """Evaluate each seed on the locked beta grid and select one beta from mean metrics."""
    beta_grid = np.arange(11, dtype=np.float64) / 10.0
    per_seed: dict[str, dict[str, Any]] = {}
    curves: dict[int, list[dict[str, float]]] = {}

    for seed, rgb_scores, pose_scores in seed_scores:
        rgb_ids, rgb_labels, rgb_probs = rgb_scores
        pose_ids, pose_labels, pose_probs = pose_scores
        pose_by_id = {str(video_id): i for i, video_id in enumerate(pose_ids)}
        if set(map(str, rgb_ids)) != set(pose_by_id):
            raise ValueError(f"RGB and Pose score video IDs differ for seed {seed}")
        order = np.array([pose_by_id[str(video_id)] for video_id in rgb_ids], dtype=np.int64)
        pose_labels = pose_labels[order]
        pose_probs = pose_probs[order]
        if not np.array_equal(rgb_labels, pose_labels):
            raise ValueError(f"RGB and Pose true_class_id differ for seed {seed}")

        curves[seed] = []
        for beta in beta_grid:
            fused = weighted_log_probabilities(pose_probs, rgb_probs, float(beta), epsilon)
            curves[seed].append({"beta": float(beta), **_metrics(fused, rgb_labels)})
        per_seed[str(seed)] = {
            "rgb": _metrics(rgb_probs, rgb_labels),
            "pose": _metrics(pose_probs, rgb_labels),
        }

    mean_curve = [
        {
            "beta": float(beta),
            **{
                f"seed_{seed}_{name}": curves[seed][index][name]
                for seed, *_ in seed_scores
                for name in METRICS
            },
            **{
                f"mean_{name}": float(
                    np.mean([curves[seed][index][name] for seed, *_ in seed_scores])
                )
                for name in METRICS
            },
        }
        for index, beta in enumerate(beta_grid)
    ]
    selected = max(
        mean_curve,
        key=lambda row: (row["mean_top1"], row["mean_macro_f1"], -row["beta"]),
    )
    selected_index = int(round(selected["beta"] * 10))
    for seed, *_ in seed_scores:
        per_seed[str(seed)]["fused"] = curves[seed][selected_index]
    return mean_curve, selected, per_seed


def run_e027_log_fusion(config: dict[str, Any], project_root: Path) -> dict[str, Any]:
    """Run the official validation-only three-seed weighted log-probability experiment."""
    epsilon = float(config["fusion"]["epsilon"])
    seed_scores = []
    checksums: dict[str, dict[str, str]] = {}
    for item in config["inputs"]:
        seed = int(item["seed"])
        rgb_path = project_root / item["rgb_scores"]
        pose_path = project_root / item["pose_scores"]
        checksums[str(seed)] = {"rgb": sha256(rgb_path), "pose": sha256(pose_path)}
        seed_scores.append((seed, load_scores(rgb_path), load_scores(pose_path)))

    mean_curve, selected, per_seed = evaluate_log_fusion(seed_scores, epsilon=epsilon)
    selected_beta = selected["beta"]
    for seed, rgb_scores, pose_scores in seed_scores:
        rgb_ids, rgb_labels, rgb_probs = rgb_scores
        pose_ids, _, pose_probs = pose_scores
        pose_by_id = {str(video_id): i for i, video_id in enumerate(pose_ids)}
        pose_probs = pose_probs[[pose_by_id[str(video_id)] for video_id in rgb_ids]]
        fused = weighted_log_probabilities(pose_probs, rgb_probs, selected_beta, epsilon)
        per_seed[str(seed)]["fused"] = _metrics(fused, rgb_labels)
        per_seed[str(seed)]["top1_delta_vs_pose"] = (
            per_seed[str(seed)]["fused"]["top1"] - per_seed[str(seed)]["pose"]["top1"]
        )
        per_seed[str(seed)]["top5_delta_vs_pose"] = (
            per_seed[str(seed)]["fused"]["top5"] - per_seed[str(seed)]["pose"]["top5"]
        )
        per_seed[str(seed)]["macro_f1_delta_vs_pose"] = (
            per_seed[str(seed)]["fused"]["macro_f1"]
            - per_seed[str(seed)]["pose"]["macro_f1"]
        )

    seeds = [int(item["seed"]) for item in config["inputs"]]
    means = {
        source: {
            metric: float(np.mean([per_seed[str(seed)][source][metric] for seed in seeds]))
            for metric in METRICS
        }
        for source in ("rgb", "pose", "fused")
    }
    deltas = {
        metric: means["fused"][metric] - means["pose"][metric] for metric in METRICS
    }
    gate_passed = (
        deltas["top1"] > 0.0 and deltas["macro_f1"] >= 0.0 and deltas["top5"] >= 0.0
    )

    output_dir = project_root / config["output"]["root"]
    output_dir.mkdir(parents=True, exist_ok=True)
    curve_path = output_dir / "beta_curve.csv"
    fieldnames = ["beta"] + [
        f"seed_{seed}_{metric}" for seed in seeds for metric in METRICS
    ] + [f"mean_{metric}" for metric in METRICS]
    with curve_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(mean_curve)

    summary = {
        "status": "ok",
        "experiment_id": config["experiment"]["id"],
        "formula": "log(max(p_pose, eps)) + beta * log(max(p_rgb, eps))",
        "epsilon": epsilon,
        "beta_grid": [i / 10.0 for i in range(11)],
        "selected_beta": selected_beta,
        "selected_mean_metrics": {
            key: selected[key] for key in ("beta", "mean_top1", "mean_top5", "mean_macro_f1")
        },
        "per_seed": per_seed,
        "means": means,
        "mean_deltas_vs_pose": deltas,
        "gate_passed": gate_passed,
        "input_checksums": checksums,
    }
    atomic_json(output_dir / "summary.json", summary)
    return summary
