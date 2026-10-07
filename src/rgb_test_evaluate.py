"""Frozen RGB test inference and fixed-beta score fusion."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .artifacts import atomic_json, sha256
from .checkpoint import load_best
from .e026_rgb_features import atomic_npz
from .e026_rgb_fusion import _rgb_loader, _score_probabilities, _validate_scores, load_scores
from .e027_log_fusion import weighted_log_probabilities
from .e028_rgb_fusion import _join_with_pose, _loader, load_temporal_rgb_split
from .evaluate import evaluate_with_scores
from .models import build_model


def _rgb_split(config: dict[str, Any], root: Path, kind: str):
    if kind == "global":
        from .e026_rgb_fusion import load_rgb_pooled_split

        split = load_rgb_pooled_split(config, root, "test")
        return split, _rgb_loader(split, int(config["batch_size"]), shuffle=False)
    split = load_temporal_rgb_split(config, root, "test")
    return split, _loader(split, int(config["batch_size"]), shuffle=False)


def run_rgb_test_evaluate(config: dict[str, Any], project_root: Path, *, device: str | None = None) -> dict[str, Any]:
    """Evaluate frozen RGB checkpoints and pre-specified beta=0.5 fusion on test."""
    target = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    beta, epsilon = float(config["fusion"]["beta"]), float(config["fusion"]["epsilon"])
    if beta != 0.5 or epsilon != 1e-12:
        raise ValueError("RGB test fusion must use beta=0.5 and epsilon=1e-12")
    output_root = project_root / config["output"]["root"]
    results: dict[str, list[dict[str, Any]]] = {}
    for branch in config["branches"]:
        name, kind = branch["name"], branch["kind"]
        branch_results: list[dict[str, Any]] = []
        for seed in (42, 43, 44):
            rgb_config = {**config, "rgb_data": branch["rgb_data"]}
            split, loader = _rgb_split(rgb_config, project_root, kind)
            checkpoint_path = project_root / branch["checkpoints"][str(seed)]
            model = build_model(branch["model"], nodes=1, num_classes=400).to(target)
            checkpoint = load_best(checkpoint_path, model, target)
            metrics, predictions, confidence, labels, probabilities = evaluate_with_scores(
                model, loader, num_classes=400, device=target
            )
            scores = probabilities.numpy().astype(np.float32, copy=False)
            _validate_scores(split.video_id, split.labels, scores)
            run_dir = output_root / name / f"seed{seed}"
            atomic_npz(run_dir / "scores.npz", video_id=split.video_id, true_class_id=np.array(labels, dtype=np.int64), probabilities=scores)
            with (run_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["video_id", "true_class_id", "pred_class_id", "confidence"])
                writer.writerows(zip(split.video_id, labels, predictions, confidence, strict=True))
            result = {"seed": seed, "rgb_metrics": metrics, "checkpoint": {"path": str(checkpoint_path.relative_to(project_root)), "sha256": sha256(checkpoint_path), "epoch": checkpoint["epoch"]}}
            pose_path = project_root / branch["pose_scores"][str(seed)]
            labels_joined, rgb_probs, pose_probs = _join_with_pose(
                (split.video_id, split.labels, scores), load_scores(pose_path)
            )
            fused = weighted_log_probabilities(pose_probs, rgb_probs, beta, epsilon)
            atomic_npz(run_dir / "fused_scores.npz", video_id=split.video_id, true_class_id=labels_joined, probabilities=fused)
            result["fusion"] = {
                "pre_specified_test_extension": bool(branch.get("pre_specified_test_extension", False)),
                "beta": beta,
                "epsilon": epsilon,
                "pose_scores": str(pose_path.relative_to(project_root)),
                "metrics": _score_probabilities(fused, labels_joined),
            }
            atomic_json(run_dir / "summary.json", result)
            branch_results.append(result)
        results[name] = branch_results
    aggregate = {
        name: {
            metric: {"mean": float(np.mean(values)), "sample_sd": float(np.std(values, ddof=1))}
            for metric, values in {
                "rgb_top1": [row["rgb_metrics"]["top1"] for row in rows],
                "rgb_top5": [row["rgb_metrics"]["top5"] for row in rows],
                "rgb_macro_f1": [row["rgb_metrics"]["macro_f1"] for row in rows],
                "fused_top1": [row["fusion"]["metrics"]["top1"] for row in rows],
                "fused_top5": [row["fusion"]["metrics"]["top5"] for row in rows],
                "fused_macro_f1": [row["fusion"]["metrics"]["macro_f1"] for row in rows],
            }.items()
        }
        for name, rows in results.items()
    }
    summary = {"status": "ok", "experiment_id": config["experiment"]["id"], "test_manifest": config["manifests"]["test"], "test_videos": 2508, "fusion": {"beta": beta, "epsilon": epsilon}, "branches": results, "aggregate": aggregate}
    atomic_json(output_root / "summary.json", summary)
    return summary
