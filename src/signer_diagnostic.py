"""Inference-only per-signer diagnostic for a frozen classifier checkpoint."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .checkpoint import load_best
from .config import output_path
from .data import FeatureSplit, load_split, make_loader, select_view
from .evaluate import evaluate
from .integrations.huggingface import download_file
from .models import build_model


def _select_rows(split: FeatureSplit, indices: np.ndarray) -> FeatureSplit:
    return FeatureSplit(
        X=split.X[indices],
        y=split.y[indices],
        video_id=split.video_id[indices],
        signer_id=split.signer_id[indices],
        frame_indices=split.frame_indices[indices],
        valid_frame_mask=split.valid_frame_mask[indices],
        keypoint_valid_mask=split.keypoint_valid_mask[indices],
    )


def _manifest_by_video(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = {row["video_id"]: row for row in csv.DictReader(handle)}
    if not rows:
        raise ValueError(f"Empty diagnostic manifest: {path}")
    return rows


def _validate_selected_labels(split: FeatureSplit, manifest: dict[str, dict[str, str]]) -> None:
    if len(split.video_id) != len(set(map(str, split.video_id))):
        raise ValueError("Diagnostic features contain duplicate video IDs")
    for video_id, class_id, signer_id in zip(
        split.video_id, split.y, split.signer_id, strict=True
    ):
        row = manifest.get(str(video_id))
        if row is None:
            raise ValueError(f"Feature video {video_id} is absent from the clean manifest")
        if int(row["class_id"]) != int(class_id):
            raise ValueError(f"Class label mismatch for video {video_id}")
        if str(row["signer_id"]).zfill(3) != str(signer_id).zfill(3):
            raise ValueError(f"Signer mismatch for video {video_id}")


def _write_predictions(
    path: Path,
    split: FeatureSplit,
    predictions: list[int],
    confidences: list[float],
    manifest: dict[str, dict[str, str]],
) -> None:
    class_to_gloss = {int(row["class_id"]): row["gloss"] for row in manifest.values()}
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "video_id",
                "signer_id",
                "true_class_id",
                "true_gloss",
                "pred_class_id",
                "pred_gloss",
                "confidence",
                "correct",
            ]
        )
        for video_id, signer_id, true, pred, confidence in zip(
            split.video_id, split.signer_id, split.y, predictions, confidences, strict=True
        ):
            true_row = manifest[str(video_id)]
            pred_gloss = class_to_gloss[int(pred)]
            writer.writerow(
                [
                    video_id,
                    signer_id,
                    true,
                    true_row["gloss"],
                    pred,
                    pred_gloss,
                    confidence,
                    int(int(true) == int(pred)),
                ]
            )


def _write_per_signer(path: Path, split: FeatureSplit, predictions: list[int]) -> None:
    counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for signer_id, true, pred in zip(split.signer_id, split.y, predictions, strict=True):
        totals = counts[str(signer_id)]
        totals[0] += 1
        totals[1] += int(int(true) == int(pred))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["signer_id", "samples", "correct", "top1"])
        for signer_id, (samples, correct) in sorted(counts.items()):
            writer.writerow([signer_id, samples, correct, correct / samples])


def run_signer_diagnostic(
    config: dict[str, Any], project_root: Path, *, device: str | None = None
) -> dict[str, Any]:
    """Evaluate one frozen checkpoint on selected clean-manifest signer rows."""
    target = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    data = config["data"]
    diagnostic = config["diagnostic"]
    cache_dir = project_root / "cache" / "hf"
    source = select_view(load_split(data, diagnostic["split"], cache_dir), config["feature"]["view"])
    manifest = _manifest_by_video(project_root / diagnostic["manifest"])
    selected_signers = {str(value).zfill(3) for value in diagnostic["signers"]}
    indices = np.flatnonzero(np.isin(source.signer_id, sorted(selected_signers)))
    selected = _select_rows(source, indices)
    _validate_selected_labels(selected, manifest)

    checkpoint = config["checkpoint"]
    checkpoint_path = download_file(
        repo_id=checkpoint["repo_id"],
        repo_type="model",
        revision=checkpoint["revision"],
        filename=checkpoint["path"],
        cache_dir=cache_dir,
    )
    model = build_model(
        config["model"], nodes=selected.X.shape[2], num_classes=int(data["num_classes"])
    ).to(target)
    checkpoint_state = load_best(checkpoint_path, model, target)
    metrics, predictions, confidences = evaluate(
        model,
        make_loader(selected, int(diagnostic["batch_size"]), shuffle=False),
        num_classes=int(data["num_classes"]),
        device=target,
    )

    output = output_path(config, project_root)
    output.mkdir(parents=True, exist_ok=True)
    _write_predictions(output / "predictions.csv", selected, predictions, confidences, manifest)
    _write_per_signer(output / "per_signer.csv", selected, predictions)
    summary = {
        "status": "ok",
        "experiment_id": config["experiment"]["id"],
        "operation": "signer_diagnostic",
        "source_experiment": checkpoint["source_experiment"],
        "checkpoint": {
            "repo_id": checkpoint["repo_id"],
            "revision": checkpoint["revision"],
            "path": checkpoint["path"],
            "epoch": checkpoint_state["epoch"],
        },
        "data": {
            "repo_id": data["repo_id"],
            "revision": data["revision"],
            "artifact_path": data["artifact_path"],
            "split": diagnostic["split"],
            "manifest": diagnostic["manifest"],
            "signers": sorted(selected_signers),
            "samples": len(selected.X),
        },
        "feature_view": config["feature"]["view"],
        "metrics": metrics,
        "environment": {
            "device": str(target),
            "torch": torch.__version__,
            "cuda": torch.version.cuda if target.type == "cuda" else None,
            "gpu": torch.cuda.get_device_name(target) if target.type == "cuda" else None,
        },
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
