"""E036 clean-label signer partition and score-only RGB retrospective."""

from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .data import FeatureSplit
from .e026_rgb_fusion import load_scores
from .metrics import as_floats, new_evaluation_metrics
from .signer_diagnostic import _manifest_by_video, _select_rows


def _validate_source(
    split: FeatureSplit,
    manifest: dict[str, dict[str, str]],
    source: str,
    excluded: set[str],
) -> None:
    ids = [str(video_id).zfill(6) for video_id in split.video_id]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{source} features contain duplicate video IDs")
    for video_id, signer_id in zip(ids, split.signer_id, strict=True):
        if video_id in excluded:
            continue
        row = manifest.get(video_id)
        if row is None:
            raise ValueError(f"{source} feature video {video_id} is absent from clean manifest")
        if str(row["signer_id"]).zfill(3) != str(signer_id).zfill(3):
            raise ValueError(f"{source} signer mismatch for video {video_id}")


def _select_clean_rows(
    split: FeatureSplit,
    manifest: dict[str, dict[str, str]],
    signers: list[str],
    excluded: set[str],
    source: str,
) -> FeatureSplit:
    selected_signers = {str(signer).zfill(3) for signer in signers}
    ids = np.array([str(video_id).zfill(6) for video_id in split.video_id], dtype="U6")
    indices = np.flatnonzero(np.isin(split.signer_id, sorted(selected_signers)))
    selected_ids = set(ids[indices]) - excluded
    expected_ids = {
        video_id
        for video_id, row in manifest.items()
        if str(row["signer_id"]).zfill(3) in selected_signers and video_id not in excluded
    }
    if selected_ids != expected_ids:
        raise ValueError(f"{source} clean signer IDs differ")
    indices = np.array([index for index in indices if ids[index] not in excluded], dtype=np.int64)
    selected = _select_rows(split, indices)
    labels = np.array([int(manifest[str(video_id)]["class_id"]) for video_id in selected.video_id])
    return replace(selected, y=labels.astype(np.int64))


def _concat(splits: list[FeatureSplit]) -> FeatureSplit:
    return FeatureSplit(
        X=np.concatenate([split.X for split in splits]),
        y=np.concatenate([split.y for split in splits]),
        video_id=np.concatenate([split.video_id for split in splits]),
        signer_id=np.concatenate([split.signer_id for split in splits]),
        frame_indices=np.concatenate([split.frame_indices for split in splits]),
        valid_frame_mask=np.concatenate([split.valid_frame_mask for split in splits]),
        keypoint_valid_mask=np.concatenate([split.keypoint_valid_mask for split in splits]),
    )


def build_clean_signer_partition(
    source_splits: dict[str, FeatureSplit], partition: dict[str, Any], project_root: Path
) -> tuple[FeatureSplit, FeatureSplit]:
    """Build the locked E036 train/validation partition in memory."""
    manifests = {
        split: _manifest_by_video(project_root / path)
        for split, path in partition["manifests"].items()
    }
    excluded = {str(video_id).zfill(6) for video_id in partition["excluded_video_ids"]}
    prohibited = {str(signer).zfill(3) for signer in partition["prohibited_signers"]}
    for source, split in source_splits.items():
        _validate_source(split, manifests[source], source, excluded)
        usable = ~np.isin(split.video_id, sorted(excluded))
        if np.isin(split.signer_id[usable], sorted(prohibited)).any():
            raise ValueError(f"{source} contains a prohibited signer")

    train_config = partition["train"]
    train = _select_clean_rows(
        source_splits["train"],
        manifests["train"],
        train_config["signers"],
        excluded,
        "train",
    )
    val_config = partition["val"]
    val = _concat(
        [
            _select_clean_rows(
                source_splits[source], manifests[source], signers, excluded, source
            )
            for source, signers in val_config["sources"].items()
        ]
    )
    if len(train.X) != train_config["expected_samples"]:
        raise ValueError("Unexpected E036 train partition sample count")
    if len(val.X) != val_config["expected_samples"]:
        raise ValueError("Unexpected E036 validation partition sample count")
    if set(map(str, train.video_id)) & set(map(str, val.video_id)):
        raise ValueError("E036 train and validation partitions overlap by video ID")
    return train, val


def score_metrics(probabilities: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    metric = new_evaluation_metrics(400, torch.device("cpu"))
    metric.update(torch.from_numpy(probabilities), torch.from_numpy(labels))
    return as_floats(metric.compute())


def _score_rows(
    source: dict[str, Any],
    manifest: dict[str, dict[str, str]],
    signers: list[str],
    project_root: Path,
) -> list[dict[str, Any]]:
    manifest_ids = np.array(list(manifest), dtype="U6")
    manifest_signers = np.array(
        [str(manifest[video_id]["signer_id"]).zfill(3) for video_id in manifest]
    )
    rows: list[dict[str, Any]] = []
    for seed, archive in source["archives"].items():
        video_ids, labels, probabilities = load_scores(project_root / archive)
        if set(map(str, video_ids)) != set(manifest_ids):
            raise ValueError(f"{source['name']} seed {seed} score IDs differ from clean validation")
        expected_labels = np.array(
            [int(manifest[str(video_id)]["class_id"]) for video_id in video_ids], dtype=np.int64
        )
        if not np.array_equal(labels, expected_labels):
            raise ValueError(f"{source['name']} seed {seed} labels differ from clean validation")
        order = {str(video_id): index for index, video_id in enumerate(video_ids)}
        for signer in signers:
            signer_ids = manifest_ids[manifest_signers == signer]
            indices = np.array([order[str(video_id)] for video_id in signer_ids])
            rows.append(
                {
                    "model": source["name"],
                    "seed": int(seed),
                    "signer_id": signer,
                    "samples": int(len(indices)),
                    **score_metrics(probabilities[indices], labels[indices]),
                }
            )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: tuple[str, ...]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def run_rgb_signer_report(config: dict[str, Any], project_root: Path) -> dict[str, Any]:
    """Write E036 RGB per-signer metrics from existing score archives."""
    report = config["report"]
    manifest = _manifest_by_video(project_root / report["manifest"])
    signers = [str(signer).zfill(3) for signer in report["signers"]]
    per_seed = [
        row
        for source in report["sources"]
        for row in _score_rows(source, manifest, signers, project_root)
    ]
    aggregate = []
    for model in sorted({row["model"] for row in per_seed}):
        for signer in signers:
            selected = [
                row for row in per_seed if row["model"] == model and row["signer_id"] == signer
            ]
            aggregate.append(
                {
                    "model": model,
                    "signer_id": signer,
                    "samples": selected[0]["samples"],
                    "seeds": len(selected),
                    **{
                        f"{metric}_mean": float(np.mean([row[metric] for row in selected]))
                        for metric in ("top1", "top5", "macro_f1")
                    },
                    **{
                        f"{metric}_std": float(np.std([row[metric] for row in selected], ddof=1))
                        for metric in ("top1", "top5", "macro_f1")
                    },
                }
            )
    output = project_root / config["output"]["root"]
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(
        output / "per_seed_signer_metrics.csv",
        per_seed,
        ("model", "seed", "signer_id", "samples", "top1", "top5", "macro_f1"),
    )
    _write_csv(
        output / "per_signer_seed_summary.csv",
        aggregate,
        (
            "model", "signer_id", "samples", "seeds", "top1_mean", "top1_std",
            "top5_mean", "top5_std", "macro_f1_mean", "macro_f1_std",
        ),
    )
    summary = {"status": "ok", "signers": signers, "aggregate": aggregate}
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
