from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path
from typing import Any

from .config import load_config, output_path

METRICS = (
    "best_validation_top1",
    "best_validation_top5",
    "best_validation_macro_f1",
)


def summarize_runs(
    config_paths: list[Path], project_root: Path, output_dir: Path
) -> dict[str, Any]:
    """Aggregate compatible three-seed classifier runs by experiment ID."""
    groups: dict[str, list[tuple[dict[str, Any], dict[str, Any], Path]]] = {}
    protocol = None

    for config_path in config_paths:
        config = load_config(config_path)
        summary_path = output_path(config, project_root) / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        current_protocol = {
            "data": {
                key: config["data"].get(key) for key in ("repo_id", "revision", "num_classes")
            },
        }
        if protocol is None:
            protocol = current_protocol
        elif current_protocol != protocol:
            raise ValueError(f"Incompatible protocol: {config_path}")
        if (
            summary["experiment_id"] != config["experiment"]["id"]
            or summary["seed"] != config["experiment"]["seed"]
            or summary["data_revision"] != config["data"].get("revision")
        ):
            raise ValueError(f"Summary does not match config: {config_path}")
        experiment_id = summary["experiment_id"]
        runs = groups.setdefault(experiment_id, [])
        if runs:
            reference = runs[0][0]
            if any(config[key] != reference[key] for key in ("feature", "model", "train")):
                raise ValueError(f"Inconsistent protocol within {experiment_id}")
        runs.append((config, summary, summary_path))

    if not groups:
        raise ValueError("At least one run config is required")

    results = []
    for experiment_id, runs in sorted(groups.items()):
        seeds = sorted(int(config["experiment"]["seed"]) for config, _, _ in runs)
        if seeds != [42, 43, 44]:
            raise ValueError(f"{experiment_id} must contain seeds 42, 43, 44")
        artifact_paths = {config["data"].get("artifact_path") for config, _, _ in runs}
        lengths = {int(summary["environment"]["input_shape"][1]) for _, summary, _ in runs}
        if len(artifact_paths) != 1 or len(lengths) != 1:
            raise ValueError(f"Inconsistent data within {experiment_id}")

        metrics = {}
        for metric in METRICS:
            values = [float(summary[metric]) for _, summary, _ in runs]
            metrics[metric] = {
                "mean": statistics.mean(values),
                "sample_sd": statistics.stdev(values),
            }
        results.append(
            {
                "experiment_id": experiment_id,
                "model": runs[0][0]["model"]["name"],
                "feature_view": runs[0][0]["feature"]["view"],
                "sequence_length": lengths.pop(),
                "artifact_path": artifact_paths.pop(),
                "seeds": seeds,
                "metrics": metrics,
                "runs": [
                    {
                        "seed": int(config["experiment"]["seed"]),
                        "summary": str(path.relative_to(project_root)),
                        **{metric: float(summary[metric]) for metric in METRICS},
                    }
                    for config, summary, path in sorted(
                        runs, key=lambda item: int(item[0]["experiment"]["seed"])
                    )
                ],
            }
        )

    best = max(results, key=lambda item: item["metrics"][METRICS[0]]["mean"])
    report = {
        "protocol": protocol,
        "best_experiment_id": best["experiment_id"],
        "selection_metric": METRICS[0],
        "experiments": results,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "comparison.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    with (output_dir / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["experiment_id", "model", "feature_view", "sequence_length", "artifact_path"] + [
            f"{metric}_{stat}" for metric in METRICS for stat in ("mean", "sample_sd")
        ]
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "experiment_id": result["experiment_id"],
                    "model": result["model"],
                    "feature_view": result["feature_view"],
                    "sequence_length": result["sequence_length"],
                    "artifact_path": result["artifact_path"],
                    **{
                        f"{metric}_{stat}": result["metrics"][metric][stat]
                        for metric in METRICS
                        for stat in ("mean", "sample_sd")
                    },
                }
            )
    return report
