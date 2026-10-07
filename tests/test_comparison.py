from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from src.comparison import summarize_runs


def _run(tmp_path: Path, length: int, seed: int, top1: float) -> Path:
    experiment_id = f"comparison_uniform{length}"
    config = {
        "experiment": {"id": experiment_id, "mode": "full", "seed": seed},
        "data": {
            "repo_id": "features",
            "revision": "revision",
            "artifact_path": f"uniform{length}",
            "num_classes": 400,
        },
        "feature": {"view": "pose59"},
        "model": {"name": "tcn", "input_channels": 2},
        "train": {"epochs": 10, "patience": 2, "batch_size": 32},
        "output": {"root": "runs"},
        "integrations": {},
    }
    config_path = tmp_path / f"{experiment_id}_s{seed}.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    run = tmp_path / "runs" / f"{experiment_id}_s{seed}"
    run.mkdir(parents=True)
    (run / "summary.json").write_text(
        json.dumps(
            {
                "experiment_id": experiment_id,
                "seed": seed,
                "data_revision": "revision",
                "environment": {"input_shape": [1, length, 59, 3]},
                "best_validation_top1": top1,
                "best_validation_top5": top1 + 0.1,
                "best_validation_macro_f1": top1 - 0.01,
            }
        ),
        encoding="utf-8",
    )
    return config_path


def test_summarize_runs_groups_seeds_and_selects_best(tmp_path: Path) -> None:
    configs = [
        _run(tmp_path, length, seed, top1)
        for length, values in ((16, (0.80, 0.81, 0.82)), (24, (0.84, 0.85, 0.86)))
        for seed, top1 in zip((42, 43, 44), values, strict=True)
    ]

    report = summarize_runs(configs, tmp_path, tmp_path / "report")

    assert report["best_experiment_id"] == "comparison_uniform24"
    assert report["experiments"][1]["metrics"]["best_validation_top1"] == {
        "mean": 0.85,
        "sample_sd": pytest.approx(0.01),
    }
    assert (tmp_path / "report/comparison.json").is_file()
    assert (tmp_path / "report/comparison.csv").is_file()


def test_summarize_runs_requires_three_seeds(tmp_path: Path) -> None:
    configs = [_run(tmp_path, 16, seed, 0.8) for seed in (42, 43)]

    with pytest.raises(ValueError, match="must contain seeds"):
        summarize_runs(configs, tmp_path, tmp_path / "report")
