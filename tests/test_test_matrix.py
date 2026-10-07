from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from src.config import _validate_test_matrix, load_config
from src.test_matrix import run_test_matrix


def test_test_matrix_runs_locked_test_configs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    test_config: dict[str, Any] = {
        "experiment": {"id": "test_source", "operation": "test_evaluate", "mode": "full", "seed": 42},
        "data": {"repo_id": "data", "revision": "revision", "artifact_path": "artifact", "num_classes": 400},
        "feature": {"view": "pose53"},
        "model": {"name": "bilstm", "input_channels": 2, "hidden_size": 128},
        "checkpoint": {"source_experiment": "source", "repo_id": "model", "revision": "pinned", "path": "best.pt"},
        "evaluation": {"split": "test", "batch_size": 32},
        "output": {"root": "artifacts/test_evaluate"},
    }
    source_path = tmp_path / "source.yaml"
    source_path.write_text(yaml.safe_dump(test_config), encoding="utf-8")
    matrix = {
        "experiment": {"id": "matrix", "operation": "test_matrix", "mode": "full", "seed": 42},
        "matrix": {"configs": ["source.yaml"]},
        "output": {"root": "artifacts/test_evaluate/matrix"},
    }
    config_path = tmp_path / "matrix.yaml"
    config_path.write_text(yaml.safe_dump(matrix), encoding="utf-8")
    config = load_config(config_path)

    def fake_run_test_evaluate(config: dict[str, Any], _: Path, *, device: str | None) -> dict[str, Any]:
        assert device == "cpu"
        return {
            "source_experiment": config["checkpoint"]["source_experiment"],
            "seed": config["experiment"]["seed"],
            "test_top1": 0.8,
            "test_top5": 0.9,
            "test_macro_f1": 0.7,
        }

    monkeypatch.setattr("src.test_matrix.run_test_evaluate", fake_run_test_evaluate)
    summary = run_test_matrix(config, tmp_path, device="cpu")

    assert summary["entries"][0]["config"] == "source.yaml"
    report = json.loads((tmp_path / "artifacts/test_evaluate/matrix/summary.json").read_text())
    assert report["entries"][0]["test_top1"] == 0.8


def test_test_matrix_rejects_empty_config_list() -> None:
    config = {
        "experiment": {"id": "matrix", "operation": "test_matrix", "mode": "full", "seed": 42},
        "matrix": {"configs": []},
        "output": {"root": "artifacts/test_evaluate/matrix"},
    }
    with pytest.raises(ValueError, match="matrix.configs"):
        _validate_test_matrix(config)
