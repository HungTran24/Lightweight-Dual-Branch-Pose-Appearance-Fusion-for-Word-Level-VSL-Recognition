from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.checkpoint import save_best
from src.config import _validate_classifier
from src.data import FeatureSplit
from src.experiment import run_test_evaluate
from src.models import build_model
from src.runtime.colab import run_on_colab


def _split() -> FeatureSplit:
    return FeatureSplit(
        X=np.zeros((5, 2, 42, 3), dtype=np.float32),
        y=np.array([0, 1, 2, 3, 4], dtype=np.int64),
        video_id=np.array(["000001", "000002", "000003", "000004", "000005"], dtype="U6"),
        signer_id=np.array(["003", "012", "019", "003", "012"], dtype="U3"),
        frame_indices=np.zeros((5, 2), dtype=np.int32),
        valid_frame_mask=np.ones((5, 2), dtype=np.uint8),
        keypoint_valid_mask=np.ones((5, 2, 42), dtype=np.uint8),
    )


def _config() -> dict[str, Any]:
    return {
        "experiment": {"id": "test_eval", "operation": "test_evaluate", "mode": "full", "seed": 42},
        "data": {"repo_id": "unused", "revision": "test-revision", "artifact_path": "unused", "num_classes": 5},
        "feature": {"view": "pose42"},
        "model": {"name": "tcn", "input_channels": 2, "channels": [4], "dilations": [1], "kernel_size": 3, "dropout": 0.0},
        "checkpoint": {"source_experiment": "source", "repo_id": "unused", "revision": "pinned", "path": "best.pt"},
        "evaluation": {"split": "test", "batch_size": 2, "predictions": True, "export_scores": True},
        "output": {"root": "artifacts/test_evaluate"},
    }


def test_test_evaluate_uses_only_frozen_test_inference(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = _config()
    _validate_classifier(config)
    checkpoint = tmp_path / "best.pt"
    model = build_model(config["model"], nodes=42, num_classes=5)
    save_best(checkpoint, model, epoch=7, metrics={})
    calls: list[str] = []

    def fake_load_split(_: object, split: str, _cache_dir: Path) -> FeatureSplit:
        assert _cache_dir.name == "hf"
        calls.append(split)
        return _split()

    monkeypatch.setattr("src.experiment.load_split", fake_load_split)
    monkeypatch.setattr("src.experiment.download_file", lambda **_: checkpoint)
    summary = run_test_evaluate(config, tmp_path, device="cpu")

    assert calls == ["test"]
    assert summary["checkpoint"]["epoch"] == 7
    assert summary["test_top1"] == summary["metrics"]["top1"]
    output = tmp_path / "artifacts/test_evaluate/test_eval_s42"
    assert json.loads((output / "summary.json").read_text())["split"] == "test"
    with (output / "predictions.csv").open(newline="", encoding="utf-8") as handle:
        assert len(list(csv.DictReader(handle))) == 5
    with np.load(output / "scores.npz", allow_pickle=False) as scores:
        assert scores["probabilities"].shape == (5, 5)
        assert scores["probabilities"].dtype == np.float32


def test_test_evaluate_rejects_non_test_split() -> None:
    config = _config()
    config["evaluation"]["split"] = "val"
    with pytest.raises(ValueError, match="evaluation.split=test"):
        _validate_classifier(config)


def test_colab_runner_syncs_local_test_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "project"
    checkpoint = project / "artifacts/runs/source_s42/best.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    config = _config()
    config["checkpoint"] = {
        "source_experiment": "source",
        "local_path": "artifacts/runs/source_s42/best.pt",
    }
    config_path = project / "test.yaml"
    config_path.write_text("unused", encoding="utf-8")
    identity = project / "identity"
    identity.write_text("unused", encoding="utf-8")
    commands: list[tuple[str, ...]] = []

    def fake_command(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
        del check
        commands.append(tuple(args))
        return subprocess.CompletedProcess(args=list(args), returncode=0)

    monkeypatch.setattr("src.runtime.colab.load_config", lambda _: config)
    monkeypatch.setattr("src.runtime.colab._command", fake_command)
    run_on_colab(
        config_path,
        project,
        session="test-session",
        gpu="T4",
        identity=identity,
        timeout=100,
        resume=False,
    )

    checkpoint_syncs = [
        command
        for command in commands
        if command and command[0] == "rsync" and command[-2] == str(checkpoint)
    ]
    assert len(checkpoint_syncs) == 1
    assert checkpoint_syncs[0][-1].endswith("/artifacts/runs/source_s42/best.pt")
    project_sync = next(command for command in commands if command and command[0] == "rsync")
    assert ".venv*" in project_sync


def test_colab_frontend_benchmark_uses_pinned_python310_runtime() -> None:
    from src.runtime import colab

    command = colab.remote_run_command(
        "/content/project_ai",
        "/content/project_ai/configs/experiments/benchmark_frontends_pose_a100.yaml",
        "",
        "frontend_benchmark",
    )

    subprocess.run(["bash", "-n", "-c", command], check=True)
    assert "export MPLBACKEND=Agg" in command
    assert "python -m uv venv --seed --python 3.10.21" in command
    assert "--index-url https://download.pytorch.org/whl/cu118" in command
    assert "torch==2.0.1+cu118 torchvision==0.15.2+cu118" in command
    assert "numpy==1.26.4 openmim==0.3.9" in command
    assert "--no-deps https://download.openmmlab.com/mmcv/dist/cu118/torch2.0.0/" in command
    assert "--no-build-isolation mmengine==0.9.0 mmdet==3.2.0 mmpose==1.3.1" in command
    assert "/content/project-ai-frontend-venv/bin/python scripts/run.py" in command
