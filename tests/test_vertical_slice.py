from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from src.config import load_config, output_path
from src.data import FeatureSplit, select_view
from src.experiment import run_experiment
from src.frame_sampling import uniform_indices


def _split(samples: int = 6, nodes: int = 53) -> FeatureSplit:
    rng = np.random.default_rng(42)
    frames = 4
    return FeatureSplit(
        X=rng.random((samples, frames, nodes, 3), dtype=np.float32),
        y=np.arange(samples, dtype=np.int64),
        video_id=np.array([f"video-{index}" for index in range(samples)]),
        signer_id=np.array([f"signer-{index}" for index in range(samples)]),
        frame_indices=np.tile(np.arange(frames, dtype=np.int32), (samples, 1)),
        valid_frame_mask=np.ones((samples, frames), dtype=np.uint8),
        keypoint_valid_mask=np.ones((samples, frames, nodes), dtype=np.uint8),
    )


def test_uniform_frame_sampling() -> None:
    assert uniform_indices(10, 4).tolist() == [0, 3, 6, 9]
    assert uniform_indices(2, 4).tolist() == [0, 1, -1, -1]


def test_config_and_pose42_contract(tmp_path: Path) -> None:
    configs_dir = Path(__file__).parents[1] / "configs/experiments"
    smoke_config = load_config(configs_dir / "e001_tcn_pose42.yaml")
    local_config = load_config(configs_dir / "frame_selection/e101_uniform24_tcn_pose53.yaml")
    assert (
        local_config["data"]["split_path"] == "artifacts/frame_selection_splits/uniform_24_pose53"
    )
    for seed in (42, 43, 44):
        full_config = load_config(configs_dir / f"e001_tcn_pose42_s{seed}.yaml")
        assert full_config["experiment"]["seed"] == seed
        assert full_config["experiment"]["mode"] == "full"

    pose42 = select_view(_split(), smoke_config["feature"]["view"])

    assert pose42.X.shape == (6, 4, 42, 3)
    assert pose42.keypoint_valid_mask.shape == (6, 4, 42)
    assert output_path(smoke_config, tmp_path).name == "e001_tcn_pose42_s42"


def test_select_spoter54_from_pose53_semantics_and_metadata() -> None:
    source = _split(samples=1, nodes=53)
    source.X[0, 0, :, 0] = np.arange(53, dtype=np.float32)
    source.X[0, 0, :, 1] = np.arange(53, dtype=np.float32) + 100
    source.X[0, 0, :, 2] = np.arange(53, dtype=np.float32) / 100
    source.keypoint_valid_mask[0, 0, 1] = 0

    view = select_view(source, "spoter54")

    assert view.X.shape == (1, 4, 54, 3)
    np.testing.assert_array_equal(view.X[0, 0, :5, 0], np.arange(48, 53))
    np.testing.assert_allclose(view.X[0, 0, 5, :2], [0.5, 100.5])
    assert view.X[0, 0, 5, 2] == 0.0
    np.testing.assert_array_equal(view.X[0, 0, 6:, 0], np.arange(48))
    assert view.keypoint_valid_mask[0, 0, 5] == 0
    np.testing.assert_array_equal(view.valid_frame_mask, source.valid_frame_mask)
    np.testing.assert_array_equal(view.frame_indices, source.frame_indices)


def test_select_pose42_and_pose53_from_pose59() -> None:
    source = _split(samples=1, nodes=59)
    source.X[:] = np.arange(59, dtype=np.float32)[None, None, :, None] / 100

    pose42 = select_view(source, "pose42")
    pose53 = select_view(source, "pose53")

    np.testing.assert_allclose(pose42.X[0, 0, :, 0] * 100, np.arange(17, 59))
    np.testing.assert_allclose(
        pose53.X[0, 0, :, 0] * 100,
        np.array([*range(5, 11), *range(17, 59), *range(5)]),
    )
    with pytest.raises(ValueError, match="Cannot select pose59 from Pose53"):
        select_view(_split(), "pose59")


def test_tcn_smoke_writes_run_outputs_and_resumes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = {
        "experiment": {"id": "test_tcn", "mode": "smoke", "seed": 42},
        "data": {
            "repo_id": "unused",
            "revision": "test-revision",
            "artifact_path": "unused",
            "num_classes": 6,
            "subset": {"train": 6, "val": 6},
        },
        "feature": {"view": "pose42"},
        "model": {
            "name": "tcn",
            "input_channels": 2,
            "channels": [4],
            "dilations": [1],
            "kernel_size": 3,
            "dropout": 0.0,
        },
        "train": {
            "epochs": 1,
            "patience": 0,
            "batch_size": 3,
            "optimizer": "adamw",
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "grad_clip_norm": 1.0,
        },
        "evaluation": {"validation": True, "predictions": True},
        "output": {"root": "artifacts/runs"},
        "integrations": {"wandb": False, "hf_checkpoints": False},
    }

    def fake_load_split(*_: object) -> FeatureSplit:
        return _split()

    monkeypatch.setattr("src.experiment.load_split", fake_load_split)

    summary = run_experiment(config, tmp_path, device="cpu")
    run_dir = output_path(config, tmp_path)

    assert summary["status"] == "ok"
    assert summary["experiment_id"] == "test_tcn"
    assert summary["seed"] == 42
    assert summary["model"] == "tcn"
    assert summary["feature_view"] == "pose42"
    assert summary["data_revision"] == "test-revision"
    assert summary["epochs_trained"] == 1
    assert summary["best_epoch"] == 1
    assert "best_validation_top1" in summary
    assert "best_validation_top5" in summary
    assert "best_validation_macro_f1" in summary
    assert "best_validation_loss" in summary
    assert summary["environment"]["device"] == "cpu"
    assert (run_dir / "best.pt").is_file()
    assert (run_dir / "last.pt").is_file()
    assert json.loads((run_dir / "summary.json").read_text())["data_revision"] == "test-revision"

    with (run_dir / "predictions.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
        assert len(rows) == 6
        assert set(rows[0].keys()) == {
            "video_id",
            "true_class_id",
            "pred_class_id",
            "confidence",
            "correct",
        }

    with pytest.raises(FileExistsError, match="Run already exists"):
        run_experiment(config, tmp_path, device="cpu", resume=False)

    config["train"]["epochs"] = 2
    resumed = run_experiment(config, tmp_path, device="cpu", resume=True)
    assert resumed["epochs_trained"] == 2
