"""Focused CPU tests for YOLO evaluation, checkpoint, coordinator dispatch, and config."""

from __future__ import annotations

import importlib
import json
import subprocess
import tarfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import torch.nn as nn

import src.yolo_pilot as yolo_pilot_shim
from src.config import load_config
from src.runtime.colab import run_on_colab
from src.yolo.checkpoint import load_checkpoint, save_checkpoint
from src.yolo.dataset import YoloDataset
from src.yolo.evaluation import evaluate_yolo
from src.yolo.experiment import run_materialized, run_yolo
from src.yolo.fit import fit_yolo
from src.yolo.training import build_yolo_scheduler


class DummyDualHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 8, kernel_size=1)
        self.head_person = nn.Linear(8, 4 + 1 + 17 * 3)
        self.head_hands = nn.Linear(8, 4 + 2 + 21 * 3)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "person": torch.zeros((len(x), 4 + 1 + 17 * 3, 10)),
            "hands": torch.zeros((len(x), 4 + 2 + 21 * 3, 10)),
        }


def dummy_loss(pred: Any, target: Any) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    value = torch.tensor(0.5, dtype=torch.float32)
    return value, {"pose_loss": value}


def test_evaluate_yolo_cpu_feed() -> None:
    model = DummyDualHead()
    batch = {
        "image": torch.zeros((2, 3, 640, 640), dtype=torch.uint8),
        "boxes": torch.tensor(
            [
                [[10.0, 10.0, 50.0, 50.0], [20.0, 20.0, 30.0, 30.0], [0.0, 0.0, 0.0, 0.0]],
                [[10.0, 10.0, 50.0, 50.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
            ]
        ),
        "classes": torch.tensor([[0, 1, -1], [0, -1, -1]], dtype=torch.int8),
        "person_keypoints": torch.zeros((2, 17, 3), dtype=torch.float32),
        "hand_keypoints": torch.zeros((2, 2, 21, 3), dtype=torch.float32),
    }
    dataloader = [batch]
    config = {
        "model": {"image_size": 640},
        "evaluation": {"metric_nms_confidence": 0.001, "metric_nms_iou": 0.7, "max_detections": 10},
    }

    results = evaluate_yolo(
        model,
        dummy_loss,
        dummy_loss,
        dataloader,  # type: ignore[arg-type]
        config=config,
        device="cpu",
        amp=False,
    )

    assert "validation_loss_per_image" in results
    assert results["total_images"] == 2
    assert "end_to_end_images_per_second" in results
    assert isinstance(results["person_bbox"]["map50"], float)
    assert isinstance(results["person_pose17_oks"]["map50"], float)
    assert isinstance(results["hand_bbox"]["map50"], float)
    assert isinstance(results["hand_pose21_oks"]["map50"], float)


def test_save_and_load_checkpoint(tmp_path: Path) -> None:
    model = DummyDualHead()
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    ckpt_path = tmp_path / "model.pt"

    state = {
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "next_sample": 100,
        "identity": "test_identity",
    }
    save_checkpoint(ckpt_path, state=state)
    assert ckpt_path.is_file()

    new_model = DummyDualHead()
    new_opt = torch.optim.SGD(new_model.parameters(), lr=0.01)
    loaded = load_checkpoint(ckpt_path, device="cpu", model=new_model, optimizer=new_opt)
    assert loaded["next_sample"] == 100
    assert loaded["identity"] == "test_identity"


def test_e005_area_audit_config_is_evaluation_only() -> None:
    config = load_config(
        Path(__file__).parents[1] / "configs/experiments/e005_yolo_hand_oks_area_audit.yaml"
    )
    assert config["experiment"]["operation"] == "evaluate"
    assert config["experiment"]["description"]
    assert config["evaluation"]["compare_hand_oks_areas"] is True
    assert config["integrations"]["hf_checkpoints"]["enabled"] is False


def test_e005_config_is_fresh_discriminative_run() -> None:
    config = load_config(
        Path(__file__).parents[1]
        / "configs/experiments/e005_yolo_person17_hand21_discriminative_max30.yaml"
    )
    train = config["train"]
    assert "checkpoint" not in config
    assert train["epochs"] == 30
    assert train["patience"] == 10
    assert train["discriminative_lr"] is True
    assert train["learning_rate"] == pytest.approx(1e-4)
    assert train["hand_learning_rate"] == pytest.approx(5e-4)
    assert train["lr_schedule"] == "warmup_cosine"
    assert train["augmentation"]["enabled"] is True
    assert train["augmentation"]["translate"] == pytest.approx(0.10)
    assert train["augmentation"]["scale"] == pytest.approx(0.20)


def test_config_validation_yolo(tmp_path: Path) -> None:
    probe_yaml = """
experiment:
  id: e003_probe
  operation: probe
  mode: full
  seed: 42
checkpoint:
  repo_id: test/repo
  revision: abcdef
  path: model.pt
data:
  train_videos: 10
  val_videos: 2
  frames_per_video: 24
model:
  name: yolo26m-pose.pt
  person_keypoints: 17
  hand_keypoints: 21
  image_size: 640
train:
  batch_size: 64
  optimizer: adamw
output:
  root: artifacts/test
"""
    config_file = tmp_path / "probe.yaml"
    config_file.write_text(probe_yaml, encoding="utf-8")
    loaded = load_config(config_file)
    assert loaded["experiment"]["operation"] == "probe"
    assert loaded["train"]["batch_size"] == 64

    # Invalid probe without checkpoint
    invalid_yaml = probe_yaml.replace("checkpoint:", "checkpoint_disabled:")
    invalid_file = tmp_path / "invalid.yaml"
    invalid_file.write_text(invalid_yaml, encoding="utf-8")
    with pytest.raises(ValueError, match="Probe operation requires checkpoint"):
        load_config(invalid_file)

    # Invalid augmentation with horizontal flip under train
    flip_yaml = probe_yaml.replace(
        "optimizer: adamw",
        "optimizer: adamw\n  augmentation:\n    horizontal_flip: true",
    )
    flip_file = tmp_path / "flip.yaml"
    flip_file.write_text(flip_yaml, encoding="utf-8")
    with pytest.raises(ValueError, match="Augmentation 'horizontal_flip' is not permitted"):
        load_config(flip_file)


def test_probe_fails_fast_on_cpu(tmp_path: Path) -> None:
    config = {
        "experiment": {"id": "test_exp", "operation": "probe", "mode": "full", "seed": 42},
        "checkpoint": {"repo_id": "test/repo", "revision": "rev1", "path": "model.pt"},
        "data": {
            "canonical_manifest_dir": "manifests/vsl400",
            "train_videos": 1,
            "val_videos": 1,
            "frames_per_video": 24,
            "known_decode_defect": "019442",
        },
        "model": {
            "name": "yolo26m-pose.pt",
            "person_keypoints": 17,
            "hand_keypoints": 21,
            "image_size": 640,
        },
        "train": {"batch_size": 64, "workers": 0, "amp": False},
        "output": {"root": "artifacts/test", "data_dir": "data", "run_dir": "run"},
        "integrations": {"wandb": False, "hf_checkpoints": False},
    }
    with pytest.raises(RuntimeError, match="YOLO capacity probe requires CUDA"):
        run_materialized(
            config, data_root=tmp_path / "data", output_root=tmp_path / "run", device="cpu"
        )


def test_probe_oom_fallback_batch_size_8(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "run"
    data_dir.mkdir(parents=True)
    output_dir.mkdir(parents=True)

    np.save(data_dir / "train_boxes.npy", np.zeros((24, 3, 4), dtype=np.float32))
    np.save(data_dir / "train_classes.npy", np.zeros((24, 3), dtype=np.int8))
    np.save(data_dir / "train_person_keypoints.npy", np.zeros((24, 17, 3), dtype=np.float32))
    np.save(data_dir / "train_hand_keypoints.npy", np.zeros((24, 2, 21, 3), dtype=np.float32))
    np.save(data_dir / "val_boxes.npy", np.zeros((24, 3, 4), dtype=np.float32))
    np.save(data_dir / "val_classes.npy", np.zeros((24, 3), dtype=np.int8))
    np.save(data_dir / "val_person_keypoints.npy", np.zeros((24, 17, 3), dtype=np.float32))
    np.save(data_dir / "val_hand_keypoints.npy", np.zeros((24, 2, 21, 3), dtype=np.float32))

    dummy_model = DummyDualHead()
    dummy_model_audit = {"channels": [8]}

    observed_batch_sizes: list[int] = []

    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.get_device_name", lambda *args: "Mocked CUDA")
    monkeypatch.setattr(
        "src.yolo.experiment.download_checkpoint",
        lambda **kwargs: tmp_path / "dummy.pt",
    )
    monkeypatch.setattr(
        "src.yolo.experiment.initialize_model",
        lambda *args, **kwargs: (dummy_model, dummy_loss, dummy_loss, dummy_model_audit),
    )
    monkeypatch.setattr(
        "src.yolo.experiment.probe_batch_capacity",
        lambda *args, **kwargs: (_ for _ in ()).throw(torch.OutOfMemoryError("CUDA out of memory")),
    )

    def mock_loader(dataset, *, batch_size, **kwargs):
        observed_batch_sizes.append(batch_size)
        return []

    monkeypatch.setattr("src.yolo.experiment.loader", mock_loader)
    monkeypatch.setattr(
        "src.yolo.experiment.evaluate_yolo",
        lambda *args, **kwargs: {
            "validation_loss_per_image": 0.1,
            "loss_components": {},
            "end_to_end_images_per_second": 100.0,
            "person_bbox": {"map50": 0.9, "map50_95": 0.8},
            "person_pose17_oks": {"map50": 0.88, "map50_95": 0.78},
            "hand_bbox": {"map50": 0.85, "map50_95": 0.75},
            "hand_pose21_oks": {"map50": 0.84, "map50_95": 0.74},
        },
    )

    config = {
        "experiment": {"id": "test_exp", "operation": "probe", "mode": "full", "seed": 42},
        "checkpoint": {"repo_id": "test/repo", "revision": "rev1", "path": "model.pt"},
        "data": {
            "canonical_manifest_dir": "manifests/vsl400",
            "train_videos": 1,
            "val_videos": 1,
            "frames_per_video": 24,
            "known_decode_defect": "019442",
        },
        "model": {
            "name": "yolo26m-pose.pt",
            "architecture": "yolo26m-pose.yaml",
            "person_keypoints": 17,
            "hand_keypoints": 21,
            "image_size": 640,
        },
        "train": {"batch_size": 64, "workers": 0, "amp": False},
        "evaluation": {"batch_size": 64},
        "output": {"root": "artifacts/test", "data_dir": "data", "run_dir": "run"},
        "integrations": {"wandb": False, "hf_checkpoints": False},
    }

    summary = run_materialized(config, data_root=data_dir, output_root=output_dir, device="cuda")
    assert summary["status"] == "completed_with_probe_error"
    assert "CUDA out of memory" in summary["probe"]["error"]
    assert observed_batch_sizes == [8]  # Fallback batch size 8 on OOM


def test_run_yolo_overwrite_guard(tmp_path: Path) -> None:
    project_root = tmp_path / "project_ai"
    run_dir = project_root / "artifacts/test/run"
    run_dir.mkdir(parents=True)
    (run_dir / "summary.json").write_text("{}", encoding="utf-8")

    config = {
        "experiment": {"id": "test_exp", "mode": "full", "seed": 42},
        "data": {"canonical_manifest_dir": "manifests/vsl400"},
        "model": {
            "name": "yolo26m-pose.pt",
            "person_keypoints": 17,
            "hand_keypoints": 21,
            "image_size": 640,
        },
        "train": {"batch_size": 8},
        "output": {"root": "artifacts/test", "data_dir": "data", "run_dir": "run"},
    }

    with pytest.raises(FileExistsError, match="exists and is not empty"):
        run_yolo(config, project_root, resume=False)


def test_version_and_keypoints_mismatch(tmp_path: Path) -> None:
    config_wrong_keypoints = {
        "experiment": {"id": "test_exp", "mode": "full", "seed": 42},
        "model": {
            "name": "yolo26m-pose.pt",
            "person_keypoints": 42,
            "hand_keypoints": 21,
            "image_size": 640,
        },
        "train": {"batch_size": 8},
        "output": {"root": "artifacts/test", "data_dir": "data", "run_dir": "run"},
    }
    with pytest.raises(ValueError, match="Expected person_keypoints 17"):
        run_materialized(config_wrong_keypoints, tmp_path / "data", tmp_path / "run")

    config_wrong_version = {
        "experiment": {"id": "test_exp", "mode": "full", "seed": 42},
        "model": {
            "name": "yolo26m-pose.pt",
            "ultralytics_version": "9.9.999",
            "person_keypoints": 17,
            "hand_keypoints": 21,
            "image_size": 640,
        },
        "train": {"batch_size": 8},
        "output": {"root": "artifacts/test", "data_dir": "data", "run_dir": "run"},
    }
    with pytest.raises(RuntimeError, match="Expected ultralytics 9.9.999"):
        run_materialized(config_wrong_version, tmp_path / "data", tmp_path / "run")


def test_fit_yolo_multi_epoch_non_divisible_batches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "run"
    data_dir.mkdir(parents=True)
    output_dir.mkdir(parents=True)

    (data_dir / "materialization.json").write_text('{"identity": "mat_id_nd"}', encoding="utf-8")
    # 10 samples total per epoch, batch size 4 -> 3 batches per epoch (4, 4, 2)
    np.save(data_dir / "train_boxes.npy", np.zeros((10, 3, 4), dtype=np.float32))
    np.save(data_dir / "train_classes.npy", np.zeros((10, 3), dtype=np.int8))
    np.save(data_dir / "train_person_keypoints.npy", np.zeros((10, 17, 3), dtype=np.float32))
    np.save(data_dir / "train_hand_keypoints.npy", np.zeros((10, 2, 21, 3), dtype=np.float32))
    np.save(data_dir / "val_boxes.npy", np.zeros((10, 3, 4), dtype=np.float32))
    np.save(data_dir / "val_classes.npy", np.zeros((10, 3), dtype=np.int8))
    np.save(data_dir / "val_person_keypoints.npy", np.zeros((10, 17, 3), dtype=np.float32))
    np.save(data_dir / "val_hand_keypoints.npy", np.zeros((10, 2, 21, 3), dtype=np.float32))

    model = DummyDualHead()
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    scaler = torch.amp.GradScaler("cpu", enabled=False)

    config = {
        "experiment": {"id": "train_nd", "operation": "train", "mode": "full", "seed": 42},
        "data": {
            "canonical_manifest_dir": "manifests/vsl400",
            "train_videos": 1,
            "val_videos": 1,
            "frames_per_video": 10,
            "known_decode_defect": "019442",
        },
        "model": {
            "name": "yolo26m-pose.pt",
            "person_keypoints": 17,
            "hand_keypoints": 21,
            "image_size": 640,
        },
        "train": {
            "epochs": 2,
            "batch_size": 4,
            "workers": 0,
            "amp": False,
            "log_every_batches": 10,
            "print_every_batches": 10,
            "checkpoint_every_batches": 10,
            "lr_schedule": "warmup_cosine",
            "warmup_epochs": 1,
        },
        "output": {"root": "artifacts/test", "data_dir": "data", "run_dir": "run"},
        "integrations": {"wandb": False, "hf_checkpoints": False},
    }

    dummy_sample = {
        "image": torch.zeros((3, 640, 640), dtype=torch.uint8),
        "boxes": torch.zeros((3, 4)),
        "classes": torch.tensor([0, -1, -1], dtype=torch.int8),
        "person_keypoints": torch.zeros((17, 3)),
        "hand_keypoints": torch.zeros((2, 21, 3)),
    }
    monkeypatch.setattr(YoloDataset, "__getitem__", lambda self, index: dummy_sample)

    def mock_train_step(*args: Any, **kwargs: Any) -> tuple[torch.Tensor, dict[str, float]]:
        args[1].step()
        return torch.tensor(0.25), {}

    monkeypatch.setattr("src.yolo.fit.train_step", mock_train_step)
    scheduler = build_yolo_scheduler(opt, config["train"], steps_per_epoch=3)

    fit_res = fit_yolo(
        model,
        opt,
        scaler,
        dummy_loss,
        dummy_loss,
        data_root=data_dir,
        output_root=output_dir,
        config=config,
        target_device=torch.device("cpu"),
        seed=42,
        scheduler=scheduler,
    )
    # Total batches = 3 (epoch 0) + 3 (epoch 1) = 6
    assert fit_res["batches"] == 6
    checkpoint_state = torch.load(output_dir / "last.pt", weights_only=True)
    assert checkpoint_state["epoch"] == 1
    assert checkpoint_state["next_sample"] == 10
    assert checkpoint_state["scheduler"]["last_epoch"] == 6

    assert "optimizer" in checkpoint_state
    assert "scaler" in checkpoint_state

    # Test resume from the completed E005 checkpoint, including scheduler state.
    resumed_model = DummyDualHead()
    resumed_opt = torch.optim.SGD(resumed_model.parameters(), lr=0.01)
    resumed_scaler = torch.amp.GradScaler("cpu", enabled=False)
    resumed_scheduler = build_yolo_scheduler(resumed_opt, config["train"], steps_per_epoch=3)
    resumed_res = fit_yolo(
        resumed_model,
        resumed_opt,
        resumed_scaler,
        dummy_loss,
        dummy_loss,
        data_root=data_dir,
        output_root=output_dir,
        config=config,
        target_device=torch.device("cpu"),
        seed=42,
        resume=True,
        scheduler=resumed_scheduler,
    )
    assert resumed_res["batches"] == 6
    assert resumed_scheduler.last_epoch == 6


def test_fit_yolo_early_stopping_and_best_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "run"
    data_dir.mkdir(parents=True)
    output_dir.mkdir(parents=True)

    (data_dir / "materialization.json").write_text('{"identity": "mat_id_es"}', encoding="utf-8")
    np.save(data_dir / "train_boxes.npy", np.zeros((4, 3, 4), dtype=np.float32))
    np.save(data_dir / "train_classes.npy", np.zeros((4, 3), dtype=np.int8))
    np.save(data_dir / "train_person_keypoints.npy", np.zeros((4, 17, 3), dtype=np.float32))
    np.save(data_dir / "train_hand_keypoints.npy", np.zeros((4, 2, 21, 3), dtype=np.float32))
    np.save(data_dir / "val_boxes.npy", np.zeros((4, 3, 4), dtype=np.float32))
    np.save(data_dir / "val_classes.npy", np.zeros((4, 3), dtype=np.int8))
    np.save(data_dir / "val_person_keypoints.npy", np.zeros((4, 17, 3), dtype=np.float32))
    np.save(data_dir / "val_hand_keypoints.npy", np.zeros((4, 2, 21, 3), dtype=np.float32))

    model = DummyDualHead()
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    scaler = torch.amp.GradScaler("cpu", enabled=False)

    config = {
        "experiment": {"id": "train_es", "operation": "train", "mode": "full", "seed": 42},
        "data": {
            "canonical_manifest_dir": "manifests/vsl400",
            "train_videos": 1,
            "val_videos": 1,
            "frames_per_video": 4,
            "known_decode_defect": "019442",
        },
        "model": {
            "name": "yolo26m-pose.pt",
            "person_keypoints": 17,
            "hand_keypoints": 21,
            "image_size": 640,
        },
        "train": {
            "epochs": 10,
            "patience": 2,
            "batch_size": 4,
            "workers": 0,
            "amp": False,
            "log_every_batches": 10,
            "print_every_batches": 10,
            "checkpoint_every_batches": 10,
            "lr_schedule": "warmup_cosine",
            "warmup_epochs": 1,
        },
        "output": {"root": "artifacts/test", "data_dir": "data", "run_dir": "run"},
        "integrations": {"wandb": False, "hf_checkpoints": False},
    }

    dummy_sample = {
        "image": torch.zeros((3, 640, 640), dtype=torch.uint8),
        "boxes": torch.zeros((3, 4)),
        "classes": torch.tensor([0, -1, -1], dtype=torch.int8),
        "person_keypoints": torch.zeros((17, 3)),
        "hand_keypoints": torch.zeros((2, 21, 3)),
    }
    monkeypatch.setattr(YoloDataset, "__getitem__", lambda self, index: dummy_sample)

    def mock_train_step(*args: Any, **kwargs: Any) -> tuple[torch.Tensor, dict[str, float]]:
        args[1].step()
        return torch.tensor(0.1), {}

    monkeypatch.setattr("src.yolo.fit.train_step", mock_train_step)
    scheduler = build_yolo_scheduler(opt, config["train"], steps_per_epoch=1)

    # Validation metric: epoch 0 = 0.5 (best), epoch 1 = 0.8 (best), epoch 2 = 0.7 (bad 1), epoch 3 = 0.6 (bad 2 -> stop)
    metrics_by_call = [0.5, 0.8, 0.7, 0.6]
    call_count = 0

    def mock_validate(eval_model: torch.nn.Module) -> dict[str, Any]:
        nonlocal call_count
        val_metric = metrics_by_call[call_count]
        call_count += 1
        return {
            "validation_loss_per_image": 0.2,
            "hand_pose21_oks": {"map50_95": val_metric},
        }

    fit_res = fit_yolo(
        model,
        opt,
        scaler,
        dummy_loss,
        dummy_loss,
        data_root=data_dir,
        output_root=output_dir,
        config=config,
        target_device=torch.device("cpu"),
        seed=42,
        scheduler=scheduler,
        validate=mock_validate,
    )

    assert fit_res["stopped_early"] is True
    assert fit_res["epochs_completed"] == 4
    assert fit_res["best_metric"] == pytest.approx(0.8)
    assert fit_res["best_epoch"] == 1
    assert fit_res["epochs_without_improvement"] == 2

    # Verify best.pt has best_epoch 1 and last.pt has epoch 3
    best_state = torch.load(output_dir / "best.pt", weights_only=True)
    assert best_state["epoch"] == 1
    assert best_state["checkpoint_role"] == "best"

    last_state = torch.load(output_dir / "last.pt", weights_only=True)
    assert last_state["epoch"] == 3
    assert last_state["best_epoch"] == 1
    assert last_state["epochs_without_improvement"] == 2
    assert last_state["checkpoint_role"] == "last"

    # Resume when early stop already triggered should not run more epochs
    resumed_model = DummyDualHead()
    resumed_opt = torch.optim.SGD(resumed_model.parameters(), lr=0.01)
    resumed_scaler = torch.amp.GradScaler("cpu", enabled=False)
    resumed_scheduler = build_yolo_scheduler(resumed_opt, config["train"], steps_per_epoch=1)
    resumed_res = fit_yolo(
        resumed_model,
        resumed_opt,
        resumed_scaler,
        dummy_loss,
        dummy_loss,
        data_root=data_dir,
        output_root=output_dir,
        config=config,
        target_device=torch.device("cpu"),
        seed=42,
        resume=True,
        scheduler=resumed_scheduler,
    )
    assert resumed_res["stopped_early"] is True
    assert resumed_res["best_epoch"] == 1
    assert resumed_res["epochs_without_improvement"] == 2


def test_materialize_part_temp_dir_cleanup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    workspace_root = tmp_path / "workspace"
    manifest_dir = workspace_root / "manifests/vsl400"
    manifest_dir.mkdir(parents=True)
    (manifest_dir / "train.csv").write_text(
        "video_id,split,signer_id,class_id,part_id,num_frames\n000001,train,001,1,1,24\n",
        encoding="utf-8",
    )
    (manifest_dir / "val.csv").write_text(
        "video_id,split,signer_id,class_id,part_id,num_frames\n000002,val,002,1,1,24\n",
        encoding="utf-8",
    )

    rtmw_dir = workspace_root / "manifests/rtmw"
    rtmw_dir.mkdir(parents=True)
    (rtmw_dir / "manifest.json").write_text(
        json.dumps({"shards": [{"shard_id": "shard_01", "video_ids": [1, 2]}]}),
        encoding="utf-8",
    )

    cache_dir = tmp_path / "cache"
    output_dir = tmp_path / "output"

    downloaded_dirs: list[Path] = []

    def mock_download_file(*, local_dir=None, **kwargs):
        if local_dir is not None:
            downloaded_dirs.append(local_dir)
            local_dir.mkdir(parents=True, exist_ok=True)
            dummy_tar = local_dir / "dummy.tar"
            dummy_tar.write_bytes(b"")
            return dummy_tar
        dummy_npz = cache_dir / "rtmw/dummy.npz"
        dummy_npz.parent.mkdir(parents=True, exist_ok=True)
        return dummy_npz

    mat_mod = importlib.import_module("src.yolo.materialize")
    monkeypatch.setattr(mat_mod, "download_file", mock_download_file)

    config = {
        "data": {
            "canonical_manifest_dir": "manifests/vsl400",
            "rtmw_manifest": "manifests/rtmw/manifest.json",
            "parts": [1],
            "train_videos": 1,
            "val_videos": 1,
            "frames_per_video": 24,
            "known_decode_defect": "019442",
            "dataset_repo": "test/repo",
            "dataset_revision": "rev",
            "rtmw_repo": "test/rtmw",
            "rtmw_revision": "rev",
            "rtmw_prefix": "prefix",
            "keypoint_score_threshold": 0.3,
        },
        "model": {"image_size": 640, "person_keypoints": 17, "hand_keypoints": 21},
    }

    # Tarfile open will fail, but finally block must clean up part_dir!
    with pytest.raises(tarfile.TarError):
        mat_mod.materialize(output_dir, cache_dir, workspace_root, config)

    assert len(downloaded_dirs) == 1
    # Check that the per-part temp directory was deleted
    assert not downloaded_dirs[0].exists()


def test_yolo_pilot_legacy_shim() -> None:
    assert hasattr(yolo_pilot_shim, "YoloDataset")
    assert hasattr(yolo_pilot_shim, "make_targets")
    assert hasattr(yolo_pilot_shim, "build_model")
    assert hasattr(yolo_pilot_shim, "build_losses")
    assert hasattr(yolo_pilot_shim, "decode")
    assert hasattr(yolo_pilot_shim, "sync_heads")
    assert hasattr(yolo_pilot_shim, "run")
    assert hasattr(yolo_pilot_shim, "run_materialized")
    assert hasattr(yolo_pilot_shim, "run_yolo")


def test_run_yolo_workspace_root_boundary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure run_yolo passes project_root as workspace_root to materialize (never project_root.parent)."""
    recorded_workspace: list[Path] = []

    def mock_materialize(
        data_root: Path, cache: Path, workspace_root: Path, config: dict[str, Any]
    ) -> None:
        recorded_workspace.append(workspace_root)

    monkeypatch.setattr("src.yolo.experiment.materialize", mock_materialize)
    monkeypatch.setattr(
        "src.yolo.experiment.run_materialized",
        lambda *args, **kwargs: {"status": "ok"},
    )

    fake_project = tmp_path / "standalone_project_ai"
    fake_project.mkdir(parents=True)
    config = {
        "experiment": {"id": "test_boundary", "operation": "train", "mode": "full", "seed": 42},
        "model": {
            "name": "yolo26m-pose.pt",
            "architecture": "yolo26m-pose.yaml",
            "person_keypoints": 17,
            "hand_keypoints": 21,
            "image_size": 640,
        },
        "output": {"root": "artifacts/runs", "run_dir": "test_run", "data_dir": "test_data"},
    }
    run_yolo(config, fake_project)
    assert len(recorded_workspace) == 1
    assert recorded_workspace[0] == fake_project
    assert recorded_workspace[0] != fake_project.parent


def test_colab_runner_standalone_sync_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ensure run_on_colab syncs project_root directly to /content/project_ai without parent paths."""
    commands: list[tuple[str, ...]] = []
    exec_scripts: list[str] = []

    def mock_command(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
        commands.append(tuple(args))
        if len(args) > 1 and args[1] == "exec":
            exec_scripts.append(Path(args[args.index("-f") + 1]).read_text(encoding="utf-8"))
        # For test -f summary.json check
        return subprocess.CompletedProcess(args=list(args), returncode=0, stdout="", stderr="")

    monkeypatch.setattr("src.runtime.colab._command", mock_command)

    fake_project = tmp_path / "isolated_project_ai"
    fake_project.mkdir(parents=True)
    cfg_file = fake_project / "config.yaml"
    cfg_file.write_text(
        "experiment:\n"
        "  id: test_colab_boundary\n"
        "  mode: full\n"
        "  seed: 42\n"
        "model:\n"
        "  name: tcn\n"
        "  feature_view: pose53\n"
        "  sequence_length: 24\n"
        "  num_classes: 400\n"
        "feature:\n"
        "  view: pose53\n"
        "data:\n"
        "  repo_id: repo\n"
        "  revision: rev\n"
        "  artifact_path: art\n"
        "train:\n"
        "  batch_size: 16\n"
        "  epochs: 1\n"
        "  patience: 0\n"
        "  learning_rate: 0.001\n"
        "evaluation:\n"
        "  validation: true\n"
        "output:\n"
        "  root: artifacts/runs\n"
        "  run_dir: my_run\n",
        encoding="utf-8",
    )
    identity_file = fake_project / "id_rsa"
    identity_file.write_text("dummy", encoding="utf-8")

    run_on_colab(
        config_path=cfg_file,
        project_root=fake_project,
        session="test-session",
        gpu="T4",
        identity=identity_file,
        timeout=100,
        resume=False,
    )

    assert len(exec_scripts) == 1
    assert "subprocess.Popen(" in exec_scripts[0]
    assert "stdout=subprocess.PIPE" in exec_scripts[0]
    assert "stderr=subprocess.STDOUT" in exec_scripts[0]
    assert "for line in process.stdout:" in exec_scripts[0]
    assert "print(line, end='', flush=True)" in exec_scripts[0]
    assert "raise subprocess.CalledProcessError(returncode, args)" in exec_scripts[0]

    # Verify rsync commands only sync from fake_project to /content/project_ai
    rsync_sources = [cmd[-2] for cmd in commands if cmd and cmd[0] == "rsync"]
    assert f"{fake_project}/" in rsync_sources
    assert f"{fake_project.parent}/src/" not in rsync_sources
    assert f"{fake_project.parent}/manifests/" not in rsync_sources
    assert str(fake_project.parent / "pyproject.toml") not in rsync_sources

    # Verify sensitive and runtime artifacts are strictly excluded from rsync
    sync_cmd = next(
        cmd for cmd in commands if cmd and cmd[0] == "rsync" and cmd[-2] == f"{fake_project}/"
    )
    for excluded in (
        ".git",
        ".env",
        ".env.*",
        ".venv",
        "artifacts",
        "cache",
        "runs",
        "output",
        "data",
        "wandb",
    ):
        assert "--exclude" in sync_cmd
        assert excluded in sync_cmd


def test_colab_runner_syncs_only_configured_rgb_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    commands: list[tuple[str, ...]] = []

    def mock_command(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
        commands.append(tuple(args))
        return subprocess.CompletedProcess(args=list(args), returncode=0, stdout="", stderr="")

    monkeypatch.setattr("src.runtime.colab._command", mock_command)
    project = tmp_path / "project"
    rgb_root = project / "artifacts/rgb/u24/data"
    rgb_root.mkdir(parents=True)
    (rgb_root / "train.npz").write_bytes(b"rgb")
    receipt = project / "artifacts/rgb/receipt.json"
    receipt.write_text("{}", encoding="utf-8")
    config = project / "config.yaml"
    config.write_text(
        """
experiment:
  id: rgb_colab
  operation: e026_rgb_train
  mode: full
  seed: 42
manifests:
  train: manifests/audit/train_clean.csv
  val: manifests/audit/val_clean.csv
rgb_data:
  root: artifacts/rgb/u24/data
  receipt: artifacts/rgb/receipt.json
  receipt_sha256: receipt
  extraction_identity: identity
model:
  name: rgb_linear
train:
  epochs: 100
  min_epochs: 60
  patience: 15
  batch_size: 256
  optimizer: adamw
  learning_rate: 0.001
  weight_decay: 0.0001
  grad_clip_norm: 1.0
output:
  root: artifacts/runs/rgb
""",
        encoding="utf-8",
    )
    identity = project / "id_rsa"
    identity.write_text("dummy", encoding="utf-8")

    run_on_colab(
        config_path=config,
        project_root=project,
        session="test-session",
        gpu="T4",
        identity=identity,
        timeout=100,
        resume=False,
    )

    rgb_syncs = [cmd for cmd in commands if cmd and cmd[0] == "rsync" and "u24/data" in cmd[-1]]
    assert len(rgb_syncs) == 1
    assert rgb_syncs[0][-2] == f"{rgb_root}/"
    receipt_syncs = [cmd for cmd in commands if cmd and cmd[0] == "rsync" and cmd[-1].endswith("receipt.json")]
    assert len(receipt_syncs) == 1
    assert receipt_syncs[0][-2] == str(receipt)

    config.write_text(
        config.read_text(encoding="utf-8").replace("artifacts/rgb/u24/data", "../outside"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="paths must stay below project root"):
        run_on_colab(
            config_path=config,
            project_root=project,
            session="test-session",
            gpu="T4",
            identity=identity,
            timeout=100,
            resume=False,
        )


def test_colab_runner_syncs_configured_local_split_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    commands: list[tuple[str, ...]] = []

    def mock_command(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
        commands.append(tuple(args))
        return subprocess.CompletedProcess(args=list(args), returncode=0, stdout="", stderr="")

    monkeypatch.setattr("src.runtime.colab._command", mock_command)
    project = tmp_path / "project"
    split_root = project / "artifacts/yolo/e029/data"
    split_root.mkdir(parents=True)
    (split_root / "train.npz").write_bytes(b"train")
    config = project / "config.yaml"
    config.write_text(
        """
experiment:
  id: local_split_colab
  mode: full
  seed: 42
data:
  split_path: artifacts/yolo/e029/data
  num_classes: 400
feature:
  view: pose53
model:
  name: tcn
  input_channels: 2
  channels: [4]
  dilations: [1]
  kernel_size: 3
  dropout: 0.0
train:
  epochs: 1
  patience: 0
  batch_size: 1
output:
  root: artifacts/runs
integrations:
  wandb:
    enabled: false
  hf_checkpoints:
    enabled: false
""",
        encoding="utf-8",
    )
    identity = project / "id_rsa"
    identity.write_text("dummy", encoding="utf-8")

    run_on_colab(
        config_path=config,
        project_root=project,
        session="test-session",
        gpu="T4",
        identity=identity,
        timeout=100,
        resume=False,
    )

    split_syncs = [cmd for cmd in commands if cmd and cmd[0] == "rsync" and "e029/data" in cmd[-1]]
    assert len(split_syncs) == 1
    assert split_syncs[0][-2] == f"{split_root}/"

    config.write_text(
        config.read_text(encoding="utf-8").replace("artifacts/yolo/e029/data", "../outside"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="data.split_path must stay below project root"):
        run_on_colab(
            config_path=config,
            project_root=project,
            session="test-session",
            gpu="T4",
            identity=identity,
            timeout=100,
            resume=False,
        )


def test_colab_runner_extract_retrieval_excludes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    commands: list[tuple[str, ...]] = []

    def mock_command(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
        commands.append(tuple(args))
        return subprocess.CompletedProcess(args=list(args), returncode=0, stdout="", stderr="")

    monkeypatch.setattr("src.runtime.colab._command", mock_command)

    fake_project = tmp_path / "colab_extract_project"
    fake_project.mkdir(parents=True)
    identity_file = fake_project / "id_rsa"
    identity_file.write_text("dummy", encoding="utf-8")

    extract_yaml = fake_project / "extract.yaml"
    extract_yaml.write_text(
        "experiment:\n"
        "  id: test_extract_colab\n"
        "  operation: extract\n"
        "  mode: full\n"
        "  seed: 42\n"
        "checkpoint:\n"
        "  repo_id: repo\n"
        "  revision: rev\n"
        "  path: pt\n"
        "data:\n"
        "  dataset_repo: d_repo\n"
        "  dataset_revision: d_rev\n"
        "  parts: [1]\n"
        "  manifests:\n"
        "    train: manifests/train.csv\n"
        "    val: manifests/val.csv\n"
        "    test: manifests/test.csv\n"
        "  expected_frames: 100\n"
        "  batch_size: 16\n"
        "  nms_confidence: 0.25\n"
        "  nms_iou: 0.7\n"
        "  max_detections: 10\n"
        "model:\n"
        "  name: yolo26m-pose.pt\n"
        "  person_keypoints: 17\n"
        "  hand_keypoints: 21\n"
        "  image_size: 640\n"
        "output:\n"
        "  root: artifacts/extract_colab\n"
        "integrations:\n"
        "  hf_artifacts:\n"
        "    enabled: true\n"
        "    repo_id: feat_repo\n",
        encoding="utf-8",
    )

    run_on_colab(
        config_path=extract_yaml,
        project_root=fake_project,
        session="extract-session",
        gpu="L4",
        identity=identity_file,
        timeout=100,
        resume=False,
    )

    # Retrieval rsync commands (remote_run/ to local_run/)
    retrieval_cmds = [
        cmd
        for cmd in commands
        if cmd
        and cmd[0] == "rsync"
        and "/content/project_ai/artifacts/extract_colab/run/" in cmd[-2]
    ]
    assert len(retrieval_cmds) >= 1
    for rsync_cmd in retrieval_cmds:
        assert ".video_cache_part_*" in rsync_cmd
        assert "part_*.npz" in rsync_cmd

    # Test extract_hand_rgb operation with hf_artifacts enabled excludes both NPZ and Parquet crops
    extract_handrgb_yaml = fake_project / "extract_handrgb.yaml"
    extract_handrgb_yaml.write_text(
        "experiment:\n"
        "  id: test_extract_hrgb_colab\n"
        "  operation: extract_hand_rgb\n"
        "  mode: full\n"
        "  seed: 42\n"
        "checkpoint:\n"
        "  repo_id: repo\n"
        "  revision: rev\n"
        "  path: pt\n"
        "data:\n"
        "  dataset_repo: d_repo\n"
        "  dataset_revision: d_rev\n"
        "  parts: [1]\n"
        "  manifests:\n"
        "    train: manifests/train.csv\n"
        "    val: manifests/val.csv\n"
        "    test: manifests/test.csv\n"
        "  expected_frames: 100\n"
        "  batch_size: 16\n"
        "  nms_confidence: 0.25\n"
        "  nms_iou: 0.7\n"
        "  max_detections: 10\n"
        "crops:\n"
        "  crop_shard_frames: 4096\n"
        "  margin: 1.3\n"
        "  quality: 95\n"
        "model:\n"
        "  name: yolo26m-pose.pt\n"
        "  person_keypoints: 17\n"
        "  hand_keypoints: 21\n"
        "  image_size: 640\n"
        "output:\n"
        "  root: artifacts/extract_hrgb_colab\n"
        "integrations:\n"
        "  hf_artifacts:\n"
        "    enabled: true\n"
        "    repo_id: feat_repo\n",
        encoding="utf-8",
    )
    commands.clear()
    run_on_colab(
        config_path=extract_handrgb_yaml,
        project_root=fake_project,
        session="extract-hrgb-session",
        gpu="L4",
        identity=identity_file,
        timeout=100,
        resume=False,
    )
    hrgb_retrieval_cmds = [
        cmd
        for cmd in commands
        if cmd
        and cmd[0] == "rsync"
        and "/content/project_ai/artifacts/extract_hrgb_colab/run/" in cmd[-2]
    ]
    assert len(hrgb_retrieval_cmds) >= 1
    for rsync_cmd in hrgb_retrieval_cmds:
        assert ".video_cache_part_*" in rsync_cmd
        assert "part_*.npz" in rsync_cmd
        assert "part_*_crops_*.parquet" in rsync_cmd
