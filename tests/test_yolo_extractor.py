from __future__ import annotations

import csv
import hashlib
import json
import tarfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.config import load_config
from src.yolo.extractor import (
    _atomic_json,
    _sha256,
    _upload_files_with_retry,
    extract_pose59,
    inverse_letterbox_keypoints,
    select_pose59,
    validate_pose59_schema,
)
from src.yolo.model_runtime import initialize_model


def test_extract_config_dispatch_contract() -> None:
    config = load_config("configs/experiments/e005_yolo_pose59_extract.yaml")
    assert config["experiment"]["operation"] == "extract"
    assert config["data"]["manifests"]["train"] == "manifests/audit/train_clean.csv"
    assert config["data"]["manifests"]["val"] == "manifests/audit/val_clean.csv"
    assert config["data"]["manifests"]["test"] == "manifests/vsl400_front_400gloss/test.csv"
    assert config["data"]["expected_frames"] == 1_587_085
    assert config["data"]["expected_videos"]["train"] == 19714
    assert config["data"]["expected_videos"]["val"] == 2529
    assert config["data"]["expected_videos"]["test"] == 2508
    assert config["data"]["known_decode_defects"] == [
        {"video_id": "019442", "reason": "corrupt mp4 container (moov atom not found)"}
    ]
    assert config["data"]["batch_size"] == 128
    assert config["data"]["cleanup_part_cache"] is False
    assert config["integrations"]["hf_artifacts"]["enabled"] is True
    assert config["integrations"]["wandb"]["enabled"] is False

    smoke_cfg = load_config("configs/experiments/e005_yolo_pose59_extract_smoke_g4_b128.yaml")
    assert smoke_cfg["experiment"]["operation"] == "extract"
    assert smoke_cfg["experiment"]["mode"] == "smoke"
    assert smoke_cfg["data"]["batch_size"] == 128
    assert smoke_cfg["data"]["expected_frames"] == 244
    assert smoke_cfg["output"]["root"] == "artifacts/yolo/e005_yolo_pose59_extract_smoke_g4_b128"
    assert smoke_cfg["integrations"]["hf_artifacts"]["enabled"] is False


def test_config_extract_validation_requires_person17_hand21_and_checkpoint(tmp_path: Path) -> None:
    base_extract_yaml = """
experiment:
  id: test_extract
  operation: extract
  mode: full
  seed: 42
checkpoint:
  repo_id: test/repo
  revision: rev1
  path: model.pt
data:
  dataset_repo: test/data
  dataset_revision: rev2
  parts: [1]
  manifests:
    train: manifests/train.csv
    val: manifests/val.csv
    test: manifests/test.csv
  expected_frames: 100
  batch_size: 16
  nms_confidence: 0.25
  nms_iou: 0.7
  max_detections: 300
model:
  name: yolo26m-pose.pt
  person_keypoints: 17
  hand_keypoints: 21
  image_size: 640
output:
  root: artifacts/extract
"""
    cfg_file = tmp_path / "valid_extract.yaml"
    cfg_file.write_text(base_extract_yaml, encoding="utf-8")
    loaded = load_config(cfg_file)
    assert loaded["experiment"]["operation"] == "extract"

    # Missing checkpoint
    no_ckpt = base_extract_yaml.replace("checkpoint:", "checkpoint_disabled:")
    (tmp_path / "no_ckpt.yaml").write_text(no_ckpt, encoding="utf-8")
    with pytest.raises(ValueError, match="Extract operation requires checkpoint"):
        load_config(tmp_path / "no_ckpt.yaml")

    # Invalid person keypoints
    bad_kpt = base_extract_yaml.replace("person_keypoints: 17", "person_keypoints: 14")
    (tmp_path / "bad_kpt.yaml").write_text(bad_kpt, encoding="utf-8")
    with pytest.raises(ValueError, match="person_keypoints=17 and hand_keypoints=21"):
        load_config(tmp_path / "bad_kpt.yaml")


def test_select_pose59_filters_hands_to_primary_person() -> None:
    person = torch.zeros((1, 57))
    person[0, :4] = torch.tensor([10, 10, 90, 90])
    person[0, 4] = 0.8
    person[0, 6:] = 0.5
    hands = torch.zeros((3, 69))
    hands[:, 5] = torch.tensor([0, 0, 1])
    hands[:, 4] = torch.tensor([0.4, 0.9, 0.7])
    hands[0, :4] = torch.tensor([20, 20, 30, 30])
    hands[1, :4] = torch.tensor([120, 120, 130, 130])  # outside person
    hands[2, :4] = torch.tensor([30, 30, 40, 40])
    hands[:, 6:] = 0.5

    x, mask, confidence = select_pose59({"person": person, "hands": hands})
    assert x.shape == (59, 3)
    assert mask[:17].sum() == 17
    assert mask[17:38].sum() == 21
    assert mask[38:].sum() == 21
    np.testing.assert_allclose(confidence, [0.8, 0.4, 0.7])


def test_select_pose59_no_person() -> None:
    x, mask, confidence = select_pose59(
        {
            "person": torch.zeros((0, 57)),
            "hands": torch.zeros((2, 69)),
        }
    )
    assert x.shape == (59, 3)
    assert np.all(x == 0.0)
    assert np.all(mask == 0)
    np.testing.assert_allclose(confidence, [0.0, 0.0, 0.0])


def test_inverse_letterbox_and_schema_missing_detections() -> None:
    points = np.array([[[30, 40, 0.7]]], dtype=np.float32)
    result = inverse_letterbox_keypoints(
        points, scale=0.5, pad_x=10, pad_y=20, width=100, height=200
    )
    np.testing.assert_allclose(result[0, 0], [0.4, 0.2, 0.7])
    arrays = {
        "X": np.zeros((2, 59, 3), dtype=np.float32),
        "keypoint_valid_mask": np.zeros((2, 59), dtype=np.uint8),
        "object_confidence": np.zeros((2, 3), dtype=np.float32),
        "frame_idx": np.arange(2, dtype=np.int32),
        "video_id": np.array(["000001"], dtype="U6"),
        "split": np.array(["train"], dtype="U5"),
        "frame_width": np.array([100], dtype=np.int32),
        "frame_height": np.array([200], dtype=np.int32),
        "video_frame_offsets": np.array([0, 2], dtype=np.int64),
    }
    validate_pose59_schema(arrays)


def test_validate_pose59_schema_rejections() -> None:
    valid = {
        "X": np.zeros((2, 59, 3), dtype=np.float32),
        "keypoint_valid_mask": np.zeros((2, 59), dtype=np.uint8),
        "object_confidence": np.zeros((2, 3), dtype=np.float32),
        "frame_idx": np.arange(2, dtype=np.int32),
        "video_id": np.array(["000001"], dtype="U6"),
        "split": np.array(["train"], dtype="U5"),
        "frame_width": np.array([100], dtype=np.int32),
        "frame_height": np.array([200], dtype=np.int32),
        "video_frame_offsets": np.array([0, 2], dtype=np.int64),
    }

    # Missing field
    with pytest.raises(ValueError, match="Pose59 fields mismatch"):
        bad = {k: v for k, v in valid.items() if k != "X"}
        validate_pose59_schema(bad)

    # Extra field (e.g. gloss or class_id)
    with pytest.raises(ValueError, match="Pose59 fields mismatch"):
        bad_extra = {**valid, "gloss": np.array(["hello"], dtype="U5")}
        validate_pose59_schema(bad_extra)

    # Malformed dtype: frame_width int64 instead of int32
    with pytest.raises(ValueError, match="frame_width must have shape .* and dtype int32"):
        bad_dt = {**valid, "frame_width": np.array([100], dtype=np.int64)}
        validate_pose59_schema(bad_dt)

    # Malformed dtype: video_id U10 instead of U6
    with pytest.raises(ValueError, match="video_id must have shape .* and dtype U6"):
        bad_dt = {**valid, "video_id": np.array(["000001"], dtype="U10")}
        validate_pose59_schema(bad_dt)

    # Malformed dtype: split U10 instead of U5
    with pytest.raises(ValueError, match="split must have shape .* and dtype U5"):
        bad_dt = {**valid, "split": np.array(["train"], dtype="U10")}
        validate_pose59_schema(bad_dt)

    # Invalid frame_idx
    with pytest.raises(ValueError, match="frame_idx must restart at zero"):
        bad = {**valid, "frame_idx": np.array([1, 2], dtype=np.int32)}
        validate_pose59_schema(bad)

    # Invalid offsets
    with pytest.raises(ValueError, match="video_frame_offsets must be strict positive prefix sums"):
        bad = {**valid, "video_frame_offsets": np.array([0, 1], dtype=np.int64)}
        validate_pose59_schema(bad)

    # Invalid non-zero keypoint when mask is 0
    with pytest.raises(ValueError, match="Invalid keypoints must be zero-filled"):
        bad_x = np.zeros((2, 59, 3), dtype=np.float32)
        bad_x[0, 0, 0] = 0.5
        bad = {**valid, "X": bad_x}
        validate_pose59_schema(bad)


def test_upload_with_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    attempts = 0

    def mock_upload(*args: Any, **kwargs: Any) -> str:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise ConnectionError("Flaky HF network")
        return "commit_oid_123"

    monkeypatch.setattr("src.artifacts.upload_files", mock_upload)
    oid = _upload_files_with_retry(
        repo_id="test/repo",
        repo_type="dataset",
        files={"remote.txt": Path("local.txt")},
        commit_message="test",
        retries=3,
        backoff_s=0.01,
    )
    assert oid == "commit_oid_123"
    assert attempts == 3


def test_extract_pose59_require_cuda_rejections(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 1. CUDA unavailable
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    config = {
        "data": {
            "require_cuda": True,
            "manifests": {},
            "expected_frames": 0,
        },
    }
    with pytest.raises(
        RuntimeError, match="extract operation requires CUDA, but CUDA is unavailable"
    ):
        extract_pose59(config, tmp_path, tmp_path)

    # 2. CUDA available, but user explicitly passes device="cpu"
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    with pytest.raises(
        RuntimeError, match="extract operation requires CUDA, but requested device is cpu"
    ):
        extract_pose59(config, tmp_path, tmp_path, device="cpu")


def test_extract_pose59_expected_videos_and_parts_mismatch(tmp_path: Path) -> None:
    manifest_dir = tmp_path / "manifests"
    manifest_dir.mkdir(parents=True)
    manifest_file = manifest_dir / "train.csv"
    with manifest_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["video_id", "part_id", "num_frames"])
        writer.writeheader()
        writer.writerow({"video_id": "000001", "part_id": "1", "num_frames": "2"})

    # 1. Test expected_videos mismatch
    config_video_mismatch = {
        "data": {
            "require_cuda": False,
            "manifests": {"train": str(manifest_file.relative_to(tmp_path))},
            "expected_videos": {"train": 5},  # Manifest only has 1
            "expected_frames": 2,
            "parts": [1],
        },
    }
    with pytest.raises(RuntimeError, match="Manifest video count mismatch for split 'train'"):
        extract_pose59(config_video_mismatch, tmp_path, tmp_path)

    # 2. Test configured parts mismatch (omits part 1, configures part 2)
    config_parts_mismatch = {
        "data": {
            "require_cuda": False,
            "manifests": {"train": str(manifest_file.relative_to(tmp_path))},
            "expected_frames": 2,
            "parts": [2],  # Manifest has part 1
        },
    }
    with pytest.raises(RuntimeError, match="Configured parts mismatch"):
        extract_pose59(config_parts_mismatch, tmp_path, tmp_path)


class DummyDualHeadModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 8, kernel_size=1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        b = len(x)
        return {
            "person": torch.zeros((b, 4 + 1 + 17 * 3, 5)),
            "hands": torch.zeros((b, 4 + 2 + 21 * 3, 5)),
        }


def test_extract_pose59_flow_and_hf_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 1. Setup minimal dummy manifest
    manifest_dir = tmp_path / "manifests"
    manifest_dir.mkdir(parents=True)
    manifest_file = manifest_dir / "train.csv"
    with manifest_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["video_id", "part_id", "num_frames"])
        writer.writeheader()
        writer.writerow({"video_id": "000001", "part_id": "1", "num_frames": "2"})

    # 2. Setup mock tar archive with 1 video
    tar_dir = tmp_path / "tar_source"
    tar_dir.mkdir()
    video_dir = tar_dir / "split_1/front_view"
    video_dir.mkdir(parents=True)
    video_mp4 = video_dir / "000001.mp4"
    video_mp4.write_bytes(b"dummy mp4 content")

    tar_path = tmp_path / "split_1_front_view.tar"
    with tarfile.open(tar_path, "w") as tar:
        tar.add(video_mp4, arcname="split_1/front_view/000001.mp4")

    # 3. Setup dummy checkpoint
    ckpt_path = tmp_path / "best.pt"
    ckpt_path.write_bytes(b"dummy checkpoint")

    # 4. Mock dependencies
    monkeypatch.setattr(
        "src.yolo.extractor.download_checkpoint",
        lambda *args, **kwargs: ckpt_path,
    )
    monkeypatch.setattr(
        "src.yolo.extractor.download_file",
        lambda *args, **kwargs: tar_path,
    )
    monkeypatch.setattr(
        "src.yolo.extractor.initialize_model",
        lambda *args, **kwargs: (DummyDualHeadModel(), None, None, {}),
    )

    uploaded_files_record: list[dict[str, Path]] = []
    monkeypatch.setattr(
        "src.yolo.extractor.ensure_repo",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "src.artifacts.upload_files",
        lambda repo_id, repo_type, files, commit_message: (
            uploaded_files_record.append(files) or "oid"
        ),
    )

    # Mock cv2.VideoCapture to return 2 fake frames
    class MockVideoCapture:
        def __init__(self, *args: Any) -> None:
            self.count = 0

        def isOpened(self) -> bool:  # noqa: N802
            return True

        def read(self) -> tuple[bool, np.ndarray | None]:
            if self.count < 2:
                self.count += 1
                return True, np.zeros((100, 100, 3), dtype=np.uint8)
            return False, None

        def release(self) -> None:
            pass

    monkeypatch.setattr("cv2.VideoCapture", MockVideoCapture)

    config = {
        "experiment": {
            "id": "e005_test_extract",
            "operation": "extract",
            "mode": "full",
            "seed": 42,
        },
        "checkpoint": {"repo_id": "test/repo", "revision": "rev1", "path": "best.pt"},
        "data": {
            "dataset_repo": "test/data_repo",
            "dataset_revision": "rev2",
            "parts": [1],
            "manifests": {"train": str(manifest_file.relative_to(tmp_path))},
            "expected_frames": 2,
            "expected_videos": {"train": 1},
            "batch_size": 2,
            "nms_confidence": 0.25,
            "nms_iou": 0.7,
            "max_detections": 10,
            "amp": False,
            "require_cuda": False,
        },
        "model": {
            "name": "yolo26m-pose.pt",
            "architecture": "yolo26m-pose.yaml",
            "ultralytics_version": "8.4.146",
            "person_keypoints": 17,
            "hand_keypoints": 21,
            "image_size": 640,
        },
        "integrations": {
            "hf_artifacts": {
                "enabled": True,
                "repo_id": "test/dest_repo",
                "private": True,
            }
        },
    }

    output_dir = tmp_path / "output"
    summary = extract_pose59(config, project_root=tmp_path, output_root=output_dir, device="cpu")

    assert summary["status"] == "completed"
    assert summary["frames"] == 2
    assert summary["videos"] == 1
    assert "provenance" in summary
    assert "checkpoint" in summary["provenance"]
    assert "protocol" in summary["provenance"]
    assert "part_artifacts" in summary["provenance"]
    assert (output_dir / "part_01.npz").is_file()
    assert (output_dir / "part_01.complete.json").is_file()
    assert (output_dir / "part_01.uploaded.json").is_file()
    assert (output_dir / "final.uploaded.json").is_file()
    assert (output_dir / "summary.json").is_file()
    assert (output_dir / "manifest.json").is_file()
    assert not (output_dir / ".video_cache_part_01").exists()

    # Verify upload was called for part 1 and final summary/manifest
    assert len(uploaded_files_record) == 2

    # Verify resume revalidates NPZ schema and skips inference & upload idempotently
    upload_count_before = len(uploaded_files_record)
    rerun_summary = extract_pose59(
        config, project_root=tmp_path, output_root=output_dir, device="cpu"
    )
    assert rerun_summary["status"] == "completed"
    assert len(uploaded_files_record) == upload_count_before  # No redundant uploads on clean resume

    # Verify that corrupting NPZ fails resume schema validation
    np.savez(output_dir / "part_01.npz", corrupt_field=np.zeros((2, 2)))
    # Update complete marker with new hash to bypass hash check and hit schema check
    _atomic_json(
        output_dir / "part_01.complete.json",
        {
            "identity": summary["identity"],
            "videos": 1,
            "frames": 2,
            "npz_sha256": _sha256(output_dir / "part_01.npz"),
        },
    )
    with pytest.raises(ValueError, match="Pose59 fields mismatch"):
        extract_pose59(config, project_root=tmp_path, output_root=output_dir, device="cpu")


def test_initialize_model_signature_and_export() -> None:
    import inspect

    sig = inspect.signature(initialize_model)
    assert "config" in sig.parameters
    assert "device" in sig.parameters
    assert "checkpoint_path" in sig.parameters


def test_identity_binds_actual_runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest_dir = tmp_path / "manifests"
    manifest_dir.mkdir(parents=True)
    manifest_file = manifest_dir / "train.csv"
    with manifest_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["video_id", "part_id", "num_frames"])
        writer.writeheader()
        writer.writerow({"video_id": "000001", "part_id": "1", "num_frames": "2"})

    tar_dir = tmp_path / "tar_source"
    tar_dir.mkdir()
    video_dir = tar_dir / "split_1/front_view"
    video_dir.mkdir(parents=True)
    video_mp4 = video_dir / "000001.mp4"
    video_mp4.write_bytes(b"dummy mp4 content")

    tar_path = tmp_path / "split_1_front_view.tar"
    with tarfile.open(tar_path, "w") as tar:
        tar.add(video_mp4, arcname="split_1/front_view/000001.mp4")

    ckpt_path = tmp_path / "best.pt"
    ckpt_path.write_bytes(b"dummy checkpoint")

    monkeypatch.setattr("src.yolo.extractor.download_checkpoint", lambda *args, **kwargs: ckpt_path)
    monkeypatch.setattr("src.yolo.extractor.download_file", lambda *args, **kwargs: tar_path)
    monkeypatch.setattr(
        "src.yolo.extractor.initialize_model",
        lambda *args, **kwargs: (DummyDualHeadModel(), None, None, {}),
    )

    class MockVideoCapture:
        def __init__(self, *args: Any) -> None:
            self.count = 0

        def isOpened(self) -> bool:  # noqa: N802
            return True

        def read(self) -> tuple[bool, np.ndarray | None]:
            if self.count < 2:
                self.count += 1
                return True, np.zeros((100, 100, 3), dtype=np.uint8)
            return False, None

        def release(self) -> None:
            pass

    monkeypatch.setattr("cv2.VideoCapture", MockVideoCapture)

    config = {
        "experiment": {
            "id": "e005_test_runtime",
            "operation": "extract",
            "mode": "full",
            "seed": 42,
        },
        "checkpoint": {"repo_id": "test/repo", "revision": "rev1", "path": "best.pt"},
        "data": {
            "dataset_repo": "test/data_repo",
            "dataset_revision": "rev2",
            "parts": [1],
            "manifests": {"train": str(manifest_file.relative_to(tmp_path))},
            "expected_frames": 2,
            "expected_videos": {"train": 1},
            "batch_size": 2,
            "nms_confidence": 0.25,
            "nms_iou": 0.7,
            "max_detections": 10,
            "amp": False,
            "require_cuda": False,
        },
        "model": {
            "name": "yolo26m-pose.pt",
            "architecture": "yolo26m-pose.yaml",
            "ultralytics_version": "8.4.146",
            "person_keypoints": 17,
            "hand_keypoints": 21,
            "image_size": 640,
        },
        "integrations": {"hf_artifacts": {"enabled": False}},
    }

    out1 = tmp_path / "out1"
    summary1 = extract_pose59(config, project_root=tmp_path, output_root=out1, device="cpu")

    # Change torch version -> identity must change
    monkeypatch.setattr("torch.__version__", "9.9.9-mocked")
    out2 = tmp_path / "out2"
    summary2 = extract_pose59(config, project_root=tmp_path, output_root=out2, device="cpu")
    assert summary1["identity"] != summary2["identity"]

    # Attempting to resume with old marker from different runtime raises Stale or mixed completion marker
    with pytest.raises(RuntimeError, match="Stale or mixed completion marker"):
        extract_pose59(config, project_root=tmp_path, output_root=out1, device="cpu")


def test_extract_smoke_mode_filtering_and_identity_separation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_dir = tmp_path / "manifests"
    manifest_dir.mkdir(parents=True)
    train_file = manifest_dir / "train.csv"
    val_file = manifest_dir / "val.csv"
    test_file = manifest_dir / "test.csv"

    with train_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["video_id", "part_id", "num_frames"])
        writer.writeheader()
        writer.writerow({"video_id": "000001", "part_id": "1", "num_frames": "2"})
        writer.writerow({"video_id": "000002", "part_id": "1", "num_frames": "2"})
        writer.writerow({"video_id": "017734", "part_id": "6", "num_frames": "2"})

    with val_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["video_id", "part_id", "num_frames"])
        writer.writeheader()
        writer.writerow({"video_id": "003414", "part_id": "1", "num_frames": "2"})

    with test_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["video_id", "part_id", "num_frames"])
        writer.writeheader()
        writer.writerow({"video_id": "001702", "part_id": "1", "num_frames": "2"})

    tar_dir = tmp_path / "tar_source"
    tar_dir.mkdir()
    for part in (1, 6):
        video_dir = tar_dir / f"split_{part}/front_view"
        video_dir.mkdir(parents=True)

    (tar_dir / "split_1/front_view/000001.mp4").write_bytes(b"dummy")
    (tar_dir / "split_1/front_view/003414.mp4").write_bytes(b"dummy")
    (tar_dir / "split_1/front_view/001702.mp4").write_bytes(b"dummy")
    (tar_dir / "split_6/front_view/017734.mp4").write_bytes(b"dummy")

    tar1_path = tmp_path / "split_1_front_view.tar"
    with tarfile.open(tar1_path, "w") as tar:
        for vid in ("000001", "003414", "001702"):
            tar.add(
                tar_dir / f"split_1/front_view/{vid}.mp4", arcname=f"split_1/front_view/{vid}.mp4"
            )

    tar6_path = tmp_path / "split_6_front_view.tar"
    with tarfile.open(tar6_path, "w") as tar:
        tar.add(tar_dir / "split_6/front_view/017734.mp4", arcname="split_6/front_view/017734.mp4")

    ckpt_path = tmp_path / "best.pt"
    ckpt_path.write_bytes(b"dummy checkpoint")

    monkeypatch.setattr("src.yolo.extractor.download_checkpoint", lambda *args, **kwargs: ckpt_path)

    def mock_download_file(*args: Any, **kwargs: Any) -> Path:
        filename = kwargs.get("filename", "")
        if "split_1" in filename:
            return tar1_path
        return tar6_path

    monkeypatch.setattr("src.yolo.extractor.download_file", mock_download_file)
    monkeypatch.setattr(
        "src.yolo.extractor.initialize_model",
        lambda *args, **kwargs: (DummyDualHeadModel(), None, None, {}),
    )

    class MockVideoCapture:
        def __init__(self, *args: Any) -> None:
            self.count = 0

        def isOpened(self) -> bool:  # noqa: N802
            return True

        def read(self) -> tuple[bool, np.ndarray | None]:
            if self.count < 2:
                self.count += 1
                return True, np.zeros((100, 100, 3), dtype=np.uint8)
            return False, None

        def release(self) -> None:
            pass

    monkeypatch.setattr("cv2.VideoCapture", MockVideoCapture)

    smoke_config = {
        "experiment": {"id": "e005_smoke", "operation": "extract", "mode": "smoke", "seed": 42},
        "checkpoint": {"repo_id": "test/repo", "revision": "rev1", "path": "best.pt"},
        "data": {
            "dataset_repo": "test/data_repo",
            "dataset_revision": "rev2",
            "parts": [1, 6],
            "manifests": {
                "train": str(train_file.relative_to(tmp_path)),
                "val": str(val_file.relative_to(tmp_path)),
                "test": str(test_file.relative_to(tmp_path)),
            },
            "video_ids": ["000001", "003414", "001702", "017734"],
            "expected_videos": {"train": 2, "val": 1, "test": 1},
            "expected_frames": 8,
            "batch_size": 2,
            "nms_confidence": 0.25,
            "nms_iou": 0.7,
            "max_detections": 10,
            "amp": False,
            "require_cuda": False,
        },
        "model": {
            "name": "yolo26m-pose.pt",
            "architecture": "yolo26m-pose.yaml",
            "ultralytics_version": "8.4.146",
            "person_keypoints": 17,
            "hand_keypoints": 21,
            "image_size": 640,
        },
        "integrations": {"hf_artifacts": {"enabled": False}},
    }

    out_smoke = tmp_path / "out_smoke"
    summary_smoke = extract_pose59(
        smoke_config, project_root=tmp_path, output_root=out_smoke, device="cpu"
    )

    assert summary_smoke["status"] == "completed"
    assert summary_smoke["mode"] == "smoke"
    assert summary_smoke["frames"] == 8
    assert summary_smoke["videos"] == 4
    assert "missing_rates" in summary_smoke
    assert "seconds" in summary_smoke
    assert summary_smoke["identity_payload"]["mode"] == "smoke"
    assert summary_smoke["identity_payload"]["subset"]["video_ids"] == [
        "000001",
        "003414",
        "001702",
        "017734",
    ]
    assert (out_smoke / "part_01.npz").is_file()
    assert (out_smoke / "part_06.npz").is_file()


def test_extract_known_decode_defects_filtering_and_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_dir = tmp_path / "manifests"
    manifest_dir.mkdir(parents=True)
    train_file = manifest_dir / "train.csv"
    val_file = manifest_dir / "val.csv"
    test_file = manifest_dir / "test.csv"

    with train_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["video_id", "part_id", "num_frames"])
        writer.writeheader()
        writer.writerow({"video_id": "000001", "part_id": "1", "num_frames": "2"})
        writer.writerow({"video_id": "019442", "part_id": "1", "num_frames": "3"})

    with val_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["video_id", "part_id", "num_frames"])
        writer.writeheader()
        writer.writerow({"video_id": "003414", "part_id": "1", "num_frames": "2"})

    with test_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["video_id", "part_id", "num_frames"])
        writer.writeheader()
        writer.writerow({"video_id": "001702", "part_id": "1", "num_frames": "2"})

    tar_dir = tmp_path / "tar_source"
    tar_dir.mkdir()
    video_dir = tar_dir / "split_1/front_view"
    video_dir.mkdir(parents=True)
    for vid in ("000001", "003414", "001702"):
        (video_dir / f"{vid}.mp4").write_bytes(b"dummy")

    tar_path = tmp_path / "split_1_front_view.tar"
    with tarfile.open(tar_path, "w") as tar:
        for vid in ("000001", "003414", "001702"):
            tar.add(video_dir / f"{vid}.mp4", arcname=f"split_1/front_view/{vid}.mp4")

    ckpt_path = tmp_path / "best.pt"
    ckpt_path.write_bytes(b"dummy checkpoint")

    monkeypatch.setattr("src.yolo.extractor.download_checkpoint", lambda *args, **kwargs: ckpt_path)
    monkeypatch.setattr("src.yolo.extractor.download_file", lambda *args, **kwargs: tar_path)
    monkeypatch.setattr(
        "src.yolo.extractor.initialize_model",
        lambda *args, **kwargs: (DummyDualHeadModel(), None, None, {}),
    )

    class MockVideoCapture:
        def __init__(self, *args: Any) -> None:
            self.count = 0

        def isOpened(self) -> bool:  # noqa: N802
            return True

        def read(self) -> tuple[bool, np.ndarray | None]:
            if self.count < 2:
                self.count += 1
                return True, np.zeros((100, 100, 3), dtype=np.uint8)
            return False, None

        def release(self) -> None:
            pass

    monkeypatch.setattr("cv2.VideoCapture", MockVideoCapture)

    config = {
        "experiment": {
            "id": "e005_test_defects",
            "operation": "extract",
            "mode": "full",
            "seed": 42,
        },
        "checkpoint": {"repo_id": "test/repo", "revision": "rev1", "path": "best.pt"},
        "data": {
            "dataset_repo": "test/data_repo",
            "dataset_revision": "rev2",
            "parts": [1],
            "known_decode_defects": [{"video_id": "019442", "reason": "moov atom corrupt"}],
            "manifests": {
                "train": str(train_file.relative_to(tmp_path)),
                "val": str(val_file.relative_to(tmp_path)),
                "test": str(test_file.relative_to(tmp_path)),
            },
            "expected_videos": {"train": 1, "val": 1, "test": 1},
            "expected_frames": 6,
            "batch_size": 2,
            "nms_confidence": 0.25,
            "nms_iou": 0.7,
            "max_detections": 10,
            "amp": False,
            "require_cuda": False,
        },
        "model": {
            "name": "yolo26m-pose.pt",
            "architecture": "yolo26m-pose.yaml",
            "ultralytics_version": "8.4.146",
            "person_keypoints": 17,
            "hand_keypoints": 21,
            "image_size": 640,
        },
        "integrations": {"hf_artifacts": {"enabled": False}},
    }

    out_dir = tmp_path / "out_defects"
    summary = extract_pose59(config, project_root=tmp_path, output_root=out_dir, device="cpu")

    assert summary["status"] == "completed"
    assert summary["frames"] == 6
    assert summary["videos"] == 3
    assert len(summary["excluded_defects"]) == 1
    assert summary["excluded_defects"][0]["video_id"] == "019442"
    assert summary["excluded_defects"][0]["split"] == "train"
    assert summary["excluded_defects"][0]["part_id"] == 1
    assert summary["excluded_defects"][0]["num_frames"] == 3
    assert summary["excluded_defects"][0]["reason"] == "moov atom corrupt"
    assert summary["provenance"]["excluded_defects"] == summary["excluded_defects"]
    assert summary["identity_payload"]["excluded_defects"] == summary["excluded_defects"]
    assert summary["counts"]["train"]["videos"] == 1
    assert summary["counts"]["train"]["frames"] == 2

    # Verify error if defect video is missing from manifests
    bad_config = {
        **config,
        "data": {
            **config["data"],
            "known_decode_defects": [{"video_id": "999999", "reason": "nonexistent"}],
        },
    }
    with pytest.raises(RuntimeError, match="was not found in any manifest"):
        extract_pose59(
            bad_config, project_root=tmp_path, output_root=tmp_path / "out_bad", device="cpu"
        )


def test_cleanup_part_cache_control(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest_dir = tmp_path / "manifests"
    manifest_dir.mkdir(parents=True)
    manifest_file = manifest_dir / "train.csv"
    with manifest_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["video_id", "part_id", "num_frames"])
        writer.writeheader()
        writer.writerow({"video_id": "000001", "part_id": "1", "num_frames": "2"})

    tar_dir = tmp_path / "tar_source"
    tar_dir.mkdir()
    video_dir = tar_dir / "split_1/front_view"
    video_dir.mkdir(parents=True)
    (video_dir / "000001.mp4").write_bytes(b"dummy mp4")

    tar_path = tmp_path / "split_1_front_view.tar"
    with tarfile.open(tar_path, "w") as tar:
        tar.add(video_dir / "000001.mp4", arcname="split_1/front_view/000001.mp4")

    ckpt_path = tmp_path / "best.pt"
    ckpt_path.write_bytes(b"dummy checkpoint")

    monkeypatch.setattr("src.yolo.extractor.download_checkpoint", lambda *args, **kwargs: ckpt_path)
    monkeypatch.setattr(
        "src.yolo.extractor.download_file",
        lambda *args, **kwargs: tar_path,
    )
    monkeypatch.setattr(
        "src.yolo.extractor.initialize_model",
        lambda *args, **kwargs: (DummyDualHeadModel(), None, None, {}),
    )

    class MockVideoCapture:
        def __init__(self, *args: Any) -> None:
            self.count = 0

        def isOpened(self) -> bool:  # noqa: N802
            return True

        def read(self) -> tuple[bool, np.ndarray | None]:
            if self.count < 2:
                self.count += 1
                return True, np.zeros((100, 100, 3), dtype=np.uint8)
            return False, None

        def release(self) -> None:
            pass

    monkeypatch.setattr("cv2.VideoCapture", MockVideoCapture)

    base_cfg = {
        "experiment": {"id": "test_cache", "operation": "extract", "mode": "full", "seed": 42},
        "checkpoint": {"repo_id": "test/repo", "revision": "rev1", "path": "best.pt"},
        "data": {
            "dataset_repo": "test/data_repo",
            "dataset_revision": "rev2",
            "parts": [1],
            "manifests": {"train": str(manifest_file.relative_to(tmp_path))},
            "expected_frames": 2,
            "expected_videos": {"train": 1},
            "batch_size": 2,
            "nms_confidence": 0.25,
            "nms_iou": 0.7,
            "max_detections": 10,
            "amp": False,
            "require_cuda": False,
            "cleanup_part_cache": False,
        },
        "model": {
            "name": "yolo26m-pose.pt",
            "architecture": "yolo26m-pose.yaml",
            "ultralytics_version": "8.4.146",
            "person_keypoints": 17,
            "hand_keypoints": 21,
            "image_size": 640,
        },
        "integrations": {"hf_artifacts": {"enabled": False}},
    }

    out_keep = tmp_path / "out_keep"
    extract_pose59(base_cfg, project_root=tmp_path, output_root=out_keep, device="cpu")
    # With cleanup_part_cache=False, directory is kept
    cache_dir = out_keep / ".video_cache_part_01"
    assert cache_dir.is_dir()
    # But individual mp4 was still deleted
    assert not (cache_dir / "000001.mp4").exists()

    # Now test with cleanup_part_cache=True
    base_cfg["data"]["cleanup_part_cache"] = True
    out_clean = tmp_path / "out_clean"
    extract_pose59(base_cfg, project_root=tmp_path, output_root=out_clean, device="cpu")
    assert not (out_clean / ".video_cache_part_01").exists()


def test_hf_cross_runtime_resume_recovered_skips_inference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_dir = tmp_path / "manifests"
    manifest_dir.mkdir(parents=True)
    manifest_file = manifest_dir / "train.csv"
    with manifest_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["video_id", "part_id", "num_frames"])
        writer.writeheader()
        writer.writerow({"video_id": "000001", "part_id": "1", "num_frames": "2"})

    ckpt_path = tmp_path / "best.pt"
    ckpt_path.write_bytes(b"dummy checkpoint")
    monkeypatch.setattr("src.yolo.extractor.download_checkpoint", lambda *args, **kwargs: ckpt_path)

    # If model is initialized, fail the test
    def fail_if_model_init(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError(
            "initialize_model should not be called when all parts are resumed from HF"
        )

    monkeypatch.setattr("src.yolo.extractor.initialize_model", fail_if_model_init)

    config = {
        "experiment": {
            "id": "e005_hf_resume_test",
            "operation": "extract",
            "mode": "full",
            "seed": 42,
        },
        "checkpoint": {"repo_id": "test/repo", "revision": "rev1", "path": "best.pt"},
        "data": {
            "dataset_repo": "test/data_repo",
            "dataset_revision": "rev2",
            "parts": [1],
            "manifests": {"train": str(manifest_file.relative_to(tmp_path))},
            "expected_frames": 2,
            "expected_videos": {"train": 1},
            "batch_size": 2,
            "nms_confidence": 0.25,
            "nms_iou": 0.7,
            "max_detections": 10,
            "amp": False,
            "require_cuda": False,
            "resume_from_hf": True,
        },
        "model": {
            "name": "yolo26m-pose.pt",
            "architecture": "yolo26m-pose.yaml",
            "ultralytics_version": "8.4.146",
            "person_keypoints": 17,
            "hand_keypoints": 21,
            "image_size": 640,
        },
        "integrations": {
            "hf_artifacts": {
                "enabled": True,
                "resume_from_hf": True,
                "repo_id": "test/dest_repo",
                "private": True,
            }
        },
    }

    # Pre-generate valid remote NPZ and marker
    remote_dir = tmp_path / "remote_hf"
    remote_dir.mkdir()
    remote_npz = remote_dir / "part_01.npz"
    valid_arrays = {
        "X": np.zeros((2, 59, 3), dtype=np.float32),
        "keypoint_valid_mask": np.zeros((2, 59), dtype=np.uint8),
        "object_confidence": np.ones((2, 3), dtype=np.float32),
        "frame_idx": np.array([0, 1], dtype=np.int32),
        "video_id": np.array(["000001"], dtype="U6"),
        "split": np.array(["train"], dtype="U5"),
        "frame_width": np.array([1080], dtype=np.int32),
        "frame_height": np.array([1080], dtype=np.int32),
        "video_frame_offsets": np.array([0, 2], dtype=np.int64),
    }
    np.savez_compressed(remote_npz, **valid_arrays)

    # Compute expected identity
    manifest_hash = _sha256(manifest_file)
    ckpt_sha = _sha256(ckpt_path)
    env_info = {
        "device": "cpu",
        "device_type": "cpu",
        "cuda_capability": None,
        "torch": torch.__version__,
        "ultralytics": "8.4.146",
        "amp_enabled": False,
    }
    identity_payload = {
        "mode": "full",
        "checkpoint_sha256": ckpt_sha,
        "checkpoint": config["checkpoint"],
        "dataset": {"repo": "test/data_repo", "revision": "rev2"},
        "manifests": {"train": manifest_hash},
        "parts": [1],
        "excluded_defects": [],
        "model": config["model"],
        "protocol": {"batch_size": 2, "max_detections": 10, "nms_confidence": 0.25, "nms_iou": 0.7},
        "runtime": env_info,
    }
    expected_identity = hashlib.sha256(
        json.dumps(identity_payload, sort_keys=True).encode()
    ).hexdigest()

    remote_marker = remote_dir / "part_01.complete.json"
    _atomic_json(
        remote_marker,
        {
            "identity": expected_identity,
            "videos": 1,
            "frames": 2,
            "npz_sha256": _sha256(remote_npz),
            "uploaded_at": 123456789.0,
        },
    )

    def mock_hf_download(*args: Any, **kwargs: Any) -> Path:
        fn = kwargs.get("filename", "")
        if fn.endswith("complete.json"):
            return remote_marker
        if fn.endswith("part_01.npz"):
            return remote_npz
        raise FileNotFoundError(f"Unexpected file: {fn}")

    monkeypatch.setattr("src.yolo.extractor.download_file", mock_hf_download)

    uploaded_calls: list[dict[str, Path]] = []
    monkeypatch.setattr("src.yolo.extractor.ensure_repo", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "src.artifacts.upload_files",
        lambda repo_id, repo_type, files, commit_message: uploaded_calls.append(files) or "oid",
    )

    out_resume = tmp_path / "out_hf_resume"
    summary = extract_pose59(config, project_root=tmp_path, output_root=out_resume, device="cpu")

    assert summary["status"] == "completed"
    assert summary["frames"] == 2
    assert (out_resume / "part_01.npz").is_file()
    assert (out_resume / "part_01.complete.json").is_file()
    assert (out_resume / "part_01.uploaded.json").is_file()
    uploaded_data = json.loads((out_resume / "part_01.uploaded.json").read_text(encoding="utf-8"))
    assert uploaded_data.get("resumed_from_hf") is True

    # Shard part_01 was NOT uploaded again, only final summary/manifest
    assert len(uploaded_calls) == 1
    assert "e005_hf_resume_test/manifest.json" in uploaded_calls[0]


def test_hf_cross_runtime_resume_missing_falls_through(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_dir = tmp_path / "manifests"
    manifest_dir.mkdir(parents=True)
    manifest_file = manifest_dir / "train.csv"
    with manifest_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["video_id", "part_id", "num_frames"])
        writer.writeheader()
        writer.writerow({"video_id": "000001", "part_id": "1", "num_frames": "2"})

    tar_dir = tmp_path / "tar_source"
    tar_dir.mkdir()
    video_dir = tar_dir / "split_1/front_view"
    video_dir.mkdir(parents=True)
    (video_dir / "000001.mp4").write_bytes(b"dummy mp4")

    tar_path = tmp_path / "split_1_front_view.tar"
    with tarfile.open(tar_path, "w") as tar:
        tar.add(video_dir / "000001.mp4", arcname="split_1/front_view/000001.mp4")

    ckpt_path = tmp_path / "best.pt"
    ckpt_path.write_bytes(b"dummy checkpoint")

    monkeypatch.setattr("src.yolo.extractor.download_checkpoint", lambda *args, **kwargs: ckpt_path)

    def mock_download_file(*args: Any, **kwargs: Any) -> Path:
        fn = kwargs.get("filename", "")
        if "e005_fallthrough" in fn:
            # 404 / EntryNotFoundError on HF
            raise FileNotFoundError("404 EntryNotFoundError: file not found on HF")
        return tar_path

    monkeypatch.setattr("src.yolo.extractor.download_file", mock_download_file)

    model_init_called = False

    def mock_init_model(*args: Any, **kwargs: Any) -> Any:
        nonlocal model_init_called
        model_init_called = True
        return (DummyDualHeadModel(), None, None, {})

    monkeypatch.setattr("src.yolo.extractor.initialize_model", mock_init_model)

    class MockVideoCapture:
        def __init__(self, *args: Any) -> None:
            self.count = 0

        def isOpened(self) -> bool:  # noqa: N802
            return True

        def read(self) -> tuple[bool, np.ndarray | None]:
            if self.count < 2:
                self.count += 1
                return True, np.zeros((100, 100, 3), dtype=np.uint8)
            return False, None

        def release(self) -> None:
            pass

    monkeypatch.setattr("cv2.VideoCapture", MockVideoCapture)
    monkeypatch.setattr("src.yolo.extractor.ensure_repo", lambda *args, **kwargs: None)
    monkeypatch.setattr("src.artifacts.upload_files", lambda *args, **kwargs: "oid")

    config = {
        "experiment": {
            "id": "e005_fallthrough",
            "operation": "extract",
            "mode": "full",
            "seed": 42,
        },
        "checkpoint": {"repo_id": "test/repo", "revision": "rev1", "path": "best.pt"},
        "data": {
            "dataset_repo": "test/data_repo",
            "dataset_revision": "rev2",
            "parts": [1],
            "manifests": {"train": str(manifest_file.relative_to(tmp_path))},
            "expected_frames": 2,
            "expected_videos": {"train": 1},
            "batch_size": 2,
            "nms_confidence": 0.25,
            "nms_iou": 0.7,
            "max_detections": 10,
            "amp": False,
            "require_cuda": False,
        },
        "model": {
            "name": "yolo26m-pose.pt",
            "architecture": "yolo26m-pose.yaml",
            "ultralytics_version": "8.4.146",
            "person_keypoints": 17,
            "hand_keypoints": 21,
            "image_size": 640,
        },
        "integrations": {
            "hf_artifacts": {
                "enabled": True,
                "resume_from_hf": True,
                "repo_id": "test/dest_repo",
                "private": True,
            }
        },
    }

    out_dir = tmp_path / "out_fallthrough"
    summary = extract_pose59(config, project_root=tmp_path, output_root=out_dir, device="cpu")
    assert summary["status"] == "completed"
    assert model_init_called is True
    assert (out_dir / "part_01.npz").is_file()


def test_hf_cross_runtime_resume_mismatch_and_corruption_fail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_dir = tmp_path / "manifests"
    manifest_dir.mkdir(parents=True)
    manifest_file = manifest_dir / "train.csv"
    with manifest_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["video_id", "part_id", "num_frames"])
        writer.writeheader()
        writer.writerow({"video_id": "000001", "part_id": "1", "num_frames": "2"})

    ckpt_path = tmp_path / "best.pt"
    ckpt_path.write_bytes(b"dummy checkpoint")
    monkeypatch.setattr("src.yolo.extractor.download_checkpoint", lambda *args, **kwargs: ckpt_path)

    remote_dir = tmp_path / "remote_hf_bad"
    remote_dir.mkdir()
    remote_npz = remote_dir / "part_01.npz"
    valid_arrays = {
        "X": np.zeros((2, 59, 3), dtype=np.float32),
        "keypoint_valid_mask": np.zeros((2, 59), dtype=np.uint8),
        "object_confidence": np.ones((2, 3), dtype=np.float32),
        "frame_idx": np.array([0, 1], dtype=np.int32),
        "video_id": np.array(["000001"], dtype="U6"),
        "split": np.array(["train"], dtype="U5"),
        "frame_width": np.array([1080], dtype=np.int32),
        "frame_height": np.array([1080], dtype=np.int32),
        "video_frame_offsets": np.array([0, 2], dtype=np.int64),
    }
    np.savez_compressed(remote_npz, **valid_arrays)

    remote_marker = remote_dir / "part_01.complete.json"
    _atomic_json(
        remote_marker,
        {
            "identity": "mismatched_identity_12345",
            "videos": 1,
            "frames": 2,
            "npz_sha256": _sha256(remote_npz),
        },
    )

    def mock_hf_download(*args: Any, **kwargs: Any) -> Path:
        fn = kwargs.get("filename", "")
        if fn.endswith("complete.json"):
            return remote_marker
        if fn.endswith("part_01.npz"):
            return remote_npz
        raise FileNotFoundError(f"Unexpected file: {fn}")

    monkeypatch.setattr("src.yolo.extractor.download_file", mock_hf_download)

    config = {
        "experiment": {"id": "e005_hf_bad", "operation": "extract", "mode": "full", "seed": 42},
        "checkpoint": {"repo_id": "test/repo", "revision": "rev1", "path": "best.pt"},
        "data": {
            "dataset_repo": "test/data_repo",
            "dataset_revision": "rev2",
            "parts": [1],
            "manifests": {"train": str(manifest_file.relative_to(tmp_path))},
            "expected_frames": 2,
            "expected_videos": {"train": 1},
            "batch_size": 2,
            "nms_confidence": 0.25,
            "nms_iou": 0.7,
            "max_detections": 10,
            "amp": False,
            "require_cuda": False,
        },
        "model": {
            "name": "yolo26m-pose.pt",
            "architecture": "yolo26m-pose.yaml",
            "ultralytics_version": "8.4.146",
            "person_keypoints": 17,
            "hand_keypoints": 21,
            "image_size": 640,
        },
        "integrations": {
            "hf_artifacts": {
                "enabled": True,
                "resume_from_hf": True,
                "repo_id": "test/dest_repo",
                "private": True,
            }
        },
    }

    # 1. Identity mismatch must raise RuntimeError
    with pytest.raises(RuntimeError, match="Remote HF completion marker mismatch"):
        extract_pose59(
            config, project_root=tmp_path, output_root=tmp_path / "out_bad_id", device="cpu"
        )

    # 2. Corrupt schema must raise ValueError
    np.savez(remote_npz, corrupt_field=np.zeros((2, 2)))
    # Compute matching identity to bypass identity check
    manifest_hash = _sha256(manifest_file)
    ckpt_sha = _sha256(ckpt_path)
    env_info = {
        "device": "cpu",
        "device_type": "cpu",
        "cuda_capability": None,
        "torch": torch.__version__,
        "ultralytics": "8.4.146",
        "amp_enabled": False,
    }
    identity_payload = {
        "mode": "full",
        "checkpoint_sha256": ckpt_sha,
        "checkpoint": config["checkpoint"],
        "dataset": {"repo": "test/data_repo", "revision": "rev2"},
        "manifests": {"train": manifest_hash},
        "parts": [1],
        "excluded_defects": [],
        "model": config["model"],
        "protocol": {"batch_size": 2, "max_detections": 10, "nms_confidence": 0.25, "nms_iou": 0.7},
        "runtime": env_info,
    }
    expected_identity = hashlib.sha256(
        json.dumps(identity_payload, sort_keys=True).encode()
    ).hexdigest()
    _atomic_json(
        remote_marker,
        {
            "identity": expected_identity,
            "videos": 1,
            "frames": 2,
            "npz_sha256": _sha256(remote_npz),
        },
    )
    with pytest.raises(ValueError, match="Pose59 fields mismatch"):
        extract_pose59(
            config, project_root=tmp_path, output_root=tmp_path / "out_corrupt", device="cpu"
        )
