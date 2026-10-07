"""Acceptance tests for E021 YOLO Pose59, native bbox, and RGB hand crops extraction."""

from __future__ import annotations

import csv
import json
import math
import tarfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
import torch.nn as nn

from src.config import load_config
from src.yolo.extractor import (
    extract_pose59,
    select_pose59,
    select_pose59_with_detections,
)
from src.yolo.hand_rgb import (
    LETTERBOX_PAD_VALUE,
    PARQUET_SCHEMA,
    crop_hand_rgb,
    inverse_letterbox_boxes,
    square_crop_rgb,
    validate_crop_shards,
    validate_extended_schema,
    write_crop_shard,
)


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


class MockDualHeadCounter(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.forward_calls = 0

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        self.forward_calls += 1
        b = len(x)
        return {
            "person": torch.zeros((b, 4 + 1 + 17 * 3, 5)),
            "hands": torch.zeros((b, 4 + 2 + 21 * 3, 5)),
        }


def _setup_synthetic_dataset(tmp_path: Path) -> tuple[Path, Path, Path]:
    manifest_dir = tmp_path / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_file = manifest_dir / "train.csv"
    with manifest_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["video_id", "part_id", "num_frames"])
        writer.writeheader()
        writer.writerow({"video_id": "000001", "part_id": "1", "num_frames": "2"})
        writer.writerow({"video_id": "000002", "part_id": "1", "num_frames": "3"})

    video_dir = tmp_path / "tar_source" / "split_1" / "front_view"
    video_dir.mkdir(parents=True, exist_ok=True)
    (video_dir / "000001.mp4").write_bytes(b"dummy1")
    (video_dir / "000002.mp4").write_bytes(b"dummy2")

    tar_path = tmp_path / "split_1_front_view.tar"
    with tarfile.open(tar_path, "w") as tar:
        tar.add(video_dir / "000001.mp4", arcname="split_1/front_view/000001.mp4")
        tar.add(video_dir / "000002.mp4", arcname="split_1/front_view/000002.mp4")

    ckpt_path = tmp_path / "best.pt"
    ckpt_path.write_bytes(b"dummy checkpoint")
    return manifest_file, tar_path, ckpt_path


def test_select_pose59_backward_compatibility_and_selected_bbox() -> None:
    """select_pose59 backward compatibility and selected bbox/valid match exactly."""
    person = torch.zeros((2, 57))
    person[0, :4] = torch.tensor([5, 5, 80, 80])
    person[0, 4] = 0.6
    person[1, :4] = torch.tensor([10, 10, 100, 100])
    person[1, 4] = 0.9
    person[1, 6:] = 0.8

    hands = torch.zeros((4, 69))
    hands[:, 5] = torch.tensor([0, 0, 1, 1])
    hands[:, 4] = torch.tensor([0.4, 0.85, 0.7, 0.95])
    hands[0, :4] = torch.tensor([20, 20, 30, 30])
    hands[1, :4] = torch.tensor([150, 150, 160, 160])
    hands[2, :4] = torch.tensor([40, 40, 55, 55])
    hands[3, :4] = torch.tensor([200, 200, 220, 220])
    hands[:, 6:] = 0.5

    decoded = {"person": person, "hands": hands}
    old_kpts, old_mask, old_conf = select_pose59(decoded)
    kpts, mask, conf, boxes, valid_mask = select_pose59_with_detections(decoded)

    np.testing.assert_allclose(kpts, old_kpts)
    np.testing.assert_array_equal(mask, old_mask)
    np.testing.assert_allclose(conf, old_conf)

    np.testing.assert_allclose(boxes[0], [10, 10, 100, 100])
    np.testing.assert_allclose(boxes[1], [20, 20, 30, 30])
    np.testing.assert_allclose(boxes[2], [40, 40, 55, 55])
    np.testing.assert_array_equal(valid_mask, [1, 1, 1])


def test_inverse_letterbox_boxes_projection_and_invalid_zero() -> None:
    """inverse-letterbox box projection and invalid zeroing."""
    boxes = np.array(
        [
            [20, 30, 60, 70],
            [0, 0, 0, 0],
            [10, 10, 10, 10],
        ],
        dtype=np.float32,
    )
    valid_mask = np.array([1, 0, 0], dtype=np.uint8)
    res = inverse_letterbox_boxes(
        boxes,
        scale=0.5,
        pad_x=10,
        pad_y=20,
        width=100,
        height=200,
        valid_mask=valid_mask,
    )
    np.testing.assert_allclose(res[0], [20, 20, 100, 100])
    np.testing.assert_allclose(res[1], [0, 0, 0, 0])
    np.testing.assert_allclose(res[2], [0, 0, 0, 0])

    res_no_mask = inverse_letterbox_boxes(
        boxes,
        scale=0.5,
        pad_x=10,
        pad_y=20,
        width=100,
        height=200,
    )
    np.testing.assert_allclose(res_no_mask[0], [20, 20, 100, 100])
    np.testing.assert_allclose(res_no_mask[1], [0, 0, 0, 0])
    np.testing.assert_allclose(res_no_mask[2], [0, 0, 0, 0])


def test_square_crop_union_margin_padding_and_rgb_color() -> None:
    """Square union bbox+Hand21, floor centering, padding 114, and RGB red/blue channels."""
    # 20x20 frame with pure Red RGB [255, 0, 0]
    frame_red = np.zeros((20, 20, 3), dtype=np.uint8)
    frame_red[:, :] = [255, 0, 0]

    bbox = [5.0, 5.0, 10.0, 10.0]
    keypoints = np.zeros((21, 3), dtype=np.float32)
    keypoints[0] = [15.0, 15.0, 2.0]  # union width=10, height=10

    # side = ceil(1.3 * 10) = 13.
    # Floor centering: left = floor((5 + 15 - 13) / 2) = 3, top = floor((5 + 15 - 13) / 2) = 3
    jpeg_bytes, roi, padding = square_crop_rgb(frame_red, bbox, keypoints, margin=1.3, quality=95)
    assert roi[2] - roi[0] == 13
    assert roi[3] - roi[1] == 13
    assert roi[0] == math.floor((5.0 + 15.0 - 13) / 2.0)
    assert roi[1] == math.floor((5.0 + 15.0 - 13) / 2.0)
    assert len(padding) == 4

    # Decode JPEG and verify RGB red channel order (detects R/B channel swap)
    dec_bgr = cv2.imdecode(np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_COLOR)
    assert dec_bgr is not None
    dec_rgb = cv2.cvtColor(dec_bgr, cv2.COLOR_BGR2RGB)
    assert dec_rgb.shape == (13, 13, 3)
    center_color = dec_rgb[6, 6]
    assert center_color[0] > 240 and center_color[1] < 20 and center_color[2] < 20

    # Test pure Blue frame [0, 0, 255]
    frame_blue = np.zeros((20, 20, 3), dtype=np.uint8)
    frame_blue[:, :] = [0, 0, 255]
    jpeg_blue, _, _ = square_crop_rgb(frame_blue, bbox, keypoints, margin=1.3, quality=95)
    dec_blue_bgr = cv2.imdecode(np.frombuffer(jpeg_blue, np.uint8), cv2.IMREAD_COLOR)
    assert dec_blue_bgr is not None
    dec_blue_rgb = cv2.cvtColor(dec_blue_bgr, cv2.COLOR_BGR2RGB)
    blue_center = dec_blue_rgb[6, 6]
    assert blue_center[2] > 240 and blue_center[0] < 20 and blue_center[1] < 20

    # Test border padding value 114
    bbox_edge = [18.0, 18.0, 22.0, 22.0]
    jpeg_edge, roi_edge, pad_edge = square_crop_rgb(
        frame_red, bbox_edge, np.zeros((21, 3)), margin=1.3
    )
    assert roi_edge[2] > 20 or roi_edge[3] > 20
    assert pad_edge[2] > 0 or pad_edge[3] > 0
    dec_edge_bgr = cv2.imdecode(np.frombuffer(jpeg_edge, np.uint8), cv2.IMREAD_COLOR)
    assert dec_edge_bgr is not None
    dec_edge_rgb = cv2.cvtColor(dec_edge_bgr, cv2.COLOR_BGR2RGB)
    pad_pixel = dec_edge_rgb[-1, -1]
    # Padded area must be near (114, 114, 114)
    assert np.all(np.abs(pad_pixel.astype(int) - np.array(LETTERBOX_PAD_VALUE)) < 5)


def test_crop_hand_rgb_rejections_and_explicit_parquet_schema(tmp_path: Path) -> None:
    """crop_hand_rgb rejects invalid/zero/non-finite boxes; explicit Parquet schema enforced."""
    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    # Missing hand / None image / invalid flag
    jpeg, roi, pad = crop_hand_rgb(None, [0, 0, 10, 10], np.zeros((21, 3)), True)
    assert jpeg is None and np.array_equal(roi, [0, 0, 0, 0]) and np.array_equal(pad, [0, 0, 0, 0])

    jpeg, roi, pad = crop_hand_rgb(frame, [0, 0, 10, 10], np.zeros((21, 3)), False)
    assert jpeg is None and np.array_equal(roi, [0, 0, 0, 0]) and np.array_equal(pad, [0, 0, 0, 0])

    # Zero-area box
    jpeg, roi, pad = crop_hand_rgb(frame, [10, 10, 10, 10], np.zeros((21, 3)), True)
    assert jpeg is None and np.array_equal(roi, [0, 0, 0, 0]) and np.array_equal(pad, [0, 0, 0, 0])

    # Inverted box
    jpeg, roi, pad = crop_hand_rgb(frame, [20, 20, 10, 10], np.zeros((21, 3)), True)
    assert jpeg is None and np.array_equal(roi, [0, 0, 0, 0]) and np.array_equal(pad, [0, 0, 0, 0])

    # Non-finite box
    jpeg, roi, pad = crop_hand_rgb(frame, [float("nan"), 0, 10, 10], np.zeros((21, 3)), True)
    assert jpeg is None and np.array_equal(roi, [0, 0, 0, 0]) and np.array_equal(pad, [0, 0, 0, 0])

    # Verify PyArrow schema nullability contract: only left_jpeg and right_jpeg are nullable
    for field in PARQUET_SCHEMA:
        if field.name in {"left_jpeg", "right_jpeg"}:
            assert field.nullable is True
        else:
            assert field.nullable is False, f"Field {field.name} must be non-nullable"

    # Write shard with all-null JPEG columns
    rows = [
        {
            "video_id": f"{i:06d}",
            "split": "train",
            "frame_idx": 0,
            "frame_width": 640,
            "frame_height": 480,
            "left_jpeg": None,
            "right_jpeg": None,
            "left_valid": 0,
            "right_valid": 0,
            "left_bbox": [0.0, 0.0, 0.0, 0.0],
            "right_bbox": [0.0, 0.0, 0.0, 0.0],
            "left_confidence": 0.0,
            "right_confidence": 0.0,
            "left_roi": [0, 0, 0, 0],
            "right_roi": [0, 0, 0, 0],
            "left_padding": [0, 0, 0, 0],
            "right_padding": [0, 0, 0, 0],
        }
        for i in range(10)
    ]
    shard_info = write_crop_shard(rows, tmp_path, 1, 0)
    assert shard_info["name"] == "part_01_crops_000.parquet"
    shard_file = tmp_path / "part_01_crops_000.parquet"
    table = pq.read_table(shard_file)
    assert table.schema == PARQUET_SCHEMA
    assert table.column("left_jpeg").null_count == 10
    assert table.column("right_jpeg").null_count == 10


def test_synthetic_integration_cross_video_batching_and_e005_per_video(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Synthetic integration: E021 cross-video batching vs E005 per-video flush, summary fields."""
    manifest_file, tar_path, ckpt_path = _setup_synthetic_dataset(tmp_path)

    mock_model = MockDualHeadCounter()
    monkeypatch.setattr("src.yolo.extractor.download_checkpoint", lambda *a, **kw: ckpt_path)
    monkeypatch.setattr("src.yolo.extractor.download_file", lambda *a, **kw: tar_path)
    monkeypatch.setattr(
        "src.yolo.extractor.initialize_model", lambda *a, **kw: (mock_model, None, None, {})
    )

    class MockVideoCapture:
        def __init__(self, filename: str) -> None:
            self.total = 2 if "000001" in filename else 3
            self.count = 0

        def isOpened(self) -> bool:  # noqa: N802
            return True

        def read(self) -> tuple[bool, np.ndarray | None]:
            if self.count < self.total:
                self.count += 1
                return True, np.zeros((100, 100, 3), dtype=np.uint8)
            return False, None

        def release(self) -> None:
            pass

    monkeypatch.setattr("cv2.VideoCapture", MockVideoCapture)

    config_e021 = {
        "experiment": {
            "id": "e021_synth",
            "operation": "extract_hand_rgb",
            "mode": "full",
            "seed": 42,
        },
        "checkpoint": {"repo_id": "test/repo", "revision": "rev1", "path": "best.pt"},
        "data": {
            "dataset_repo": "test/data_repo",
            "dataset_revision": "rev2",
            "parts": [1],
            "manifests": {
                "train": str(manifest_file.relative_to(tmp_path)),
                "val": str(manifest_file.relative_to(tmp_path)),
                "test": str(manifest_file.relative_to(tmp_path)),
            },
            "expected_frames": 15,
            "expected_videos": {"train": 2, "val": 2, "test": 2},
            "batch_size": 32,
            "nms_confidence": 0.25,
            "nms_iou": 0.7,
            "max_detections": 10,
            "amp": False,
            "require_cuda": False,
        },
        "crops": {"crop_shard_frames": 4096, "margin": 1.3, "quality": 95},
        "model": {
            "name": "yolo26m-pose.pt",
            "architecture": "yolo26m-pose.yaml",
            "ultralytics_version": "8.4.146",
            "person_keypoints": 17,
            "hand_keypoints": 21,
            "image_size": 640,
        },
    }

    out_e021 = tmp_path / "out_e021"
    summary_e021 = extract_pose59(
        config_e021, project_root=tmp_path, output_root=out_e021, device="cpu"
    )

    # 15 frames total across 6 videos, batch_size=32 -> exactly 1 model forward call for E021
    assert mock_model.forward_calls == 1
    assert summary_e021["forward_calls"] == 1
    assert summary_e021["mean_effective_batch"] == 15.0

    # Verify summary fields
    assert "timings" in summary_e021
    timings = summary_e021["timings"]
    for k in ("download", "decode", "letterbox", "inference_nms", "crop_encode", "write", "total"):
        assert k in timings and isinstance(timings[k], float)
    assert "crop_counts" in summary_e021
    assert "crop_bytes" in summary_e021 and summary_e021["crop_bytes"] > 0

    # Verify NPZ contents
    npz_file = out_e021 / "part_01.npz"
    assert npz_file.is_file()
    with np.load(npz_file, allow_pickle=False) as loaded:
        arrays = dict(loaded)
        validate_extended_schema(arrays)
        np.testing.assert_array_equal(
            arrays["frame_idx"], [0, 1, 0, 1, 2, 0, 1, 0, 1, 2, 0, 1, 0, 1, 2]
        )
        np.testing.assert_array_equal(arrays["video_frame_offsets"], [0, 2, 5, 7, 10, 12, 15])
        for forbidden in ("label", "gloss", "class_id"):
            assert forbidden not in arrays

    # Verify Parquet contents
    parquet_file = out_e021 / "part_01_crops_000.parquet"
    assert parquet_file.is_file()
    pq_table = pq.read_table(parquet_file)
    assert len(pq_table) == 15
    for forbidden in ("label", "gloss", "class_id"):
        assert forbidden not in pq_table.column_names

    # Test E005 per-video batching regression: operation="extract" flushes per video -> exactly 6 forward calls
    config_e005 = {
        **config_e021,
        "experiment": {
            "id": "e005_synth",
            "operation": "extract",
            "mode": "full",
            "seed": 42,
        },
    }
    mock_model.forward_calls = 0
    out_e005 = tmp_path / "out_e005"
    summary_e005 = extract_pose59(
        config_e005, project_root=tmp_path, output_root=out_e005, device="cpu"
    )
    assert summary_e005["status"] == "completed"
    assert mock_model.forward_calls == 6


def test_resume_validation_corruption_and_alignment_rejections(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Resume validation rejects corrupt markers, mismatched hashes, schemas, and row-wise null alignments."""
    manifest_file, tar_path, ckpt_path = _setup_synthetic_dataset(tmp_path)

    monkeypatch.setattr("src.yolo.extractor.download_checkpoint", lambda *a, **kw: ckpt_path)
    monkeypatch.setattr("src.yolo.extractor.download_file", lambda *a, **kw: tar_path)
    monkeypatch.setattr(
        "src.yolo.extractor.initialize_model",
        lambda *a, **kw: (MockDualHeadCounter(), None, None, {}),
    )

    class MockVideoCapture:
        def __init__(self, filename: str) -> None:
            self.total = 2 if "000001" in filename else 3
            self.count = 0

        def isOpened(self) -> bool:  # noqa: N802
            return True

        def read(self) -> tuple[bool, np.ndarray | None]:
            if self.count < self.total:
                self.count += 1
                return True, np.zeros((100, 100, 3), dtype=np.uint8)
            return False, None

        def release(self) -> None:
            pass

    monkeypatch.setattr("cv2.VideoCapture", MockVideoCapture)

    config = {
        "experiment": {
            "id": "e021_resume_test",
            "operation": "extract_hand_rgb",
            "mode": "full",
            "seed": 42,
        },
        "checkpoint": {"repo_id": "test/repo", "revision": "rev1", "path": "best.pt"},
        "data": {
            "dataset_repo": "test/data_repo",
            "dataset_revision": "rev2",
            "parts": [1],
            "manifests": {
                "train": str(manifest_file.relative_to(tmp_path)),
                "val": str(manifest_file.relative_to(tmp_path)),
                "test": str(manifest_file.relative_to(tmp_path)),
            },
            "expected_frames": 15,
            "expected_videos": {"train": 2, "val": 2, "test": 2},
            "batch_size": 16,
            "nms_confidence": 0.25,
            "nms_iou": 0.7,
            "max_detections": 10,
            "amp": False,
            "require_cuda": False,
        },
        "crops": {"crop_shard_frames": 4096, "margin": 1.3, "quality": 95},
        "model": {
            "name": "yolo26m-pose.pt",
            "architecture": "yolo26m-pose.yaml",
            "ultralytics_version": "8.4.146",
            "person_keypoints": 17,
            "hand_keypoints": 21,
            "image_size": 640,
        },
    }

    out_dir = tmp_path / "out_resume"
    summary = extract_pose59(config, project_root=tmp_path, output_root=out_dir, device="cpu")
    assert summary["status"] == "completed"

    marker_file = out_dir / "part_01.complete.json"
    marker_orig = json.loads(marker_file.read_text(encoding="utf-8"))
    shard_file = out_dir / "part_01_crops_000.parquet"
    shard_orig_bytes = shard_file.read_bytes()

    # 1. Corrupt parquet shard hash
    shard_file.write_bytes(b"corrupted parquet bytes")
    with pytest.raises(RuntimeError, match="Corrupt or missing parquet shard on resume"):
        extract_pose59(config, project_root=tmp_path, output_root=out_dir, device="cpu")
    shard_file.write_bytes(shard_orig_bytes)

    # 2. Stale completion marker identity
    marker_bad_id = {**marker_orig, "identity": "stale_identity_123"}
    marker_file.write_text(json.dumps(marker_bad_id), encoding="utf-8")
    with pytest.raises(RuntimeError, match="Stale or mixed completion marker"):
        extract_pose59(config, project_root=tmp_path, output_root=out_dir, device="cpu")

    # 3. Incomplete / empty shards list
    marker_no_shards = {**marker_orig, "shards": []}
    marker_file.write_text(json.dumps(marker_no_shards), encoding="utf-8")
    with pytest.raises(RuntimeError, match="Incomplete crop shard marker"):
        extract_pose59(config, project_root=tmp_path, output_root=out_dir, device="cpu")

    # 4. Extra shard on disk not in marker
    extra_shard = out_dir / "part_01_crops_001.parquet"
    extra_shard.write_bytes(shard_orig_bytes)
    marker_file.write_text(json.dumps(marker_orig), encoding="utf-8")
    with pytest.raises(RuntimeError, match="Crop shard set mismatch on resume"):
        extract_pose59(config, project_root=tmp_path, output_root=out_dir, device="cpu")
    extra_shard.unlink()

    # 5. Shard crop count mismatch
    marker_bad_crop = {**marker_orig, "crop_counts": {"left": 999, "right": 999}}
    marker_file.write_text(json.dumps(marker_bad_crop), encoding="utf-8")
    with pytest.raises(RuntimeError, match="Crop count marker mismatch"):
        extract_pose59(config, project_root=tmp_path, output_root=out_dir, device="cpu")

    # 6. Direct unit test for row-wise null JPEG alignment in validate_crop_shards
    table = pq.read_table(shard_file)
    # Construct misaligned table where null_count is correct but row indices are inverted
    pydict = table.to_pydict()
    pydict["left_valid"] = [1 - v for v in pydict["left_valid"]]
    bad_table = pa.Table.from_pydict(pydict, schema=PARQUET_SCHEMA)
    bad_dir = tmp_path / "bad_dir"
    bad_dir.mkdir(parents=True, exist_ok=True)
    bad_shard_file = bad_dir / "part_01_crops_000.parquet"
    pq.write_table(bad_table, bad_shard_file)

    bad_marker = {
        "shards": [
            {
                "name": bad_shard_file.name,
                "frames": len(bad_table),
                "sha256": _sha256(bad_shard_file),
            }
        ],
        "crop_counts": {
            "left": int(np.sum(pydict["left_valid"])),
            "right": int(np.sum(pydict["right_valid"])),
        },
        "missing_counts": {
            "left": len(bad_table) - int(np.sum(pydict["left_valid"])),
            "right": len(bad_table) - int(np.sum(pydict["right_valid"])),
            "person": 0,
        },
    }
    with pytest.raises(RuntimeError, match="Crop JPEG null alignment mismatch"):
        validate_crop_shards(bad_marker, bad_dir, 1, len(bad_table), _sha256)


def test_config_validation_e005_vs_e021_and_hf_rejection(tmp_path: Path) -> None:
    """Config validation: E005 subset zero rejected vs E021 test=0 accepted, E021 HF rejected."""
    # E005 config with test=0 must be rejected
    base_e005 = {
        "experiment": {"id": "e005_test", "operation": "extract", "mode": "smoke", "seed": 42},
        "model": {"name": "yolo26m-pose.pt", "person_keypoints": 17, "hand_keypoints": 21},
        "checkpoint": {"repo_id": "r", "revision": "rev", "path": "p"},
        "output": {"root": "out"},
        "data": {
            "dataset_repo": "dr",
            "dataset_revision": "drev",
            "parts": [1],
            "manifests": {"train": "t.csv", "val": "v.csv", "test": "te.csv"},
            "subset": {"train": 10, "val": 10, "test": 0},
            "batch_size": 32,
            "nms_confidence": 0.25,
            "nms_iou": 0.7,
            "max_detections": 100,
        },
    }
    e005_yaml = tmp_path / "e005.yaml"
    import yaml

    e005_yaml.write_text(yaml.safe_dump(base_e005), encoding="utf-8")
    with pytest.raises(
        ValueError,
        match="smoke extract data.subset must be a non-empty mapping of split to positive",
    ):
        load_config(e005_yaml)

    # E021 config with test=0 must be accepted
    base_e021 = {
        **base_e005,
        "experiment": {
            "id": "e021_test",
            "operation": "extract_hand_rgb",
            "mode": "smoke",
            "seed": 42,
        },
        "crops": {"crop_shard_frames": 4096, "margin": 1.3, "quality": 95},
    }
    e021_yaml = tmp_path / "e021.yaml"
    e021_yaml.write_text(yaml.safe_dump(base_e021), encoding="utf-8")
    loaded_e021 = load_config(e021_yaml)
    assert loaded_e021["data"]["subset"]["test"] == 0

    # E021 config with hf_artifacts.enabled=true in smoke mode must be rejected
    base_e021_hf = {
        **base_e021,
        "integrations": {"hf_artifacts": {"enabled": True, "repo_id": "test/repo"}},
    }
    e021_hf_yaml = tmp_path / "e021_hf.yaml"
    e021_hf_yaml.write_text(yaml.safe_dump(base_e021_hf), encoding="utf-8")
    with pytest.raises(
        ValueError,
        match="smoke extract_hand_rgb does not support integrations.hf_artifacts.enabled=true",
    ):
        load_config(e021_hf_yaml)

    # Verify smoke config covers parts 1..7
    repo_root = Path(__file__).resolve().parent.parent
    smoke_cfg_path = repo_root / "configs/experiments/e021_yolo_pose59_bbox_handrgb_smoke.yaml"
    smoke_cfg = load_config(smoke_cfg_path)
    assert smoke_cfg["data"]["cleanup_part_cache"] is True
    assert smoke_cfg["data"]["expected_frames"] == 25668
    train_manifest = repo_root / smoke_cfg["data"]["manifests"]["train"]
    with train_manifest.open(newline="", encoding="utf-8") as f:
        train_rows = [r for r in csv.DictReader(f) if r["video_id"].zfill(6) != "019442"][:300]
    val_manifest = repo_root / smoke_cfg["data"]["manifests"]["val"]
    with val_manifest.open(newline="", encoding="utf-8") as f:
        val_rows = [r for r in csv.DictReader(f) if r["video_id"].zfill(6) != "019442"][:100]
    all_parts = {int(r["part_id"]) for r in train_rows + val_rows}
    assert all_parts == {1, 2, 3, 4, 5, 6, 7}


def test_e021_hf_upload_mapping_and_full_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """E021 HF upload commits NPZ + complete marker + exact Parquet shards in single commit."""
    manifest_file, tar_path, ckpt_path = _setup_synthetic_dataset(tmp_path)

    uploaded_files_map: dict[str, Path] = {}

    def mock_upload(*, repo_id: str, files: dict[str, Path], commit_message: str, **kw: Any) -> str:
        uploaded_files_map.update(files)
        return "mock_revision_hash_123"

    monkeypatch.setattr("src.yolo.extractor._upload_files_with_retry", mock_upload)
    monkeypatch.setattr("src.yolo.extractor.ensure_repo", lambda *a, **kw: None)
    monkeypatch.setattr("src.yolo.extractor.download_checkpoint", lambda *a, **kw: ckpt_path)
    monkeypatch.setattr("src.yolo.extractor.download_file", lambda *a, **kw: tar_path)
    monkeypatch.setattr(
        "src.yolo.extractor.initialize_model",
        lambda *a, **kw: (MockDualHeadCounter(), None, None, {}),
    )

    class MockVideoCapture:
        def __init__(self, filename: str) -> None:
            self.total = 2 if "000001" in filename else 3
            self.count = 0

        def isOpened(self) -> bool:  # noqa: N802
            return True

        def read(self) -> tuple[bool, np.ndarray | None]:
            if self.count < self.total:
                self.count += 1
                return True, np.zeros((100, 100, 3), dtype=np.uint8)
            return False, None

        def release(self) -> None:
            pass

    monkeypatch.setattr("cv2.VideoCapture", MockVideoCapture)

    config = {
        "experiment": {
            "id": "e021_full_hf_upload",
            "operation": "extract_hand_rgb",
            "mode": "full",
            "seed": 42,
        },
        "checkpoint": {"repo_id": "test/repo", "revision": "rev1", "path": "best.pt"},
        "data": {
            "dataset_repo": "test/data_repo",
            "dataset_revision": "rev2",
            "parts": [1],
            "manifests": {
                "train": str(manifest_file.relative_to(tmp_path)),
                "val": str(manifest_file.relative_to(tmp_path)),
                "test": str(manifest_file.relative_to(tmp_path)),
            },
            "expected_frames": 15,
            "expected_videos": {"train": 2, "val": 2, "test": 2},
            "batch_size": 32,
            "nms_confidence": 0.25,
            "nms_iou": 0.7,
            "max_detections": 10,
            "amp": False,
            "require_cuda": False,
        },
        "crops": {"crop_shard_frames": 4096, "margin": 1.3, "quality": 95},
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
                "repo_id": "test/upload_repo",
                "private": True,
            },
        },
    }

    out_dir = tmp_path / "out_hf_upload"
    summary = extract_pose59(config, project_root=tmp_path, output_root=out_dir, device="cpu")
    assert summary["status"] == "completed"

    # Verify uploaded files map contains NPZ, marker, and all parquet shards
    exp_id = config["experiment"]["id"]
    assert f"{exp_id}/part_01.npz" in uploaded_files_map
    assert f"{exp_id}/part_01.complete.json" in uploaded_files_map
    assert f"{exp_id}/part_01_crops_000.parquet" in uploaded_files_map

    # Verify upload receipt file
    uploaded_file = out_dir / "part_01.uploaded.json"
    assert uploaded_file.is_file()
    uploaded_data = json.loads(uploaded_file.read_text(encoding="utf-8"))
    assert "identity" in uploaded_data
    assert "npz_sha256" in uploaded_data
    assert uploaded_data.get("revision") == "mock_revision_hash_123"

    # Verify full YAML file
    repo_root = Path(__file__).resolve().parent.parent
    full_yaml_path = repo_root / "configs/experiments/e021_yolo_pose59_bbox_handrgb_full.yaml"
    full_cfg = load_config(full_yaml_path)
    assert full_cfg["experiment"]["operation"] == "extract_hand_rgb"
    assert full_cfg["experiment"]["mode"] == "full"
    assert full_cfg["integrations"]["hf_artifacts"]["enabled"] is True
    assert (
        "resume_from_hf" not in full_cfg["integrations"]["hf_artifacts"]
        or full_cfg["integrations"]["hf_artifacts"]["resume_from_hf"] is False
    )
    assert full_cfg["data"]["expected_frames"] == 1587085
