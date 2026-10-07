"""Focused unit and synthetic integration tests for E026 Phase C visual crop audit."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.config import load_config
from src.e026_visual_audit import (
    REASON_TARGETS,
    collect_aligned_candidates,
    create_contact_sheet,
    retrieve_and_validate_crops,
    run_e026_visual_audit,
    select_visual_sample,
    write_visual_sample_csv,
)
from src.yolo.hand_rgb import PARQUET_SCHEMA


def _encode_rgb_jpeg(rgb_array: np.ndarray, quality: int = 95) -> bytes:
    bgr = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return buf.tobytes()


def _make_parquet_shard(
    rows: list[dict[str, Any]],
    path: Path,
) -> Path:
    table = pa.Table.from_pylist(rows, schema=PARQUET_SCHEMA)
    pq.write_table(table, path)
    return path


def test_collect_aligned_candidates_filters_and_formats(tmp_path: Path) -> None:
    """collect_aligned_candidates extracts valid hands for U30 frames and clean video allowlist."""
    u30_splits = {
        "train": {
            "video_id": np.array(["000001", "019442"], dtype="U6"),
            "frame_indices": np.arange(30, dtype=np.int32).reshape(1, 30).repeat(2, axis=0),
            "valid_frame_mask": np.ones((2, 30), dtype=np.uint8),
        },
        "val": {
            "video_id": np.array(["000002"], dtype="U6"),
            "frame_indices": np.arange(30, dtype=np.int32).reshape(1, 30),
            "valid_frame_mask": np.ones((1, 30), dtype=np.uint8),
        },
    }

    # Video 000001 (clean train), 019442 (defect, excluded), 000002 (clean val), 000003 (test)
    rows = [
        {
            "video_id": "000001",
            "split": "train",
            "frame_idx": 0,
            "frame_width": 1080,
            "frame_height": 1080,
            "left_jpeg": None,
            "right_jpeg": None,
            "left_valid": 1,
            "right_valid": 1,
            "left_bbox": [10.0, 10.0, 50.0, 50.0],
            "right_bbox": [10.0, 10.0, 50.0, 50.0],
            "left_confidence": 0.9,
            "right_confidence": 0.85,
            "left_roi": [0, 0, 80, 80],
            "right_roi": [0, 0, 80, 80],
            "left_padding": [5, 5, 5, 5],
            "right_padding": [0, 0, 0, 0],
        },
        {
            "video_id": "019442",
            "split": "train",
            "frame_idx": 0,
            "frame_width": 1080,
            "frame_height": 1080,
            "left_jpeg": None,
            "right_jpeg": None,
            "left_valid": 1,
            "right_valid": 0,
            "left_bbox": [10.0, 10.0, 50.0, 50.0],
            "right_bbox": [0.0, 0.0, 0.0, 0.0],
            "left_confidence": 0.9,
            "right_confidence": 0.0,
            "left_roi": [0, 0, 80, 80],
            "right_roi": [0, 0, 0, 0],
            "left_padding": [0, 0, 0, 0],
            "right_padding": [0, 0, 0, 0],
        },
        {
            "video_id": "000003",
            "split": "test",
            "frame_idx": 0,
            "frame_width": 1080,
            "frame_height": 1080,
            "left_jpeg": None,
            "right_jpeg": None,
            "left_valid": 1,
            "right_valid": 0,
            "left_bbox": [10.0, 10.0, 50.0, 50.0],
            "right_bbox": [0.0, 0.0, 0.0, 0.0],
            "left_confidence": 0.9,
            "right_confidence": 0.0,
            "left_roi": [0, 0, 80, 80],
            "right_roi": [0, 0, 0, 0],
            "left_padding": [0, 0, 0, 0],
            "right_padding": [0, 0, 0, 0],
        },
    ]
    shard = _make_parquet_shard(rows, tmp_path / "candidates.parquet")
    allowed = {"train": {"000001"}, "val": {"000002"}}

    candidates = collect_aligned_candidates([shard], u30_splits, allowed_video_ids=allowed)
    # Only 000001 train left and right should be included (2 candidates)
    assert len(candidates) == 2
    assert all(c["video_id"] == "000001" for c in candidates)
    left_cand = next(c for c in candidates if c["hand"] == "left")
    assert left_cand["crop_side"] == 80
    assert left_cand["padding_ratio"] > 0.0


def test_deterministic_selection_and_reason_counts() -> None:
    """Selection returns exactly 32 unique candidates with requested reason quotas and train/val balance."""
    candidates: list[dict[str, Any]] = []

    # Generate 40 train candidates and 40 val candidates with varied metrics
    for s in ("train", "val"):
        for i in range(40):
            side = 50 + i * 5
            conf = 0.50 + (i % 20) * 0.02
            pad_ratio = 0.01 * (i % 15)
            candidates.append({
                "split": s,
                "video_id": f"{i:06d}",
                "frame_idx": i % 30,
                "hand": "left" if i % 2 == 0 else "right",
                "crop_side": side,
                "confidence": conf,
                "padding_ratio": pad_ratio,
            })

    # Add test split and unlisted items to ensure select_visual_sample only picks from train/val
    candidates.append({
        "split": "test",
        "video_id": "999999",
        "frame_idx": 0,
        "hand": "left",
        "crop_side": 20,
        "confidence": 0.99,
        "padding_ratio": 0.0,
    })

    selected = select_visual_sample(candidates, seed=42)
    assert len(selected) == 32

    # Check 100% uniqueness of keys
    keys = [(c["split"], c["video_id"], c["frame_idx"], c["hand"]) for c in selected]
    assert len(keys) == len(set(keys))

    # Check reason counts
    expected_counts = dict(REASON_TARGETS)
    actual_counts = {r: sum(1 for c in selected if c["selection_reason"] == r) for r, _ in REASON_TARGETS}
    assert actual_counts == expected_counts

    # Check balanced train/val splits: 16 train, 16 val
    train_count = sum(1 for c in selected if c["split"] == "train")
    val_count = sum(1 for c in selected if c["split"] == "val")
    assert train_count == 16
    assert val_count == 16

    # Test split must never be selected
    assert all(c["split"] in {"train", "val"} for c in selected)

    # Repeatability with seed=42
    selected_repeat = select_visual_sample(candidates, seed=42)
    assert selected == selected_repeat


def test_selected_jpeg_null_valid_mismatch_fails(tmp_path: Path) -> None:
    """Selected hand with valid=1 but null JPEG bytes fails validation."""
    selected_items = [
        {
            "split": "train",
            "video_id": "000001",
            "frame_idx": 0,
            "hand": "left",
            "selection_reason": "random_control",
            "crop_side": 100,
            "confidence": 0.9,
            "padding_ratio": 0.0,
        }
    ]

    # Row has left_valid=1, but left_jpeg=None (null mismatch)
    bad_row = {
        "video_id": "000001",
        "split": "train",
        "frame_idx": 0,
        "frame_width": 1080,
        "frame_height": 1080,
        "left_jpeg": None,
        "right_jpeg": None,
        "left_valid": 1,
        "right_valid": 0,
        "left_bbox": [10.0, 10.0, 90.0, 90.0],
        "right_bbox": [0.0, 0.0, 0.0, 0.0],
        "left_confidence": 0.9,
        "right_confidence": 0.0,
        "left_roi": [0, 0, 100, 100],
        "right_roi": [0, 0, 0, 0],
        "left_padding": [0, 0, 0, 0],
        "right_padding": [0, 0, 0, 0],
    }
    shard = _make_parquet_shard([bad_row], tmp_path / "bad_shard.parquet")

    with pytest.raises(ValueError, match="Selected hand must have valid=1 and non-null JPEG"):
        retrieve_and_validate_crops(selected_items, [shard])


def test_synthetic_rgb_channels_and_square_roi_validation(tmp_path: Path) -> None:
    """Synthetic RGB JPEG proves color channel preservation and validates square/ROI dimensions."""
    # Create pure red crop (R=255, G=0, B=0), size 100x100
    red_rgb = np.zeros((100, 100, 3), dtype=np.uint8)
    red_rgb[:, :, 0] = 255
    red_jpeg = _encode_rgb_jpeg(red_rgb)

    selected_item = {
        "split": "train",
        "video_id": "000001",
        "frame_idx": 5,
        "hand": "left",
        "selection_reason": "random_control",
        "crop_side": 100,
        "confidence": 0.95,
        "padding_ratio": 0.0,
    }

    valid_row = {
        "video_id": "000001",
        "split": "train",
        "frame_idx": 5,
        "frame_width": 1080,
        "frame_height": 1080,
        "left_jpeg": red_jpeg,
        "right_jpeg": None,
        "left_valid": 1,
        "right_valid": 0,
        "left_bbox": [10.0, 10.0, 90.0, 90.0],
        "right_bbox": [0.0, 0.0, 0.0, 0.0],
        "left_confidence": 0.95,
        "right_confidence": 0.0,
        "left_roi": [0, 0, 100, 100],
        "right_roi": [0, 0, 0, 0],
        "left_padding": [0, 0, 0, 0],
        "right_padding": [0, 0, 0, 0],
    }
    shard = _make_parquet_shard([valid_row], tmp_path / "rgb_shard.parquet")

    validated = retrieve_and_validate_crops([selected_item], [shard])
    assert len(validated) == 1
    _, jpeg_bytes, img_bgr = validated[0]
    assert jpeg_bytes == red_jpeg

    # Verify BGR decoded image: Blue is at channel 0, Red is at channel 2
    assert img_bgr[50, 50, 2] > 200  # Red channel in BGR
    assert img_bgr[50, 50, 0] < 50   # Blue channel in BGR

    # Check contact sheet RGB representation and persistence
    sheet = create_contact_sheet(validated, tile_size=100, cols=1)
    sheet_path = tmp_path / "test_sheet.jpg"
    assert cv2.imwrite(str(sheet_path), sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    reloaded_sheet_bgr = cv2.imread(str(sheet_path))
    assert reloaded_sheet_bgr is not None
    assert reloaded_sheet_bgr[62, 62, 2] > 200  # Red channel in BGR
    assert reloaded_sheet_bgr[62, 62, 0] < 50   # Blue channel in BGR

    # 1. Non-square image rejection
    non_square_bgr = np.zeros((100, 80, 3), dtype=np.uint8)
    ok, non_square_buf = cv2.imencode(".jpg", non_square_bgr)
    assert ok
    non_sq_row = dict(valid_row, left_jpeg=non_square_buf.tobytes(), left_roi=[0, 0, 80, 80])
    shard_non_sq = _make_parquet_shard([non_sq_row], tmp_path / "non_sq.parquet")
    with pytest.raises(ValueError, match="Decoded crop is not square"):
        retrieve_and_validate_crops([selected_item], [shard_non_sq])

    # 2. ROI side mismatch rejection (image is 100x100, but ROI side is 120)
    mismatch_row = dict(valid_row, left_roi=[0, 0, 120, 120])
    shard_mismatch = _make_parquet_shard([mismatch_row], tmp_path / "mismatch.parquet")
    with pytest.raises(ValueError, match="does not equal ROI side"):
        retrieve_and_validate_crops([selected_item], [shard_mismatch])


def test_vertical_visual_audit_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Full vertical run writes visual_sample.csv, crop files, contact sheet, and summary.json."""
    # 1. Prepare clean manifests
    train_manifest = tmp_path / "train.csv"
    train_manifest.write_text(
        "video_id,video_path,gloss,class_id,signer_id,fps,resolution,num_frames,length_seconds,part_id,split\n"
        "000001,000001.mp4,g1,0,001,25.0,1080,30,1.2,1,train\n",
        encoding="utf-8",
    )
    val_manifest = tmp_path / "val.csv"
    val_manifest.write_text(
        "video_id,video_path,gloss,class_id,signer_id,fps,resolution,num_frames,length_seconds,part_id,split\n"
        "000002,000002.mp4,g2,1,002,25.0,1080,30,1.2,1,val\n",
        encoding="utf-8",
    )

    # 2. Prepare mock Uniform-30 splits
    u30_train_npz = tmp_path / "train.npz"
    np.savez(
        u30_train_npz,
        video_id=np.array(["000001"], dtype="U6"),
        frame_indices=np.arange(30, dtype=np.int32).reshape(1, 30),
        valid_frame_mask=np.ones((1, 30), dtype=np.uint8),
    )
    u30_val_npz = tmp_path / "val.npz"
    np.savez(
        u30_val_npz,
        video_id=np.array(["000002"], dtype="U6"),
        frame_indices=np.arange(30, dtype=np.int32).reshape(1, 30),
        valid_frame_mask=np.ones((1, 30), dtype=np.uint8),
    )

    # 3. Create synthetic crop JPEG (64x64)
    crop_rgb = np.full((64, 64, 3), 128, dtype=np.uint8)
    crop_jpeg = _encode_rgb_jpeg(crop_rgb)

    # 4. Generate 30 train frames and 30 val frames with left and right hands
    parquet_rows = []
    for f in range(30):
        parquet_rows.append({
            "video_id": "000001",
            "split": "train",
            "frame_idx": f,
            "frame_width": 1080,
            "frame_height": 1080,
            "left_jpeg": crop_jpeg,
            "right_jpeg": crop_jpeg,
            "left_valid": 1,
            "right_valid": 1,
            "left_bbox": [10.0, 10.0, 50.0, 50.0],
            "right_bbox": [10.0, 10.0, 50.0, 50.0],
            "left_confidence": float(0.80 + f * 0.005),
            "right_confidence": float(0.80 + f * 0.005),
            "left_roi": [0, 0, 64, 64],
            "right_roi": [0, 0, 64, 64],
            "left_padding": [0, 0, 0, 0],
            "right_padding": [0, 0, 0, 0],
        })
        parquet_rows.append({
            "video_id": "000002",
            "split": "val",
            "frame_idx": f,
            "frame_width": 1080,
            "frame_height": 1080,
            "left_jpeg": crop_jpeg,
            "right_jpeg": crop_jpeg,
            "left_valid": 1,
            "right_valid": 1,
            "left_bbox": [10.0, 10.0, 50.0, 50.0],
            "right_bbox": [10.0, 10.0, 50.0, 50.0],
            "left_confidence": float(0.85 + f * 0.003),
            "right_confidence": float(0.85 + f * 0.003),
            "left_roi": [0, 0, 64, 64],
            "right_roi": [0, 0, 64, 64],
            "left_padding": [0, 0, 0, 0],
            "right_padding": [0, 0, 0, 0],
        })

    shard_path = _make_parquet_shard(parquet_rows, tmp_path / "part_01_crops_000.parquet")

    from src.artifacts import sha256

    marker_path = tmp_path / "part_01.complete.json"
    marker_path.write_text(
        json.dumps({
            "identity": "mock_e021_identity",
            "shards": [{"name": shard_path.name, "sha256": sha256(shard_path), "frames": 60}],
        }),
        encoding="utf-8",
    )

    def mock_download_file(**kwargs: Any) -> Path:
        fn = str(kwargs.get("filename", ""))
        if fn.endswith("train.npz"):
            return u30_train_npz
        if fn.endswith("val.npz"):
            return u30_val_npz
        if fn.endswith("part_01.complete.json"):
            return marker_path
        if fn.endswith("part_01_crops_000.parquet"):
            return shard_path
        raise AssertionError(f"Unexpected download filename: {fn}")

    monkeypatch.setattr("src.e026_visual_audit.download_file", mock_download_file)

    config_yaml = tmp_path / "visual_audit.yaml"
    config_yaml.write_text(
        f"""
experiment:
  id: e026_visual_audit_test
  operation: e026_visual_audit
  mode: full
e021_source:
  repo_id: test/e021
  revision: rev1
  prefix: e021_prefix/
  identity: mock_e021_identity
  parts: [1]
uniform30_source:
  repo_id: test/u30
  revision: rev2
  artifact_path: u30_prefix/
manifests:
  train: {train_manifest.relative_to(tmp_path)}
  val: {val_manifest.relative_to(tmp_path)}
expected_videos:
  train: 1
  val: 1
sampling:
  seed: 42
output:
  root: reports/visual_audit
""",
        encoding="utf-8",
    )

    loaded_cfg = load_config(config_yaml)
    result = run_e026_visual_audit(loaded_cfg, project_root=tmp_path)
    assert result["status"] == "completed"
    assert result["total_selected"] == 32
    assert result["validation_checks"]["all_jpegs_valid"] is True
    assert result["validation_checks"]["all_crops_square"] is True
    assert result["validation_checks"]["all_roi_sides_matched"] is True

    out_dir = tmp_path / "reports/visual_audit"
    assert (out_dir / "summary.json").is_file()
    assert (out_dir / "contact_sheet.jpg").is_file()
    assert (out_dir / "contact_sheet.jpg").stat().st_size > 0
    assert (out_dir / "visual_sample.csv").is_file()

    # Verify visual_sample.csv has header and 32 rows with empty tag and note
    with (out_dir / "visual_sample.csv").open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        assert len(rows) == 32
        for r in rows:
            assert r["tag"] == ""
            assert r["note"] == ""
            assert r["selection_reason"] in dict(REASON_TARGETS)

    # Verify selected_crops directory has 32 jpeg files
    crop_files = list((out_dir / "selected_crops").glob("*.jpg"))
    assert len(crop_files) == 32

    # Verify write_visual_sample_csv independently
    custom_csv = tmp_path / "custom_sample.csv"
    write_visual_sample_csv(rows[:5], custom_csv)
    assert custom_csv.is_file()
