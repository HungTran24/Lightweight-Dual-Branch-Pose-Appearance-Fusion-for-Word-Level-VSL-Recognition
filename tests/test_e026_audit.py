"""Focused unit and synthetic integration tests for E026 Phases A–B audit."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.config import load_config
from src.e026_audit import (
    DEV_SPLITS,
    audit_crop_metadata,
    audit_uniform30_alignment,
    format_crop_audit_markdown,
    run_e026_audit,
)
from src.rtmw.manifest import RTMWVideoRecord
from src.yolo.hand_rgb import PARQUET_SCHEMA


def _make_dummy_video(video_id: str, split: str = "train", num_frames: int = 30) -> RTMWVideoRecord:
    return RTMWVideoRecord(
        video_id=video_id,
        video_path=f"path/{video_id}.mp4",
        gloss="test_gloss",
        class_id=0,
        signer_id="001",
        fps=25.0,
        resolution="1080",
        num_frames=num_frames,
        length_seconds=float(num_frames / 25.0),
        part_id=1,
        split=split,
    )


def _make_parquet_rows(
    entries: list[tuple[str, str, int, int, int]],  # (split, video_id, frame_idx, left_valid, right_valid)
    *,
    side: int = 120,
    pad: list[int] | None = None,
    bbox: list[float] | None = None,
    conf: float = 0.9,
) -> list[dict[str, Any]]:
    padding = pad if pad is not None else [0, 0, 0, 0]
    box = bbox if bbox is not None else [10.0, 10.0, 100.0, 100.0]
    rows = []
    for split, vid, fidx, lv, rv in entries:
        rows.append(
            {
                "video_id": vid,
                "split": split,
                "frame_idx": fidx,
                "frame_width": 1080,
                "frame_height": 1080,
                "left_jpeg": b"fake_jpeg_binary_payload" if lv else None,
                "right_jpeg": b"fake_jpeg_binary_payload" if rv else None,
                "left_valid": lv,
                "right_valid": rv,
                "left_bbox": box if lv else [0.0, 0.0, 0.0, 0.0],
                "right_bbox": box if rv else [0.0, 0.0, 0.0, 0.0],
                "left_confidence": conf if lv else 0.0,
                "right_confidence": conf if rv else 0.0,
                "left_roi": [0, 0, side, side] if lv else [0, 0, 0, 0],
                "right_roi": [0, 0, side, side] if rv else [0, 0, 0, 0],
                "left_padding": padding if lv else [0, 0, 0, 0],
                "right_padding": padding if rv else [0, 0, 0, 0],
            }
        )
    return rows


def _write_parquet_shard(rows: list[dict[str, Any]], path: Path) -> Path:
    table = pa.Table.from_pylist(rows, schema=PARQUET_SCHEMA)
    pq.write_table(table, path)
    return path


def test_metadata_audit_excludes_test_and_computes_statistics(tmp_path: Path) -> None:
    """Metadata audit processes train and val, ignores test rows, and computes exact Phase A metrics."""
    rows = []
    # 10 train frames: sides from 50 to 140 (some <64, some <96, some >96)
    for i in range(10):
        side = 50 + i * 10
        # side=50 has pad=5 -> unpadded=40x40, area=2500, inside=1600, pad_area=900, ratio=0.36
        pad = [5, 5, 5, 5] if i == 0 else [0, 0, 0, 0]
        box = [10.0, 10.0, float(10 + side / 1.3), float(10 + side / 1.3)]
        rows.extend(
            _make_parquet_rows(
                [("train", "000001", i, 1, 1 if i < 8 else 0)],
                side=side,
                pad=pad,
                bbox=box,
                conf=0.85 + i * 0.01,
            )
        )

    # 5 val frames: side=250 (>224)
    for i in range(5):
        rows.extend(_make_parquet_rows([("val", "000002", i, 1, 1)], side=250, conf=0.95))

    # 10 test frames: should be completely ignored from development audit
    for i in range(10):
        rows.extend(_make_parquet_rows([("test", "000003", i, 1, 1)], side=40, conf=0.5))

    shard_path = _write_parquet_shard(rows, tmp_path / "part_01_crops_000.parquet")
    stats = audit_crop_metadata([shard_path], valid_splits=DEV_SPLITS)

    assert "train" in stats and "val" in stats and "overall" in stats
    assert "test" not in stats

    # Train left hand
    tr_left = stats["train"]["left"]
    assert tr_left["total_frames"] == 10
    assert tr_left["valid_count"] == 10
    assert tr_left["missing_count"] == 0
    assert tr_left["missing_rate"] == 0.0
    assert tr_left["median_crop_side"] == 95.0
    assert tr_left["fraction_lt_64"] == 0.2  # 50, 60
    assert tr_left["fraction_lt_96"] == 0.5  # 50, 60, 70, 80, 90
    assert tr_left["fraction_gt_224"] == 0.0
    assert tr_left["crops_with_padding_rate"] == 0.1
    assert tr_left["padding_area_ratio_mean"] > 0.0
    assert tr_left["confidence_median"] == pytest.approx(0.895, rel=1e-3)

    # Train right hand has 2 missing (i=8, 9)
    tr_right = stats["train"]["right"]
    assert tr_right["total_frames"] == 10
    assert tr_right["valid_count"] == 8
    assert tr_right["missing_count"] == 2
    assert tr_right["missing_rate"] == 0.2

    # Val left hand has side=250
    val_left = stats["val"]["left"]
    assert val_left["total_frames"] == 5
    assert val_left["valid_count"] == 5
    assert val_left["median_crop_side"] == 250.0
    assert val_left["fraction_gt_224"] == 1.0

    # Overall combines train (10 frames) + val (5 frames) = 15 frames
    overall_left = stats["overall"]["left"]
    assert overall_left["total_frames"] == 15
    assert overall_left["valid_count"] == 15


def test_metadata_audit_filters_by_allowed_video_ids(tmp_path: Path) -> None:
    """audit_crop_metadata filters out excluded defects like 019442 and unlisted video IDs."""
    rows = []
    # Allowed train video: 000001
    rows.extend(_make_parquet_rows([("train", "000001", i, 1, 1) for i in range(5)], side=100))
    # Excluded defect video: 019442 (in train split in raw parquet)
    rows.extend(_make_parquet_rows([("train", "019442", i, 1, 1) for i in range(5)], side=50))
    # Unlisted video: 999999
    rows.extend(_make_parquet_rows([("train", "999999", i, 1, 1) for i in range(5)], side=200))

    shard_path = _write_parquet_shard(rows, tmp_path / "defect_filter.parquet")
    allowed = {"train": {"000001"}, "val": set()}

    stats = audit_crop_metadata([shard_path], valid_splits=DEV_SPLITS, allowed_video_ids=allowed)
    assert stats["train"]["left"]["total_frames"] == 5
    assert stats["train"]["left"]["median_crop_side"] == 100.0


def test_metadata_audit_rejects_zero_area_bbox(tmp_path: Path) -> None:
    """audit_crop_metadata raises ValueError if a valid crop has non-positive bbox dimension."""
    # Degenerate bbox: x0=50, x1=50 (width=0)
    rows = _make_parquet_rows(
        [("train", "000001", 0, 1, 1)],
        side=100,
        bbox=[50.0, 10.0, 50.0, 100.0],
    )
    shard_path = _write_parquet_shard(rows, tmp_path / "degenerate_bbox.parquet")
    with pytest.raises(ValueError, match="Valid hand crop has non-positive bbox dimension"):
        audit_crop_metadata([shard_path], valid_splits=DEV_SPLITS)


def test_metadata_audit_handles_empty_split_shards(tmp_path: Path) -> None:
    """audit_crop_metadata gracefully handles shards with 0 rows for a given split."""
    # Shard 1: only train rows
    s1 = _write_parquet_shard(
        _make_parquet_rows([("train", "000001", i, 1, 1) for i in range(5)]),
        tmp_path / "s1.parquet",
    )
    # Shard 2: only val rows
    s2 = _write_parquet_shard(
        _make_parquet_rows([("val", "000002", i, 1, 1) for i in range(5)]),
        tmp_path / "s2.parquet",
    )

    stats = audit_crop_metadata([s1, s2], valid_splits=DEV_SPLITS)
    assert stats["train"]["left"]["valid_count"] == 5
    assert stats["val"]["left"]["valid_count"] == 5
    assert stats["overall"]["left"]["valid_count"] == 10


def test_exact_keyed_join_succeeds_with_shuffled_rows(tmp_path: Path) -> None:
    """Phase B exact join succeeds when Parquet rows and manifests are in shuffled/non-consecutive order."""
    # Manifest has 000002 then 000001
    manifest_rows = {
        "train": [_make_dummy_video("000002", "train"), _make_dummy_video("000001", "train")],
        "val": [_make_dummy_video("000003", "val")],
    }

    # U30 array has 000001 then 000002 (different order than manifest, but same set)
    u30_splits = {
        "train": {
            "video_id": np.array(["000001", "000002"], dtype="U6"),
            "frame_indices": np.tile(np.arange(30, dtype=np.int32), (2, 1)),
            "valid_frame_mask": np.ones((2, 30), dtype=np.uint8),
        },
        "val": {
            "video_id": np.array(["000003"], dtype="U6"),
            "frame_indices": np.arange(30, dtype=np.int32).reshape(1, 30),
            "valid_frame_mask": np.ones((1, 30), dtype=np.uint8),
        },
    }

    # Generate all 90 required frame rows and shuffle them
    all_entries = []
    for vid, split in (("000001", "train"), ("000002", "train"), ("000003", "val")):
        for fidx in range(30):
            all_entries.append((split, vid, fidx, 1, 1))

    rng = np.random.default_rng(42)
    shuffled_indices = rng.permutation(len(all_entries))
    shuffled_entries = [all_entries[i] for i in shuffled_indices]

    rows = _make_parquet_rows(shuffled_entries)
    shard_path = _write_parquet_shard(rows, tmp_path / "shuffled_shard.parquet")

    result = audit_uniform30_alignment(manifest_rows, u30_splits, [shard_path], valid_splits=DEV_SPLITS)

    assert result["status"] == "passed"
    assert result["total_videos"] == 3
    assert result["total_uniform30_slots"] == 90
    assert result["matched_frames"] == 90
    assert result["missing_keys"] == 0
    assert result["duplicate_keys"] == 0


def test_alignment_rejects_duplicates_missing_and_split_mismatch(tmp_path: Path) -> None:
    """Phase B rejects duplicate frame keys, missing keys, and set mismatches."""
    manifest_rows = {
        "train": [_make_dummy_video("000001", "train")],
        "val": [_make_dummy_video("000002", "val")],
    }

    u30_splits = {
        "train": {
            "video_id": np.array(["000001"], dtype="U6"),
            "frame_indices": np.arange(30, dtype=np.int32).reshape(1, 30),
            "valid_frame_mask": np.ones((1, 30), dtype=np.uint8),
        },
        "val": {
            "video_id": np.array(["000002"], dtype="U6"),
            "frame_indices": np.arange(30, dtype=np.int32).reshape(1, 30),
            "valid_frame_mask": np.ones((1, 30), dtype=np.uint8),
        },
    }

    # 1. Missing key (only 29 frames for video 000001)
    missing_entries = [("train", "000001", i, 1, 1) for i in range(29)] + [
        ("val", "000002", i, 1, 1) for i in range(30)
    ]
    p_missing = _write_parquet_shard(_make_parquet_rows(missing_entries), tmp_path / "missing.parquet")
    with pytest.raises(ValueError, match="Missing Uniform-30 frame key in E021 Parquet"):
        audit_uniform30_alignment(manifest_rows, u30_splits, [p_missing])

    # 2. Duplicate key in Parquet
    dup_entries = (
        [("train", "000001", i, 1, 1) for i in range(30)]
        + [("train", "000001", 0, 1, 1)]  # duplicate of frame 0
        + [("val", "000002", i, 1, 1) for i in range(30)]
    )
    p_dup = _write_parquet_shard(_make_parquet_rows(dup_entries), tmp_path / "dup.parquet")
    with pytest.raises(ValueError, match="Duplicate Parquet frame key"):
        audit_uniform30_alignment(manifest_rows, u30_splits, [p_dup])

    # 3. Set mismatch between manifest and U30
    bad_manifest = {
        "train": [_make_dummy_video("000003", "train")],  # wrong video for train
        "val": [_make_dummy_video("000002", "val")],
    }
    valid_entries = [("train", "000001", i, 1, 1) for i in range(30)] + [
        ("val", "000002", i, 1, 1) for i in range(30)
    ]
    p_valid = _write_parquet_shard(_make_parquet_rows(valid_entries), tmp_path / "valid.parquet")
    with pytest.raises(ValueError, match="Manifest video IDs and Uniform-30 video IDs set mismatch"):
        audit_uniform30_alignment(bad_manifest, u30_splits, [p_valid])


def test_markdown_report_formatting() -> None:
    """format_crop_audit_markdown produces valid markdown with headers, table rows, and status."""
    meta_stats = {
        s: {
            h: {
                "total_frames": 100,
                "valid_count": 99,
                "missing_count": 1,
                "missing_rate": 0.01,
                "p5_crop_side": 50.0,
                "median_crop_side": 100.0,
                "p95_crop_side": 180.0,
                "p99_crop_side": 210.0,
                "fraction_lt_64": 0.15,
                "fraction_lt_96": 0.45,
                "fraction_gt_224": 0.02,
                "padding_area_ratio_mean": 0.05,
                "crops_with_padding_rate": 0.08,
                "roi_expansion_ratio_median": 1.30,
                "confidence_median": 0.92,
            }
            for h in ("left", "right")
        }
        for s in ("train", "val", "overall")
    }

    alignment = {
        "status": "passed",
        "total_videos": 22243,
        "total_uniform30_slots": 667290,
        "matched_frames": 667290,
        "missing_keys": 0,
        "duplicate_keys": 0,
        "split_mismatches": 0,
        "deviation_note": "verified",
    }

    provenance = {
        "e021_repo": "test_e021_repo",
        "e021_revision": "rev_e021",
        "e021_path": "e021_path",
        "e021_identity": "ident_e021",
        "u30_repo": "test_u30_repo",
        "u30_revision": "rev_u30",
        "u30_artifact": "u30_art",
        "train_videos": 19714,
        "val_videos": 2529,
    }

    md = format_crop_audit_markdown(meta_stats, alignment, provenance)

    assert "# E026 Hand-RGB Crop and Alignment Audit" in md
    assert "Phase A: Metadata Crop Distribution" in md
    assert "Phase B: Exact Uniform-30 Key Alignment" in md
    assert "| train | left |" in md
    assert "| overall | right |" in md
    assert "667290" in md
    assert "19714" in md


def test_e026_config_validation_and_runner_dispatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Config validation rejects invalid IDs or test split, and runner executes full mock pipeline."""
    # 1. Config validation rejection when experiment.id is missing/empty
    no_id_yaml = tmp_path / "no_id.yaml"
    no_id_yaml.write_text(
        """
experiment:
  operation: e026_audit
  mode: full
e021_source:
  repo_id: repo1
  revision: rev1
  prefix: prefix1
  identity: id1
  parts: [1]
uniform30_source:
  repo_id: repo2
  revision: rev2
  artifact_path: art2
manifests:
  train: t.csv
  val: v.csv
expected_videos:
  train: 10
  val: 5
output:
  root: out
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="e026_audit requires a non-empty experiment.id"):
        load_config(no_id_yaml)

    # 2. Config validation rejection when test split is present in manifests
    bad_yaml = tmp_path / "bad_config.yaml"
    bad_yaml.write_text(
        """
experiment:
  id: e026_bad
  operation: e026_audit
  mode: full
e021_source:
  repo_id: repo1
  revision: rev1
  prefix: prefix1
  identity: id1
  parts: [1]
uniform30_source:
  repo_id: repo2
  revision: rev2
  artifact_path: art2
manifests:
  train: t.csv
  val: v.csv
  test: test.csv
expected_videos:
  train: 10
  val: 5
output:
  root: out
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="e026_audit manifests must define train and val without test split"):
        load_config(bad_yaml)

    # 3. Synthetic end-to-end run_e026_audit
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

    parquet_entries = [("train", "000001", i, 1, 1) for i in range(30)] + [
        ("val", "000002", i, 1, 1) for i in range(30)
    ]
    shard_file = _write_parquet_shard(_make_parquet_rows(parquet_entries), tmp_path / "part_01_crops_000.parquet")

    from src.artifacts import sha256

    marker_file = tmp_path / "part_01.complete.json"
    marker_file.write_text(
        json.dumps(
            {
                "identity": "mock_e021_identity",
                "shards": [{"name": shard_file.name, "sha256": sha256(shard_file), "frames": 60}],
            }
        ),
        encoding="utf-8",
    )

    def mock_download_file(**kwargs: Any) -> Path:
        fn = str(kwargs.get("filename", ""))
        if fn.endswith("train.npz"):
            return u30_train_npz
        if fn.endswith("val.npz"):
            return u30_val_npz
        if fn.endswith("part_01.complete.json"):
            return marker_file
        if fn.endswith("part_01_crops_000.parquet"):
            return shard_file
        raise AssertionError(f"Unexpected download filename: {fn}")

    monkeypatch.setattr("src.e026_audit.download_file", mock_download_file)

    good_config = {
        "experiment": {"id": "e026_audit_test", "operation": "e026_audit", "mode": "smoke"},
        "e021_source": {
            "repo_id": "test/e021",
            "revision": "rev1",
            "prefix": "e021_prefix/",
            "identity": "mock_e021_identity",
            "parts": [1],
        },
        "uniform30_source": {"repo_id": "test/u30", "revision": "rev2", "artifact_path": "u30_prefix/"},
        "manifests": {"train": str(train_manifest.relative_to(tmp_path)), "val": str(val_manifest.relative_to(tmp_path))},
        "expected_videos": {"train": 1, "val": 1},
        "output": {"root": "reports/audit_test"},
    }

    result = run_e026_audit(good_config, project_root=tmp_path)

    assert result["status"] == "completed"
    assert result["alignment_audit"]["status"] == "passed"
    assert (tmp_path / "reports/audit_test/crop_audit.json").is_file()
    assert (tmp_path / "reports/audit_test/summary.json").is_file()
    assert (tmp_path / "reports/audit_test/crop_audit.md").is_file()


def test_audit_never_reads_binary_jpeg_columns(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that neither Phase A nor Phase B projects binary JPEG columns from Parquet."""
    rows = _make_parquet_rows([("train", "000001", i, 1, 1) for i in range(30)])
    shard_path = _write_parquet_shard(rows, tmp_path / "check_columns.parquet")

    requested_columns_list: list[list[str] | None] = []
    original_read_table = pq.read_table

    def tracking_read_table(source: Any, *args: Any, **kwargs: Any) -> pa.Table:
        columns = kwargs.get("columns")
        requested_columns_list.append(columns)
        return original_read_table(source, *args, **kwargs)

    monkeypatch.setattr(pq, "read_table", tracking_read_table)

    # Run Phase A
    audit_crop_metadata([shard_path], valid_splits=DEV_SPLITS)

    # Run Phase B
    manifest_rows = {"train": [_make_dummy_video("000001", "train")], "val": []}
    u30_splits = {
        "train": {
            "video_id": np.array(["000001"], dtype="U6"),
            "frame_indices": np.arange(30, dtype=np.int32).reshape(1, 30),
            "valid_frame_mask": np.ones((1, 30), dtype=np.uint8),
        },
        "val": {
            "video_id": np.array([], dtype="U6"),
            "frame_indices": np.zeros((0, 30), dtype=np.int32),
            "valid_frame_mask": np.zeros((0, 30), dtype=np.uint8),
        },
    }
    audit_uniform30_alignment(manifest_rows, u30_splits, [shard_path], valid_splits=DEV_SPLITS)

    assert len(requested_columns_list) >= 2
    for cols in requested_columns_list:
        assert cols is not None, "pq.read_table must project explicit columns, not read full table"
        assert "left_jpeg" not in cols, "Must not load binary left_jpeg"
        assert "right_jpeg" not in cols, "Must not load binary right_jpeg"
