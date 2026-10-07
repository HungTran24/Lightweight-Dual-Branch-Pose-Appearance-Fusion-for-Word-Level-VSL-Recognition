"""Focused unit and synthetic integration tests for E026 one-pass U24+U30 RGB extraction."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
import torchvision

from src.artifacts import sha256
from src.config import load_config
from src.e026_rgb_features import (
    atomic_npz,
    build_mobilenet_extractor,
    compute_extraction_identity,
    preprocess_crop_bgr,
    run_e026_rgb_features,
    run_feature_forward,
)
from src.yolo.hand_rgb import PARQUET_SCHEMA


def _encode_bgr_jpeg(bgr_array: np.ndarray, quality: int = 95) -> bytes:
    ok, buf = cv2.imencode(".jpg", bgr_array, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return buf.tobytes()


def _make_shard(rows: list[dict[str, Any]], path: Path) -> Path:
    table = pa.Table.from_pylist(rows, schema=PARQUET_SCHEMA)
    pq.write_table(table, path)
    return path


def _make_mock_encoder(dim: int = 576) -> torch.nn.Module:
    class MockEncoder(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.eval()

        def features(self, x: torch.Tensor) -> torch.Tensor:
            return x

        def avgpool(self, x: torch.Tensor) -> torch.Tensor:
            b = x.shape[0]
            val = x.mean(dim=(1, 2, 3), keepdim=False).view(b, 1, 1, 1)
            return val.expand(b, dim, 1, 1)

    return MockEncoder()


def _setup_synthetic_sources(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path, Path]:
    """Helper creating minimal synthetic manifests and U24/U30 NPZ files."""
    tr_man = tmp_path / "train.csv"
    tr_man.write_text(
        "video_id,video_path,gloss,class_id,signer_id,fps,resolution,num_frames,length_seconds,part_id,split\n"
        "000001,000001.mp4,g1,0,001,25.0,1080,30,1.2,1,train\n",
        encoding="utf-8",
    )
    va_man = tmp_path / "val.csv"
    va_man.write_text(
        "video_id,video_path,gloss,class_id,signer_id,fps,resolution,num_frames,length_seconds,part_id,split\n"
        "000002,000002.mp4,g2,1,002,25.0,1080,30,1.2,1,val\n",
        encoding="utf-8",
    )
    # Video 1: U24 has frames 0 and 2; U30 has frames 1 and 2 (frame 2 is shared)
    idx24 = np.full((1, 24), -1, dtype=np.int32)
    idx24[0, 0], idx24[0, 1] = 0, 2
    mask24 = np.zeros((1, 24), dtype=np.uint8)
    mask24[0, 0], mask24[0, 1] = 1, 1

    idx30 = np.full((1, 30), -1, dtype=np.int32)
    idx30[0, 0], idx30[0, 1] = 1, 2
    mask30 = np.zeros((1, 30), dtype=np.uint8)
    mask30[0, 0], mask30[0, 1] = 1, 1

    u24_tr = tmp_path / "u24_tr.npz"
    np.savez(u24_tr, video_id=np.array(["000001"], dtype="U6"), frame_indices=idx24, valid_frame_mask=mask24)
    u24_va = tmp_path / "u24_va.npz"
    np.savez(u24_va, video_id=np.array(["000002"], dtype="U6"), frame_indices=np.zeros((1, 24), dtype=np.int32), valid_frame_mask=np.ones((1, 24), dtype=np.uint8))

    u30_tr = tmp_path / "u30_tr.npz"
    np.savez(u30_tr, video_id=np.array(["000001"], dtype="U6"), frame_indices=idx30, valid_frame_mask=mask30)
    u30_va = tmp_path / "u30_va.npz"
    np.savez(u30_va, video_id=np.array(["000002"], dtype="U6"), frame_indices=np.zeros((1, 30), dtype=np.int32), valid_frame_mask=np.ones((1, 30), dtype=np.uint8))

    return tr_man, va_man, u24_tr, u24_va, u30_tr, u30_va


def test_atomic_npz_writes_and_replaces(tmp_path: Path) -> None:
    dest = tmp_path / "test.npz"
    arr = np.arange(10, dtype=np.int32)
    atomic_npz(dest, x=arr)
    assert dest.is_file()
    assert not (tmp_path / ".test.npz.tmp").exists()
    with np.load(dest) as loaded:
        assert np.array_equal(loaded["x"], arr)


def test_preprocessing_shape_colors_and_576d_feature_path() -> None:
    red_rgb = np.zeros((100, 100, 3), dtype=np.uint8)
    red_rgb[:, :, 0] = 255
    red_bgr = cv2.cvtColor(red_rgb, cv2.COLOR_RGB2BGR)

    chw = preprocess_crop_bgr(red_bgr, target_size=128)
    assert chw.shape == (3, 128, 128)
    assert chw[0, 64, 64] > 1.0
    assert chw[2, 64, 64] < -1.0

    non_square = np.zeros((100, 80, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="Crop must be square"):
        preprocess_crop_bgr(non_square, target_size=128)

    device = torch.device("cpu")
    model = build_mobilenet_extractor(device, weights_name=None)
    features = run_feature_forward(model, torch.randn(2, 3, 128, 128, device=device))
    assert features.shape == (2, 576)


def test_synthetic_vertical_union_flow_shared_encode_and_zeros(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Full vertical pipeline: shared frame encoded once, missing hands give exact zero, outputs valid."""
    tr_m, va_m, u24_tr, u24_va, u30_tr, u30_va = _setup_synthetic_sources(tmp_path)

    jpeg_0 = _encode_bgr_jpeg(np.full((64, 64, 3), 10, dtype=np.uint8))
    jpeg_1 = _encode_bgr_jpeg(np.full((64, 64, 3), 20, dtype=np.uint8))
    jpeg_2 = _encode_bgr_jpeg(np.full((64, 64, 3), 30, dtype=np.uint8))
    jpeg_v = _encode_bgr_jpeg(np.full((64, 64, 3), 40, dtype=np.uint8))

    # Video 1: left hand valid=0 across all frames (tests missing-hand zero vector)
    # Right hand valid=1 in train frames 0, 1, 2 (frame 2 is shared between U24 and U30)
    rows = [
        {"video_id": "000001", "split": "train", "frame_idx": 0, "frame_width": 1080, "frame_height": 1080, "left_jpeg": None, "right_jpeg": jpeg_0, "left_valid": 0, "right_valid": 1, "left_bbox": [0,0,0,0], "right_bbox": [0,0,50,50], "left_confidence": 0.0, "right_confidence": 0.9, "left_roi": [0,0,0,0], "right_roi": [0,0,64,64], "left_padding": [0,0,0,0], "right_padding": [0,0,0,0]},
        {"video_id": "000001", "split": "train", "frame_idx": 1, "frame_width": 1080, "frame_height": 1080, "left_jpeg": None, "right_jpeg": jpeg_1, "left_valid": 0, "right_valid": 1, "left_bbox": [0,0,0,0], "right_bbox": [0,0,50,50], "left_confidence": 0.0, "right_confidence": 0.9, "left_roi": [0,0,0,0], "right_roi": [0,0,64,64], "left_padding": [0,0,0,0], "right_padding": [0,0,0,0]},
        {"video_id": "000001", "split": "train", "frame_idx": 2, "frame_width": 1080, "frame_height": 1080, "left_jpeg": None, "right_jpeg": jpeg_2, "left_valid": 0, "right_valid": 1, "left_bbox": [0,0,0,0], "right_bbox": [0,0,50,50], "left_confidence": 0.0, "right_confidence": 0.9, "left_roi": [0,0,0,0], "right_roi": [0,0,64,64], "left_padding": [0,0,0,0], "right_padding": [0,0,0,0]},
        {"video_id": "000002", "split": "val", "frame_idx": 0, "frame_width": 1080, "frame_height": 1080, "left_jpeg": jpeg_v, "right_jpeg": jpeg_v, "left_valid": 1, "right_valid": 1, "left_bbox": [0,0,50,50], "right_bbox": [0,0,50,50], "left_confidence": 0.9, "right_confidence": 0.9, "left_roi": [0,0,64,64], "right_roi": [0,0,64,64], "left_padding": [0,0,0,0], "right_padding": [0,0,0,0]},
    ]
    shard_path = _make_shard(rows, tmp_path / "part_01_crops_000.parquet")
    marker_path = tmp_path / "part_01.complete.json"
    marker_path.write_text(json.dumps({"identity": "mock_e021", "shards": [{"name": shard_path.name, "sha256": sha256(shard_path), "frames": 4}]}), encoding="utf-8")

    monkeypatch.setattr(
        "src.e026_rgb_features.download_file",
        lambda **kw: marker_path if "complete.json" in str(kw.get("filename"))
        else shard_path if "parquet" in str(kw.get("filename"))
        else u24_tr if "u24" in str(kw.get("filename")) and "train" in str(kw.get("filename"))
        else u24_va if "u24" in str(kw.get("filename"))
        else u30_tr if "u30" in str(kw.get("filename")) and "train" in str(kw.get("filename"))
        else u30_va,
    )

    encode_counts = []
    def track_forward(_model: Any, x: torch.Tensor) -> torch.Tensor:
        del _model
        encode_counts.append(x.shape[0])
        b = x.shape[0]
        val = x.mean(dim=(1, 2, 3), keepdim=False).view(b, 1)
        return val.expand(b, 576)

    monkeypatch.setattr("src.e026_rgb_features.run_feature_forward", track_forward)

    cfg_yaml = tmp_path / "exp.yaml"
    cfg_yaml.write_text(
        f"""
experiment:
  id: test_vertical
  operation: e026_rgb_features
  mode: smoke
e021_source:
  repo_id: r1
  revision: rev1
  prefix: e021/
  identity: mock_e021
  parts: [1]
u24_source:
  repo_id: r2
  revision: rev2
  artifact_path: u24/
u30_source:
  repo_id: r3
  revision: rev3
  artifact_path: u30/
manifests:
  train: {tr_m.relative_to(tmp_path)}
  val: {va_m.relative_to(tmp_path)}
expected_videos:
  train: 1
  val: 1
preprocessing:
  backbone: mobilenet_v3_small
  weights: IMAGENET1K_V1
  feature_dim: 576
  target_size: 128
  interpolation: bilinear
  antialias: true
  mean: [0.485, 0.456, 0.406]
  std: [0.229, 0.224, 0.225]
  output_dtype: float16
data:
  batch_size: 10
smoke:
  train_videos: 1
  val_videos: 1
output:
  root: out_test
""",
        encoding="utf-8",
    )

    loaded_cfg = load_config(cfg_yaml)
    summary = run_e026_rgb_features(
        loaded_cfg,
        project_root=tmp_path,
        encoder_factory=lambda _: _make_mock_encoder(),
        device=torch.device("cpu"),
    )
    assert summary["status"] == "completed"

    # Shared train frame 2 was encoded once: 3 crops in train (frames 0, 1, 2 right hand) + 2 crops in val (frame 0 left+right) = 5 total
    assert sum(encode_counts) == 5

    out_root = tmp_path / "out_test"
    with np.load(out_root / "u24/data/train.npz") as l24:
        # Left hand has 0 valid -> exact zero vector
        assert l24["hand_valid_count"][0, 0] == 0
        assert np.array_equal(l24["X_rgb"][0, 0], np.zeros(576, dtype=np.float16))
        # Right hand has 2 valid frames (0 and 2)
        assert l24["hand_valid_count"][0, 1] == 2
        assert np.isfinite(l24["X_rgb"][0, 1]).all()

    receipt = json.loads((out_root / "receipt.json").read_text(encoding="utf-8"))
    assert receipt["final_files"]["u24"]["train"]["video_count"] == 1
    assert "sha256" in receipt["final_files"]["u24"]["train"]
    assert receipt["performance"]["valid_crops_encoded"] == 5


@pytest.mark.parametrize(
    ("corrupt_type", "err_match"),
    [
        ("duplicate_frame", "Duplicate Parquet frame key"),
        ("missing_frame", "Missing required union frame"),
        ("split_mismatch", "Split mismatch for 000001"),
    ],
)
def test_data_contract_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, corrupt_type: str, err_match: str
) -> None:
    tr_m, va_m, u24_tr, u24_va, u30_tr, u30_va = _setup_synthetic_sources(tmp_path)
    jpeg = _encode_bgr_jpeg(np.full((64, 64, 3), 50, dtype=np.uint8))

    if corrupt_type == "duplicate_frame":
        rows = [
            {"video_id": "000001", "split": "train", "frame_idx": 0, "frame_width": 1080, "frame_height": 1080, "left_jpeg": jpeg, "right_jpeg": None, "left_valid": 1, "right_valid": 0, "left_bbox": [0,0,50,50], "right_bbox": [0,0,0,0], "left_confidence": 0.9, "right_confidence": 0.0, "left_roi": [0,0,64,64], "right_roi": [0,0,0,0], "left_padding": [0,0,0,0], "right_padding": [0,0,0,0]},
            {"video_id": "000001", "split": "train", "frame_idx": 0, "frame_width": 1080, "frame_height": 1080, "left_jpeg": jpeg, "right_jpeg": None, "left_valid": 1, "right_valid": 0, "left_bbox": [0,0,50,50], "right_bbox": [0,0,0,0], "left_confidence": 0.9, "right_confidence": 0.0, "left_roi": [0,0,64,64], "right_roi": [0,0,0,0], "left_padding": [0,0,0,0], "right_padding": [0,0,0,0]},
        ]
    elif corrupt_type == "missing_frame":
        rows = [
            {"video_id": "000001", "split": "train", "frame_idx": 0, "frame_width": 1080, "frame_height": 1080, "left_jpeg": jpeg, "right_jpeg": None, "left_valid": 1, "right_valid": 0, "left_bbox": [0,0,50,50], "right_bbox": [0,0,0,0], "left_confidence": 0.9, "right_confidence": 0.0, "left_roi": [0,0,64,64], "right_roi": [0,0,0,0], "left_padding": [0,0,0,0], "right_padding": [0,0,0,0]},
        ]
    else:  # split_mismatch
        rows = [
            {"video_id": "000001", "split": "val", "frame_idx": 0, "frame_width": 1080, "frame_height": 1080, "left_jpeg": jpeg, "right_jpeg": None, "left_valid": 1, "right_valid": 0, "left_bbox": [0,0,50,50], "right_bbox": [0,0,0,0], "left_confidence": 0.9, "right_confidence": 0.0, "left_roi": [0,0,64,64], "right_roi": [0,0,0,0], "left_padding": [0,0,0,0], "right_padding": [0,0,0,0]},
        ]

    shard = _make_shard(rows, tmp_path / f"{corrupt_type}.parquet")
    marker = tmp_path / f"{corrupt_type}.json"
    marker.write_text(json.dumps({"identity": "mock_e021", "shards": [{"name": shard.name, "sha256": sha256(shard), "frames": len(rows)}]}), encoding="utf-8")

    monkeypatch.setattr(
        "src.e026_rgb_features.download_file",
        lambda **kw: marker if "complete.json" in str(kw.get("filename"))
        else shard if "parquet" in str(kw.get("filename"))
        else u24_tr if "u24" in str(kw.get("filename")) and "train" in str(kw.get("filename"))
        else u24_va if "u24" in str(kw.get("filename"))
        else u30_tr if "u30" in str(kw.get("filename")) and "train" in str(kw.get("filename"))
        else u30_va,
    )

    cfg = {
        "experiment": {"id": "test_err", "operation": "e026_rgb_features", "mode": "smoke"},
        "e021_source": {"repo_id": "r1", "revision": "rev1", "prefix": "e021/", "identity": "mock_e021", "parts": [1]},
        "u24_source": {"repo_id": "r2", "revision": "rev2", "artifact_path": "u24/"},
        "u30_source": {"repo_id": "r3", "revision": "rev3", "artifact_path": "u30/"},
        "manifests": {"train": str(tr_m.relative_to(tmp_path)), "val": str(va_m.relative_to(tmp_path))},
        "output": {"root": f"out_{corrupt_type}"},
    }

    with pytest.raises(ValueError, match=err_match):
        run_e026_rgb_features(cfg, project_root=tmp_path, encoder_factory=lambda _: _make_mock_encoder(), device=torch.device("cpu"))


def test_marker_cache_skip_and_identity_checksum_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Marker skip runs without downloads; identity or checksum mismatch fails."""
    tr_m, va_m, u24_tr, u24_va, u30_tr, u30_va = _setup_synthetic_sources(tmp_path)
    monkeypatch.setattr(
        "src.e026_rgb_features.download_file",
        lambda **kw: u24_tr if "u24" in str(kw.get("filename")) and "train" in str(kw.get("filename"))
        else u24_va if "u24" in str(kw.get("filename"))
        else u30_tr if "u30" in str(kw.get("filename")) and "train" in str(kw.get("filename"))
        else u30_va,
    )

    cfg = {
        "experiment": {"id": "test_marker", "operation": "e026_rgb_features", "mode": "smoke"},
        "e021_source": {"repo_id": "r1", "revision": "rev1", "prefix": "e021/", "identity": "mock_e021", "parts": [1]},
        "u24_source": {"repo_id": "r2", "revision": "rev2", "artifact_path": "u24/"},
        "u30_source": {"repo_id": "r3", "revision": "rev3", "artifact_path": "u30/"},
        "manifests": {"train": str(tr_m.relative_to(tmp_path)), "val": str(va_m.relative_to(tmp_path))},
        "output": {"root": "out_marker"},
    }

    identity = compute_extraction_identity(cfg, {"train": sha256(tr_m), "val": sha256(va_m)}, torch.__version__, torchvision.__version__)
    parts_dir = tmp_path / "out_marker/parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    p24 = parts_dir / "part_01_u24.npz"
    p30 = parts_dir / "part_01_u30.npz"
    np.savez(p24, video_id=np.array(["000001", "000002"], dtype="U6"), split=np.array(["train", "val"], dtype="U5"), selected_frame_indices=np.zeros((2, 24), dtype=np.int32), hand_valid_count=np.zeros((2, 2), dtype=np.uint8), X_rgb=np.zeros((2, 2, 576), dtype=np.float16))
    np.savez(p30, video_id=np.array(["000001", "000002"], dtype="U6"), split=np.array(["train", "val"], dtype="U5"), selected_frame_indices=np.zeros((2, 30), dtype=np.int32), hand_valid_count=np.zeros((2, 2), dtype=np.uint8), X_rgb=np.zeros((2, 2, 576), dtype=np.float16))

    marker = parts_dir / "part_01.complete.json"
    marker.write_text(json.dumps({"extraction_identity": identity, "part": 1, "videos": 2, "u24_sha256": sha256(p24), "u30_sha256": sha256(p30)}), encoding="utf-8")

    # 1. Skip works
    res = run_e026_rgb_features(cfg, project_root=tmp_path, encoder_factory=lambda _: _make_mock_encoder(), device=torch.device("cpu"))
    assert res["status"] == "completed"

    # 2. Checksum mismatch raises
    marker.write_text(json.dumps({"extraction_identity": identity, "part": 1, "videos": 2, "u24_sha256": "bad", "u30_sha256": sha256(p30)}), encoding="utf-8")
    with pytest.raises(ValueError, match="identity/checksum mismatch"):
        run_e026_rgb_features(cfg, project_root=tmp_path, encoder_factory=lambda _: _make_mock_encoder(), device=torch.device("cpu"))
