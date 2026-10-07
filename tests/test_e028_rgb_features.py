"""Focused tests for E028 temporal RGB feature extraction."""

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from src.artifacts import sha256
from src.config import load_config
from src.e028_rgb_features import _flush, _validate_output, run_e028_rgb_features, temporal_bin_map
from src.yolo.hand_rgb import PARQUET_SCHEMA


class _Encoder(torch.nn.Module):
    def features(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def avgpool(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        return x.mean(dim=(1, 2, 3)).reshape(batch, 1, 1, 1).expand(batch, 576, 1, 1)


def test_slot_bin_mapping_accumulation_and_duplicate_rejection() -> None:
    u24_indices = np.arange(100, 124, dtype=np.int32)
    u24_mapping = temporal_bin_map(
        u24_indices, np.ones(24, dtype=np.uint8), sampling_slots=24
    )
    assert [u24_mapping[int(index)] for index in u24_indices] == [0] * 8 + [1] * 8 + [2] * 8

    indices = np.arange(100, 130, dtype=np.int32)
    mask = np.ones(30, dtype=np.uint8)
    mapping = temporal_bin_map(indices, mask)
    assert mapping[100] == 0
    assert mapping[110] == 1
    assert mapping[129] == 2

    duplicate = indices.copy()
    duplicate[20] = duplicate[0]
    with pytest.raises(ValueError, match="Duplicate valid U30 frame index"):
        temporal_bin_map(duplicate, mask)

    sums = np.zeros((1, 2, 3, 576), dtype=np.float32)
    counts = np.zeros((1, 2, 3), dtype=np.uint8)
    batch = [
        (0, 0, 0, torch.ones(3, 4, 4)),
        (0, 0, 0, torch.full((3, 4, 4), 3.0)),
        (0, 1, 2, torch.full((3, 4, 4), 2.0)),
    ]
    _flush(batch, _Encoder(), torch.device("cpu"), sums, counts)
    assert counts.tolist() == [[[2, 0, 0], [0, 0, 1]]]
    assert np.allclose(sums[0, 0, 0], 4.0)
    assert np.allclose(sums[0, 1, 2], 2.0)


def test_output_schema_zero_bins_and_locked_config(tmp_path: Path) -> None:
    path = tmp_path / "val.npz"
    features = np.zeros((1, 2, 3, 576), dtype=np.float16)
    counts = np.zeros((1, 2, 3), dtype=np.uint8)
    counts[0, 0, 1] = 1
    features[0, 0, 1] = 2.0
    np.savez(
        path,
        video_id=np.array(["000001"], dtype="U6"),
        split=np.array(["val"], dtype="U5"),
        selected_frame_indices=np.arange(30, dtype=np.int32)[None],
        hand_valid_count=counts,
        X_rgb=features,
    )
    _validate_output(path, ["000001"])
    assert np.all(features[counts == 0] == 0)

    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs/experiments/e028_rgb_temporal3_features_full.yaml")
    assert set(config["manifests"]) == {"train", "val"}
    assert config["output"]["root"] == "artifacts/rgb/e028_rgb_temporal3_full"

    u24_path = tmp_path / "train.npz"
    np.savez(
        u24_path,
        video_id=np.array(["000001"], dtype="U6"),
        split=np.array(["train"], dtype="U5"),
        selected_frame_indices=np.arange(24, dtype=np.int32)[None],
        hand_valid_count=np.full((1, 2, 3), 8, dtype=np.uint8),
        X_rgb=np.ones((1, 2, 3, 576), dtype=np.float16),
    )
    _validate_output(u24_path, ["000001"], sampling_slots=24)

    e033 = load_config(root / "configs/experiments/e033_rgb_temporal3_u24_features_full.yaml")
    assert e033["sampling_slots"] == 24
    assert e033["u24_source"]["artifact_path"].endswith("uniform24-fbbaf7a173a950b2")
    test = load_config(root / "configs/experiments/test/e033_rgb_temporal3_u24_test.yaml")
    assert test["manifests"] == {"test": "manifests/vsl400_front_400gloss/test.csv"}


@pytest.mark.parametrize("sampling_slots", [24, 30])
def test_synthetic_extractor_keyed_permuted_sampling_and_bins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sampling_slots: int
) -> None:
    header = "video_id,video_path,gloss,class_id,signer_id,fps,resolution,num_frames,length_seconds,part_id,split\n"
    (tmp_path / "train.csv").write_text(
        header
        + "000001,1.mp4,g1,0,001,25,1080,30,1.2,1,train\n"
        + "000002,2.mp4,g2,1,001,25,1080,30,1.2,1,train\n",
        encoding="utf-8",
    )
    (tmp_path / "val.csv").write_text(
        header + "000003,3.mp4,g3,2,001,25,1080,30,1.2,1,val\n", encoding="utf-8"
    )

    uniform_files: dict[str, Path] = {}
    for split, ids, frames in (
        ("train", ["000002", "000001"], [21, 1]),
        ("val", ["000003"], [11]),
    ):
        indices = np.full((len(ids), sampling_slots), -1, dtype=np.int32)
        masks = np.zeros((len(ids), sampling_slots), dtype=np.uint8)
        bin_width = sampling_slots // 3
        for i, frame in enumerate(frames):
            slot = (frame // bin_width) * bin_width
            indices[i, slot] = frame
            masks[i, slot] = 1
        path = tmp_path / f"u{sampling_slots}_{split}.npz"
        np.savez(
            path,
            video_id=np.array(ids, dtype="U6"),
            frame_indices=indices,
            valid_frame_mask=masks,
        )
        uniform_files[split] = path

    ok, jpeg_buf = cv2.imencode(".jpg", np.full((32, 32, 3), 80, dtype=np.uint8))
    assert ok
    rows = []
    for video_id, split, frame in (
        ("000001", "train", 1),
        ("000002", "train", 21),
        ("000003", "val", 11),
    ):
        rows.append({
            "video_id": video_id, "split": split, "frame_idx": frame,
            "frame_width": 1080, "frame_height": 1080,
            "left_jpeg": jpeg_buf.tobytes(), "right_jpeg": None,
            "left_valid": 1, "right_valid": 0,
            "left_bbox": [0, 0, 20, 20], "right_bbox": [0, 0, 0, 0],
            "left_confidence": 0.9, "right_confidence": 0.0,
            "left_roi": [0, 0, 32, 32], "right_roi": [0, 0, 0, 0],
            "left_padding": [0, 0, 0, 0], "right_padding": [0, 0, 0, 0],
        })
    shard = tmp_path / "shard.parquet"
    pq.write_table(pa.Table.from_pylist(rows, schema=PARQUET_SCHEMA), shard)
    marker = tmp_path / "part_01.complete.json"
    marker.write_text(
        json.dumps({
            "identity": "e021",
            "shards": [{"name": shard.name, "sha256": sha256(shard)}],
        }),
        encoding="utf-8",
    )

    def download(**kwargs: Any) -> Path:
        filename = str(kwargs["filename"])
        if filename.endswith("train.npz"):
            return uniform_files["train"]
        if filename.endswith("val.npz"):
            return uniform_files["val"]
        return marker if filename.endswith("complete.json") else shard

    monkeypatch.setattr("src.e028_rgb_features.download_file", download)
    monkeypatch.setattr(
        "src.e028_rgb_features._load_uniform_splits",
        lambda *_args: {
            split: {
                name: np.array(values)
                for name, values in np.load(path, allow_pickle=False).items()
            }
            for split, path in uniform_files.items()
        },
    )
    config = {
        "experiment": {"id": "vertical", "mode": "full"},
        "manifests": {"train": "train.csv", "val": "val.csv"},
        "expected_videos": {"train": 2, "val": 1},
        "excluded_video_ids": ["019442"],
        f"u{sampling_slots}_source": {
            "repo_id": "r", "revision": "rev", "artifact_path": f"u{sampling_slots}"
        },
        "sampling_slots": sampling_slots,
        "e021_source": {
            "repo_id": "r", "revision": "rev", "prefix": "e021",
            "identity": "e021", "parts": [1],
        },
        "data": {"batch_size": 8},
        "output": {"root": "output"},
    }
    result = run_e028_rgb_features(
        config, tmp_path, encoder_factory=lambda _: _Encoder(), device=torch.device("cpu")
    )
    assert result["status"] == "completed"
    with np.load(tmp_path / "output/data/train.npz") as loaded:
        assert loaded["video_id"].tolist() == ["000001", "000002"]
        assert loaded["hand_valid_count"][:, 0].tolist() == [[1, 0, 0], [0, 0, 1]]
        assert np.all(loaded["X_rgb"][:, 1] == 0)
