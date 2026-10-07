from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.artifacts import sha256 as _sha256
from src.config import load_config
from src.data import FeatureSplit, select_view, validate_split
from src.frame_sampling import uniform_indices
from src.pose59_uniform import _missingness, derive_uniform_pose59


def _write_manifest(path: Path, split: str, video_id: str, signer_id: str, frames: int) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "video_id",
                "video_path",
                "gloss",
                "class_id",
                "signer_id",
                "fps",
                "resolution",
                "num_frames",
                "length_seconds",
                "part_id",
                "split",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "video_id": video_id,
                "video_path": f"{video_id}.mp4",
                "gloss": "test",
                "class_id": {"train": 0, "val": 1, "test": 399}[split],
                "signer_id": signer_id,
                "fps": 25.0,
                "resolution": "1080x1080",
                "num_frames": frames,
                "length_seconds": frames / 25.0,
                "part_id": 1,
                "split": split,
            }
        )


def _source_shard(
    path: Path,
    frames: int = 34,
    video_ids: tuple[str, ...] = ("000001", "000002", "000003"),
    splits: tuple[str, ...] = ("train", "val", "test"),
) -> None:
    video_ids_array = np.array(video_ids, dtype="U6")
    offsets = np.arange(len(video_ids) + 1, dtype=np.int64) * frames
    total = frames * len(video_ids)
    x = np.zeros((total, 59, 3), dtype=np.float32)
    x[..., 0] = np.arange(total, dtype=np.float32)[:, None] / total
    x[..., 1] = 0.5
    x[..., 2] = 0.9
    np.savez_compressed(
        path,
        X=x,
        keypoint_valid_mask=np.ones((total, 59), dtype=np.uint8),
        object_confidence=np.ones((total, 3), dtype=np.float32),
        frame_idx=np.tile(np.arange(frames, dtype=np.int32), 3),
        video_id=video_ids_array,
        split=np.array(splits, dtype="U5"),
        frame_width=np.full(len(video_ids), 1080, dtype=np.int32),
        frame_height=np.full(len(video_ids), 1080, dtype=np.int32),
        video_frame_offsets=offsets,
    )


def _config(source_identity: str = "source-identity") -> dict[str, Any]:
    return {
        "experiment": {
            "id": "test_pose59_uniform",
            "operation": "derive_pose59",
            "mode": "full",
            "seed": 42,
        },
        "source": {
            "repo_id": "test/features",
            "revision": "source-revision",
            "prefix": "e005_yolo_pose59_extract",
            "identity": source_identity,
            "parts": [1],
            "manifests": {
                "train": "manifests/train.csv",
                "val": "manifests/val.csv",
                "test": "manifests/test.csv",
            },
            "excluded_video_ids": [],
            "expected_videos": {"train": 1, "val": 1, "test": 1},
            "expected_frames": 102,
        },
        "sampling": {"method": "uniform", "sequence_lengths": [16, 24, 30]},
        "output": {"root": "artifacts/derived"},
        "integrations": {
            "hf_artifacts": {
                "enabled": False,
                "repo_id": "test/features",
                "private": True,
            }
        },
    }


def _setup_source(tmp_path: Path, source_identity: str = "source-identity") -> tuple[Path, Path]:
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    _write_manifest(manifests / "train.csv", "train", "000001", "001", 34)
    _write_manifest(manifests / "val.csv", "val", "000002", "002", 34)
    _write_manifest(manifests / "test.csv", "test", "000003", "003", 34)

    remote = tmp_path / "remote"
    remote.mkdir()
    shard = remote / "part_01.npz"
    _source_shard(shard)
    summary = remote / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "identity": source_identity,
                "videos": 3,
                "frames": 102,
                "excluded_defects": [],
                "provenance": {
                    "manifests": {
                        split: {
                            "path": f"manifests/{split}.csv",
                            "sha256": _sha256(manifests / f"{split}.csv"),
                        }
                        for split in ("train", "val", "test")
                    },
                    "part_artifacts": {
                        "part_01": {
                            "videos": 3,
                            "frames": 102,
                            "npz_sha256": _sha256(shard),
                        }
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    return summary, shard


def test_uniform_indices_for_requested_lengths() -> None:
    for length in (16, 24, 30):
        indices = uniform_indices(34, length)
        assert indices.dtype == np.int32
        assert indices.shape == (length,)
        assert indices[0] == 0
        assert indices[-1] == 33
        assert np.all(np.diff(indices) >= 0)


def test_uniform_derivation_clean_join_shapes_dtypes_and_masks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    summary_path, shard_path = _setup_source(tmp_path)

    def fake_download(**kwargs: Any) -> Path:
        filename = str(kwargs["filename"])
        if filename.endswith("summary.json"):
            return summary_path
        if filename.endswith("part_01.npz"):
            return shard_path
        raise AssertionError(filename)

    monkeypatch.setattr("src.pose59_uniform.download_file", fake_download)
    result = derive_uniform_pose59(_config(), tmp_path)

    assert result["status"] == "complete"
    identities = set()
    for length in (16, 24, 30):
        artifact = result["artifacts"][str(length)]
        identities.add(artifact["identity"])
        artifact_root = tmp_path / "artifacts/derived" / artifact["artifact"]
        for split, expected_video, expected_label, signer in (
            ("train", "000001", 0, "001"),
            ("val", "000002", 1, "002"),
            ("test", "000003", 399, "003"),
        ):
            with np.load(artifact_root / "data" / f"{split}.npz", allow_pickle=False) as loaded:
                arrays = {name: loaded[name] for name in loaded.files}
            validate_split(FeatureSplit(**arrays), num_classes=400, nodes=59)
            assert arrays["X"].shape == (1, length, 59, 3)
            assert arrays["video_id"].tolist() == [expected_video]
            assert arrays["signer_id"].tolist() == [signer]
            assert arrays["y"].tolist() == [expected_label]
            assert arrays["valid_frame_mask"].tolist() == [[1] * length]
            expected_indices = uniform_indices(34, length)
            np.testing.assert_array_equal(arrays["frame_indices"][0], expected_indices)
            source_offset = {"train": 0, "val": 34, "test": 68}[split]
            expected_x = (source_offset + expected_indices).astype(np.float32) / 102
            np.testing.assert_allclose(
                arrays["X"][0, :, :, 0], np.repeat(expected_x[:, None], 59, axis=1)
            )
            np.testing.assert_array_equal(
                arrays["X"][0, :, :, 1], np.full((length, 59), 0.5, dtype=np.float32)
            )
            np.testing.assert_array_equal(
                arrays["X"][0, :, :, 2], np.full((length, 59), 0.9, dtype=np.float32)
            )
            assert np.all(arrays["keypoint_valid_mask"] == 1)

        receipt = json.loads((artifact_root / "receipt.json").read_text(encoding="utf-8"))
        success = json.loads((artifact_root / "_SUCCESS.json").read_text(encoding="utf-8"))
        assert receipt["identity_payload"]["sequence_length"] == length
        assert receipt["identity_payload"]["manifests"].keys() == {"train", "val", "test"}
        assert success["identity"] == artifact["identity"]
    assert len(identities) == 3

    first_artifact = result["artifacts"]["16"]
    first_root = tmp_path / "artifacts/derived" / first_artifact["artifact"]
    first_hash = _sha256(first_root / "data/train.npz")
    with pytest.raises(FileExistsError, match="already exists"):
        derive_uniform_pose59(_config(), tmp_path)
    assert _sha256(first_root / "data/train.npz") == first_hash


def test_dev_only_uniform60_skips_unconfigured_source_split(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    summary_path, shard_path = _setup_source(tmp_path)
    config = _config()
    config["source"]["manifests"].pop("test")
    config["source"]["expected_videos"].pop("test")
    config["source"]["expected_frames"] = 68
    config["source"]["source_totals"] = {"videos": 3, "frames": 102}
    config["sampling"] = {"method": "uniform", "sequence_lengths": [60]}
    config["output"] = {"root": "artifacts/dev_uniform60"}
    monkeypatch.setattr(
        "src.pose59_uniform.download_file",
        lambda **kwargs: (
            summary_path if str(kwargs["filename"]).endswith("summary.json") else shard_path
        ),
    )

    result = derive_uniform_pose59(config, tmp_path)
    artifact_root = tmp_path / "artifacts/dev_uniform60" / result["artifacts"]["60"]["artifact"]
    assert (artifact_root / "data/train.npz").is_file()
    assert (artifact_root / "data/val.npz").is_file()
    assert not (artifact_root / "data/test.npz").exists()


def test_sampled_hand_missingness_audit() -> None:
    arrays = {
        "valid_frame_mask": np.ones((2, 5), dtype=np.uint8),
        "keypoint_valid_mask": np.ones((2, 5, 59), dtype=np.uint8),
    }
    # Clip 0: left missing at 1,2; right missing at 2,4 -> either runs 1..2 and 4.
    arrays["keypoint_valid_mask"][0, 1:3, 17:38] = 0
    arrays["keypoint_valid_mask"][0, [2, 4], 38:59] = 0
    audit = _missingness(arrays["valid_frame_mask"], arrays["keypoint_valid_mask"])
    assert audit["missing_left"] == {"count": 2, "rate": 0.2}
    assert audit["missing_right"] == {"count": 2, "rate": 0.2}
    assert audit["missing_both"] == {"count": 1, "rate": 0.1}
    assert audit["clips_with_missing_hand"] == {"count": 1, "rate": 0.5}
    assert audit["max_consecutive_missing_frames"] == 2


def test_uniform_derivation_counts_post_exclusion_source_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    summary_path, shard_path = _setup_source(tmp_path)
    train = tmp_path / "manifests/train.csv"
    with train.open("a", newline="", encoding="utf-8") as handle:
        handle.write("019442,019442.mp4,test,2,004,25.0,1080x1080,34,1.36,1,train\n")
    config = _config()
    config["source"]["excluded_video_ids"] = ["019442"]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["excluded_defects"] = [{"video_id": "019442", "split": "train", "num_frames": 34}]
    summary["provenance"]["manifests"]["train"]["sha256"] = _sha256(train)
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    monkeypatch.setattr(
        "src.pose59_uniform.download_file",
        lambda **kwargs: (
            summary_path if str(kwargs["filename"]).endswith("summary.json") else shard_path
        ),
    )
    assert derive_uniform_pose59(config, tmp_path)["status"] == "complete"


def test_uniform_derivation_rejects_source_checksum_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    summary_path, shard_path = _setup_source(tmp_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["provenance"]["part_artifacts"]["part_01"]["npz_sha256"] = "0" * 64
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    monkeypatch.setattr(
        "src.pose59_uniform.download_file",
        lambda **kwargs: (
            summary_path if str(kwargs["filename"]).endswith("summary.json") else shard_path
        ),
    )
    with pytest.raises(ValueError, match="Source checksum mismatch"):
        derive_uniform_pose59(_config(), tmp_path)


def test_pose59_derivation_and_tcn_config_validation(tmp_path: Path) -> None:
    config = load_config("configs/experiments/e005_yolo_pose59_uniform.yaml")
    assert config["source"]["revision"] == "f94c83a148f01630d7091e394a461acb91bad9c9"
    assert config["sampling"]["sequence_lengths"] == [16, 24, 30]
    assert config["integrations"]["hf_artifacts"]["enabled"] is True
    dev_config = load_config("configs/experiments/e029_pose59_uniform60_dev.yaml")
    assert dev_config["sampling"]["sequence_lengths"] == [60]
    assert set(dev_config["source"]["manifests"]) == {"train", "val"}
    assert dev_config["integrations"]["hf_artifacts"]["enabled"] is False
    for model, experiment in (("tcn", "e030_tcn_pose53_uniform60"), ("stgcn", "e031_stgcn_pose53_uniform60")):
        for seed in (42, 43, 44):
            matrix = load_config(f"configs/experiments/{experiment}_s{seed}.yaml")
            assert matrix["experiment"]["id"] == experiment
            assert matrix["experiment"]["seed"] == seed
            assert matrix["data"]["split_path"].endswith(
                "yolo-e005-pose59-uniform60-350f147fe65e2755/data"
            )
            assert matrix["data"]["num_classes"] == 400
            assert matrix["feature"]["view"] == "pose53"
            assert matrix["model"]["name"] == model
            expected_integrations = (
                {
                    "wandb": {"enabled": True, "project": "vsl400-recognition"},
                    "hf_checkpoints": {
                        "enabled": True,
                        "repo_id": "hungchill24/vsl400-kp-selection-models",
                        "private": True,
                        "upload_every_epochs": 10,
                    },
                }
                if model == "stgcn"
                else {"wandb": {"enabled": False}, "hf_checkpoints": {"enabled": False}}
            )
            assert matrix["integrations"] == expected_integrations

    base = """
experiment:
  id: classifier
  mode: full
  seed: 42
data:
  repo_id: repo
  revision: revision
  artifact_path: artifact
feature:
  view: pose59
model:
  name: {model}
train:
  epochs: 1
  patience: 0
  batch_size: 1
output:
  root: artifacts/runs
"""
    tcn_path = tmp_path / "tcn.yaml"
    tcn_path.write_text(base.format(model="tcn"), encoding="utf-8")
    assert load_config(tcn_path)["feature"]["view"] == "pose59"
    pose59 = FeatureSplit(
        X=np.zeros((1, 16, 59, 3), dtype=np.float32),
        y=np.zeros(1, dtype=np.int64),
        video_id=np.array(["000001"], dtype="U6"),
        signer_id=np.array(["001"], dtype="U3"),
        frame_indices=np.zeros((1, 16), dtype=np.int32),
        valid_frame_mask=np.ones((1, 16), dtype=np.uint8),
        keypoint_valid_mask=np.ones((1, 16, 59), dtype=np.uint8),
    )
    assert select_view(pose59, "pose59") is pose59
    validate_split(pose59, num_classes=400, nodes=59)
    malformed = FeatureSplit(
        **{
            **pose59.__dict__,
            "video_id": np.array(["too-long"], dtype="U8"),
        }
    )
    with pytest.raises(ValueError, match="video_id must be U6"):
        validate_split(malformed, num_classes=400, nodes=59)

    stgcn_path = tmp_path / "stgcn.yaml"
    stgcn_path.write_text(base.format(model="stgcn"), encoding="utf-8")
    assert load_config(stgcn_path)["feature"]["view"] == "pose59"
