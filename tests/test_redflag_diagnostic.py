from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from src.config import load_config
from src.data import FeatureSplit, validate_split
from src.redflag_diagnostic import build_clean_signer_partition, run_rgb_signer_report


def _split(rows: list[tuple[str, str, int]]) -> FeatureSplit:
    count = len(rows)
    return FeatureSplit(
        X=np.zeros((count, 2, 53, 3), dtype=np.float32),
        y=np.array([label for _, _, label in rows], dtype=np.int64),
        video_id=np.array([video_id for video_id, _, _ in rows], dtype="U6"),
        signer_id=np.array([signer for _, signer, _ in rows], dtype="U3"),
        frame_indices=np.zeros((count, 2), dtype=np.int32),
        valid_frame_mask=np.ones((count, 2), dtype=np.uint8),
        keypoint_valid_mask=np.ones((count, 2, 53), dtype=np.uint8),
    )


def _manifest(path: Path, rows: list[tuple[str, str, int]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("video_id", "signer_id", "class_id"))
        writer.writeheader()
        writer.writerows(
            {"video_id": video_id, "signer_id": signer, "class_id": label}
            for video_id, signer, label in rows
        )


def _scores(path: Path, rows: list[tuple[str, int]]) -> None:
    probabilities = np.full((len(rows), 400), 1e-6, dtype=np.float32)
    for index, (_, label) in enumerate(rows):
        probabilities[index, label] = 1.0 - 399e-6
    np.savez(
        path,
        video_id=np.array([video_id for video_id, _ in rows], dtype="U6"),
        true_class_id=np.array([label for _, label in rows], dtype=np.int64),
        probabilities=probabilities,
    )


def test_rtmw_raw_score_channel_can_exceed_one() -> None:
    split = _split([("000001", "001", 1)])
    split.X[:, :, :, 2] = 1.2
    validate_split(split, num_classes=400, nodes=53)


def test_clean_signer_partition_relabels_and_rejects_overlap(tmp_path: Path) -> None:
    _manifest(tmp_path / "train.csv", [("000001", "001", 1), ("019442", "024", 2)])
    _manifest(tmp_path / "val.csv", [("000002", "006", 3), ("000003", "028", 4)])
    config = {
        "manifests": {"train": "train.csv", "val": "val.csv"},
        "excluded_video_ids": ["019442", "024029"],
        "prohibited_signers": ["003", "012", "019"],
        "train": {"source": "train", "signers": ["001", "024"], "expected_samples": 1},
        "val": {"sources": {"val": ["006", "028"]}, "expected_samples": 2},
    }
    train, val = build_clean_signer_partition(
        {"train": _split([("000001", "001", 9)]), "val": _split([("000002", "006", 8), ("000003", "028", 7)])},
        config,
        tmp_path,
    )
    assert train.y.tolist() == [1]
    assert val.y.tolist() == [3, 4]

    _manifest(tmp_path / "val.csv", [("000001", "001", 1)])
    overlap_config = {**config, "val": {"sources": {"val": ["001"]}, "expected_samples": 1}}
    with pytest.raises(ValueError, match="overlap"):
        build_clean_signer_partition(
            {"train": _split([("000001", "001", 9)]), "val": _split([("000001", "001", 9)])},
            overlap_config,
            tmp_path,
        )


def test_rgb_signer_report_writes_seed_metrics_and_sample_sd(tmp_path: Path) -> None:
    _manifest(
        tmp_path / "val.csv",
        [("000001", "006", 1), ("000002", "020", 2), ("000003", "028", 3)],
    )
    archives: dict[int, str] = {}
    for seed in (42, 43, 44):
        path = tmp_path / f"scores_{seed}.npz"
        _scores(path, [("000001", 1), ("000002", 2), ("000003", 3)])
        archives[seed] = str(path)
    config = {
        "report": {
            "manifest": "val.csv",
            "signers": ["006", "020", "028"],
            "sources": [{"name": "rgb", "archives": archives}],
        },
        "output": {"root": "report"},
    }
    summary = run_rgb_signer_report(config, tmp_path)
    assert len(summary["aggregate"]) == 3
    assert all(row["top1_std"] == 0.0 for row in summary["aggregate"])
    seed_rows = list(csv.DictReader((tmp_path / "report" / "per_seed_signer_metrics.csv").open()))
    assert len(seed_rows) == 9
    assert (tmp_path / "report" / "per_seed_signer_metrics.csv").is_file()
    assert (tmp_path / "report" / "per_signer_seed_summary.csv").is_file()
    assert (tmp_path / "report" / "summary.json").is_file()


def test_e036_configs_validate() -> None:
    root = Path(__file__).parents[1] / "configs/experiments"
    for seed in (42, 43, 44):
        config = load_config(root / f"e036_redflag_rtmw_tcn_s{seed}.yaml")
        assert config["data"]["signer_partition"]["val"]["expected_samples"] == 5181
    assert load_config(root / "e036_redflag_rgb_signer_report.yaml")["report"]["signers"] == [
        "006",
        "020",
        "028",
    ]
