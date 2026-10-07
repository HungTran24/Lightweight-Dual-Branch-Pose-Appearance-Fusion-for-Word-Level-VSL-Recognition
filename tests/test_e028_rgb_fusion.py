"""Focused tests for E028 temporal RGB head, loader, training, and fixed fusion."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.artifacts import sha256
from src.config import load_config
from src.e028_rgb_fusion import (
    load_temporal_rgb_split,
    run_e028_fixed_fusion,
    run_e028_rgb_train,
)
from src.models import build_model


def _manifest(path: Path, split: str, ids: list[str]) -> None:
    header = "video_id,video_path,gloss,class_id,signer_id,fps,resolution,num_frames,length_seconds,part_id,split\n"
    rows = "".join(
        f"{video_id},{video_id}.mp4,g{i},{i},001,25,1080,30,1.2,1,{split}\n"
        for i, video_id in enumerate(ids)
    )
    path.write_text(header + rows, encoding="utf-8")


def _temporal_fixture(
    tmp_path: Path, *, seed: int = 42, sampling_slots: int = 30
) -> dict[str, object]:
    ids = {"train": ["000001", "000002"], "val": ["000003", "000004"]}
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    files = {}
    for split, video_ids in ids.items():
        _manifest(tmp_path / f"{split}.csv", split, video_ids)
        features = np.ones((2, 2, 3, 576), dtype=np.float16)
        counts = np.ones((2, 2, 3), dtype=np.uint8)
        counts[0, 1, 2] = 0
        features[0, 1, 2] = 0
        path = data_dir / f"{split}.npz"
        np.savez(
            path,
            video_id=np.array(video_ids, dtype="U6"),
            split=np.array([split, split], dtype="U5"),
            selected_frame_indices=np.tile(
                np.arange(sampling_slots, dtype=np.int32), (2, 1)
            ),
            hand_valid_count=counts,
            X_rgb=features,
        )
        files[split] = {"sha256": sha256(path)}
    receipt = tmp_path / "receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "extraction_identity": "e028-id",
                "sampling_slots": sampling_slots,
                "files": files,
            }
        ),
        encoding="utf-8",
    )
    return {
        "experiment": {"id": "tiny", "operation": "e028_rgb_train", "mode": "full", "seed": seed},
        "manifests": {"train": "train.csv", "val": "val.csv"},
        "expected_videos": {"train": 2, "val": 2},
        "excluded_video_ids": ["019442"],
        "rgb_data": {
            "root": "data",
            "receipt": "receipt.json",
            "receipt_sha256": sha256(receipt),
            "extraction_identity": "e028-id",
            "sampling_slots": sampling_slots,
        },
        "model": {"name": "rgb_temporal3"},
        "train": {"epochs": 1, "min_epochs": 0, "patience": 0, "batch_size": 2, "optimizer": "adamw", "learning_rate": 0.001, "weight_decay": 0.0001, "grad_clip_norm": 1.0},
        "output": {"root": "run"},
    }


def test_u24_loader_rejects_more_than_eight_frames_per_bin(tmp_path: Path) -> None:
    fixture_root = tmp_path / "u24"
    fixture_root.mkdir()
    config = _temporal_fixture(fixture_root, sampling_slots=24)
    split_path = fixture_root / "data/train.npz"
    with np.load(split_path, allow_pickle=False) as loaded:
        arrays = {name: np.array(loaded[name]) for name in loaded.files}
    arrays["hand_valid_count"][0, 0, 0] = 9
    np.savez(
        split_path,
        video_id=arrays["video_id"],
        split=arrays["split"],
        selected_frame_indices=arrays["selected_frame_indices"],
        hand_valid_count=arrays["hand_valid_count"],
        X_rgb=arrays["X_rgb"],
    )
    receipt_path = fixture_root / "receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["files"]["train"]["sha256"] = sha256(split_path)
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    config["rgb_data"]["receipt_sha256"] = sha256(receipt_path)  # type: ignore[index]
    with pytest.raises(ValueError, match="feature validity contract failed"):
        load_temporal_rgb_split(config, fixture_root, "train")


def test_official_seed_configs_and_temporal_loader_head_mask_and_tiny_train(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = Path(__file__).resolve().parents[1]
    for seed in (42, 43, 44):
        train_config = load_config(root / f"configs/experiments/e028_rgb_temporal3_s{seed}.yaml")
        fusion_config = load_config(root / f"configs/experiments/e028_fixed_log_fusion_s{seed}.yaml")
        assert train_config["experiment"]["seed"] == fusion_config["experiment"]["seed"] == seed
        assert fusion_config["candidate_scores"] == f"artifacts/runs/e028_rgb_temporal3_s{seed}/scores.npz"
        assert fusion_config["baseline_rgb_scores"] == f"artifacts/runs/e026_rgb_linear_u30_s{seed}/scores.npz"
        assert fusion_config["pose_scores"] == f"artifacts/runs/e026_fusion_u30_s{seed}/pose_scores.npz"

        e033 = load_config(
            root / f"configs/experiments/e033_rgb_temporal3_u24_s{seed}.yaml"
        )
        assert e033["rgb_data"]["sampling_slots"] == 24
        assert e033["integrations"]["wandb"] == {
            "enabled": True,
            "project": "vsl400-recognition",
        }
        e035 = load_config(
            root / f"configs/experiments/e035_rgb_shared_hand128_temporal3_s{seed}.yaml"
        )
        assert e035["model"] == {"name": "rgb_shared_hand128_temporal3"}
        assert e035["rgb_data"]["extraction_identity"] == (
            "9c5d05a21ade20db15ee466bbbcfb9a87bf4ac382fa387b7e8dc2759318f3c62"
        )
        assert e035["integrations"]["wandb"] == {
            "enabled": True,
            "project": "vsl400-recognition",
        }

    bad_path = tmp_path / "bad_fusion.yaml"
    bad_path.write_text(
        (root / "configs/experiments/e028_fixed_log_fusion_s42.yaml")
        .read_text(encoding="utf-8")
        .replace("beta: 0.5", "beta: 0.6"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="locked official"):
        load_config(bad_path)

    config = _temporal_fixture(tmp_path, seed=43)
    config["integrations"] = {"wandb": {"enabled": True, "project": "test"}}
    seeded: list[int] = []
    logged_epochs: list[int] = []
    finished: list[dict[str, object]] = []
    tracker = object()
    monkeypatch.setattr("src.e028_rgb_fusion.set_seed", seeded.append)
    monkeypatch.setattr("src.e028_rgb_fusion.start_run", lambda *_args: tracker)
    monkeypatch.setattr(
        "src.e028_rgb_fusion.log_epoch",
        lambda run, _metrics, epoch: logged_epochs.append(epoch) if run is tracker else None,
    )
    monkeypatch.setattr(
        "src.e028_rgb_fusion.finish_run",
        lambda run, summary: finished.append(summary) if run is tracker else None,
    )
    split = load_temporal_rgb_split(config, tmp_path, "train")
    assert split.features.shape == (2, 2, 3, 576)
    assert split.features.dtype == np.float32
    assert split.token_present.shape == (2, 6)
    assert split.token_present[0, 5] == 0

    model = build_model({"name": "rgb_temporal3"}, nodes=1, num_classes=400)
    assert sum(parameter.numel() for parameter in model.parameters()) == 190_928
    shared_model = build_model(
        {"name": "rgb_shared_hand128_temporal3"}, nodes=1, num_classes=400
    )
    assert sum(parameter.numel() for parameter in shared_model.parameters()) == 381_712
    shared_model.eval()
    shared_features = torch.randn(2, 2, 3, 576)
    shared_present = torch.ones(2, 6)
    shared_present[:, 5] = 0
    changed_shared = shared_features.clone()
    changed_shared[:, 1, 2] = 10_000
    assert torch.allclose(
        shared_model(shared_features, shared_present),
        shared_model(changed_shared, shared_present),
    )
    model.eval()
    features = torch.randn(2, 2, 3, 576)
    present = torch.ones(2, 6)
    present[:, 5] = 0
    changed = features.clone()
    changed[:, 1, 2] = 10_000
    assert torch.allclose(model(features, present), model(changed, present))

    summary = run_e028_rgb_train(config, tmp_path, device="cpu")
    assert seeded == [43]
    assert logged_epochs == [1]
    assert finished == [summary]
    assert summary["seed"] == 43
    assert summary["parameters"] == 190_928
    for name in ("best.pt", "last.pt", "scores.npz", "predictions.csv", "summary.json"):
        assert (tmp_path / "run" / name).is_file()


def _scores(path: Path, ids: np.ndarray, probabilities: np.ndarray) -> None:
    np.savez(
        path,
        video_id=ids,
        true_class_id=np.array([0, 1], dtype=np.int64),
        probabilities=probabilities.astype(np.float32),
    )


def test_fixed_beta_fusion_gate_and_test_path_rejection(tmp_path: Path) -> None:
    ids = np.array(["000001", "000002"], dtype="U6")
    pose = np.full((2, 400), 0.3 / 398, dtype=np.float32)
    pose[0, 0] = pose[1, 1] = 0.4
    pose[:, 2] = 0.3
    candidate = np.full((2, 400), 1e-6, dtype=np.float32)
    candidate[0, 0] = candidate[1, 1] = 1 - 399e-6
    baseline = np.full((2, 400), 1e-6, dtype=np.float32)
    baseline[0, 2] = baseline[1, 2] = 1 - 399e-6
    for name, probs in (("pose", pose), ("candidate", candidate)):
        _scores(tmp_path / f"{name}.npz", ids, probs)
    np.savez(
        tmp_path / "baseline.npz",
        video_id=ids[::-1],
        true_class_id=np.array([1, 0], dtype=np.int64),
        probabilities=baseline[::-1],
    )

    config = {
        "experiment": {"id": "tiny_s42", "seed": 42},
        "candidate_scores": "candidate.npz",
        "baseline_rgb_scores": "baseline.npz",
        "pose_scores": "pose.npz",
        "output": {"root": "fusion"},
    }
    summary = run_e028_fixed_fusion(config, tmp_path)
    assert summary["experiment_id"] == "tiny_s42"
    assert summary["seed"] == 42
    assert summary["beta"] == 0.5
    assert summary["gate_passed"]
    assert summary["continuation"] == "eligible_for_manual_seed_43_44_authorization"
    assert summary["deltas"]["top1"] >= 0.005
    assert (tmp_path / "fusion/summary.json").is_file()

    for seed in (43, 44):
        config["experiment"] = {"id": f"tiny_s{seed}", "seed": seed}
        config["output"] = {"root": f"fusion_s{seed}"}
        summary = run_e028_fixed_fusion(config, tmp_path)
        assert summary["experiment_id"] == f"tiny_s{seed}"
        assert summary["seed"] == seed
        assert summary["continuation"] == "diagnostic_only"

    np.savez(
        tmp_path / "baseline.npz",
        video_id=np.array(["000001", "999999"], dtype="U6"),
        true_class_id=np.array([0, 1], dtype=np.int64),
        probabilities=baseline,
    )
    config["output"]["root"] = "fusion_mismatch"
    with pytest.raises(ValueError, match="Candidate and baseline score video IDs differ"):
        run_e028_fixed_fusion(config, tmp_path)
