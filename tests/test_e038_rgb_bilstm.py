"""Focused tests for E038 per-frame RGB BiLSTM architecture, loader, and training."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.e038_rgb_frame_fusion import (
    build_model,
    run_e038_fixed_fusion,
    run_e038_rgb_bilstm_train,
)


def _manifest(path: Path, split: str, ids: list[str]) -> None:
    header = "video_id,video_path,gloss,class_id,signer_id,fps,resolution,num_frames,length_seconds,part_id,split\n"
    rows = "".join(
        f"{video_id},{video_id}.mp4,g{i},{i},001,25,1080,30,1.2,1,{split}\n"
        for i, video_id in enumerate(ids)
    )
    path.write_text(header + rows, encoding="utf-8")


def _fixture(tmp_path: Path, *, seed: int = 42) -> dict[str, object]:
    ids = {"train": ["000001", "000002"], "val": ["000003", "000004"]}
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    for split, video_ids in ids.items():
        _manifest(tmp_path / f"{split}.csv", split, video_ids)
        features = np.ones((2, 30, 2, 576), dtype=np.float16)
        valid_mask = np.ones((2, 30), dtype=np.float32)
        np.savez(
            data_dir / f"{split}.npz",
            video_id=np.array(video_ids, dtype="U6"),
            split=np.array([split, split], dtype="U5"),
            selected_frame_indices=np.tile(np.arange(30, dtype=np.int32), (2, 1)),
            valid_frame_mask=valid_mask,
            hand_valid_mask=np.ones((2, 30, 2), dtype=np.uint8),
            X_rgb=features,
        )
    return {
        "experiment": {
            "id": f"test_e038_s{seed}",
            "operation": "e038_rgb_bilstm_train",
            "mode": "full",
            "seed": seed,
        },
        "manifests": {"train": str(tmp_path / "train.csv"), "val": str(tmp_path / "val.csv")},
        "rgb_data": {"root": str(data_dir)},
        "model": {
            "name": "rgb_bilstm",
            "input_dim": 1152,
            "projection_dim": 256,
            "hidden_size": 128,
            "dropout": 0.0,
        },
        "train": {
            "epochs": 1,
            "min_epochs": 0,
            "patience": 0,
            "batch_size": 2,
            "optimizer": "adamw",
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "grad_clip_norm": 1.0,
        },
        "output": {"root": str(tmp_path / f"run_s{seed}")},
    }


def test_bilstm_forward_and_parameter_count() -> None:
    model = build_model(
        {"name": "rgb_bilstm", "input_dim": 1152, "projection_dim": 256, "hidden_size": 128},
        nodes=1,
        num_classes=400,
    )
    assert sum(p.numel() for p in model.parameters()) == 793_744
    x = torch.randn(4, 30, 2, 576)
    mask = torch.ones(4, 30)
    out = model(x, mask)
    assert out.shape == (4, 400)


def test_tiny_train_and_fusion(tmp_path: Path) -> None:
    config = _fixture(tmp_path, seed=42)
    summary = run_e038_rgb_bilstm_train(config, tmp_path, device="cpu")
    assert summary["status"] == "ok"
    assert summary["seed"] == 42
    assert summary["parameters"] == 793_744
    assert (tmp_path / "run_s42/best.pt").is_file()
    assert (tmp_path / "run_s42/scores.npz").is_file()

    # Create dummy pose scores
    pose_scores_path = tmp_path / "pose_scores.npz"
    np.savez(
        pose_scores_path,
        video_id=np.array(["000003", "000004"], dtype="U6"),
        true_class_id=np.array([0, 1], dtype=np.int64),
        probabilities=np.full((2, 400), 1.0 / 400.0, dtype=np.float32),
    )
    fusion_cfg = {
        "experiment": {"id": "fusion_test", "seed": 42},
        "candidate_scores": str(tmp_path / "run_s42/scores.npz"),
        "pose_scores": str(pose_scores_path),
        "fusion": {"beta": 0.5},
        "output": {"root": str(tmp_path / "fusion_run")},
    }
    fusion_res = run_e038_fixed_fusion(fusion_cfg, tmp_path)
    assert fusion_res["status"] == "ok"
    assert "fused_metrics" in fusion_res
