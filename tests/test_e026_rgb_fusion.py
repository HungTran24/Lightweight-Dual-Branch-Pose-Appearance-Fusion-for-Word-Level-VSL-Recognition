"""Focused tests for E026 RGB linear training and probability fusion."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.artifacts import sha256
from src.config import load_config
from src.data import FeatureSplit
from src.e026_rgb_fusion import (
    aggregate_alpha_curves,
    continuation_gate,
    continuation_status,
    fuse_probability_scores,
    load_rgb_pooled_split,
    run_e026_fusion,
    run_e026_fusion_summary,
    run_e026_rgb_train,
)
from src.evaluate import evaluate, evaluate_with_scores
from src.models import build_model


def _write_manifest(path: Path, split: str, rows: list[tuple[str, int]]) -> None:
    header = "video_id,video_path,gloss,class_id,signer_id,fps,resolution,num_frames,length_seconds,part_id,split\n"
    body = "".join(
        f"{vid},{vid}.mp4,g{label},{label},001,25.0,1080,30,1.2,1,{split}\n"
        for vid, label in rows
    )
    path.write_text(header + body, encoding="utf-8")


def _rgb_fixture(tmp_path: Path) -> dict[str, object]:
    train_rows = [("000001", 1), ("000002", 2), ("000003", 3), ("000004", 4)]
    val_rows = [("000005", 1), ("000006", 2)]
    _write_manifest(tmp_path / "train.csv", "train", train_rows)
    _write_manifest(tmp_path / "val.csv", "val", val_rows)
    data_dir = tmp_path / "rgb"
    data_dir.mkdir()
    rng = np.random.default_rng(42)
    for split, rows in (("train", train_rows), ("val", val_rows)):
        np.savez(
            data_dir / f"{split}.npz",
            video_id=np.array([r[0] for r in rows], dtype="U6"),
            split=np.array([split] * len(rows), dtype="U5"),
            selected_frame_indices=np.zeros((len(rows), 30), dtype=np.int32),
            hand_valid_count=np.ones((len(rows), 2), dtype=np.uint8),
            X_rgb=rng.normal(size=(len(rows), 2, 576)).astype(np.float16),
        )
    receipt = tmp_path / "receipt.json"
    receipt.write_text(
        json.dumps({
            "extraction_identity": "rgb-id",
            "final_files": {
                view: {
                    split: {"sha256": sha256(data_dir / f"{split}.npz")}
                    for split in ("train", "val")
                }
                for view in ("u24", "u30")
            },
        }),
        encoding="utf-8",
    )
    return {
        "experiment": {"id": "tiny", "operation": "e026_rgb_train", "mode": "full", "seed": 42},
        "manifests": {"train": "train.csv", "val": "val.csv"},
        "expected_videos": {"train": 4, "val": 2},
        "excluded_video_ids": ["019442"],
        "rgb_data": {
            "root": "rgb",
            "receipt": "receipt.json",
            "receipt_sha256": sha256(receipt),
            "extraction_identity": "rgb-id",
        },
        "model": {"name": "rgb_linear"},
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
        "output": {"root": "run"},
    }


def test_rgb_loader_keyed_join_and_shape(tmp_path: Path) -> None:
    cfg = _rgb_fixture(tmp_path)
    split = load_rgb_pooled_split(cfg, tmp_path, "train")
    assert split.features.shape == (4, 1152)
    assert split.features.dtype == np.float32
    assert split.labels.tolist() == [1, 2, 3, 4]
    assert split.video_id.tolist() == ["000001", "000002", "000003", "000004"]

    rgb_data = cfg["rgb_data"]
    assert isinstance(rgb_data, dict)
    rgb_data["view"] = "u24"
    assert load_rgb_pooled_split(cfg, tmp_path, "train").features.shape == (4, 1152)
    rgb_data["view"] = "unsupported"
    with pytest.raises(ValueError, match="view must be u24 or u30"):
        load_rgb_pooled_split(cfg, tmp_path, "train")


def test_rgb_linear_parameter_count_and_tiny_fit(tmp_path: Path) -> None:
    cfg = _rgb_fixture(tmp_path)
    model = build_model({"name": "rgb_linear"}, nodes=1, num_classes=400)
    assert sum(p.numel() for p in model.parameters()) == 461_200
    assert model(torch.randn(2, 1152), torch.ones(2, 1)).shape == (2, 400)

    summary = run_e026_rgb_train(cfg, tmp_path, device="cpu")
    assert summary["parameters"] == 461_200
    for name in ("best.pt", "last.pt", "predictions.csv", "scores.npz", "summary.json"):
        assert (tmp_path / "run" / name).is_file()
    with np.load(tmp_path / "run" / "scores.npz") as scores:
        assert scores["probabilities"].shape == (2, 400)
        assert scores["probabilities"].dtype == np.float32
        assert np.allclose(scores["probabilities"].sum(axis=1), 1.0, atol=1e-5)


def test_shared_hand128_model_and_official_contract(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = Path(__file__).resolve().parents[1]
    for view in ("u24", "u30"):
        for seed in (42, 43, 44):
            config = load_config(
                root / f"configs/experiments/e034_rgb_shared_hand128_{view}_s{seed}.yaml"
            )
            assert config["model"] == {"name": "rgb_shared_hand128"}
            assert config["rgb_data"]["view"] == view
            assert config["experiment"]["seed"] == seed
            assert config["integrations"]["wandb"] == {
                "enabled": True,
                "project": "vsl400-recognition",
            }
    frozen = load_config(
        root / "configs/experiments/test/e034_rgb_shared_hand128_frozen_test.yaml"
    )
    assert [branch["name"] for branch in frozen["branches"]] == [
        "shared_hand128_u24",
        "shared_hand128_u30",
    ]
    assert all(branch["model"] == {"name": "rgb_shared_hand128"} for branch in frozen["branches"])

    model = build_model({"name": "rgb_shared_hand128"}, nodes=1, num_classes=400)
    assert sum(parameter.numel() for parameter in model.parameters()) == 176_912
    assert model(torch.randn(2, 2, 576), torch.ones(2, 1)).shape == (2, 400)

    config = _rgb_fixture(tmp_path)
    config["model"] = {"name": "rgb_shared_hand128"}
    config["integrations"] = {"wandb": {"enabled": True, "project": "test"}}
    tracker = object()
    logged: list[int] = []
    finished: list[dict[str, object]] = []
    monkeypatch.setattr("src.e026_rgb_fusion.start_run", lambda *_args: tracker)
    monkeypatch.setattr(
        "src.e026_rgb_fusion.log_epoch",
        lambda run, _metrics, epoch: logged.append(epoch) if run is tracker else None,
    )
    monkeypatch.setattr(
        "src.e026_rgb_fusion.finish_run",
        lambda run, summary: finished.append(summary) if run is tracker else None,
    )
    summary = run_e026_rgb_train(config, tmp_path, device="cpu")
    assert summary["model"] == "rgb_shared_hand128"
    assert summary["parameters"] == 176_912
    assert logged == [1]
    assert finished == [summary]


def test_shared_evaluation_preserves_contract_and_exports_scores() -> None:
    model = build_model({"name": "rgb_linear"}, nodes=1, num_classes=400)
    dataset = TensorDataset(
        torch.randn(3, 1152), torch.tensor([1, 2, 3]), torch.ones(3, 1, dtype=torch.uint8)
    )
    loader = DataLoader(dataset, batch_size=2, shuffle=False)
    old_result = evaluate(model, loader, num_classes=400, device=torch.device("cpu"))
    new_result = evaluate_with_scores(model, loader, num_classes=400, device=torch.device("cpu"))
    assert len(old_result) == 3
    assert old_result == new_result[:3]
    assert new_result[3] == [1, 2, 3]
    assert new_result[4].shape == (3, 400)


def test_fusion_keyed_endpoints_tie_and_mismatch() -> None:
    ids = np.array(["000001", "000002"], dtype="U6")
    labels = np.array([0, 1], dtype=np.int64)
    rgb = np.full((2, 400), 1e-6, dtype=np.float32)
    rgb[0, 0], rgb[1, 1] = 1.0 - 399e-6, 1.0 - 399e-6
    pose = rgb.copy()

    curve, alpha, selected, _, _ = fuse_probability_scores(
        (ids, labels, rgb), (ids[::-1], labels[::-1], pose[::-1])
    )
    assert alpha == 1.0
    assert np.allclose(curve[0]["top1"], 1.0)
    assert np.array_equal(selected, pose)

    with pytest.raises(ValueError, match="video IDs differ"):
        fuse_probability_scores((ids, labels, rgb), (np.array(["000001", "000003"]), labels, pose))
    with pytest.raises(ValueError, match="true_class_id differ"):
        fuse_probability_scores((ids, labels, rgb), (ids, labels[::-1], pose))

    assert continuation_gate(
        {"top1": 0.805, "macro_f1": 0.79}, {"top1": 0.8, "macro_f1": 0.79}
    )
    assert not continuation_gate(
        {"top1": 0.804, "macro_f1": 0.80}, {"top1": 0.8, "macro_f1": 0.79}
    )
    assert not continuation_gate(
        {"top1": 0.81, "macro_f1": 0.78}, {"top1": 0.8, "macro_f1": 0.79}
    )
    assert continuation_status(42, True) == "eligible_for_manual_seed_43_44_authorization"
    assert continuation_status(42, False) == "stop_after_seed_42"
    assert continuation_status(43, True) == "diagnostic_only"
    assert continuation_status(44, False) == "diagnostic_only"


def test_three_seed_configs_and_shared_alpha_selection(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    for seed in (42, 43, 44):
        cfg = load_config(root / f"configs/experiments/e026_rgb_linear_u30_s{seed}.yaml")
        assert cfg["experiment"]["seed"] == seed

    bad = root / "configs/experiments/e026_rgb_linear_u30_s42.yaml"
    bad_text = bad.read_text(encoding="utf-8").replace("seed: 42", "seed: 45")
    bad_path = tmp_path / "bad_seed.yaml"
    bad_path.write_text(bad_text, encoding="utf-8")
    with pytest.raises(ValueError, match="seed 42/43/44"):
        load_config(bad_path)

    grid = [i / 10.0 for i in range(11)]
    curves = [
        [
            {"alpha": alpha, "top1": 0.8, "top5": 0.9, "macro_f1": 0.7}
            for alpha in grid
        ]
        for _ in range(3)
    ]
    # Complete tie selects larger alpha.
    mean_curve, selected = aggregate_alpha_curves(curves)
    assert len(mean_curve) == 11
    assert selected["alpha"] == 1.0

    # Mean Top-1 wins first, even when another alpha has higher Macro-F1.
    for curve in curves:
        curve[4]["top1"] = 0.85
        curve[4]["macro_f1"] = 0.6
        curve[5]["top1"] = 0.84
        curve[5]["macro_f1"] = 0.9
    _, selected = aggregate_alpha_curves(curves)
    assert selected["alpha"] == 0.4

    bad_curve = [dict(row) for row in curves[0]]
    bad_curve[1]["alpha"] = 0.15
    with pytest.raises(ValueError, match="α grid"):
        aggregate_alpha_curves([bad_curve, curves[1], curves[2]])

    summary_cfg = load_config(
        root / "configs/experiments/e026_fusion_u30_three_seed_summary.yaml"
    )
    assert [run["seed"] for run in summary_cfg["fusion_runs"]] == [42, 43, 44]

    for run, curve in zip(summary_cfg["fusion_runs"], curves, strict=True):
        run_dir = tmp_path / run["root"]
        run_dir.mkdir(parents=True)
        seed = run["seed"]
        with (run_dir / "alpha_curve.csv").open("w", encoding="utf-8") as handle:
            handle.write("alpha,top1,top5,macro_f1\n")
            for row in curve:
                handle.write(
                    f"{row['alpha']},{row['top1']},{row['top5']},{row['macro_f1']}\n"
                )
        (run_dir / "summary.json").write_text(
            json.dumps({
                "experiment_id": f"e026_fusion_u30_s{seed}",
                **({} if seed == 42 else {
                    "seed": seed,
                    "rgb_scores": f"artifacts/runs/e026_rgb_linear_u30_s{seed}/scores.npz",
                }),
                "checkpoint": {"path": f"runs/e008_tcn_pose53_uniform30_s{seed}/best.pt"},
                "rgb_metrics": {"top1": 0.4, "top5": 0.7, "macro_f1": 0.3},
                "pose_metrics": {"top1": 0.8, "top5": 0.9, "macro_f1": 0.7},
            }),
            encoding="utf-8",
        )
    summary_cfg["output"]["root"] = "three_seed"
    result = run_e026_fusion_summary(summary_cfg, tmp_path)
    assert result["selected_alpha"] == 0.4
    assert set(result["per_seed"]) == {"42", "43", "44"}
    assert result["means"]["pose"]["top1"] == pytest.approx(0.8)
    assert result["means"]["fused"]["top1"] == pytest.approx(0.85)
    assert (tmp_path / "three_seed/mean_alpha_curve.csv").is_file()
    assert (tmp_path / "three_seed/summary.json").is_file()


def test_tiny_fusion_run_writes_pose_curve_and_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ids = np.array(["000001", "000002"], dtype="U6")
    labels = np.array([0, 1], dtype=np.int64)
    rgb_probs = np.full((2, 400), 1e-6, dtype=np.float32)
    rgb_probs[0, 0] = rgb_probs[1, 1] = 1.0 - 399e-6
    rgb_path = tmp_path / "rgb_scores.npz"
    np.savez(rgb_path, video_id=ids, true_class_id=labels, probabilities=rgb_probs)

    pose_split = FeatureSplit(
        X=np.zeros((2, 1, 53, 3), dtype=np.float32),
        y=labels,
        video_id=ids,
        signer_id=np.array(["001", "002"], dtype="U3"),
        frame_indices=np.zeros((2, 1), dtype=np.int32),
        valid_frame_mask=np.ones((2, 1), dtype=np.uint8),
        keypoint_valid_mask=np.ones((2, 1, 53), dtype=np.uint8),
    )
    monkeypatch.setattr("src.e026_rgb_fusion.load_split", lambda *_args, **_kwargs: pose_split)
    monkeypatch.setattr("src.e026_rgb_fusion.select_view", lambda split, _view: split)
    monkeypatch.setattr(
        "src.e026_rgb_fusion.build_model",
        lambda *_args, **_kwargs: build_model({"name": "rgb_linear"}, nodes=1, num_classes=400),
    )
    checkpoint_path = tmp_path / "best.pt"
    checkpoint_path.write_bytes(b"placeholder")
    monkeypatch.setattr("src.e026_rgb_fusion.download_file", lambda **_kwargs: checkpoint_path)
    monkeypatch.setattr("src.e026_rgb_fusion.load_best", lambda *_args: {"epoch": 55})
    pose_probs = torch.from_numpy(rgb_probs.copy())
    monkeypatch.setattr(
        "src.e026_rgb_fusion.evaluate_with_scores",
        lambda *_args, **_kwargs: ({}, [0, 1], [1.0, 1.0], labels.tolist(), pose_probs),
    )

    cfg = {
        "experiment": {"id": "tiny_fusion", "operation": "e026_fusion", "mode": "full", "seed": 42},
        "rgb_scores": "rgb_scores.npz",
        "pose": {
            "data": {"num_classes": 400},
            "model": {"name": "tcn"},
            "checkpoint": {
                "repo_id": "repo",
                "revision": "rev",
                "path": "best.pt",
                "expected_epoch": 55,
            },
        },
        "evaluation": {"batch_size": 2},
        "output": {"root": "fusion"},
    }
    summary = run_e026_fusion(cfg, tmp_path, device="cpu")
    assert summary["selected_alpha"] == 1.0
    for name in ("pose_scores.npz", "fused_scores.npz", "alpha_curve.csv", "summary.json"):
        assert (tmp_path / "fusion" / name).is_file()
