"""Focused tests for E027 shared-beta weighted log-probability fusion."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.config import load_config
from src.e027_log_fusion import (
    evaluate_log_fusion,
    run_e027_log_fusion,
    weighted_log_probabilities,
)


def _probabilities(correct_class: int, strength: float) -> np.ndarray:
    probs = np.full((1, 400), (1.0 - strength) / 399.0, dtype=np.float32)
    probs[0, correct_class] = strength
    return probs


def _scores(seed: int, *, pose_strength: float = 0.8, rgb_strength: float = 0.7):
    video_id = np.array([f"{seed:06d}"], dtype="U6")
    labels = np.array([0], dtype=np.int64)
    return (
        seed,
        (video_id, labels, _probabilities(0, rgb_strength)),
        (video_id, labels, _probabilities(0, pose_strength)),
    )


def test_weighted_log_probabilities_stable_and_beta_zero_is_pose() -> None:
    pose = _probabilities(0, 0.8)
    rgb = _probabilities(0, 0.7)
    fused = weighted_log_probabilities(pose, rgb, beta=0.0, epsilon=1e-12)
    assert fused.dtype == np.float32
    assert np.isfinite(fused).all()
    assert np.allclose(fused.sum(axis=1), 1.0)
    assert np.allclose(fused, pose, atol=1e-7)

    zero_safe = weighted_log_probabilities(
        np.eye(1, 400, dtype=np.float32), np.eye(1, 400, dtype=np.float32), 1.0, 1e-12
    )
    assert np.isfinite(zero_safe).all()
    assert np.allclose(zero_safe.sum(axis=1), 1.0)


def test_shared_beta_tie_prefers_smaller_and_rejects_mismatch() -> None:
    mean_curve, selected, per_seed = evaluate_log_fusion(
        [_scores(seed) for seed in (42, 43, 44)], epsilon=1e-12
    )
    assert len(mean_curve) == 11
    assert selected["beta"] == 0.0
    assert set(per_seed) == {"42", "43", "44"}

    seed, rgb_scores, pose_scores = _scores(44)
    _, labels, probabilities = pose_scores
    bad_scores = (
        seed,
        rgb_scores,
        (np.array(["999999"], dtype="U6"), labels, probabilities),
    )
    with pytest.raises(ValueError, match="video IDs differ"):
        evaluate_log_fusion([_scores(42), _scores(43), bad_scores], epsilon=1e-12)


def test_config_and_synthetic_official_run(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    config_path = root / "configs/experiments/e027_weighted_log_prob_fusion.yaml"
    config = load_config(config_path)
    assert config["experiment"]["operation"] == "e027_log_fusion"

    bad_path = tmp_path / "bad.yaml"
    bad_path.write_text(
        config_path.read_text(encoding="utf-8").replace(
            "artifacts/runs/e026_fusion_u30_s42/pose_scores.npz",
            "artifacts/test_evaluate/pose_scores.npz",
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="locked validation inputs"):
        load_config(bad_path)

    inputs = []
    for seed in (42, 43, 44):
        rgb_path = tmp_path / f"rgb_{seed}.npz"
        pose_path = tmp_path / f"pose_{seed}.npz"
        video_id = np.array([f"{seed:06d}"], dtype="U6")
        labels = np.array([0], dtype=np.int64)
        np.savez(
            rgb_path,
            video_id=video_id,
            true_class_id=labels,
            probabilities=_probabilities(0, 0.7),
        )
        np.savez(
            pose_path,
            video_id=video_id,
            true_class_id=labels,
            probabilities=_probabilities(0, 0.8),
        )
        inputs.append(
            {"seed": seed, "rgb_scores": rgb_path.name, "pose_scores": pose_path.name}
        )

    config["inputs"] = inputs
    config["output"]["root"] = "report"
    summary = run_e027_log_fusion(config, tmp_path)
    assert summary["selected_beta"] == 0.0
    assert summary["formula"].startswith("log(max(p_pose")
    assert set(summary["input_checksums"]) == {"42", "43", "44"}
    assert (tmp_path / "report/beta_curve.csv").is_file()
    assert (tmp_path / "report/summary.json").is_file()
