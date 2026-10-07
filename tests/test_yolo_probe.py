from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch

from src.yolo.probe import probe_batch_capacity, summarize_probe_measurements


def test_summarize_probe_measurements() -> None:
    durations = [0.010, 0.012, 0.015, 0.020, 0.025]  # in seconds: 10, 12, 15, 20, 25 ms
    e2e_duration = 0.15
    losses = [1.5, 1.4, 1.3, 1.2, 1.1]

    result = summarize_probe_measurements(
        batch_size=8,
        warmup_steps=2,
        measured_steps=5,
        compute_step_durations_s=durations,
        e2e_duration_s=e2e_duration,
        peak_allocated_bytes=1024 * 1024 * 500,  # 500 MB
        peak_reserved_bytes=1024 * 1024 * 800,  # 800 MB
        total_vram_bytes=1024 * 1024 * 16000,  # 16 GB
        losses=losses,
    )

    assert result["batch_size"] == 8
    assert result["warmup_steps"] == 2
    assert result["measured_steps"] == 5
    assert result["warmup_samples"] == 16
    assert result["measured_samples"] == 40
    assert pytest.approx(result["compute_step_p50_ms"], rel=1e-3) == 15.0
    assert result["compute_step_p95_ms"] > 20.0
    assert result["compute_images_per_second"] == pytest.approx(40 / sum(durations), rel=1e-3)
    assert result["end_to_end_images_per_second"] == pytest.approx(40 / e2e_duration, rel=1e-3)
    assert result["peak_allocated_vram_mb"] == pytest.approx(500.0)
    assert result["peak_reserved_vram_mb"] == pytest.approx(800.0)
    assert result["total_vram_mb"] == pytest.approx(16000.0)
    assert result["finite_loss"] is True


def test_summarize_probe_measurements_detects_non_finite_loss() -> None:
    durations = [0.010, 0.012]
    losses = [1.5, float("inf")]

    result = summarize_probe_measurements(
        batch_size=4,
        warmup_steps=1,
        measured_steps=2,
        compute_step_durations_s=durations,
        e2e_duration_s=0.05,
        peak_allocated_bytes=1024 * 1024,
        peak_reserved_bytes=1024 * 1024,
        total_vram_bytes=1024 * 1024 * 1000,
        losses=losses,
    )
    assert result["finite_loss"] is False


def test_probe_batch_capacity_requires_cuda_when_no_cuda() -> None:
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError, match="YOLO capacity probe requires CUDA"):
            probe_batch_capacity(
                model=MagicMock(),
                optimizer=MagicMock(),
                scaler=MagicMock(),
                pose_loss=MagicMock(),
                hand_loss=MagicMock(),
                dataset=MagicMock(),
                batch_size=8,
                image_size=640,
                warmup_steps=10,
                measured_steps=100,
            )
