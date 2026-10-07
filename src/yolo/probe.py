"""Bounded batch capacity probe for YOLO dual-head training."""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch

from .dataset import YoloDataset, loader, make_targets
from .training import train_step


def summarize_probe_measurements(
    *,
    batch_size: int,
    warmup_steps: int,
    measured_steps: int,
    compute_step_durations_s: Sequence[float],
    e2e_duration_s: float,
    peak_allocated_bytes: int,
    peak_reserved_bytes: int,
    total_vram_bytes: int,
    losses: Sequence[float | torch.Tensor],
) -> dict[str, Any]:
    """Summarize step durations, throughput, VRAM, and finite loss into a report dictionary."""
    if len(compute_step_durations_s) == 0:
        raise ValueError("compute_step_durations_s cannot be empty")

    durations_ms = [d * 1000.0 for d in compute_step_durations_s]
    warmup_samples = warmup_steps * batch_size
    measured_samples = measured_steps * batch_size
    total_compute_s = sum(compute_step_durations_s)

    p50_ms = float(np.percentile(durations_ms, 50))
    p95_ms = float(np.percentile(durations_ms, 95))
    compute_fps = measured_samples / max(total_compute_s, 1e-9)
    e2e_fps = measured_samples / max(e2e_duration_s, 1e-9)

    finite_loss = all(
        math.isfinite(float(val.item() if isinstance(val, torch.Tensor) else val))
        for val in losses
    )

    return {
        "batch_size": batch_size,
        "warmup_steps": warmup_steps,
        "measured_steps": measured_steps,
        "warmup_samples": warmup_samples,
        "measured_samples": measured_samples,
        "compute_step_p50_ms": p50_ms,
        "compute_step_p95_ms": p95_ms,
        "compute_images_per_second": compute_fps,
        "end_to_end_images_per_second": e2e_fps,
        "peak_allocated_vram_bytes": peak_allocated_bytes,
        "peak_allocated_vram_mb": peak_allocated_bytes / (1024.0 * 1024.0),
        "peak_reserved_vram_bytes": peak_reserved_bytes,
        "peak_reserved_vram_mb": peak_reserved_bytes / (1024.0 * 1024.0),
        "total_vram_bytes": total_vram_bytes,
        "total_vram_mb": total_vram_bytes / (1024.0 * 1024.0),
        "finite_loss": finite_loss,
    }


def probe_batch_capacity(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    pose_loss: Any,
    hand_loss: Any,
    dataset: YoloDataset,
    *,
    batch_size: int,
    image_size: int,
    warmup_steps: int,
    measured_steps: int,
    workers: int = 0,
    amp: bool = True,
    device: torch.device | str = "cuda",
) -> dict[str, Any]:
    """Execute a bounded capacity benchmark with real YoloDataset and full train step."""
    if not torch.cuda.is_available():
        raise RuntimeError("YOLO capacity probe requires CUDA")
    if warmup_steps < 0 or measured_steps <= 0:
        raise ValueError("warmup_steps must be >= 0 and measured_steps must be > 0")

    target_device = torch.device(device) if isinstance(device, str) else device
    model.train()
    data_loader = loader(dataset, batch_size=batch_size, workers=workers)
    data_iter = iter(data_loader)

    def next_batch() -> dict[str, torch.Tensor]:
        nonlocal data_iter
        try:
            return next(data_iter)
        except StopIteration:
            data_iter = iter(data_loader)
            return next(data_iter)

    losses: list[float] = []

    for _ in range(warmup_steps):
        batch = next_batch()
        pose, hands, _ = make_targets(batch, device=target_device, image_size=image_size)
        loss = train_step(
            model,
            optimizer,
            scaler,
            pose_loss,
            hand_loss,
            pose,
            hands,
            amp=amp,
            return_components=False,
        )
        losses.append(float(loss.detach().item()))

    torch.cuda.synchronize(target_device)
    torch.cuda.reset_peak_memory_stats(target_device)

    compute_durations: list[float] = []
    e2e_start = time.perf_counter()

    for _ in range(measured_steps):
        batch = next_batch()
        pose, hands, _ = make_targets(batch, device=target_device, image_size=image_size)
        torch.cuda.synchronize(target_device)
        step_start = time.perf_counter()
        loss = train_step(
            model,
            optimizer,
            scaler,
            pose_loss,
            hand_loss,
            pose,
            hands,
            amp=amp,
            return_components=False,
        )
        torch.cuda.synchronize(target_device)
        compute_durations.append(time.perf_counter() - step_start)
        losses.append(float(loss.detach().item()))

    e2e_duration = time.perf_counter() - e2e_start

    device_props = torch.cuda.get_device_properties(target_device)
    total_vram = int(device_props.total_memory)
    peak_alloc = int(torch.cuda.max_memory_allocated(target_device))
    peak_res = int(torch.cuda.max_memory_reserved(target_device))

    return summarize_probe_measurements(
        batch_size=batch_size,
        warmup_steps=warmup_steps,
        measured_steps=measured_steps,
        compute_step_durations_s=compute_durations,
        e2e_duration_s=e2e_duration,
        peak_allocated_bytes=peak_alloc,
        peak_reserved_bytes=peak_res,
        total_vram_bytes=total_vram,
        losses=losses,
    )
