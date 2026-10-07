"""Controlled inference microbenchmark for frozen pose classifiers."""

from __future__ import annotations

import os
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .artifacts import atomic_json, sha256
from .checkpoint import load_best
from .models import build_model


def _cuda_events(
    model: torch.nn.Module, x: torch.Tensor, mask: torch.Tensor, iterations: int
) -> list[float]:
    pairs: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
    with torch.inference_mode():
        for _ in range(iterations):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            model(x, mask)
            end.record()
            pairs.append((start, end))
    torch.cuda.synchronize()
    return [start.elapsed_time(end) for start, end in pairs]


def _cpu_events(
    model: torch.nn.Module, x: torch.Tensor, mask: torch.Tensor, iterations: int
) -> list[float]:
    elapsed: list[float] = []
    with torch.inference_mode():
        for _ in range(iterations):
            started = time.perf_counter_ns()
            model(x, mask)
            elapsed.append((time.perf_counter_ns() - started) / 1_000_000)
    return elapsed


def _cpu_name() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                return line.partition(":")[2].strip()
    return platform.processor() or platform.machine()


def _set_cpu_threads(hardware: dict[str, Any]) -> dict[str, Any]:
    intra = int(hardware["cpu_threads"])
    interop = int(hardware["interop_threads"])
    torch.set_num_threads(intra)
    if torch.get_num_interop_threads() != interop:
        try:
            torch.set_num_interop_threads(interop)
        except RuntimeError as exc:
            raise RuntimeError("Set CPU inter-op threads before running PyTorch work") from exc
    if torch.get_num_threads() != intra or torch.get_num_interop_threads() != interop:
        raise RuntimeError("CPU benchmark thread settings did not take effect")
    affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    return {
        "intra_op_threads": torch.get_num_threads(),
        "inter_op_threads": torch.get_num_interop_threads(),
        "available_cpu_ids": affinity,
        "logical_cpus": os.cpu_count(),
        "parallel_info": torch.__config__.parallel_info(),
        "mkldnn_available": torch.backends.mkldnn.is_available(),
        "mkldnn_enabled": torch.backends.mkldnn.enabled,
        "platform": platform.platform(),
    }


def run_classifier_benchmark(
    config: dict[str, Any], project_root: Path, *, device: str | None = None
) -> dict[str, Any]:
    hardware = config["hardware"]
    kind = str(hardware.get("kind", "cuda"))
    target = torch.device(device or kind)
    if target.type != kind:
        raise RuntimeError(f"Benchmark config requires {kind}, got {target}")

    cpu_metadata: dict[str, Any] | None = None
    if kind == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Classifier benchmark requires CUDA")
        gpu = torch.cuda.get_device_name(target)
        expected_gpu = str(hardware["gpu_contains"])
        if expected_gpu not in gpu:
            raise RuntimeError(f"Expected GPU containing {expected_gpu!r}, got {gpu!r}")
        environment: dict[str, Any] = {"gpu": gpu, "torch": torch.__version__, "cuda": torch.version.cuda}
        timer = _cuda_events
    elif kind == "cpu":
        cpu = _cpu_name()
        expected_cpu = str(hardware["cpu_contains"])
        if expected_cpu not in cpu:
            raise RuntimeError(f"Expected CPU containing {expected_cpu!r}, got {cpu!r}")
        cpu_metadata = _set_cpu_threads(hardware)
        environment = {"cpu": cpu, "torch": torch.__version__, "threading": cpu_metadata}
        timer = _cpu_events
    else:
        raise ValueError(f"Unsupported benchmark hardware.kind: {kind}")

    warmup = int(config["protocol"]["warmup_iterations"])
    batch1_iterations = int(config["protocol"]["batch1_iterations"])
    batch32_iterations = int(config["protocol"]["batch32_iterations"])
    rows = []
    for entry in config["entries"]:
        model_config = entry["model"]
        nodes = int(entry.get("nodes", 53))
        input_channels = int(model_config.get("input_channels", entry.get("input_channels", 2)))
        model = build_model(model_config, nodes=nodes, num_classes=400).to(target).eval()
        checkpoint = project_root / entry["checkpoint"]
        state = load_best(checkpoint, model, target)
        frames = int(entry["frames"])

        x1 = torch.zeros((1, frames, nodes, input_channels), dtype=torch.float32, device=target)
        mask1 = torch.ones((1, frames), dtype=torch.uint8, device=target)
        with torch.inference_mode():
            for _ in range(warmup):
                model(x1, mask1)
        if kind == "cuda":
            torch.cuda.synchronize()
        latency = timer(model, x1, mask1, batch1_iterations)

        x32 = torch.zeros((32, frames, nodes, input_channels), dtype=torch.float32, device=target)
        mask32 = torch.ones((32, frames), dtype=torch.uint8, device=target)
        with torch.inference_mode():
            for _ in range(warmup):
                model(x32, mask32)
        if kind == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(target)
        throughput_times = timer(model, x32, mask32, batch32_iterations)
        row = {
            "name": entry["name"],
            "model": entry["model"]["name"],
            "frames": frames,
            "checkpoint": {
                "path": entry["checkpoint"],
                "sha256": sha256(checkpoint),
                "epoch": int(state["epoch"]),
            },
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "model_state_bytes_fp32": sum(
                parameter.numel() * parameter.element_size() for parameter in model.parameters()
            ),
            "batch1_latency_ms": {
                "median": float(np.median(latency)),
                "p95": float(np.percentile(latency, 95)),
            },
            "batch32_throughput_samples_s": float(
                32 * batch32_iterations / (sum(throughput_times) / 1000.0)
            ),
        }
        if kind == "cuda":
            row["batch32_peak_vram_bytes"] = int(torch.cuda.max_memory_allocated(target))
        rows.append(row)
        del model, x1, mask1, x32, mask32
        if kind == "cuda":
            torch.cuda.empty_cache()

    summary = {
        "status": "ok",
        "experiment_id": config["experiment"]["id"],
        "hardware": environment,
        "precision": "float32",
        "protocol": config["protocol"],
        "rows": rows,
    }
    atomic_json(project_root / config["output"]["root"] / "summary.json", summary)
    return summary
