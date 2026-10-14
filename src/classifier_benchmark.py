"""Controlled inference microbenchmark for frozen pose classifiers."""

from __future__ import annotations

import os
import platform
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.flop_counter import FlopCounterMode

from .artifacts import atomic_json, sha256
from .checkpoint import load_best
from .models import build_model


def _cuda_events(
    model: torch.nn.Module, x: torch.Tensor, mask: torch.Tensor, iterations: int
) -> list[float]:
    pairs = [
        (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        for _ in range(iterations)
    ]
    with torch.inference_mode():
        for start, end in pairs:
            start.record()
            model(x, mask)
            end.record()
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


def _block_seconds(
    model: torch.nn.Module, x: torch.Tensor, mask: torch.Tensor, iterations: int, kind: str
) -> float:
    with torch.inference_mode():
        if kind == "cuda":
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iterations):
                model(x, mask)
            end.record()
            torch.cuda.synchronize()
            return start.elapsed_time(end) / 1000.0
        started = time.perf_counter()
        for _ in range(iterations):
            model(x, mask)
        return time.perf_counter() - started


def _gflops(model: torch.nn.Module, x: torch.Tensor, mask: torch.Tensor) -> float:
    """Count batch-1 forward FLOPs (2 x MACs); some models only trace under one grad mode."""
    for context in (torch.inference_mode, torch.enable_grad):
        counter = FlopCounterMode(display=False)
        try:
            with context(), counter:
                model(x, mask)
        except (RuntimeError, AttributeError):
            continue
        if counter.get_total_flops() > 0:
            return counter.get_total_flops() / 1e9
    raise RuntimeError("FLOP counter found no operations")


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
    protocol = config["protocol"]
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
        environment: dict[str, Any] = {
            "gpu": gpu,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "driver": torch.cuda.driver_version()
            if hasattr(torch.cuda, "driver_version")
            else None,
            "cudnn": torch.backends.cudnn.version(),
            "cpu": _cpu_name(),
            "intra_op_threads": torch.get_num_threads(),
            "inter_op_threads": torch.get_num_interop_threads(),
        }
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

    warmup = int(protocol["warmup_iterations"])
    batch1_iterations = int(protocol["batch1_iterations"])
    batch32_iterations = int(protocol["batch32_iterations"])
    effective_protocol = {
        **protocol,
        "repeats": int(protocol.get("repeats", 1)),
        "order_seed": int(protocol.get("order_seed", 42)),
    }
    repeats = effective_protocol["repeats"]
    order_seed = int(protocol.get("order_seed", 42))
    entries = list(config["entries"])
    order_indices = []
    rng = random.Random(order_seed)
    for _ in range(repeats):
        order = list(range(len(entries)))
        if repeats > 1:
            rng.shuffle(order)
        order_indices.append(order)
    orders = [[entries[index]["name"] for index in order] for order in order_indices]

    old_tf32 = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    if "allow_tf32" in protocol:
        torch.backends.cuda.matmul.allow_tf32 = protocol["allow_tf32"]
        torch.backends.cudnn.allow_tf32 = protocol["allow_tf32"]
    round_results: list[list[dict[str, Any]]] = [[] for _ in range(repeats)]
    runtime_flags: dict[str, Any] = {}
    try:
        for round_index, order in enumerate(order_indices):
            for entry_index in order:
                entry = entries[entry_index]
                model_config = entry["model"]
                nodes = int(entry.get("nodes", 53))
                input_channels = int(
                    model_config.get("input_channels", entry.get("input_channels", 2))
                )
                frames = int(entry["frames"])
                checkpoint = project_root / entry["checkpoint"]
                checkpoint_hash = sha256(checkpoint)
                declared_hash = entry.get("checkpoint_sha256")
                if declared_hash is not None and checkpoint_hash != declared_hash:
                    raise ValueError(
                        f"Checkpoint SHA-256 mismatch for {entry['name']}: {checkpoint}"
                    )

                model = build_model(model_config, nodes=nodes, num_classes=400).eval()
                state = load_best(checkpoint, model, torch.device("cpu"))
                epoch = int(state["epoch"])
                del state
                model = model.to(target)
                x1 = torch.zeros(
                    (1, frames, nodes, input_channels), dtype=torch.float32, device=target
                )
                mask1 = torch.ones((1, frames), dtype=torch.uint8, device=target)
                gflops = _gflops(model, x1, mask1)
                with torch.inference_mode():
                    for _ in range(warmup):
                        model(x1, mask1)
                if kind == "cuda":
                    torch.cuda.synchronize()
                latency = timer(model, x1, mask1, batch1_iterations)

                x32 = torch.zeros(
                    (32, frames, nodes, input_channels), dtype=torch.float32, device=target
                )
                mask32 = torch.ones((32, frames), dtype=torch.uint8, device=target)
                with torch.inference_mode():
                    for _ in range(warmup):
                        model(x32, mask32)
                if kind == "cuda":
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats(target)
                block_seconds = _block_seconds(model, x32, mask32, batch32_iterations, kind)
                peak_vram = int(torch.cuda.max_memory_allocated(target)) if kind == "cuda" else None
                result = {
                    "name": entry["name"],
                    "model": entry["model"]["name"],
                    "frames": frames,
                    "checkpoint": {
                        "path": entry["checkpoint"],
                        "sha256": checkpoint_hash,
                        "epoch": epoch,
                    },
                    "parameters": sum(parameter.numel() for parameter in model.parameters()),
                    "model_state_bytes_fp32": sum(
                        parameter.numel() * parameter.element_size()
                        for parameter in model.parameters()
                    ),
                    "gflops_batch1": gflops,
                    "batch1_raw_ms": latency,
                    "batch1_latency_ms": {
                        "median": float(np.median(latency)),
                        "p95": float(np.percentile(latency, 95)),
                    },
                    "batch32_block_seconds": block_seconds,
                    "batch32_throughput_samples_s": 32 * batch32_iterations / block_seconds,
                    "batch32_peak_vram_bytes": peak_vram,
                    "input_shape_batch1": list(x1.shape),
                    "input_shape_batch32": list(x32.shape),
                    "input_dtype": str(x1.dtype),
                }
                round_results[round_index].append(result)
                del model, x1, mask1, x32, mask32
                if kind == "cuda":
                    torch.cuda.empty_cache()
        runtime_flags = {
            "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
            "tf32_cudnn": torch.backends.cudnn.allow_tf32,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
        }
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = old_tf32

    rows = []
    for entry in entries:
        results = [
            next(row for row in round_rows if row["name"] == entry["name"])
            for round_rows in round_results
        ]
        raw = [value for result in results for value in result["batch1_raw_ms"]]
        block_seconds = sum(result["batch32_block_seconds"] for result in results)
        row = {
            key: results[0][key]
            for key in (
                "name",
                "model",
                "frames",
                "checkpoint",
                "parameters",
                "gflops_batch1",
                "model_state_bytes_fp32",
                "input_shape_batch1",
                "input_shape_batch32",
                "input_dtype",
            )
        }
        row.update(
            {
                "batch1_latency_ms": {
                    "median": float(np.median(raw)),
                    "p95": float(np.percentile(raw, 95)),
                },
                "batch1_round_medians_ms": [
                    float(np.median(result["batch1_raw_ms"])) for result in results
                ],
                "batch1_round_median_sd_ms": float(
                    np.std([np.median(result["batch1_raw_ms"]) for result in results], ddof=1)
                )
                if repeats > 1
                else 0.0,
                "batch1_raw_ms_by_round": [result["batch1_raw_ms"] for result in results],
                "batch32_throughput_samples_s": 32 * batch32_iterations * repeats / block_seconds,
                "batch32_block_seconds_by_round": [
                    result["batch32_block_seconds"] for result in results
                ],
                "batch32_throughput_samples_s_by_round": [
                    result["batch32_throughput_samples_s"] for result in results
                ],
            }
        )
        peaks = [
            result["batch32_peak_vram_bytes"]
            for result in results
            if result["batch32_peak_vram_bytes"] is not None
        ]
        if peaks:
            row["batch32_peak_vram_bytes"] = max(peaks)
        rows.append(row)

    summary = {
        "status": "ok",
        "experiment_id": config["experiment"]["id"],
        "hardware": environment,
        "precision": "float32",
        "protocol": effective_protocol,
        "runtime": {**runtime_flags, "source_sha256": sha256(Path(__file__))},
        "entries_order_by_round": orders,
        "rows": rows,
    }
    if cpu_metadata is not None:
        summary["hardware"]["threading"] = cpu_metadata
    atomic_json(project_root / config["output"]["root"] / "summary.json", summary)
    return summary
