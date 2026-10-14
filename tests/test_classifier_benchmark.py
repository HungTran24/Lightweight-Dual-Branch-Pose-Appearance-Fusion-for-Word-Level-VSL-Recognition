from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch

from src.checkpoint import save_best
from src.classifier_benchmark import run_classifier_benchmark
from src.config import _validate_classifier_benchmark, load_config
from src.models import build_model


def _cpu_config() -> dict[str, Any]:
    return {
        "experiment": {
            "id": "cpu_benchmark",
            "operation": "classifier_benchmark",
            "mode": "full",
        },
        "hardware": {
            "kind": "cpu",
            "cpu_contains": "Test CPU",
            "cpu_threads": 1,
            "interop_threads": 1,
        },
        "protocol": {
            "precision": "float32",
            "inputs_on_cpu": True,
            "warmup_iterations": 1,
            "batch1_iterations": 2,
            "batch32_iterations": 2,
        },
        "entries": [
            {
                "name": "tiny_tcn",
                "frames": 2,
                "nodes": 42,
                "checkpoint": "checkpoint.pt",
                "model": {
                    "name": "tcn",
                    "input_channels": 2,
                    "channels": [4],
                    "dilations": [1],
                    "kernel_size": 3,
                    "dropout": 0.0,
                },
            }
        ],
        "output": {"root": "artifacts/cpu_benchmark"},
    }


def test_cpu_benchmark_records_cpu_metrics(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = _cpu_config()
    _validate_classifier_benchmark(config)
    model = build_model(config["entries"][0]["model"], nodes=42, num_classes=400)
    save_best(tmp_path / "checkpoint.pt", model, epoch=3, metrics={})
    monkeypatch.setattr("src.classifier_benchmark._cpu_name", lambda: "Test CPU")
    original_load_best = __import__("src.classifier_benchmark", fromlist=["load_best"]).load_best
    load_devices: list[torch.device] = []

    def capture_load_device(
        path: Path, target_model: Any, load_device: torch.device
    ) -> dict[str, Any]:
        load_devices.append(load_device)
        return original_load_best(path, target_model, load_device)

    monkeypatch.setattr("src.classifier_benchmark.load_best", capture_load_device)
    monkeypatch.setattr(
        "src.classifier_benchmark._set_cpu_threads",
        lambda _: {"intra_op_threads": 1, "inter_op_threads": 1},
    )

    summary = run_classifier_benchmark(config, tmp_path, device="cpu")

    assert load_devices == [torch.device("cpu")]
    assert summary["hardware"]["cpu"] == "Test CPU"
    assert summary["hardware"]["threading"] == {"intra_op_threads": 1, "inter_op_threads": 1}
    row = summary["rows"][0]
    assert row["checkpoint"]["epoch"] == 3
    assert row["batch1_latency_ms"]["median"] > 0
    assert row["batch32_throughput_samples_s"] > 0
    assert "batch32_peak_vram_bytes" not in row
    assert (tmp_path / "artifacts/cpu_benchmark/summary.json").is_file()


def test_repeats_preserve_raw_latency_and_aggregate_block_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _cpu_config()
    config["protocol"]["repeats"] = 2
    config["protocol"]["order_seed"] = 7
    model_config = config["entries"][0]["model"]
    config["entries"].append(
        {
            "name": "second_tcn",
            "frames": 2,
            "nodes": 42,
            "checkpoint": "second.pt",
            "model": model_config,
        }
    )
    for entry, epoch in zip(config["entries"], (1, 2), strict=True):
        model = build_model(entry["model"], nodes=42, num_classes=400)
        save_best(tmp_path / entry["checkpoint"], model, epoch=epoch, metrics={})
    monkeypatch.setattr("src.classifier_benchmark._cpu_name", lambda: "Test CPU")
    monkeypatch.setattr("src.classifier_benchmark._set_cpu_threads", lambda _: {})
    monkeypatch.setattr("src.classifier_benchmark._cpu_events", lambda *args: [10.0, 20.0])
    durations = iter((1.0, 2.0, 3.0, 4.0))
    monkeypatch.setattr("src.classifier_benchmark._block_seconds", lambda *args: next(durations))

    summary = run_classifier_benchmark(config, tmp_path, device="cpu")
    expected_order = summary["entries_order_by_round"]
    rows = {row["name"]: row for row in summary["rows"]}
    assert all(sorted(order) == ["second_tcn", "tiny_tcn"] for order in expected_order)
    durations = iter((1.0, 2.0, 3.0, 4.0))
    repeated = run_classifier_benchmark(config, tmp_path, device="cpu")
    assert repeated["entries_order_by_round"] == expected_order
    assert rows["tiny_tcn"]["batch1_raw_ms_by_round"] == [[10.0, 20.0], [10.0, 20.0]]
    assert rows["tiny_tcn"]["batch32_throughput_samples_s"] == pytest.approx(
        32 * 2 * 2 / sum(rows["tiny_tcn"]["batch32_block_seconds_by_round"])
    )
    assert len(rows["tiny_tcn"]["batch32_block_seconds_by_round"]) == 2


def test_cpu_benchmark_rejects_invalid_thread_count() -> None:
    config = _cpu_config()
    config["hardware"]["cpu_threads"] = 0
    with pytest.raises(ValueError, match="positive thread counts"):
        _validate_classifier_benchmark(config)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("protocol", "repeats"), True),
        (("protocol", "order_seed"), 1.5),
        (("protocol", "allow_tf32"), 0),
        (("entries", 0, "frames"), 0),
        (("protocol", "precision"), "float16"),
    ],
)
def test_benchmark_rejects_invalid_optional_settings(path: tuple[Any, ...], value: Any) -> None:
    config = _cpu_config()
    target: Any = config
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValueError):
        _validate_classifier_benchmark(config)


def test_benchmark_rejects_checkpoint_hash_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _cpu_config()
    model = build_model(config["entries"][0]["model"], nodes=42, num_classes=400)
    save_best(tmp_path / "checkpoint.pt", model, epoch=1, metrics={})
    config["entries"][0]["checkpoint_sha256"] = "0" * 64
    monkeypatch.setattr("src.classifier_benchmark._cpu_name", lambda: "Test CPU")
    monkeypatch.setattr("src.classifier_benchmark._set_cpu_threads", lambda _: {})
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        run_classifier_benchmark(config, tmp_path, device="cpu")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_tiny_benchmark_loads_checkpoint_on_cpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _cpu_config()
    config["hardware"] = {"kind": "cuda", "gpu_contains": torch.cuda.get_device_name(0)}
    config["protocol"]["allow_tf32"] = False
    model = build_model(config["entries"][0]["model"], nodes=42, num_classes=400)
    save_best(tmp_path / "checkpoint.pt", model, epoch=2, metrics={})
    devices: list[torch.device] = []
    original = __import__("src.classifier_benchmark", fromlist=["load_best"]).load_best

    def capture(path: Path, target_model: Any, load_device: torch.device) -> dict[str, Any]:
        devices.append(load_device)
        return original(path, target_model, load_device)

    monkeypatch.setattr("src.classifier_benchmark.load_best", capture)
    summary = run_classifier_benchmark(config, tmp_path, device="cuda:0")
    assert devices == [torch.device("cpu")]
    assert summary["rows"][0]["batch32_throughput_samples_s"] > 0
    assert summary["runtime"]["tf32_matmul"] is False


def test_cpu_benchmark_config_loads() -> None:
    config = load_config(
        Path(__file__).parents[1]
        / "configs/experiments/benchmark_slr_controlled_cpu_ryzen7_7735hs.yaml"
    )
    assert config["hardware"] == {
        "kind": "cpu",
        "cpu_contains": "AMD Ryzen 7 7735HS",
        "cpu_threads": 8,
        "interop_threads": 1,
    }
    assert [entry["name"] for entry in config["entries"]] == [
        "tcn_pose53_u30_seed42",
        "stgcn_pose53_u30_seed42",
        "spoter_pose54_u30_seed42",
        "slgcn_bone_pose27_u30_epoch10",
    ]
