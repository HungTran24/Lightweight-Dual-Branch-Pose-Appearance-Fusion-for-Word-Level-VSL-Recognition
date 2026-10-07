from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

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
    monkeypatch.setattr(
        "src.classifier_benchmark._set_cpu_threads",
        lambda _: {"intra_op_threads": 1, "inter_op_threads": 1},
    )

    summary = run_classifier_benchmark(config, tmp_path, device="cpu")

    assert summary["hardware"]["cpu"] == "Test CPU"
    assert summary["hardware"]["threading"] == {"intra_op_threads": 1, "inter_op_threads": 1}
    row = summary["rows"][0]
    assert row["checkpoint"]["epoch"] == 3
    assert row["batch1_latency_ms"]["median"] > 0
    assert row["batch32_throughput_samples_s"] > 0
    assert "batch32_peak_vram_bytes" not in row
    assert (tmp_path / "artifacts/cpu_benchmark/summary.json").is_file()


def test_cpu_benchmark_rejects_invalid_thread_count() -> None:
    config = _cpu_config()
    config["hardware"]["cpu_threads"] = 0
    with pytest.raises(ValueError, match="positive thread counts"):
        _validate_classifier_benchmark(config)


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
