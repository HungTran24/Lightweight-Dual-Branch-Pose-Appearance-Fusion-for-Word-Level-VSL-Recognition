from __future__ import annotations

import hashlib
import importlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from src.config import load_config
from src.frontend_benchmark import _measure, _selected_rows, _yolo_onnx_backend

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_measure_records_latency_throughput_and_presence() -> None:
    calls: list[int] = []

    def infer(_: np.ndarray, index: int) -> bool:
        calls.append(index)
        return index % 2 == 0

    frames = [np.zeros((2, 2, 3), dtype=np.uint8) for _ in range(4)]
    summary = _measure("fake", frames, 1, infer, device=torch.device("cpu"))

    assert calls == [0, 1, 2, 3, 0, 1, 2, 3]
    assert summary["measured_frames"] == 4
    assert summary["warmup_frames"] == 4
    assert summary["latency_ms"]["median"] >= 0.0
    assert summary["throughput_frames_s"] > 0.0
    assert summary["output_presence_rate"] == 0.5


def test_frontend_benchmark_configs_load() -> None:
    for name in (
        "benchmark_frontends_pose_cpu_ryzen7_7735hs.yaml",
        "benchmark_frontends_pose_a100.yaml",
    ):
        config = load_config(PROJECT_ROOT / "configs/experiments" / name)
        assert config["experiment"]["operation"] == "frontend_benchmark"
    assert config["backends"] == ["rtmw", "yolo"]
    cpu_config = load_config(
        PROJECT_ROOT / "configs/experiments/benchmark_frontends_pose_cpu_ryzen7_7735hs.yaml"
    )
    assert cpu_config["backends"] == ["mediapipe", "rtmw", "yolo", "yolo_onnx"]
    assert cpu_config["yolo_onnx"]["sha256"] == (
        "b9d477485901ba79cab5c73dc831f4623bf116ce0a0b72daec27bcded8bf1736"
    )


@pytest.mark.skipif(
    not (PROJECT_ROOT / "manifests/audit/val_clean.csv").is_file(),
    reason="needs the licensed VSL400 manifest",
)
def test_frontend_benchmark_uses_ten_clean_validation_videos() -> None:
    config = load_config(
        PROJECT_ROOT / "configs/experiments/benchmark_frontends_pose_cpu_ryzen7_7735hs.yaml"
    )
    selected = _selected_rows(config["data"], PROJECT_ROOT)

    assert len(selected) == 10
    assert [row["signer_id"] for row in selected].count("006") == 4
    assert [row["signer_id"] for row in selected].count("020") == 3
    assert [row["signer_id"] for row in selected].count("028") == 3
    assert all(int(row["num_frames"]) >= 30 for row in selected)


def test_yolo_onnx_backend_uses_cpu_threads_and_existing_postprocess(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"onnx")
    digest = hashlib.sha256(b"onnx").hexdigest()
    captured: dict[str, Any] = {}

    class SessionOptions:
        intra_op_num_threads = 0
        inter_op_num_threads = 0

    class InferenceSession:
        def __init__(self, path: str, *, sess_options: Any, providers: list[str]) -> None:
            captured.update(path=path, options=sess_options, providers=providers)

        def run(self, outputs: list[str], inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
            captured.update(outputs=outputs, images=inputs["images"])
            return [
                np.zeros((1, 56, 1), dtype=np.float32),
                np.zeros((1, 69, 1), dtype=np.float32),
            ]

    fake_ort = SimpleNamespace(SessionOptions=SessionOptions, InferenceSession=InferenceSession)
    original_import = importlib.import_module
    monkeypatch.setattr(
        "src.frontend_benchmark.importlib.import_module",
        lambda name: fake_ort if name == "onnxruntime" else original_import(name),
    )

    def fake_decode(output: dict[str, torch.Tensor], **_: Any) -> list[dict[str, Any]]:
        assert all(isinstance(value, torch.Tensor) for value in output.values())
        return [{"decoded": True}]

    monkeypatch.setattr("src.frontend_benchmark.decode", fake_decode)
    monkeypatch.setattr(
        "src.frontend_benchmark.select_pose59",
        lambda _: (None, None, np.array([1.0], dtype=np.float32)),
    )
    config = {
        "hardware": {"cpu_threads": 8, "interop_threads": 1},
        "protocol": {"nms_confidence": 0.25, "nms_iou": 0.7, "max_detections": 300},
        "yolo_model": {"image_size": 2},
        "yolo_onnx": {"path": "model.onnx", "sha256": digest},
    }

    infer, details = _yolo_onnx_backend(config, tmp_path)
    assert infer(np.zeros((2, 2, 3), dtype=np.uint8), 0)
    assert captured["providers"] == ["CPUExecutionProvider"]
    assert captured["options"].intra_op_num_threads == 8
    assert captured["options"].inter_op_num_threads == 1
    assert captured["outputs"] == ["person_raw", "hands_raw"]
    assert captured["images"].shape == (1, 3, 2, 2)
    assert captured["images"].dtype == np.float32
    assert details["artifact_sha256"] == digest
