"""Fixed-shape ONNX export and parity proof for the dual-head YOLO model."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from ..artifacts import atomic_json, sha256
from ..config import load_config
from ..models.yolo_dual_head import decode
from .checkpoint import download_checkpoint
from .extractor import select_pose59_with_detections
from .model_runtime import initialize_model


class RawDualHeadWrapper(nn.Module):
    """Expose the custom heads as stable raw person and hand tensors."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        output = self.model(images)
        person = output["person"]
        hands = output["hands"]
        return (
            person[0] if isinstance(person, tuple) else person,
            hands[0] if isinstance(hands, tuple) else hands,
        )


def _max_abs(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left - right).abs().max().item())


def _pose59_from_raw(
    person: torch.Tensor, hands: torch.Tensor, config: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    decoded = decode(
        {"person": person, "hands": hands},
        confidence=float(config["data"]["nms_confidence"]),
        iou=float(config["data"]["nms_iou"]),
        max_detections=int(config["data"]["max_detections"]),
    )
    return select_pose59_with_detections(decoded[0])


def export_onnx(
    config_path: Path,
    project_root: Path,
    *,
    output_dir: Path,
    opset: int = 18,
    tolerance: float = 1e-3,
) -> dict[str, Any]:
    """Export the pinned dual-head model and verify CPU ONNX Runtime raw-output parity."""
    import onnx  # type: ignore[import-not-found]
    import onnxruntime as ort  # type: ignore[import-not-found]
    import ultralytics

    config_path = config_path.resolve() if config_path.is_absolute() else project_root / config_path
    config = load_config(config_path)
    if config["experiment"].get("id") != "e005_yolo_pose59_extract":
        raise ValueError("ONNX export requires the pinned E005 extraction config")

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_config = config["checkpoint"]
    checkpoint = download_checkpoint(
        repo_id=str(checkpoint_config["repo_id"]),
        revision=str(checkpoint_config["revision"]),
        path_in_repo=str(checkpoint_config["path"]),
        cache_dir=project_root / "cache" / "hf",
    )
    model, _, _, _ = initialize_model(config, torch.device("cpu"), checkpoint)
    model.eval()
    wrapper = RawDualHeadWrapper(model).eval()
    size = int(config["model"]["image_size"])
    generator = torch.Generator(device="cpu").manual_seed(42)
    images = torch.rand((1, 3, size, size), generator=generator, dtype=torch.float32)
    with torch.no_grad():
        torch_person, torch_hands = wrapper(images)

    model_path = output_dir / "model.onnx"
    torch.onnx.export(
        wrapper,
        (images,),
        model_path,
        input_names=["images"],
        output_names=["person_raw", "hands_raw"],
        opset_version=opset,
        dynamo=False,
    )
    onnx_model = onnx.load(model_path)
    onnx.checker.check_model(onnx_model)
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    person_raw, hands_raw = session.run(
        ["person_raw", "hands_raw"], {"images": np.ascontiguousarray(images.numpy())}
    )
    onnx_person = torch.from_numpy(person_raw)
    onnx_hands = torch.from_numpy(hands_raw)
    raw_error = {
        "person_max_abs": _max_abs(torch_person, onnx_person),
        "hands_max_abs": _max_abs(torch_hands, onnx_hands),
    }
    if max(raw_error.values()) > tolerance:
        raise RuntimeError(f"ONNX raw-output parity exceeded tolerance {tolerance}: {raw_error}")

    torch_pose = _pose59_from_raw(torch_person, torch_hands, config)
    onnx_pose = _pose59_from_raw(onnx_person, onnx_hands, config)
    pose_error = float(np.abs(torch_pose[0] - onnx_pose[0]).max())
    if not np.array_equal(torch_pose[1], onnx_pose[1]) or pose_error > tolerance:
        raise RuntimeError(f"ONNX Pose59 parity failed: max_abs={pose_error}")

    summary = {
        "status": "ok",
        "scope": "fixed-shape raw dual-head ONNX export and CPU Runtime parity proof only",
        "source": {
            "config": str(config_path.relative_to(project_root)),
            "checkpoint": checkpoint_config,
            "checkpoint_sha256": sha256(checkpoint),
        },
        "artifact": {"path": "model.onnx", "sha256": sha256(model_path)},
        "input": {"name": "images", "shape": list(images.shape), "dtype": "float32"},
        "outputs": {
            "person_raw": list(torch_person.shape),
            "hands_raw": list(torch_hands.shape),
        },
        "runtime": {
            "torch": torch.__version__,
            "ultralytics": ultralytics.__version__,
            "onnx": onnx.__version__,
            "onnxruntime": ort.__version__,
            "provider": "CPUExecutionProvider",
            "opset": opset,
        },
        "parity": {
            "tolerance": tolerance,
            "raw": raw_error,
            "pose59_max_abs": pose_error,
            "pose59_mask_equal": True,
        },
        "limitations": [
            "NMS, Pose59 selection, letterboxing, video decode, and crop handling remain outside ONNX.",
            "This proof does not measure end-to-end CPU latency or detector accuracy across the dataset.",
        ],
    }
    atomic_json(output_dir / "summary.json", summary)
    return summary
