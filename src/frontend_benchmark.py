"""Small fixed-frame runtime benchmark for the project pose front ends."""

from __future__ import annotations

import csv
import hashlib
import importlib
import importlib.metadata
import tarfile
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from .artifacts import atomic_json, sha256
from .frame_sampling import uniform_indices
from .integrations.huggingface import download_file
from .models.yolo_dual_head import decode
from .yolo.checkpoint import download_checkpoint
from .yolo.extractor import select_pose59
from .yolo.model_runtime import initialize_model
from .yolo.transforms import _letterbox

RTMW_DETECTOR_URL = (
    "https://download.openmmlab.com/mmpose/v1/projects/rtmpose/"
    "rtmdet_nano_8xb32-100e_coco-obj365-person-05d8511e.pth"
)
RTMW_POSE_URL = (
    "https://download.openmmlab.com/mmpose/v1/projects/rtmw/"
    "rtmw-dw-l-m_simcc-cocktail14_270e-256x192-20231122.pth"
)


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not_installed"


def _selected_rows(data: dict[str, Any], project_root: Path) -> list[dict[str, str]]:
    with (project_root / data["manifest"]).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    frames_per_video = int(data["frames_per_video"])
    selected: list[dict[str, str]] = []
    for signer, count in data["signer_counts"].items():
        candidates = [
            row
            for row in rows
            if row["signer_id"] == signer and int(row["num_frames"]) >= frames_per_video
        ]
        candidates.sort(
            key=lambda row: hashlib.sha256(
                f"{data['selection_seed']}:{row['video_id']}".encode()
            ).hexdigest()
        )
        if len(candidates) < int(count):
            raise RuntimeError(f"Insufficient eligible validation videos for signer {signer}")
        selected.extend(candidates[: int(count)])
    return sorted(selected, key=lambda row: (int(row["part_id"]), row["video_id"]))


def _decode_frames(
    config: dict[str, Any], project_root: Path
) -> tuple[list[np.ndarray], list[dict[str, Any]]]:
    data = config["data"]
    rows = _selected_rows(data, project_root)
    frames: list[np.ndarray] = []
    selected: list[dict[str, Any]] = []
    cache_dir = project_root / "cache" / "frontend_benchmark"
    frames_per_video = int(data["frames_per_video"])
    with tempfile.TemporaryDirectory(prefix="frontend-benchmark-") as temporary:
        temporary_dir = Path(temporary)
        for part in sorted({int(row["part_id"]) for row in rows}):
            archive = download_file(
                repo_id=data["dataset_repo"],
                repo_type="dataset",
                revision=data["dataset_revision"],
                filename=f"data/split_{part}_front_view.tar",
                cache_dir=cache_dir,
            )
            with tarfile.open(archive, "r:") as contents:
                for row in (row for row in rows if int(row["part_id"]) == part):
                    video_id = row["video_id"].zfill(6)
                    source = contents.extractfile(f"split_{part}/front_view/{video_id}.mp4")
                    if source is None:
                        raise RuntimeError(f"Benchmark video is missing: {video_id}")
                    video_path = temporary_dir / f"{video_id}.mp4"
                    video_path.write_bytes(source.read())
                    capture = cv2.VideoCapture(str(video_path))
                    video_frames: list[np.ndarray] = []
                    try:
                        while True:
                            ok, frame = capture.read()
                            if not ok:
                                break
                            video_frames.append(frame)
                    finally:
                        capture.release()
                    video_path.unlink()
                    if len(video_frames) != int(row["num_frames"]):
                        raise RuntimeError(f"Decoded frame count mismatch for {video_id}")
                    indices = uniform_indices(len(video_frames), frames_per_video)
                    if np.any(indices < 0):
                        raise RuntimeError(f"Benchmark video is shorter than Uniform-{frames_per_video}: {video_id}")
                    frames.extend(video_frames[index] for index in indices)
                    selected.append(
                        {
                            "video_id": video_id,
                            "signer_id": row["signer_id"],
                            "source_frames": len(video_frames),
                            "uniform_indices": indices.tolist(),
                        }
                    )
    if len(frames) != int(data["expected_frames"]):
        raise RuntimeError(f"Expected {data['expected_frames']} selected frames, got {len(frames)}")
    return frames, selected


def _measure(
    name: str,
    frames: list[np.ndarray],
    warmup_passes: int,
    infer: Callable[[np.ndarray, int], bool],
    *,
    device: torch.device,
) -> dict[str, Any]:
    for warmup_pass in range(warmup_passes):
        for index, frame in enumerate(frames):
            infer(frame, warmup_pass * len(frames) + index)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    latencies: list[float] = []
    detections = 0
    for index, frame in enumerate(frames):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter_ns()
        detections += int(infer(frame, index))
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        latencies.append((time.perf_counter_ns() - started) / 1_000_000)
    elapsed = sum(latencies)
    result: dict[str, Any] = {
        "name": name,
        "measured_frames": len(latencies),
        "warmup_passes": warmup_passes,
        "warmup_frames": warmup_passes * len(frames),
        "latency_ms": {"median": float(np.median(latencies)), "p95": float(np.percentile(latencies, 95))},
        "throughput_frames_s": len(latencies) / (elapsed / 1000.0),
        "output_presence_rate": detections / len(latencies),
    }
    if device.type == "cuda":
        result["peak_vram_bytes"] = int(torch.cuda.max_memory_allocated(device))
    return result


def _mediapipe_backend() -> tuple[Callable[[np.ndarray, int], bool], dict[str, Any]]:
    mp = importlib.import_module("mediapipe")

    options = mp.solutions.holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        refine_face_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    def infer(frame: np.ndarray, _: int) -> bool:
        result = options.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        return result.pose_landmarks is not None

    return infer, {"topology": "MediaPipe Holistic: 33 pose + 468 face + 21 per hand"}


def _rtmw_backend(device: torch.device) -> tuple[Callable[[np.ndarray, int], bool], dict[str, Any]]:
    mmdet_apis = importlib.import_module("mmdet.apis")
    mmpose_apis = importlib.import_module("mmpose.apis")
    mmpose_utils = importlib.import_module("mmpose.utils")
    mmpose = importlib.import_module("mmpose")
    package_root = Path(str(mmpose.__file__)).parent / ".mim"
    detector_config = package_root / "demo/mmdetection_cfg/rtmdet_nano_320-8xb32_coco-person.py"
    pose_config = package_root / "configs/wholebody_2d_keypoint/rtmpose/cocktail14/rtmw-m_8xb1024-270e_cocktail14-256x192.py"
    detector = mmdet_apis.init_detector(str(detector_config), RTMW_DETECTOR_URL, device=str(device))
    detector.cfg = mmpose_utils.adapt_mmdet_pipeline(detector.cfg)
    pose = mmpose_apis.init_model(str(pose_config), RTMW_POSE_URL, device=str(device))

    def infer(frame: np.ndarray, _: int) -> bool:
        detections = mmdet_apis.inference_detector(detector, frame).pred_instances
        person = (detections.labels == 0) & (detections.scores >= 0.3)
        boxes = detections.bboxes[person].detach().cpu().numpy()
        return bool(mmpose_apis.inference_topdown(pose, frame, bboxes=boxes, bbox_format="xyxy"))

    return infer, {
        "topology": "RTMW-m: 133 whole-body keypoints",
        "detector": "RTMDet-nano person 320x320",
        "detector_url": RTMW_DETECTOR_URL,
        "pose_url": RTMW_POSE_URL,
    }


def _yolo_backend(
    config: dict[str, Any], project_root: Path, device: torch.device
) -> tuple[Callable[[np.ndarray, int], bool], dict[str, Any]]:
    checkpoint_cfg = config["yolo_checkpoint"]
    checkpoint = download_checkpoint(
        repo_id=checkpoint_cfg["repo_id"],
        revision=checkpoint_cfg["revision"],
        path_in_repo=checkpoint_cfg["path"],
        cache_dir=project_root / "cache" / "frontend_benchmark",
    )
    model_config = {"model": config["yolo_model"]}
    model, _, _, _ = initialize_model(model_config, device, checkpoint)
    model.eval()
    image_size = int(config["yolo_model"]["image_size"])
    protocol = config["protocol"]

    def infer(frame: np.ndarray, _: int) -> bool:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        letterboxed = _letterbox(rgb, image_size)[0]
        tensor = torch.from_numpy(letterboxed).permute(2, 0, 1).unsqueeze(0).to(device).float().div_(255.0)
        with torch.inference_mode():
            prediction = decode(
                model(tensor),
                confidence=float(protocol["nms_confidence"]),
                iou=float(protocol["nms_iou"]),
                max_detections=int(protocol["max_detections"]),
            )[0]
        confidence = select_pose59(prediction)[2]
        return bool(confidence[0] > 0.0)

    return infer, {
        "topology": "Pose59: Person17 + Left Hand21 + Right Hand21",
        "checkpoint_sha256": sha256(checkpoint),
        "checkpoint": checkpoint_cfg,
    }


def _yolo_onnx_backend(
    config: dict[str, Any], project_root: Path
) -> tuple[Callable[[np.ndarray, int], bool], dict[str, Any]]:
    ort = importlib.import_module("onnxruntime")
    artifact = config["yolo_onnx"]
    model_path = project_root / artifact["path"]
    if not model_path.is_file():
        raise FileNotFoundError(f"YOLO ONNX artifact is missing: {model_path}")
    actual_sha = sha256(model_path)
    if actual_sha != artifact["sha256"]:
        raise RuntimeError(f"YOLO ONNX SHA-256 mismatch: {actual_sha}")
    options = ort.SessionOptions()
    options.intra_op_num_threads = int(config["hardware"]["cpu_threads"])
    options.inter_op_num_threads = int(config["hardware"]["interop_threads"])
    session = ort.InferenceSession(
        str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
    )
    image_size = int(config["yolo_model"]["image_size"])
    protocol = config["protocol"]

    def infer(frame: np.ndarray, _: int) -> bool:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        letterboxed = _letterbox(rgb, image_size)[0]
        images = np.ascontiguousarray(
            letterboxed.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        )
        person_raw, hands_raw = session.run(
            ["person_raw", "hands_raw"], {"images": images}
        )
        prediction = decode(
            {"person": torch.from_numpy(person_raw), "hands": torch.from_numpy(hands_raw)},
            confidence=float(protocol["nms_confidence"]),
            iou=float(protocol["nms_iou"]),
            max_detections=int(protocol["max_detections"]),
        )[0]
        confidence = select_pose59(prediction)[2]
        return bool(confidence[0] > 0.0)

    return infer, {
        "topology": "Pose59: Person17 + Left Hand21 + Right Hand21",
        "artifact": artifact,
        "artifact_sha256": actual_sha,
        "provider": "CPUExecutionProvider",
        "threading": {
            "intra_op_threads": options.intra_op_num_threads,
            "inter_op_threads": options.inter_op_num_threads,
        },
    }


def run_frontend_benchmark(
    config: dict[str, Any], project_root: Path, *, device: str | None = None
) -> dict[str, Any]:
    """Benchmark enabled pose front ends over the fixed decoded smoke frames."""
    target = torch.device(device or config["hardware"]["kind"])
    if target.type != config["hardware"]["kind"]:
        raise RuntimeError(f"Config requires {config['hardware']['kind']}, got {target}")
    if target.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA frontend benchmark requested but CUDA is unavailable")
    if target.type == "cuda" and config["hardware"]["gpu_contains"] not in torch.cuda.get_device_name(target):
        raise RuntimeError(f"Unexpected benchmark GPU: {torch.cuda.get_device_name(target)}")
    if target.type == "cpu":
        torch.set_num_threads(int(config["hardware"]["cpu_threads"]))
        torch.set_num_interop_threads(int(config["hardware"]["interop_threads"]))
    frames, selected_videos = _decode_frames(config, project_root)
    warmup_passes = int(config["protocol"]["warmup_passes"])
    requested = config["backends"]
    rows: list[dict[str, Any]] = []
    runtimes: dict[str, Any] = {}
    if "mediapipe" in requested:
        if target.type != "cpu":
            raise RuntimeError("MediaPipe benchmark is CPU-only")
        infer, details = _mediapipe_backend()
        rows.append(_measure("mediapipe_holistic", frames, warmup_passes, infer, device=target) | details)
        runtimes["mediapipe"] = _package_version("mediapipe")
    if "rtmw" in requested:
        infer, details = _rtmw_backend(target)
        rows.append(_measure("rtmdet_nano_rtmw_m", frames, warmup_passes, infer, device=target) | details)
        runtimes |= {name: _package_version(name) for name in ("mmpose", "mmdet", "mmcv", "mmengine")}
    if "yolo" in requested:
        infer, details = _yolo_backend(config, project_root, target)
        rows.append(_measure("e005_dual_head_yolo", frames, warmup_passes, infer, device=target) | details)
        runtimes["ultralytics"] = _package_version("ultralytics")
    if "yolo_onnx" in requested:
        if target.type != "cpu":
            raise RuntimeError("YOLO ONNX benchmark is CPU-only")
        infer, details = _yolo_onnx_backend(config, project_root)
        rows.append(
            _measure(
                "e005_dual_head_yolo_onnxruntime",
                frames,
                warmup_passes,
                infer,
                device=target,
            )
            | details
        )
        runtimes["onnxruntime"] = _package_version("onnxruntime")
    environment = {"torch": torch.__version__, "device": str(target), "packages": runtimes}
    if target.type == "cuda":
        environment["gpu"] = torch.cuda.get_device_name(target)
    else:
        environment["threading"] = {
            "intra_op_threads": torch.get_num_threads(),
            "inter_op_threads": torch.get_num_interop_threads(),
        }
    summary = {
        "status": "ok",
        "scope": "fixed clean validation Uniform-30 frontend runtime benchmark; decode and model load excluded",
        "experiment_id": config["experiment"]["id"],
        "data": {
            "manifest": config["data"]["manifest"],
            "manifest_sha256": sha256(project_root / config["data"]["manifest"]),
            "selection_seed": config["data"]["selection_seed"],
            "selected_videos": selected_videos,
            "selected_frames": len(frames),
        },
        "protocol": config["protocol"],
        "environment": environment,
        "rows": rows,
    }
    atomic_json(project_root / config["output"]["root"] / "summary.json", summary)
    return summary
