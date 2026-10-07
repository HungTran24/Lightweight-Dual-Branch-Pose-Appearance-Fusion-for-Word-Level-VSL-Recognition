"""All-frame YOLO dual-head inference and Pose59 artifact storage."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import tarfile
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import ultralytics

from ..artifacts import (
    atomic_json as _atomic_json,
)
from ..artifacts import (
    sha256 as _sha256,
)
from ..artifacts import (
    upload_with_retry as _upload_files_with_retry,
)
from ..integrations.huggingface import download_file, ensure_repo
from ..models.yolo_dual_head import decode
from ..pose_views.pose59 import validate_pose59_schema
from .checkpoint import download_checkpoint
from .hand_rgb import (
    HandRgbSink,
    inverse_letterbox_boxes,
    validate_crop_shards,
    validate_extended_schema,
)
from .model_runtime import initialize_model
from .transforms import _letterbox


def inverse_letterbox_keypoints(
    keypoints: np.ndarray,
    *,
    scale: float,
    pad_x: int,
    pad_y: int,
    width: int,
    height: int,
) -> np.ndarray:
    """Project letterboxed coordinates back to normalized [0, 1] original frame space."""
    result = np.asarray(keypoints, dtype=np.float32).copy()
    result[..., 0] = np.clip((result[..., 0] - pad_x) / scale, 0, width) / max(width, 1)
    result[..., 1] = np.clip((result[..., 1] - pad_y) / scale, 0, height) / max(height, 1)
    return result


def select_pose59_with_detections(
    decoded: dict[str, torch.Tensor],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Select primary person and inside-person hands, returning keypoints, masks, and boxes."""
    person, hands = decoded["person"], decoded["hands"]
    output = np.zeros((59, 3), dtype=np.float32)
    mask = np.zeros(59, dtype=np.uint8)
    confidence = np.zeros(3, dtype=np.float32)
    boxes = np.zeros((3, 4), dtype=np.float32)
    valid_mask = np.zeros(3, dtype=np.uint8)
    if not len(person):
        return output, mask, confidence, boxes, valid_mask

    primary = person[int(torch.argmax(person[:, 4]).item())]
    output[:17] = primary[6:].detach().cpu().numpy().reshape(17, 3)
    mask[:17] = (output[:17, 2] > 0).astype(np.uint8)
    output[:17][mask[:17] == 0] = 0.0
    confidence[0] = float(primary[4])
    boxes[0] = primary[:4].detach().cpu().numpy()
    valid_mask[0] = 1
    x0, y0, x1, y1 = primary[:4]

    for cls, start in ((0, 17), (1, 38)):
        candidates = hands[hands[:, 5].long() == cls]
        if len(candidates):
            centers = (candidates[:, :2] + candidates[:, 2:4]) / 2.0
            candidates = candidates[
                (centers[:, 0] >= x0)
                & (centers[:, 0] <= x1)
                & (centers[:, 1] >= y0)
                & (centers[:, 1] <= y1)
            ]
        if len(candidates):
            hand = candidates[int(torch.argmax(candidates[:, 4]).item())]
            output[start : start + 21] = hand[6:].detach().cpu().numpy().reshape(21, 3)
            mask[start : start + 21] = (output[start : start + 21, 2] > 0).astype(np.uint8)
            output[start : start + 21][mask[start : start + 21] == 0] = 0.0
            confidence[cls + 1] = float(hand[4])
            boxes[cls + 1] = hand[:4].detach().cpu().numpy()
            valid_mask[cls + 1] = 1
    return output, mask, confidence, boxes, valid_mask


def select_pose59(decoded: dict[str, torch.Tensor]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Select the primary person and corresponding inside-person hands into 59-keypoint layout."""
    output, mask, confidence, _, _ = select_pose59_with_detections(decoded)
    return output, mask, confidence


def _process_batch(
    input_buffers: list[torch.Tensor],
    records: list[tuple[np.ndarray | None, float, int, int, int, int, int, str, str]],
    model: Any,
    target_device: torch.device,
    data: dict[str, Any],
    arrays: dict[str, np.ndarray],
    frame_cursor: int,
    sink: HandRgbSink | None = None,
    stage_timings: dict[str, float] | None = None,
) -> int:
    t_inf = time.perf_counter()
    with (
        torch.inference_mode(),
        torch.autocast(
            target_device.type,
            enabled=target_device.type == "cuda" and bool(data.get("amp", True)),
        ),
    ):
        batch_tensor = torch.stack(input_buffers).to(target_device).float().div_(255.0)
        predictions = decode(
            model(batch_tensor),
            confidence=float(data["nms_confidence"]),
            iou=float(data["nms_iou"]),
            max_detections=int(data["max_detections"]),
        )
    if stage_timings is not None:
        stage_timings["inference_nms"] += time.perf_counter() - t_inf
    for prediction, record in zip(predictions, records, strict=True):
        image, scale, pad_x, pad_y, width, height, index, video_id, split = record
        boxes = np.zeros((3, 4), dtype=np.float32)
        obj_valid = np.zeros(3, dtype=np.uint8)
        if sink is None:
            x, mask, confidence = select_pose59(prediction)
        else:
            x, mask, confidence, lb_boxes, obj_valid = select_pose59_with_detections(prediction)
            boxes = inverse_letterbox_boxes(
                lb_boxes,
                scale=scale,
                pad_x=pad_x,
                pad_y=pad_y,
                width=width,
                height=height,
                valid_mask=obj_valid,
            )
            positive = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
            obj_valid &= positive.astype(np.uint8)
            boxes[obj_valid == 0] = 0.0
            confidence[obj_valid == 0] = 0.0
            arrays["object_bbox_xyxy"][frame_cursor] = boxes
            arrays["object_valid_mask"][frame_cursor] = obj_valid
        inv_kpts = inverse_letterbox_keypoints(
            x, scale=scale, pad_x=pad_x, pad_y=pad_y, width=width, height=height
        )
        inv_kpts[mask == 0] = 0.0
        arrays["X"][frame_cursor] = inv_kpts
        arrays["keypoint_valid_mask"][frame_cursor] = mask
        arrays["object_confidence"][frame_cursor] = confidence
        arrays["frame_idx"][frame_cursor] = index
        if sink is not None:
            assert image is not None
            sink.append_frame(
                image, video_id, split, index, width, height, inv_kpts, boxes, confidence, obj_valid
            )
        frame_cursor += 1
    input_buffers.clear()
    records.clear()
    return frame_cursor


def extract_pose59(
    config: dict[str, Any],
    project_root: Path,
    output_root: Path,
    device: str | None = None,
) -> dict[str, Any]:
    """Execute all-frame YOLO Pose59 feature extraction across dataset splits."""
    start_time = time.perf_counter()
    operation = str(config.get("experiment", {}).get("operation", "extract"))
    hand_rgb = operation == "extract_hand_rgb"
    mode = str(config.get("experiment", {}).get("mode", "full"))
    data = config["data"]
    crop_config = config.get("crops", {"crop_shard_frames": 4096, "margin": 1.3, "quality": 95})
    target_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if bool(data.get("require_cuda", False)):
        if not torch.cuda.is_available():
            raise RuntimeError("extract operation requires CUDA, but CUDA is unavailable")
        if target_device.type != "cuda":
            raise RuntimeError(
                f"extract operation requires CUDA, but requested device is {target_device}"
            )

    rows_by_split: dict[str, list[dict[str, str]]] = {}
    manifest_hashes: dict[str, str] = {}
    for split, manifest in data["manifests"].items():
        path = project_root / str(manifest)
        manifest_hashes[split] = _sha256(path)
        with path.open(newline="", encoding="utf-8") as handle:
            rows_by_split[split] = list(csv.DictReader(handle))

    excluded_defects: list[dict[str, Any]] = []
    if "known_decode_defects" in data:
        raw_defects = data["known_decode_defects"]
        for item in raw_defects:
            defect_id = str(item["video_id"] if isinstance(item, dict) else item).zfill(6)
            defect_reason = str(
                item.get("reason", "known decode defect")
                if isinstance(item, dict)
                else "known decode defect"
            )
            found_matches: list[tuple[str, dict[str, str]]] = []
            for split, rows in rows_by_split.items():
                for row in rows:
                    if str(row["video_id"]).zfill(6) == defect_id:
                        found_matches.append((split, row))

            if len(found_matches) == 0:
                raise RuntimeError(
                    f"known_decode_defects video_id '{defect_id}' was not found in any manifest"
                )
            if len(found_matches) > 1:
                raise RuntimeError(
                    f"known_decode_defects video_id '{defect_id}' matched multiple manifest rows ({len(found_matches)})"
                )

            split, row = found_matches[0]
            rows_by_split[split] = [
                r for r in rows_by_split[split] if str(r["video_id"]).zfill(6) != defect_id
            ]
            excluded_defects.append(
                {
                    "video_id": defect_id,
                    "split": split,
                    "part_id": int(row["part_id"]),
                    "num_frames": int(row["num_frames"]),
                    "reason": defect_reason,
                }
            )

    subset_info: dict[str, Any] | None = None
    if mode == "smoke":
        if "video_ids" in data:
            target_ids = [str(vid).zfill(6) for vid in data["video_ids"]]
            target_set = set(target_ids)
            for split in rows_by_split:
                rows_by_split[split] = [
                    row
                    for row in rows_by_split[split]
                    if str(row["video_id"]).zfill(6) in target_set
                ]
            found_ids = {
                str(row["video_id"]).zfill(6) for rows in rows_by_split.values() for row in rows
            }
            missing_ids = target_set - found_ids
            if missing_ids:
                raise RuntimeError(
                    f"Requested smoke video_ids not found in manifests: {sorted(missing_ids)}"
                )
            subset_info = {
                "type": "video_ids",
                "video_ids": target_ids,
                "videos_per_split": {split: len(rows) for split, rows in rows_by_split.items()},
            }
        elif "subset" in data:
            subset_cfg = data["subset"]
            for split, count in subset_cfg.items():
                if split in rows_by_split:
                    rows_by_split[split] = rows_by_split[split][: int(count)]
            subset_info = {
                "type": "counts",
                "counts": {s: int(c) for s, c in subset_cfg.items()},
                "video_ids": sorted(
                    str(row["video_id"]).zfill(6) for rows in rows_by_split.values() for row in rows
                ),
                "videos_per_split": {split: len(rows) for split, rows in rows_by_split.items()},
            }
        else:
            raise ValueError("Smoke mode extract requires either data.video_ids or data.subset")

    expected_videos = data.get("expected_videos")
    if expected_videos:
        for split, expected_count in expected_videos.items():
            actual_count = len(rows_by_split.get(split, []))
            if actual_count != int(expected_count):
                raise RuntimeError(
                    f"Manifest video count mismatch for split '{split}': expected {expected_count}, got {actual_count}"
                )

    manifest_parts = {int(row["part_id"]) for rows in rows_by_split.values() for row in rows}
    configured_parts = set(map(int, data["parts"]))
    if configured_parts != manifest_parts:
        missing_parts = manifest_parts - configured_parts
        extra_parts = configured_parts - manifest_parts
        err_msg = []
        if missing_parts:
            err_msg.append(f"missing parts from manifests: {sorted(missing_parts)}")
        if extra_parts:
            err_msg.append(f"extra parts not in manifests: {sorted(extra_parts)}")
        raise RuntimeError(f"Configured parts mismatch: {', '.join(err_msg)}")

    actual_frames = sum(int(row["num_frames"]) for rows in rows_by_split.values() for row in rows)
    if "expected_frames" in data and actual_frames != int(data["expected_frames"]):
        raise RuntimeError(
            f"Manifest frame count mismatch: expected {data['expected_frames']}, got {actual_frames}"
        )

    output_root.mkdir(parents=True, exist_ok=True)
    checkpoint = config["checkpoint"]
    checkpoint_path = download_checkpoint(
        repo_id=str(checkpoint["repo_id"]),
        revision=str(checkpoint["revision"]),
        path_in_repo=str(checkpoint["path"]),
        cache_dir=output_root.parent / "cache",
    )

    model = None
    model_audit: dict[str, Any] = {}

    runtime_amp = target_device.type == "cuda" and bool(data.get("amp", True))
    cuda_capability = (
        list(torch.cuda.get_device_capability(target_device))
        if target_device.type == "cuda"
        else None
    )
    env_info = {
        "device": torch.cuda.get_device_name(target_device)
        if target_device.type == "cuda"
        else "cpu",
        "device_type": target_device.type,
        "cuda_capability": cuda_capability,
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "amp_enabled": runtime_amp,
    }

    checkpoint_sha = _sha256(checkpoint_path)
    identity_payload: dict[str, Any] = {
        "mode": mode,
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint": checkpoint,
        "dataset": {"repo": data["dataset_repo"], "revision": data["dataset_revision"]},
        "manifests": manifest_hashes,
        "parts": sorted(map(int, data["parts"])),
        "excluded_defects": excluded_defects,
        "model": config["model"],
        "protocol": {
            k: data[k] for k in ("batch_size", "nms_confidence", "nms_iou", "max_detections")
        },
        "runtime": env_info,
    }
    if hand_rgb:
        identity_payload["crops"] = crop_config
    if mode == "smoke":
        identity_payload["subset"] = subset_info
    identity = hashlib.sha256(json.dumps(identity_payload, sort_keys=True).encode()).hexdigest()
    batch_size = int(data["batch_size"])
    cleanup_part_cache = bool(data.get("cleanup_part_cache", True))
    hf_settings = config.get("integrations", {}).get("hf_artifacts", {})
    resume_from_hf = bool(
        hf_settings.get("enabled", False)
        and (hf_settings.get("resume_from_hf", False) or data.get("resume_from_hf", False))
    )
    part_artifacts: dict[str, dict[str, Any]] = {}
    forward_calls = 0
    stage_timings: dict[str, float] = {
        "download": 0.0,
        "decode": 0.0,
        "letterbox": 0.0,
        "inference_nms": 0.0,
        "crop_encode": 0.0,
        "write": 0.0,
    }

    for part in sorted(map(int, data["parts"])):
        part_rows = [
            (split, row)
            for split, rows in rows_by_split.items()
            for row in rows
            if int(row["part_id"]) == part
        ]
        expected_frames_part = sum(int(row["num_frames"]) for _, row in part_rows)
        npz_path = output_root / f"part_{part:02d}.npz"
        marker = output_root / f"part_{part:02d}.complete.json"
        uploaded = output_root / f"part_{part:02d}.uploaded.json"

        is_completed = False
        part_missing_rates: dict[str, float] = {}
        marker_state: dict[str, Any] = {}
        marker_payload: dict[str, Any] = {}

        if marker.is_file() and npz_path.is_file():
            marker_state = json.loads(marker.read_text(encoding="utf-8"))
            if (
                marker_state.get("identity") != identity
                or marker_state.get("videos") != len(part_rows)
                or marker_state.get("frames") != expected_frames_part
                or marker_state.get("npz_sha256") != _sha256(npz_path)
            ):
                raise RuntimeError(f"Stale or mixed completion marker: {marker}")
            with np.load(npz_path, allow_pickle=False) as loaded_npz:
                arrays_loaded = dict(loaded_npz)
                if hand_rgb:
                    validate_extended_schema(arrays_loaded, validate_pose59_schema)
                    validate_crop_shards(
                        marker_state,
                        output_root,
                        part,
                        expected_frames_part,
                        _sha256,
                        arrays_loaded,
                    )
                else:
                    validate_pose59_schema(arrays_loaded)
                part_missing_rates = {
                    "person": float(np.mean(arrays_loaded["object_confidence"][:, 0] == 0.0)),
                    "left_hand": float(np.mean(arrays_loaded["object_confidence"][:, 1] == 0.0)),
                    "right_hand": float(np.mean(arrays_loaded["object_confidence"][:, 2] == 0.0)),
                }
            is_completed = True
        elif resume_from_hf:
            repo_id = str(hf_settings["repo_id"])
            revision = str(hf_settings.get("revision", "main"))
            marker_fn = f"{config['experiment']['id']}/part_{part:02d}.complete.json"
            npz_fn = f"{config['experiment']['id']}/part_{part:02d}.npz"
            downloaded_marker: Path | None = None
            downloaded_npz: Path | None = None
            try:
                downloaded_marker = download_file(
                    repo_id=repo_id,
                    repo_type="dataset",
                    revision=revision,
                    filename=marker_fn,
                )
                downloaded_npz = download_file(
                    repo_id=repo_id,
                    repo_type="dataset",
                    revision=revision,
                    filename=npz_fn,
                )
            except Exception as exc:
                err_type = type(exc).__name__.lower()
                err_msg = str(exc).lower()
                if (
                    isinstance(exc, FileNotFoundError)
                    or "entrynotfound" in err_type
                    or "404" in err_msg
                    or "not found" in err_msg
                ):
                    downloaded_marker = None
                    downloaded_npz = None
                else:
                    raise

            if downloaded_marker is not None and downloaded_npz is not None:
                marker_state = json.loads(downloaded_marker.read_text(encoding="utf-8"))
                if (
                    marker_state.get("identity") != identity
                    or marker_state.get("videos") != len(part_rows)
                    or marker_state.get("frames") != expected_frames_part
                    or marker_state.get("npz_sha256") != _sha256(downloaded_npz)
                ):
                    raise RuntimeError(
                        f"Remote HF completion marker mismatch for part {part:02d}: "
                        f"identity={marker_state.get('identity')} vs expected={identity}, "
                        f"videos={marker_state.get('videos')} vs expected={len(part_rows)}, "
                        f"frames={marker_state.get('frames')} vs expected={expected_frames_part}"
                    )
                with np.load(downloaded_npz, allow_pickle=False) as loaded_npz:
                    arrays_loaded = dict(loaded_npz)
                    validate_pose59_schema(arrays_loaded)
                    part_missing_rates = {
                        "person": float(np.mean(arrays_loaded["object_confidence"][:, 0] == 0.0)),
                        "left_hand": float(
                            np.mean(arrays_loaded["object_confidence"][:, 1] == 0.0)
                        ),
                        "right_hand": float(
                            np.mean(arrays_loaded["object_confidence"][:, 2] == 0.0)
                        ),
                    }
                tmp_npz = npz_path.with_suffix(".tmp.npz")
                tmp_marker = marker.with_suffix(".tmp.json")
                shutil.copyfile(downloaded_npz, tmp_npz)
                shutil.copyfile(downloaded_marker, tmp_marker)
                os.replace(tmp_npz, npz_path)
                os.replace(tmp_marker, marker)
                _atomic_json(
                    uploaded,
                    {
                        "identity": identity,
                        "npz_sha256": _sha256(npz_path),
                        "uploaded_at": marker_state.get("uploaded_at", time.time()),
                        "resumed_from_hf": True,
                    },
                )
                is_completed = True

        if not is_completed:
            if model is None:
                model, _, _, model_audit = initialize_model(config, target_device, checkpoint_path)
                model.eval()

            shape = (expected_frames_part, 59, 3)
            arrays = {
                "X": np.zeros(shape, np.float32),
                "keypoint_valid_mask": np.zeros((expected_frames_part, 59), np.uint8),
                "object_confidence": np.zeros((expected_frames_part, 3), np.float32),
                "frame_idx": np.zeros(expected_frames_part, np.int32),
                "video_id": np.empty(len(part_rows), "U6"),
                "split": np.empty(len(part_rows), "U5"),
                "frame_width": np.zeros(len(part_rows), np.int32),
                "frame_height": np.zeros(len(part_rows), np.int32),
                "video_frame_offsets": np.zeros(len(part_rows) + 1, np.int64),
            }
            if hand_rgb:
                arrays["object_bbox_xyxy"] = np.zeros((expected_frames_part, 3, 4), np.float32)
                arrays["object_valid_mask"] = np.zeros((expected_frames_part, 3), np.uint8)
            hand_sink = HandRgbSink(output_root, part, config, _sha256) if hand_rgb else None
            frame_cursor = 0
            video_cursor = 0
            input_buffers: list[torch.Tensor] = []
            records: list[tuple[np.ndarray | None, float, int, int, int, int, int, str, str]] = []
            part_cache = output_root / f".video_cache_part_{part:02d}"
            part_cache.mkdir(parents=True, exist_ok=True)
            try:
                t_dl = time.perf_counter()
                archive = download_file(
                    repo_id=str(data["dataset_repo"]),
                    repo_type="dataset",
                    revision=str(data["dataset_revision"]),
                    filename=f"data/split_{part}_front_view.tar",
                    local_dir=part_cache,
                )
                stage_timings["download"] += time.perf_counter() - t_dl
                with tarfile.open(archive, "r:") as tar:
                    for split, row in part_rows:
                        video_id = str(row["video_id"]).zfill(6)
                        source = tar.extractfile(f"split_{part}/front_view/{video_id}.mp4")
                        if source is None:
                            raise RuntimeError(f"Cannot extract video {video_id}")
                        video_file = part_cache / f"{video_id}.mp4"
                        with source, video_file.open("wb") as destination:
                            shutil.copyfileobj(source, destination)

                        capture = cv2.VideoCapture(str(video_file))
                        width: int | None = None
                        height: int | None = None
                        decoded_count = 0
                        try:
                            if not capture.isOpened():
                                raise RuntimeError(f"Cannot open video {video_id}")
                            while True:
                                t_dec = time.perf_counter()
                                ok, frame = capture.read()
                                stage_timings["decode"] += time.perf_counter() - t_dec
                                if not ok:
                                    break
                                frame_height, frame_width = frame.shape[:2]
                                if width is None or height is None:
                                    width, height = frame_width, frame_height
                                elif (frame_width, frame_height) != (width, height):
                                    raise RuntimeError(
                                        f"Frame dimensions mismatch for {video_id}:{decoded_count}"
                                    )
                                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                t_lb = time.perf_counter()
                                image, scale, pad_x, pad_y = _letterbox(
                                    rgb, int(config["model"]["image_size"])
                                )
                                stage_timings["letterbox"] += time.perf_counter() - t_lb
                                assert width is not None and height is not None
                                input_buffers.append(torch.from_numpy(image).permute(2, 0, 1))
                                records.append(
                                    (
                                        rgb if hand_rgb else None,
                                        scale,
                                        pad_x,
                                        pad_y,
                                        width,
                                        height,
                                        decoded_count,
                                        video_id,
                                        split,
                                    )
                                )
                                decoded_count += 1
                                if len(input_buffers) == batch_size:
                                    forward_calls += 1
                                    frame_cursor = _process_batch(
                                        input_buffers,
                                        records,
                                        model,
                                        target_device,
                                        data,
                                        arrays,
                                        frame_cursor,
                                        sink=hand_sink,
                                        stage_timings=stage_timings,
                                    )
                        finally:
                            capture.release()
                            video_file.unlink(missing_ok=True)
                            if not hand_rgb and input_buffers:
                                forward_calls += 1
                                frame_cursor = _process_batch(
                                    input_buffers,
                                    records,
                                    model,
                                    target_device,
                                    data,
                                    arrays,
                                    frame_cursor,
                                    sink=hand_sink,
                                    stage_timings=stage_timings,
                                )

                        if decoded_count != int(row["num_frames"]):
                            raise RuntimeError(
                                f"Decode failed for {video_id}: expected {row['num_frames']}, got {decoded_count}"
                            )
                        assert width is not None and height is not None
                        arrays["video_id"][video_cursor] = video_id
                        arrays["split"][video_cursor] = split
                        arrays["frame_width"][video_cursor] = width
                        arrays["frame_height"][video_cursor] = height
                        arrays["video_frame_offsets"][video_cursor + 1] = frame_cursor + len(
                            records
                        )
                        video_cursor += 1

                if input_buffers:
                    forward_calls += 1
                    frame_cursor = _process_batch(
                        input_buffers,
                        records,
                        model,
                        target_device,
                        data,
                        arrays,
                        frame_cursor,
                        sink=hand_sink,
                        stage_timings=stage_timings,
                    )
                if hand_sink is not None:
                    hand_sink.flush()
                    validate_extended_schema(arrays, validate_pose59_schema)
                    stage_timings["crop_encode"] += hand_sink.crop_time
                    stage_timings["write"] += hand_sink.write_time
                else:
                    validate_pose59_schema(arrays)
                part_missing_rates = {
                    "person": float(np.mean(arrays["object_confidence"][:, 0] == 0.0)),
                    "left_hand": float(np.mean(arrays["object_confidence"][:, 1] == 0.0)),
                    "right_hand": float(np.mean(arrays["object_confidence"][:, 2] == 0.0)),
                }
                t_w = time.perf_counter()
                temporary_npz = npz_path.with_suffix(".tmp.npz")
                np.savez_compressed(temporary_npz, **arrays)
                os.replace(temporary_npz, npz_path)
                stage_timings["write"] += time.perf_counter() - t_w
                marker_payload: dict[str, Any] = {
                    "identity": identity,
                    "videos": len(part_rows),
                    "frames": expected_frames_part,
                    "npz_sha256": _sha256(npz_path),
                }
                if hand_sink is not None:
                    marker_payload.update(hand_sink.marker_fields(arrays))
                _atomic_json(marker, marker_payload)
            finally:
                if cleanup_part_cache:
                    shutil.rmtree(part_cache, ignore_errors=True)

        part_artifacts[f"part_{part:02d}"] = {
            "videos": len(part_rows),
            "frames": expected_frames_part,
            "npz_sha256": _sha256(npz_path),
            "missing_rates": part_missing_rates,
            "marker": marker_state if is_completed else marker_payload,
        }

        settings = config.get("integrations", {}).get("hf_artifacts", {})
        if settings.get("enabled", False):
            need_upload = True
            if uploaded.is_file():
                try:
                    uploaded_state = json.loads(uploaded.read_text(encoding="utf-8"))
                    if uploaded_state.get("identity") == identity and uploaded_state.get(
                        "npz_sha256"
                    ) == _sha256(npz_path):
                        need_upload = False
                except Exception:
                    need_upload = True

            if need_upload:
                ensure_repo(
                    repo_id=str(settings["repo_id"]),
                    repo_type="dataset",
                    private=bool(settings.get("private", True)),
                )
                files_to_upload: dict[str, Path] = {
                    f"{config['experiment']['id']}/part_{part:02d}.npz": npz_path,
                    f"{config['experiment']['id']}/part_{part:02d}.complete.json": marker,
                }
                active_marker = marker_state if is_completed else marker_payload
                if hand_rgb:
                    for shard in active_marker.get("shards", []):
                        shard_name = str(shard["name"])
                        files_to_upload[f"{config['experiment']['id']}/{shard_name}"] = (
                            output_root / shard_name
                        )
                upload_res = _upload_files_with_retry(
                    repo_id=str(settings["repo_id"]),
                    repo_type="dataset",
                    files=files_to_upload,
                    commit_message=f"extract {'Pose59 HandRGB' if hand_rgb else 'Pose59'} part {part:02d}",
                )
                receipt_payload: dict[str, Any] = {
                    "identity": identity,
                    "npz_sha256": _sha256(npz_path),
                    "uploaded_at": time.time(),
                }
                if upload_res:
                    receipt_payload["revision"] = upload_res
                _atomic_json(uploaded, receipt_payload)

    provenance_payload: dict[str, Any] = {
        "operation": operation,
        "mode": mode,
        "identity": identity,
        "identity_payload": identity_payload,
        "checkpoint": {
            **checkpoint,
            "sha256": checkpoint_sha,
        },
        "dataset": {
            "repo": data["dataset_repo"],
            "revision": data["dataset_revision"],
        },
        "manifests": {
            split: {
                "path": str(manifest),
                "sha256": manifest_hashes[split],
            }
            for split, manifest in data["manifests"].items()
        },
        "parts": sorted(map(int, data["parts"])),
        "excluded_defects": excluded_defects,
        "model": {
            "name": config["model"]["name"],
            "architecture": config["model"].get("architecture"),
            "ultralytics_version": config["model"].get("ultralytics_version"),
            "person_keypoints": int(config["model"]["person_keypoints"]),
            "hand_keypoints": int(config["model"]["hand_keypoints"]),
            "image_size": int(config["model"]["image_size"]),
        },
        "model_audit": model_audit,
        "protocol": {
            "batch_size": int(data["batch_size"]),
            "nms_confidence": float(data["nms_confidence"]),
            "nms_iou": float(data["nms_iou"]),
            "max_detections": int(data["max_detections"]),
            "normalization": "coordinates in [0, 1] normalized by original frame dimensions with inverse letterbox",
            "selection": "highest confidence primary person + hands filtered within person bounding box",
        },
        "runtime": env_info,
        "part_artifacts": part_artifacts,
    }
    if mode == "smoke":
        provenance_payload["subset"] = subset_info

    total_frames = sum(a["frames"] for a in part_artifacts.values())
    total_missing_person = sum(
        a["missing_rates"]["person"] * a["frames"] for a in part_artifacts.values()
    ) / max(1, total_frames)
    total_missing_left = sum(
        a["missing_rates"]["left_hand"] * a["frames"] for a in part_artifacts.values()
    ) / max(1, total_frames)
    total_missing_right = sum(
        a["missing_rates"]["right_hand"] * a["frames"] for a in part_artifacts.values()
    ) / max(1, total_frames)

    summary = {
        "status": "completed",
        "operation": operation,
        "mode": mode,
        "identity": identity,
        "identity_payload": identity_payload,
        "provenance": provenance_payload,
        "runtime": env_info,
        "counts": {
            split: {"videos": len(rows), "frames": sum(int(row["num_frames"]) for row in rows)}
            for split, rows in rows_by_split.items()
        },
        "frames": actual_frames,
        "videos": sum(map(len, rows_by_split.values())),
        "excluded_defects": excluded_defects,
        "missing_rates": {
            "person": total_missing_person,
            "left_hand": total_missing_left,
            "right_hand": total_missing_right,
        },
    }
    if hand_rgb:
        total_crops_left = int(
            sum(
                p.get("marker", {}).get("crop_counts", {}).get("left", 0)
                for p in part_artifacts.values()
            )
        )
        total_crops_right = int(
            sum(
                p.get("marker", {}).get("crop_counts", {}).get("right", 0)
                for p in part_artifacts.values()
            )
        )
        total_crop_bytes = int(
            sum(
                shard.get("bytes", 0)
                for p in part_artifacts.values()
                for shard in p.get("marker", {}).get("shards", [])
            )
        )
        stage_timings["total"] = time.perf_counter() - start_time
        summary["timings"] = stage_timings
        summary["crop_counts"] = {
            "left": total_crops_left,
            "right": total_crops_right,
            "total": total_crops_left + total_crops_right,
        }
        summary["crop_bytes"] = total_crop_bytes
        summary["forward_calls"] = forward_calls
        summary["mean_effective_batch"] = actual_frames / max(1, forward_calls)
    manifest_path = output_root / "manifest.json"
    summary_path = output_root / "summary.json"
    _atomic_json(manifest_path, summary)
    _atomic_json(summary_path, summary)

    settings = config.get("integrations", {}).get("hf_artifacts", {})
    if settings.get("enabled", False):
        final_uploaded = output_root / "final.uploaded.json"
        need_final_upload = True
        if final_uploaded.is_file():
            try:
                final_state = json.loads(final_uploaded.read_text(encoding="utf-8"))
                need_final_upload = not (
                    final_state.get("identity") == identity
                    and final_state.get("summary_sha256") == _sha256(summary_path)
                )
            except Exception:
                need_final_upload = True

        if need_final_upload:
            ensure_repo(
                repo_id=str(settings["repo_id"]),
                repo_type="dataset",
                private=bool(settings.get("private", True)),
            )
            _upload_files_with_retry(
                repo_id=str(settings["repo_id"]),
                repo_type="dataset",
                files={
                    f"{config['experiment']['id']}/manifest.json": manifest_path,
                    f"{config['experiment']['id']}/summary.json": summary_path,
                },
                commit_message="extract Pose59 complete summary and manifest",
            )
            _atomic_json(
                final_uploaded,
                {
                    "identity": identity,
                    "summary_sha256": _sha256(summary_path),
                    "uploaded_at": time.time(),
                },
            )

    elapsed_seconds = time.perf_counter() - start_time
    return {**summary, "seconds": elapsed_seconds}
