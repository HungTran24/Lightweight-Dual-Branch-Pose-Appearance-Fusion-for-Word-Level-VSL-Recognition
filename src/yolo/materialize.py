"""Materialize canonical VSL400 targets for YOLO dual-head training."""

from __future__ import annotations

import json
import os
import shutil
import tarfile
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.frame_sampling.uniform import uniform_indices
from src.integrations.huggingface import download_file
from src.rtmw.diagnostics import (
    COCO_WHOLEBODY_LEFT_HAND_SLICE,
    COCO_WHOLEBODY_RIGHT_HAND_SLICE,
)
from src.rtmw.schema import RawRTMWShard

from .selection import _identity, canonical_selection
from .transforms import _labels, _letterbox, _open_arrays


def materialize(
    output: Path,
    cache: Path,
    workspace_root: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Decode Uniform-N frames while retaining only one video TAR at a time."""
    data = config["data"]
    image_size = int(config["model"]["image_size"])
    frames = int(data["frames_per_video"])
    manifest_dir = workspace_root / data["canonical_manifest_dir"]
    selected = canonical_selection(
        manifest_dir,
        known_decode_defect=str(data["known_decode_defect"]),
    )
    expected = {"train": int(data["train_videos"]), "val": int(data["val_videos"])}
    if {split: len(rows) for split, rows in selected.items()} != expected:
        raise RuntimeError("Canonical pilot selection count mismatch")
    if {row["signer_id"] for row in selected["train"]} & {
        row["signer_id"] for row in selected["val"]
    }:
        raise RuntimeError("Train and validation signers overlap")
    if config.get("experiment", {}).get("operation") == "evaluate":
        selected = {"val": selected["val"]}
        expected = {"val": expected["val"]}

    raw_manifest = json.loads((workspace_root / data["rtmw_manifest"]).read_text(encoding="utf-8"))
    shard_for = {
        str(video_id).zfill(6): shard["shard_id"]
        for shard in raw_manifest["shards"]
        for video_id in shard["video_ids"]
    }
    identity = _identity(config, selected)
    output.mkdir(parents=True, exist_ok=True)
    materialize_started = time.perf_counter()
    arrays = _open_arrays(output, selected, frames)
    by_part: dict[int, list[tuple[str, str, dict[str, Any]]]] = {
        int(part): [] for part in data["parts"]
    }
    for split, rows in selected.items():
        for row in rows:
            by_part[row["part_id"]].append((shard_for[row["video_id"]], split, row))
    for rows in by_part.values():
        rows.sort(key=lambda item: (item[0], item[2]["video_id"]))

    for part, rows in by_part.items():
        marker = output / f"part_{part:02d}.complete.json"
        if marker.is_file():
            state = json.loads(marker.read_text(encoding="utf-8"))
            if state != {"identity": identity, "videos": len(rows), "images": len(rows) * frames}:
                raise RuntimeError(f"Invalid completion marker: {marker}")
            if not all(
                (
                    output / split / "images" / f"{row['selection_index']:05d}_{frame:02d}.jpg"
                ).is_file()
                for _, split, row in rows
                for frame in range(frames)
            ):
                raise RuntimeError(f"Completed part {part} has missing images")
            print(json.dumps({"part": part, "status": "already_complete"}), flush=True)
            continue

        started = time.perf_counter()
        part_dir = cache / "video" / f"part_{part:02d}"
        part_dir.mkdir(parents=True, exist_ok=True)
        extracted = output / ".current_video.mp4"
        try:
            archive = download_file(
                repo_id=data["dataset_repo"],
                repo_type="dataset",
                revision=data["dataset_revision"],
                filename=f"data/split_{part}_front_view.tar",
                local_dir=part_dir,
            )
            with tarfile.open(archive, "r:") as tar:
                current_shard_id = ""
                shard = None
                for shard_id, split, row in rows:
                    if shard_id != current_shard_id:
                        raw_path = download_file(
                            repo_id=data["rtmw_repo"],
                            repo_type="dataset",
                            revision=data["rtmw_revision"],
                            filename=f"{data['rtmw_prefix']}/data/{shard_id}.npz",
                            cache_dir=cache / "rtmw",
                        )
                        shard = RawRTMWShard.load_npz(raw_path)
                        current_shard_id = shard_id
                    if shard is None:
                        raise RuntimeError(f"Raw shard was not loaded: {shard_id}")
                    video_id = row["video_id"]
                    matches = np.flatnonzero(shard.video_id.astype(str) == video_id)
                    if len(matches) != 1:
                        raise RuntimeError(f"RTMW join failed for {video_id}")
                    video_position = int(matches[0])
                    expected_width = int(shard.frame_width[video_position])
                    expected_height = int(shard.frame_height[video_position])
                    raw_start = int(shard.video_frame_offsets[video_position])
                    raw_stop = int(shard.video_frame_offsets[video_position + 1])
                    if raw_stop - raw_start != row["num_frames"]:
                        raise RuntimeError(f"Frame count mismatch for {video_id}")
                    raw_frame_indices = shard.frame_idx[raw_start:raw_stop]
                    if not np.array_equal(
                        raw_frame_indices, np.arange(row["num_frames"], dtype=np.int32)
                    ):
                        raise RuntimeError(
                            f"RTMW frame indices are not decoder-sequential for {video_id}"
                        )
                    if row["num_frames"] < frames:
                        raise RuntimeError(f"Video {video_id} has fewer than {frames} frames")
                    source = tar.extractfile(f"split_{part}/front_view/{video_id}.mp4")
                    if source is None:
                        raise RuntimeError(f"Cannot extract video {video_id}")
                    with source, extracted.open("wb") as destination:
                        shutil.copyfileobj(source, destination, length=4 * 1024 * 1024)
                    positions = uniform_indices(row["num_frames"], frames)
                    wanted = {int(position): index for index, position in enumerate(positions)}
                    capture = cv2.VideoCapture(str(extracted))
                    try:
                        if not capture.isOpened():
                            raise RuntimeError(f"Cannot open video {video_id}")
                        for frame_index in range(int(positions[-1]) + 1):
                            ok, frame = capture.read()
                            if not ok:
                                raise RuntimeError(f"Decode failed for {video_id}:{frame_index}")
                            if frame_index not in wanted:
                                continue
                            if (frame.shape[1], frame.shape[0]) != (
                                expected_width,
                                expected_height,
                            ):
                                raise RuntimeError(f"Frame dimensions mismatch for {video_id}")
                            frame_number = wanted[frame_index]
                            image, scale, pad_x, pad_y = _letterbox(frame, image_size)
                            index = row["selection_index"] * frames + frame_number
                            image_path = (
                                output
                                / split
                                / "images"
                                / f"{row['selection_index']:05d}_{frame_number:02d}.jpg"
                            )
                            if not cv2.imwrite(str(image_path), image):
                                raise RuntimeError(f"Cannot write {image_path}")
                            boxes, classes, person_keypoints, hand_keypoints = _labels(
                                shard,
                                raw_start + frame_index,
                                frame.shape[1],
                                frame.shape[0],
                                float(data["keypoint_score_threshold"]),
                                slice(0, 17),
                                (COCO_WHOLEBODY_LEFT_HAND_SLICE, COCO_WHOLEBODY_RIGHT_HAND_SLICE),
                            )
                            valid_boxes = classes >= 0
                            boxes[valid_boxes, 0] = boxes[valid_boxes, 0] * scale + pad_x
                            boxes[valid_boxes, 2] = boxes[valid_boxes, 2] * scale + pad_x
                            boxes[valid_boxes, 1] = boxes[valid_boxes, 1] * scale + pad_y
                            boxes[valid_boxes, 3] = boxes[valid_boxes, 3] * scale + pad_y
                            visible_person = person_keypoints[:, 2] != 0
                            person_keypoints[visible_person, 0] = person_keypoints[visible_person, 0] * scale + pad_x
                            person_keypoints[visible_person, 1] = person_keypoints[visible_person, 1] * scale + pad_y
                            visible_hands = hand_keypoints[..., 2] != 0
                            hand_keypoints[..., 0][visible_hands] = (
                                hand_keypoints[..., 0][visible_hands] * scale + pad_x
                            )
                            hand_keypoints[..., 1][visible_hands] = (
                                hand_keypoints[..., 1][visible_hands] * scale + pad_y
                            )
                            arrays[f"{split}_boxes"][index] = boxes
                            arrays[f"{split}_classes"][index] = classes
                            arrays[f"{split}_person_keypoints"][index] = person_keypoints
                            arrays[f"{split}_hand_keypoints"][index] = hand_keypoints
                    finally:
                        capture.release()
                        extracted.unlink(missing_ok=True)
            for array in arrays.values():
                array.flush()
            state = {"identity": identity, "videos": len(rows), "images": len(rows) * frames}
            temporary = marker.with_suffix(".tmp")
            temporary.write_text(json.dumps(state), encoding="utf-8")
            os.replace(temporary, marker)
            print(
                json.dumps(
                    {
                        "part": part,
                        **state,
                        "elapsed_seconds": round(time.perf_counter() - started, 1),
                    }
                ),
                flush=True,
            )
        finally:
            extracted.unlink(missing_ok=True)
            shutil.rmtree(part_dir, ignore_errors=True)

    inventory = {
        "identity": identity,
        "splits": {
            split: [
                {
                    key: row[key]
                    for key in (
                        "video_id",
                        "signer_id",
                        "class_id",
                        "part_id",
                        "num_frames",
                        "selection_index",
                    )
                }
                for row in rows
            ]
            for split, rows in selected.items()
        },
    }
    (output / "selection_inventory.json").write_text(
        json.dumps(inventory, indent=2), encoding="utf-8"
    )
    summary = {
        "status": "materialized",
        "identity": identity,
        "train_images": expected.get("train", 0) * frames,
        "val_images": expected.get("val", 0) * frames,
        "dataset_revision": data["dataset_revision"],
        "rtmw_revision": data["rtmw_revision"],
        "seconds_this_invocation": time.perf_counter() - materialize_started,
    }
    (output / "materialization.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
