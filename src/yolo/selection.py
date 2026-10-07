"""Canonical VSL400 video selection and dataset identity hashing."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any


def canonical_selection(
    manifest_dir: Path,
    *,
    known_decode_defect: str,
) -> dict[str, list[dict[str, Any]]]:
    """Choose the first usable video for each signer/class pair."""
    selected: dict[str, list[dict[str, Any]]] = {"train": [], "val": []}
    for split in selected:
        with (manifest_dir / f"{split}.csv").open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        seen: set[tuple[str, str]] = set()
        for source in sorted(rows, key=lambda row: str(row["video_id"]).zfill(6)):
            video_id = str(source["video_id"]).zfill(6)
            if source["split"] != split or video_id == known_decode_defect:
                continue
            key = str(source["signer_id"]).zfill(3), str(source["class_id"])
            if key in seen:
                continue
            seen.add(key)
            row = dict(source)
            row.update(
                video_id=video_id,
                signer_id=key[0],
                part_id=int(source["part_id"]),
                num_frames=int(source["num_frames"]),
            )
            selected[split].append(row)
    for rows in selected.values():
        rows.sort(key=lambda row: row["video_id"])
        for index, row in enumerate(rows):
            row["selection_index"] = index
    return selected


def _identity(config: dict[str, Any], selected: dict[str, list[dict[str, Any]]]) -> str:
    model = config["model"]
    model_identity = {"image_size": model["image_size"]}
    if "person_keypoints" in model:
        model_identity.update(
            person_keypoints=model["person_keypoints"],
            hand_keypoints=model["hand_keypoints"],
        )
    else:
        model_identity["pose_keypoints"] = model["pose_keypoints"]
    payload = {
        "data": config["data"],
        "model": model_identity,
        "videos": {split: [row["video_id"] for row in rows] for split, rows in selected.items()},
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
