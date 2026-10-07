"""Per-frame hand-RGB feature extraction for U30 sampling."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pyarrow.parquet as pq
import torch

from .artifacts import atomic_json, sha256, upload_with_retry
from .e026_audit import load_dev_manifests
from .e026_rgb_features import (
    LOCKED_PREPROCESSING,
    PARQUET_COLUMNS,
    _load_uniform_splits,
    atomic_npz,
    build_mobilenet_extractor,
    preprocess_crop_bgr,
    run_feature_forward,
)
from .integrations.huggingface import download_file


def frame_slot_map(
    frame_indices: np.ndarray, valid_mask: np.ndarray, *, sampling_slots: int = 30
) -> dict[int, int]:
    """Map each valid absolute frame index to its sampling slot index (0..T-1)."""
    if frame_indices.shape != (sampling_slots,) or valid_mask.shape != (sampling_slots,):
        raise ValueError(f"U{sampling_slots} indices and mask must have shape ({sampling_slots},)")
    mapping: dict[int, int] = {}
    for slot in range(sampling_slots):
        if int(valid_mask[slot]) == 0:
            continue
        frame_idx = int(frame_indices[slot])
        if frame_idx < 0:
            continue
        if frame_idx in mapping:
            raise ValueError(f"Duplicate valid U{sampling_slots} frame index: {frame_idx}")
        mapping[frame_idx] = slot
    return mapping


def _flush_frame_batch(
    batch: list[tuple[int, int, int, torch.Tensor]],
    model: Any,
    device: torch.device,
    features: np.ndarray,
    valid_mask: np.ndarray,
) -> None:
    if not batch:
        return
    tensors = torch.stack([item[3] for item in batch]).to(device)
    with torch.inference_mode():
        embeddings = run_feature_forward(model, tensors).cpu().numpy().astype(np.float16)
    for embedding, (video_idx, hand_idx, slot_idx, _) in zip(embeddings, batch, strict=True):
        features[video_idx, slot_idx, hand_idx] = embedding
        valid_mask[video_idx, slot_idx, hand_idx] = 1
    batch.clear()


def run_e038_rgb_frame_features(
    config: dict[str, Any],
    project_root: Path,
    *,
    encoder_factory: Any | None = None,
    device: torch.device | str | None = None,
) -> dict[str, Any]:
    """Extract per-frame frozen MobileNet embeddings [N, T, 2, 576]."""
    sampling_slots = int(config.get("sampling_slots", 30))
    source_key = f"u{sampling_slots}_source"
    if source_key not in config:
        raise ValueError(f"Config must define {source_key}")

    split_names = tuple(config["manifests"])
    records = load_dev_manifests(config, project_root, splits=split_names)
    clean_ids = {split: [row.video_id for row in records[split]] for split in split_names}
    cache_dir = project_root / "cache" / "hf"
    uniform = _load_uniform_splits(config[source_key], cache_dir, split_names)

    metadata: dict[str, dict[str, Any]] = {}
    source_sets: dict[str, set[str]] = {}
    for split in split_names:
        source_ids = [str(v).zfill(6) for v in uniform[split]["video_id"]]
        if len(source_ids) != len(set(source_ids)):
            raise ValueError(f"Duplicate video IDs in {split}")
        source_sets[split] = set(source_ids)
        if source_sets[split] != set(clean_ids[split]):
            raise ValueError(f"{split} IDs must exactly cover the clean manifest")
        source_index = {video_id: i for i, video_id in enumerate(source_ids)}
        for video_id in clean_ids[split]:
            i = source_index[video_id]
            indices = uniform[split]["frame_indices"][i].astype(np.int32)
            valid_m = uniform[split]["valid_frame_mask"][i]
            metadata[video_id] = {
                "split": split,
                "indices": indices,
                "valid_mask": valid_m,
                "frame_slot_map": frame_slot_map(
                    indices,
                    valid_m,
                    sampling_slots=sampling_slots,
                ),
            }

    all_records = [row for split in split_names for row in records[split]]
    video_ids = [row.video_id for row in all_records]
    video_to_index = {video_id: i for i, video_id in enumerate(video_ids)}

    # features: [N, T, 2, 576]
    features = np.zeros((len(video_ids), sampling_slots, 2, 576), dtype=np.float16)
    hand_valid = np.zeros((len(video_ids), sampling_slots, 2), dtype=np.uint8)
    seen: set[tuple[str, str, int]] = set()

    target_device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model = (
        encoder_factory(target_device)
        if encoder_factory is not None
        else build_mobilenet_extractor(target_device)
    )
    batch_size = int(config.get("data", {}).get("batch_size", 256))
    batch: list[tuple[int, int, int, torch.Tensor]] = []
    e021 = config["e021_source"]
    prefix = str(e021["prefix"]).rstrip("/")

    for part in sorted(map(int, e021["parts"])):
        marker_path = download_file(
            repo_id=e021["repo_id"], repo_type="dataset", revision=e021["revision"],
            filename=f"{prefix}/part_{part:02d}.complete.json", cache_dir=cache_dir,
        )
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if marker.get("identity") != e021["identity"]:
            raise ValueError(f"E021 identity mismatch for part {part:02d}")
        for shard in marker["shards"]:
            shard_path = download_file(
                repo_id=e021["repo_id"], repo_type="dataset", revision=e021["revision"],
                filename=f"{prefix}/{shard['name']}", cache_dir=cache_dir,
            )
            if sha256(shard_path) != shard["sha256"]:
                raise ValueError(f"E021 shard checksum mismatch: {shard['name']}")
            table = pq.read_table(shard_path, columns=PARQUET_COLUMNS)
            columns = {name: table.column(name).to_pylist() for name in PARQUET_COLUMNS}
            for row in range(table.num_rows):
                video_id = str(columns["video_id"][row]).zfill(6)
                if video_id not in video_to_index:
                    continue
                split = str(columns["split"][row])
                frame_idx = int(columns["frame_idx"][row])
                if split != metadata[video_id]["split"]:
                    raise ValueError(f"Split mismatch for {video_id}")
                slot_map = metadata[video_id]["frame_slot_map"]
                if frame_idx not in slot_map:
                    continue
                slot_idx = slot_map[frame_idx]
                key = (split, video_id, frame_idx)
                if key in seen:
                    raise ValueError(f"Duplicate frame key: {key}")
                seen.add(key)
                for hand, side in enumerate(("left", "right")):
                    if int(columns[f"{side}_valid"][row]) == 0:
                        continue
                    jpeg = columns[f"{side}_jpeg"][row]
                    if jpeg is None:
                        raise ValueError(f"Valid {side} hand has null JPEG at {key}")
                    crop = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
                    if crop is None:
                        raise ValueError(f"Failed JPEG decode at {key}")
                    batch.append((video_to_index[video_id], hand, slot_idx, preprocess_crop_bgr(crop)))
                    if len(batch) >= batch_size:
                        _flush_frame_batch(batch, model, target_device, features, hand_valid)
    _flush_frame_batch(batch, model, target_device, features, hand_valid)

    output_root = project_root / config["output"]["root"]
    data_dir = output_root / "data"
    files: dict[str, dict[str, Any]] = {}
    for split in split_names:
        indices = [i for i, row in enumerate(all_records) if row.split == split]
        path = data_dir / f"{split}.npz"
        atomic_npz(
            path,
            video_id=np.array([video_ids[i] for i in indices], dtype="U6"),
            split=np.array([split] * len(indices), dtype="U5"),
            selected_frame_indices=np.stack([metadata[video_ids[i]]["indices"] for i in indices]),
            valid_frame_mask=np.stack([metadata[video_ids[i]]["valid_mask"] for i in indices]),
            hand_valid_mask=hand_valid[indices],
            X_rgb=features[indices],
        )
        files[split] = {"path": str(path.relative_to(output_root)), "sha256": sha256(path), "videos": len(indices)}

    receipt = {
        "status": "completed",
        "experiment_id": config["experiment"]["id"],
        "sampling_slots": sampling_slots,
        "preprocessing": LOCKED_PREPROCESSING,
        "files": files,
    }
    atomic_json(output_root / "receipt.json", receipt)
    atomic_json(output_root / "summary.json", receipt)

    hf = config.get("integrations", {}).get("hf_artifacts", {})
    if isinstance(hf, dict) and hf.get("enabled", False):
        prefix = str(hf.get("prefix", config["experiment"]["id"])).strip("/")
        upload_files = {
            f"{prefix}/data/{split}.npz": output_root / files[split]["path"]
            for split in split_names
        }
        upload_files[f"{prefix}/receipt.json"] = output_root / "receipt.json"
        revision = upload_with_retry(
            repo_id=str(hf["repo_id"]),
            repo_type=str(hf.get("repo_type", "model")),
            files=upload_files,
            commit_message=f"artifact: {config['experiment']['id']} frozen RGB features",
        )
        receipt["hf_artifact"] = {
            "repo_id": hf["repo_id"],
            "repo_type": hf.get("repo_type", "model"),
            "prefix": prefix,
            "revision": revision,
        }
        atomic_json(output_root / "receipt.json", receipt)
        atomic_json(output_root / "summary.json", receipt)
    return receipt
