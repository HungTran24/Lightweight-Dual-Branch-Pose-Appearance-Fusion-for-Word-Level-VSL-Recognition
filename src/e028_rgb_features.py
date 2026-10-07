"""Coarse temporal hand-RGB extraction for U24 and U30 sampling."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pyarrow.parquet as pq
import torch

from .artifacts import atomic_json, sha256
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


def temporal_bin_map(
    frame_indices: np.ndarray, valid_mask: np.ndarray, *, sampling_slots: int = 30
) -> dict[int, int]:
    """Map each valid absolute frame index to one of three equal sampling-slot bins."""
    if sampling_slots not in {24, 30}:
        raise ValueError("Temporal RGB sampling_slots must be 24 or 30")
    if frame_indices.shape != (sampling_slots,) or valid_mask.shape != (sampling_slots,):
        raise ValueError(f"U{sampling_slots} indices and mask must each have {sampling_slots} slots")
    bin_width = sampling_slots // 3
    mapping: dict[int, int] = {}
    for slot in range(sampling_slots):
        if int(valid_mask[slot]) == 0:
            continue
        frame_idx = int(frame_indices[slot])
        if frame_idx < 0:
            raise ValueError(f"Valid U{sampling_slots} slot must have a non-negative frame index")
        if frame_idx in mapping:
            raise ValueError(f"Duplicate valid U{sampling_slots} frame index: {frame_idx}")
        mapping[frame_idx] = slot // bin_width
    return mapping


def _flush(
    batch: list[tuple[int, int, int, torch.Tensor]],
    model: Any,
    device: torch.device,
    sums: np.ndarray,
    counts: np.ndarray,
) -> None:
    if not batch:
        return
    tensors = torch.stack([item[3] for item in batch]).to(device)
    with torch.inference_mode():
        embeddings = run_feature_forward(model, tensors).cpu().numpy().astype(np.float32)
    for embedding, (video, hand, bin_idx, _) in zip(embeddings, batch, strict=True):
        sums[video, hand, bin_idx] += embedding
        counts[video, hand, bin_idx] += 1
    batch.clear()


def _validate_output(
    path: Path, expected_ids: list[str], *, sampling_slots: int = 30
) -> None:
    if sampling_slots not in {24, 30}:
        raise ValueError("Temporal RGB sampling_slots must be 24 or 30")
    bin_width = sampling_slots // 3
    with np.load(path, allow_pickle=False) as loaded:
        count = len(expected_ids)
        video_ids = loaded["video_id"]
        if video_ids.dtype != np.dtype("U6") or [str(v).zfill(6) for v in video_ids] != expected_ids:
            raise ValueError(f"{path.name} video IDs mismatch, dtype, or order")
        split = loaded["split"]
        expected_split = path.stem
        if (
            split.shape != (count,)
            or split.dtype != np.dtype("U5")
            or not np.all(split == expected_split)
        ):
            raise ValueError(f"{path.name} split must be U5 [N] filled with {expected_split}")
        selected = loaded["selected_frame_indices"]
        if selected.shape != (count, sampling_slots) or selected.dtype != np.int32:
            raise ValueError(
                f"{path.name} selected indices must be int32 [N,{sampling_slots}]"
            )
        valid = loaded["hand_valid_count"]
        features = loaded["X_rgb"]
        if (
            valid.shape != (count, 2, 3)
            or valid.dtype != np.uint8
            or np.any(valid > bin_width)
        ):
            raise ValueError(
                f"{path.name} hand_valid_count must be uint8 [N,2,3] "
                f"bounded by {bin_width}"
            )
        if features.shape != (count, 2, 3, 576) or features.dtype != np.float16:
            raise ValueError(f"{path.name} X_rgb must be float16 [N,2,3,576]")
        if not np.isfinite(features).all() or np.any(features[valid == 0] != 0):
            raise ValueError(f"{path.name} contains invalid or non-zero missing-bin features")


def run_e028_rgb_features(
    config: dict[str, Any],
    project_root: Path,
    *,
    encoder_factory: Any | None = None,
    device: torch.device | str | None = None,
) -> dict[str, Any]:
    """Extract frozen MobileNet embeddings pooled over three sampling-slot bins."""
    import torchvision

    sampling_slots = int(config.get("sampling_slots", 30))
    source_key = f"u{sampling_slots}_source"
    if source_key not in config:
        raise ValueError(f"Temporal RGB config must define {source_key}")
    bin_width = sampling_slots // 3
    temporal_bins = [
        [start, start + bin_width - 1]
        for start in range(0, sampling_slots, bin_width)
    ]
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
            raise ValueError(f"Duplicate temporal RGB video IDs in {split}")
        source_sets[split] = set(source_ids)
        if source_sets[split] != set(clean_ids[split]):
            raise ValueError(f"temporal RGB {split} IDs must exactly cover the clean manifest")
        source_index = {video_id: i for i, video_id in enumerate(source_ids)}
        for video_id in clean_ids[split]:
            i = source_index[video_id]
            indices = uniform[split]["frame_indices"][i].astype(np.int32)
            metadata[video_id] = {
                "split": split,
                "indices": indices,
                "frame_bins": temporal_bin_map(
                    indices,
                    uniform[split]["valid_frame_mask"][i],
                    sampling_slots=sampling_slots,
                ),
            }
    if "train" in source_sets and "val" in source_sets and source_sets["train"] & source_sets["val"]:
        raise ValueError("temporal RGB train and val video IDs overlap")

    all_records = [row for split in split_names for row in records[split]]
    video_ids = [row.video_id for row in all_records]
    video_to_index = {video_id: i for i, video_id in enumerate(video_ids)}
    sums = np.zeros((len(video_ids), 2, 3, 576), dtype=np.float32)
    counts = np.zeros((len(video_ids), 2, 3), dtype=np.uint8)
    seen: set[tuple[str, str, int]] = set()

    target_device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model = (
        encoder_factory(target_device)
        if encoder_factory is not None
        else build_mobilenet_extractor(target_device)
    )
    batch_size = int(config["data"]["batch_size"])
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
                frame_bins = metadata[video_id]["frame_bins"]
                if frame_idx not in frame_bins:
                    continue
                key = (split, video_id, frame_idx)
                if key in seen:
                    raise ValueError(f"Duplicate E021 frame key: {key}")
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
                    batch.append((video_to_index[video_id], hand, frame_bins[frame_idx], preprocess_crop_bgr(crop)))
                    if len(batch) == batch_size:
                        _flush(batch, model, target_device, sums, counts)
    _flush(batch, model, target_device, sums, counts)

    expected = {
        (metadata[video_id]["split"], video_id, frame_idx)
        for video_id in video_ids
        for frame_idx in metadata[video_id]["frame_bins"]
    }
    if seen != expected:
        raise ValueError(f"E021/temporal RGB key coverage mismatch: missing={len(expected - seen)}")

    features = np.zeros_like(sums, dtype=np.float16)
    present = counts > 0
    features[present] = (sums[present] / counts[present][:, None]).astype(np.float16)
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
            hand_valid_count=counts[indices],
            X_rgb=features[indices],
        )
        _validate_output(path, clean_ids[split], sampling_slots=sampling_slots)
        files[split] = {"path": str(path.relative_to(output_root)), "sha256": sha256(path), "videos": len(indices)}

    identity_payload = json.dumps(
        {
            "code_sha256": sha256(Path(__file__).resolve()),
            "e021_source": e021,
            source_key: config[source_key],
            "manifests": {
                split: sha256(project_root / config["manifests"][split]) for split in split_names
            },
            "preprocessing": LOCKED_PREPROCESSING,
            "torch_base": torch.__version__.split("+")[0],
            "torchvision_base": torchvision.__version__.split("+")[0],
            "temporal_bins": temporal_bins,
        },
        sort_keys=True,
    )
    receipt = {
        "status": "completed",
        "experiment_id": config["experiment"]["id"],
        "extraction_identity": hashlib.sha256(identity_payload.encode()).hexdigest(),
        "e021_source": e021,
        source_key: config[source_key],
        "sampling_slots": sampling_slots,
        "preprocessing": LOCKED_PREPROCESSING,
        "runtime_versions": {
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
        },
        "temporal_bins": temporal_bins,
        "files": files,
    }
    atomic_json(output_root / "receipt.json", receipt)
    atomic_json(output_root / "summary.json", receipt)
    return receipt
