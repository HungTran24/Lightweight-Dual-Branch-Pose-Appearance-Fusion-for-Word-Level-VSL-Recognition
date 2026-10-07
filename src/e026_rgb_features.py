"""E026 One-pass U24 and U30 RGB feature extraction using frozen MobileNetV3-Small."""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pyarrow.parquet as pq
import torch
from torchvision.transforms.functional import InterpolationMode, normalize, resize

from .artifacts import atomic_json, sha256
from .e026_audit import DEV_SPLITS, load_dev_manifests
from .integrations.huggingface import download_file

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

LOCKED_PREPROCESSING = {
    "backbone": "mobilenet_v3_small",
    "weights": "IMAGENET1K_V1",
    "feature_dim": 576,
    "target_size": 128,
    "interpolation": "bilinear",
    "antialias": True,
    "mean": IMAGENET_MEAN,
    "std": IMAGENET_STD,
    "output_dtype": "float16",
}

PARQUET_COLUMNS = [
    "split",
    "video_id",
    "frame_idx",
    "left_valid",
    "right_valid",
    "left_jpeg",
    "right_jpeg",
]


def atomic_npz(path: Path | str, **arrays: Any) -> Path:
    """Atomically write numpy NPZ archive via sibling temporary file."""
    dest = Path(path).resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)
    temp = dest.parent / f".{dest.name}.tmp"
    with temp.open("wb") as handle:
        np.savez(handle, **arrays)
    os.replace(temp, dest)
    return dest


def compute_extraction_identity(
    config: dict[str, Any],
    manifest_checksums: dict[str, str],
    torch_version: str,
    torchvision_version: str,
) -> str:
    """Compute deterministic sha256 binding inputs, manifests, preprocessing, code, and versions."""
    module_path = Path(__file__).resolve()
    code_hash = sha256(module_path)
    raw = json.dumps(
        {
            "code_sha256": code_hash,
            "e021_source": {
                "repo_id": config["e021_source"]["repo_id"],
                "revision": config["e021_source"]["revision"],
                "prefix": config["e021_source"]["prefix"],
                "identity": config["e021_source"]["identity"],
                "parts": sorted(map(int, config["e021_source"]["parts"])),
            },
            "u24_source": {
                "repo_id": config["u24_source"]["repo_id"],
                "revision": config["u24_source"]["revision"],
                "artifact_path": config["u24_source"]["artifact_path"],
            },
            "u30_source": {
                "repo_id": config["u30_source"]["repo_id"],
                "revision": config["u30_source"]["revision"],
                "artifact_path": config["u30_source"]["artifact_path"],
            },
            "manifest_checksums": manifest_checksums,
            "excluded_video_ids": sorted(config.get("excluded_video_ids", [])),
            "mode": config["experiment"].get("mode", "full"),
            "smoke": config.get("smoke", {}),
            "batch_size": int(config.get("data", {}).get("batch_size", 256)),
            "preprocessing": LOCKED_PREPROCESSING,
            "torch_base": torch_version.split("+")[0].strip(),
            "torchvision_base": torchvision_version.split("+")[0].strip(),
        },
        sort_keys=True,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def preprocess_crop_bgr(bgr_img: np.ndarray, target_size: int = 128) -> torch.Tensor:
    """Convert decoded BGR square crop to RGB float32 [3, 128, 128] antialiased normalized tensor."""
    h, w, _ = bgr_img.shape
    if h != w:
        raise ValueError(f"Crop must be square, got shape ({w}x{h})")
    rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).to(torch.float32) / 255.0
    if h != target_size or w != target_size:
        tensor = resize(
            tensor,
            size=[target_size, target_size],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
    return normalize(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD)


def build_mobilenet_extractor(
    device: torch.device,
    *,
    weights_name: str | None = "IMAGENET1K_V1",
) -> torch.nn.Module:
    """Build frozen MobileNetV3-Small feature extractor on target device."""
    import torchvision.models as models
    from torchvision.models import MobileNet_V3_Small_Weights

    if weights_name in {"IMAGENET1K_V1", "DEFAULT"}:
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
    elif weights_name is None:
        weights = None
    else:
        raise ValueError(f"Unsupported weights identifier: {weights_name}")

    model = models.mobilenet_v3_small(weights=weights)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model.to(device)


def run_feature_forward(model: Any, batch_tensor: torch.Tensor) -> torch.Tensor:
    """Compute 576-D flattened feature representations from [B, 3, 128, 128]."""
    x = model.features(batch_tensor)
    x = model.avgpool(x)
    return torch.flatten(x, 1)


def _flush_feature_batch(
    items: list[tuple[int, int, bool, bool, torch.Tensor]],
    *,
    encoder: Any,
    target_device: torch.device,
    sum_u24: np.ndarray,
    count_u24: np.ndarray,
    sum_u30: np.ndarray,
    count_u30: np.ndarray,
) -> tuple[int, float]:
    """Execute batched forward pass and accumulate into U24 and U30 sums."""
    if not items:
        return 0, 0.0
    batch_len = len(items)
    tensors = torch.stack([item[4] for item in items]).to(target_device)
    t0 = time.perf_counter()
    with torch.inference_mode():
        embs = run_feature_forward(encoder, tensors).cpu().numpy().astype(np.float32)
    t_fwd = time.perf_counter() - t0

    for k, (v_idx, h_idx, in_24, in_30, _) in enumerate(items):
        emb = embs[k]
        if in_24:
            sum_u24[v_idx, h_idx] += emb
            count_u24[v_idx, h_idx] += 1
        if in_30:
            sum_u30[v_idx, h_idx] += emb
            count_u30[v_idx, h_idx] += 1
    items.clear()
    return batch_len, t_fwd


def _validate_final_archive(
    path: Path,
    expected_vids: list[str],
    expected_t: int,
) -> None:
    """Compact scientific validation of final split NPZ schema, shape, and zero vectors."""
    with np.load(path, allow_pickle=False) as loaded:
        n = len(expected_vids)
        vids = [str(v).zfill(6) for v in loaded["video_id"]]
        if vids != expected_vids:
            raise ValueError(f"{path.name} video ID mismatch or out of order")
        indices = loaded["selected_frame_indices"]
        if indices.shape != (n, expected_t) or indices.dtype != np.int32:
            raise ValueError(f"{path.name} invalid selected_frame_indices shape/dtype")
        counts = loaded["hand_valid_count"]
        if counts.shape != (n, 2) or counts.dtype != np.uint8 or np.any(counts > expected_t):
            raise ValueError(f"{path.name} invalid hand_valid_count shape/dtype/bounds")
        x_rgb = loaded["X_rgb"]
        if x_rgb.shape != (n, 2, 576) or x_rgb.dtype != np.float16 or not np.isfinite(x_rgb).all():
            raise ValueError(f"{path.name} invalid X_rgb shape/dtype or non-finite values")
        zero_mask = counts == 0
        if np.any(zero_mask) and not np.all(x_rgb[zero_mask] == 0.0):
            raise ValueError(f"{path.name} zero-valid hands must have exact zero vectors")


def _load_uniform_splits(
    artifact_cfg: dict[str, Any], cache_dir: Path, splits: tuple[str, ...] = DEV_SPLITS
) -> dict[str, dict[str, np.ndarray]]:
    prefix = str(artifact_cfg["artifact_path"]).rstrip("/")
    loaded_splits: dict[str, dict[str, np.ndarray]] = {}
    for s in splits:
        npz_file = download_file(
            repo_id=artifact_cfg["repo_id"],
            repo_type="dataset",
            revision=artifact_cfg["revision"],
            filename=f"{prefix}/data/{s}.npz",
            cache_dir=cache_dir,
        )
        with np.load(npz_file, allow_pickle=False) as loaded:
            vids = [str(v).zfill(6) for v in loaded["video_id"]]
            if len(vids) != len(set(vids)):
                raise ValueError(f"Duplicate video IDs in {artifact_cfg['artifact_path']} split {s}")
            loaded_splits[s] = {
                "video_id": np.array(vids, dtype="U6"),
                "frame_indices": loaded["frame_indices"],
                "valid_frame_mask": loaded["valid_frame_mask"],
            }
    return loaded_splits


def run_e026_rgb_features(
    config: dict[str, Any],
    project_root: Path,
    *,
    encoder_factory: Callable[[torch.device], torch.nn.Module] | None = None,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Execute one-pass U24+U30 RGB feature extraction with per-part caching and atomic merges."""
    import torchvision

    torch_mod = torch
    torchvision_mod = torchvision

    split_names = tuple(config["manifests"])
    manifest_records = load_dev_manifests(config, project_root, splits=split_names)
    clean_vids = {s: [r.video_id for r in manifest_records[s]] for s in split_names}
    if "train" in clean_vids and "val" in clean_vids and set(clean_vids["train"]) & set(clean_vids["val"]):
        raise ValueError("Overlap detected between clean train and val video IDs")

    manifest_checksums = {s: sha256(project_root / config["manifests"][s]) for s in split_names}
    cache_dir = project_root / "cache" / "hf"
    u24_splits = _load_uniform_splits(config["u24_source"], cache_dir, split_names)
    u30_splits = _load_uniform_splits(config["u30_source"], cache_dir, split_names)

    video_meta: dict[str, dict[str, Any]] = {}
    for s in split_names:
        u24_map = {str(v).zfill(6): i for i, v in enumerate(u24_splits[s]["video_id"])}
        u30_map = {str(v).zfill(6): i for i, v in enumerate(u30_splits[s]["video_id"])}
        if set(clean_vids[s]) != set(u24_map) or set(clean_vids[s]) != set(u30_map):
            raise ValueError(f"Manifest video IDs mismatch with U24 or U30 in split '{s}'")

        for vid in clean_vids[s]:
            idx24 = u24_splits[s]["frame_indices"][u24_map[vid]]
            mask24 = u24_splits[s]["valid_frame_mask"][u24_map[vid]]
            idx30 = u30_splits[s]["frame_indices"][u30_map[vid]]
            mask30 = u30_splits[s]["valid_frame_mask"][u30_map[vid]]

            u24_f = {int(idx24[t]) for t in range(24) if int(mask24[t]) == 1 and int(idx24[t]) >= 0}
            u30_f = {int(idx30[t]) for t in range(30) if int(mask30[t]) == 1 and int(idx30[t]) >= 0}
            video_meta[vid] = {
                "split": s,
                "u24_indices": idx24,
                "u30_indices": idx30,
                "u24_frames": u24_f,
                "u30_frames": u30_f,
                "union_frames": u24_f | u30_f,
            }

    mode = str(config["experiment"].get("mode", "full"))
    e021_cfg = config["e021_source"]
    configured_parts = sorted(map(int, e021_cfg.get("parts", [1, 2, 3, 4, 5, 6, 7])))
    smoke_cfg = config.get("smoke", {})
    selected_by_part: dict[int, list[Any]] = {p: [] for p in configured_parts}

    for s in split_names:
        split_records = [r for r in manifest_records[s] if r.part_id in selected_by_part]
        if mode == "smoke":
            max_s = smoke_cfg.get(f"{s}_videos")
            if max_s is not None:
                split_records = split_records[: int(max_s)]
        for r in split_records:
            selected_by_part[r.part_id].append(r)

    target_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if encoder_factory is not None:
        encoder = encoder_factory(target_device)
    else:
        encoder = build_mobilenet_extractor(target_device)

    batch_size = int(config.get("data", {}).get("batch_size", 256))
    target_size = int(config.get("data", {}).get("target_size", 128))
    output_root = project_root / config["output"]["root"]
    parts_dir = output_root / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)

    extraction_identity = compute_extraction_identity(
        config, manifest_checksums, torch_mod.__version__, torchvision_mod.__version__
    )

    part_summaries: list[dict[str, Any]] = []

    for part in configured_parts:
        part_records = selected_by_part[part]
        part_vids = [r.video_id for r in part_records]
        n_videos = len(part_vids)
        part_splits = [video_meta[v]["split"] for v in part_vids]

        marker_path = parts_dir / f"part_{part:02d}.complete.json"
        u24_part_path = parts_dir / f"part_{part:02d}_u24.npz"
        u30_part_path = parts_dir / f"part_{part:02d}_u30.npz"

        if marker_path.is_file():
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            if (
                marker.get("extraction_identity") == extraction_identity
                and u24_part_path.is_file()
                and u30_part_path.is_file()
                and sha256(u24_part_path) == marker.get("u24_sha256")
                and sha256(u30_part_path) == marker.get("u30_sha256")
            ):
                part_summaries.append(marker)
                continue
            raise ValueError(f"Part {part:02d} marker present but identity/checksum mismatch")

        e021_prefix = str(e021_cfg["prefix"]).rstrip("/")
        e021_marker_file = download_file(
            repo_id=e021_cfg["repo_id"],
            repo_type="dataset",
            revision=e021_cfg["revision"],
            filename=f"{e021_prefix}/part_{part:02d}.complete.json",
            cache_dir=cache_dir,
        )
        e021_marker = json.loads(e021_marker_file.read_text(encoding="utf-8"))
        if e021_marker.get("identity") != e021_cfg.get("identity"):
            raise ValueError(f"E021 identity mismatch for part {part:02d}")

        shard_paths = [
            download_file(
                repo_id=e021_cfg["repo_id"],
                repo_type="dataset",
                revision=e021_cfg["revision"],
                filename=f"{e021_prefix}/{s['name']}",
                cache_dir=cache_dir,
            )
            for s in e021_marker.get("shards", [])
        ]
        for s_idx, s in enumerate(e021_marker.get("shards", [])):
            if sha256(shard_paths[s_idx]) != s.get("sha256"):
                raise ValueError(f"Corrupt shard checksum: {s['name']}")

        t_part_start = time.perf_counter()
        vid_to_idx = {vid: i for i, vid in enumerate(part_vids)}
        part_splits_arr = np.array(part_splits, dtype="U5")
        sum_u24 = np.zeros((n_videos, 2, 576), dtype=np.float32)
        count_u24 = np.zeros((n_videos, 2), dtype=np.uint32)
        sum_u30 = np.zeros((n_videos, 2, 576), dtype=np.float32)
        count_u30 = np.zeros((n_videos, 2), dtype=np.uint32)

        seen_parquet_keys: set[tuple[str, str, int]] = set()
        seen_union_keys: set[tuple[str, str, int]] = set()
        batch_items: list[tuple[int, int, bool, bool, torch.Tensor]] = []
        part_backbone_seconds = 0.0
        part_crops_encoded = 0
        part_largest_batch = 0

        for s_path in shard_paths:
            table = pq.read_table(s_path, columns=PARQUET_COLUMNS)
            p_splits = table.column("split").to_pylist()
            p_vids = [str(v).zfill(6) for v in table.column("video_id").to_pylist()]
            p_fidxs = table.column("frame_idx").to_pylist()
            p_lval = table.column("left_valid").to_pylist()
            p_rval = table.column("right_valid").to_pylist()
            p_ljpeg = table.column("left_jpeg").to_pylist()
            p_rjpeg = table.column("right_jpeg").to_pylist()

            for i in range(len(p_splits)):
                s, vid, fidx = p_splits[i], p_vids[i], int(p_fidxs[i])
                if vid not in vid_to_idx:
                    continue
                expected_s = video_meta[vid]["split"]
                if s != expected_s:
                    raise ValueError(f"Split mismatch for {vid}: expected {expected_s}, got {s}")

                key = (s, vid, fidx)
                if key in seen_parquet_keys:
                    raise ValueError(f"Duplicate Parquet frame key in part {part:02d}: {key}")
                seen_parquet_keys.add(key)

                v_info = video_meta[vid]
                if fidx not in v_info["union_frames"]:
                    continue
                seen_union_keys.add(key)

                in_24 = fidx in v_info["u24_frames"]
                in_30 = fidx in v_info["u30_frames"]
                v_idx = vid_to_idx[vid]

                for h_idx, (valid_val, raw_jpeg, name) in enumerate([
                    (p_lval[i], p_ljpeg[i], "left"),
                    (p_rval[i], p_rjpeg[i], "right"),
                ]):
                    if valid_val == 1:
                        if raw_jpeg is None:
                            raise ValueError(f"Valid hand has null JPEG at {key} hand={name}")
                        buf = np.frombuffer(raw_jpeg, dtype=np.uint8)
                        bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                        if bgr is None:
                            raise ValueError(f"Failed to decode JPEG at {key} hand={name}")
                        tensor_chw = preprocess_crop_bgr(bgr, target_size=target_size)
                        batch_items.append((v_idx, h_idx, in_24, in_30, tensor_chw))
                        part_crops_encoded += 1

                        if len(batch_items) >= batch_size:
                            cur_b, t_fwd = _flush_feature_batch(
                                batch_items,
                                encoder=encoder,
                                target_device=target_device,
                                sum_u24=sum_u24,
                                count_u24=count_u24,
                                sum_u30=sum_u30,
                                count_u30=count_u30,
                            )
                            part_backbone_seconds += t_fwd
                            part_largest_batch = max(part_largest_batch, cur_b)

        if batch_items:
            cur_b, t_fwd = _flush_feature_batch(
                batch_items,
                encoder=encoder,
                target_device=target_device,
                sum_u24=sum_u24,
                count_u24=count_u24,
                sum_u30=sum_u30,
                count_u30=count_u30,
            )
            part_backbone_seconds += t_fwd
            part_largest_batch = max(part_largest_batch, cur_b)

        expected_union_frames = sum(len(video_meta[v]["union_frames"]) for v in part_vids)
        for vid in part_vids:
            s = video_meta[vid]["split"]
            for f in video_meta[vid]["union_frames"]:
                if (s, vid, f) not in seen_union_keys:
                    raise ValueError(f"Missing required union frame in Parquet: {(s, vid, f)}")

        x_rgb_u24 = np.zeros((n_videos, 2, 576), dtype=np.float16)
        x_rgb_u30 = np.zeros((n_videos, 2, 576), dtype=np.float16)
        for i in range(n_videos):
            for h in (0, 1):
                if count_u24[i, h] > 0:
                    x_rgb_u24[i, h] = (sum_u24[i, h] / float(count_u24[i, h])).astype(np.float16)
                if count_u30[i, h] > 0:
                    x_rgb_u30[i, h] = (sum_u30[i, h] / float(count_u30[i, h])).astype(np.float16)

        idx24_arr = (
            np.stack([video_meta[v]["u24_indices"] for v in part_vids], axis=0).astype(np.int32)
            if n_videos > 0
            else np.zeros((0, 24), dtype=np.int32)
        )
        idx30_arr = (
            np.stack([video_meta[v]["u30_indices"] for v in part_vids], axis=0).astype(np.int32)
            if n_videos > 0
            else np.zeros((0, 30), dtype=np.int32)
        )

        atomic_npz(
            u24_part_path,
            video_id=np.array(part_vids, dtype="U6"),
            split=part_splits_arr,
            selected_frame_indices=idx24_arr,
            hand_valid_count=count_u24.astype(np.uint8),
            X_rgb=x_rgb_u24,
        )
        atomic_npz(
            u30_part_path,
            video_id=np.array(part_vids, dtype="U6"),
            split=part_splits_arr,
            selected_frame_indices=idx30_arr,
            hand_valid_count=count_u30.astype(np.uint8),
            X_rgb=x_rgb_u30,
        )

        part_proc_seconds = time.perf_counter() - t_part_start
        part_marker = {
            "extraction_identity": extraction_identity,
            "part": part,
            "videos": n_videos,
            "expected_union_frames": expected_union_frames,
            "seen_union_frames": len(seen_union_keys),
            "valid_crops_encoded": part_crops_encoded,
            "configured_batch_size": batch_size,
            "largest_executed_batch": part_largest_batch,
            "processing_seconds": part_proc_seconds,
            "backbone_forward_seconds": part_backbone_seconds,
            "processing_crops_per_second": float(part_crops_encoded / max(part_proc_seconds, 1e-6)),
            "backbone_crops_per_second": float(
                part_crops_encoded / max(part_backbone_seconds, 1e-6)
            ),
            "device": str(target_device),
            "u24_sha256": sha256(u24_part_path),
            "u30_sha256": sha256(u30_part_path),
        }
        atomic_json(marker_path, part_marker)
        part_summaries.append(part_marker)

    # Final in-memory merge across parts
    part_loaded_u24: dict[str, dict[str, Any]] = {}
    part_loaded_u30: dict[str, dict[str, Any]] = {}

    for part in configured_parts:
        u24_part_path = parts_dir / f"part_{part:02d}_u24.npz"
        u30_part_path = parts_dir / f"part_{part:02d}_u30.npz"

        with np.load(u24_part_path, allow_pickle=False) as l24:
            vids = [str(v).zfill(6) for v in l24["video_id"]]
            splits = [str(v) for v in l24["split"]]
            indices = np.array(l24["selected_frame_indices"])
            valid = np.array(l24["hand_valid_count"])
            x_arr = np.array(l24["X_rgb"])
            for i, vid in enumerate(vids):
                if vid in part_loaded_u24:
                    raise ValueError(f"Duplicate video ID across parts: {vid}")
                part_loaded_u24[vid] = {
                    "split": splits[i],
                    "indices": indices[i],
                    "valid": valid[i],
                    "x": x_arr[i],
                }

        with np.load(u30_part_path, allow_pickle=False) as l30:
            vids = [str(v).zfill(6) for v in l30["video_id"]]
            splits = [str(v) for v in l30["split"]]
            indices = np.array(l30["selected_frame_indices"])
            valid = np.array(l30["hand_valid_count"])
            x_arr = np.array(l30["X_rgb"])
            for i, vid in enumerate(vids):
                if vid in part_loaded_u30:
                    raise ValueError(f"Duplicate video ID across parts: {vid}")
                part_loaded_u30[vid] = {
                    "split": splits[i],
                    "indices": indices[i],
                    "valid": valid[i],
                    "x": x_arr[i],
                }

    final_split_files: dict[str, dict[str, str]] = {"u24": {}, "u30": {}}
    receipt_final_files: dict[str, dict[str, dict[str, Any]]] = {"u24": {}, "u30": {}}

    for view, part_dict, exp_t in (("u24", part_loaded_u24, 24), ("u30", part_loaded_u30, 30)):
        view_dir = output_root / view / "data"
        view_dir.mkdir(parents=True, exist_ok=True)

        for s in split_names:
            split_vids = [r.video_id for r in manifest_records[s] if r.video_id in part_dict]
            n_split = len(split_vids)
            if n_split == 0:
                continue

            arr_vids = np.array(split_vids, dtype="U6")
            arr_splits = np.array([part_dict[v]["split"] for v in split_vids], dtype="U5")
            arr_indices = np.stack([part_dict[v]["indices"] for v in split_vids], axis=0).astype(
                np.int32
            )
            arr_valid = np.stack([part_dict[v]["valid"] for v in split_vids], axis=0).astype(np.uint8)
            arr_x = np.stack([part_dict[v]["x"] for v in split_vids], axis=0).astype(np.float16)

            final_path = view_dir / f"{s}.npz"
            atomic_npz(
                final_path,
                video_id=arr_vids,
                split=arr_splits,
                selected_frame_indices=arr_indices,
                hand_valid_count=arr_valid,
                X_rgb=arr_x,
            )
            _validate_final_archive(final_path, split_vids, exp_t)

            rel_path = str(final_path.relative_to(output_root))
            final_split_files[view][s] = rel_path
            receipt_final_files[view][s] = {
                "path": rel_path,
                "sha256": sha256(final_path),
                "video_count": n_split,
            }

    tot_crops = sum(p.get("valid_crops_encoded", 0) for p in part_summaries)
    tot_proc_sec = sum(p.get("processing_seconds", 0.0) for p in part_summaries)
    tot_fwd_sec = sum(p.get("backbone_forward_seconds", 0.0) for p in part_summaries)
    max_batch = max((p.get("largest_executed_batch", 0) for p in part_summaries), default=0)

    perf_summary = {
        "expected_union_frames": sum(p.get("expected_union_frames", 0) for p in part_summaries),
        "seen_union_frames": sum(p.get("seen_union_frames", 0) for p in part_summaries),
        "valid_crops_encoded": tot_crops,
        "configured_batch_size": batch_size,
        "largest_executed_batch": max_batch,
        "processing_seconds": tot_proc_sec,
        "backbone_forward_seconds": tot_fwd_sec,
        "processing_crops_per_second": float(tot_crops / max(tot_proc_sec, 1e-6)),
        "backbone_crops_per_second": float(tot_crops / max(tot_fwd_sec, 1e-6)),
        "device": str(target_device),
    }

    receipt = {
        "extraction_identity": extraction_identity,
        "mode": mode,
        "code_sha256": sha256(Path(__file__).resolve()),
        "runtime_versions": {
            "torch": torch_mod.__version__,
            "torchvision": torchvision_mod.__version__,
        },
        "e021_source": e021_cfg,
        "u24_source": config["u24_source"],
        "u30_source": config["u30_source"],
        "preprocessing": LOCKED_PREPROCESSING,
        "performance": perf_summary,
        "configured_parts": configured_parts,
        "parts": part_summaries,
        "final_files": receipt_final_files,
    }
    atomic_json(output_root / "receipt.json", receipt)

    summary = {
        "status": "completed",
        "experiment_id": config["experiment"]["id"],
        "mode": mode,
        "extraction_identity": extraction_identity,
        "code_sha256": receipt["code_sha256"],
        "runtime_versions": receipt["runtime_versions"],
        "performance": perf_summary,
        "parts": configured_parts,
        "total_videos": len(part_loaded_u24),
        "output_files": final_split_files,
    }
    atomic_json(output_root / "summary.json", summary)

    return summary
