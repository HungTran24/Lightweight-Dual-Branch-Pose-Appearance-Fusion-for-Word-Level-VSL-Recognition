"""E026 Phase C visual crop audit implementation."""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pyarrow.parquet as pq

from .artifacts import atomic_json, sha256
from .e026_audit import DEV_SPLITS, load_dev_manifests
from .integrations.huggingface import download_file

CANDIDATE_COLUMNS = [
    "split",
    "video_id",
    "frame_idx",
    "left_valid",
    "right_valid",
    "left_roi",
    "right_roi",
    "left_padding",
    "right_padding",
    "left_confidence",
    "right_confidence",
]

RETRIEVAL_COLUMNS = [
    "split",
    "video_id",
    "frame_idx",
    "left_valid",
    "right_valid",
    "left_jpeg",
    "right_jpeg",
    "left_roi",
    "right_roi",
]

REASON_TARGETS: list[tuple[str, int]] = [
    ("random_control", 8),
    ("smallest_crop_side", 6),
    ("largest_crop_side", 6),
    ("lowest_confidence", 6),
    ("highest_padding_ratio", 6),
]


def collect_aligned_candidates(
    parquet_sources: Sequence[Path | str] | Sequence[Mapping[str, Any]],
    u30_splits: Mapping[str, Mapping[str, np.ndarray]],
    *,
    valid_splits: tuple[str, ...] = DEV_SPLITS,
    allowed_video_ids: Mapping[str, set[str]] | None = None,
) -> list[dict[str, Any]]:
    """Extract candidate valid hand crops matching exact Uniform-30 keyed frames."""
    u30_keys: set[tuple[str, str, int]] = set()
    for s in valid_splits:
        if s not in u30_splits:
            continue
        vids = [str(v).zfill(6) for v in u30_splits[s]["video_id"]]
        indices = u30_splits[s]["frame_indices"]
        masks = u30_splits[s]["valid_frame_mask"]
        for i, vid in enumerate(vids):
            if allowed_video_ids is not None and (s not in allowed_video_ids or vid not in allowed_video_ids[s]):
                continue
            for t in range(len(masks[i])):
                if int(masks[i][t]) == 1:
                    fidx = int(indices[i][t])
                    if fidx >= 0:
                        u30_keys.add((s, vid, fidx))

    candidates: list[dict[str, Any]] = []

    def _process_rows(rows: Mapping[str, Any]) -> None:
        splits = np.asarray(rows["split"], dtype=str)
        vids = np.array([str(v).zfill(6) for v in rows["video_id"]], dtype=str)
        fidxs = np.asarray(rows["frame_idx"], dtype=np.int32)
        l_valid = np.asarray(rows["left_valid"], dtype=np.uint8)
        r_valid = np.asarray(rows["right_valid"], dtype=np.uint8)
        l_roi = np.asarray(rows["left_roi"])
        r_roi = np.asarray(rows["right_roi"])
        l_pad = np.asarray(rows["left_padding"])
        r_pad = np.asarray(rows["right_padding"])
        l_conf = np.asarray(rows["left_confidence"], dtype=np.float32)
        r_conf = np.asarray(rows["right_confidence"], dtype=np.float32)

        for i in range(len(splits)):
            s = splits[i]
            vid = vids[i]
            fidx = int(fidxs[i])
            if (s, vid, fidx) not in u30_keys:
                continue

            if l_valid[i] == 1:
                roi = l_roi[i]
                pad = l_pad[i]
                conf = float(l_conf[i])
                side = int(roi[2] - roi[0])
                unpadded_w = max(0, side - int(pad[0]) - int(pad[2]))
                unpadded_h = max(0, side - int(pad[1]) - int(pad[3]))
                crop_area = side * side
                inside_area = unpadded_w * unpadded_h
                pad_ratio = float(max(0.0, float(crop_area - inside_area)) / max(float(crop_area), 1.0))
                candidates.append({
                    "split": s,
                    "video_id": vid,
                    "frame_idx": fidx,
                    "hand": "left",
                    "crop_side": side,
                    "confidence": conf,
                    "padding_ratio": pad_ratio,
                })

            if r_valid[i] == 1:
                roi = r_roi[i]
                pad = r_pad[i]
                conf = float(r_conf[i])
                side = int(roi[2] - roi[0])
                unpadded_w = max(0, side - int(pad[0]) - int(pad[2]))
                unpadded_h = max(0, side - int(pad[1]) - int(pad[3]))
                crop_area = side * side
                inside_area = unpadded_w * unpadded_h
                pad_ratio = float(max(0.0, float(crop_area - inside_area)) / max(float(crop_area), 1.0))
                candidates.append({
                    "split": s,
                    "video_id": vid,
                    "frame_idx": fidx,
                    "hand": "right",
                    "crop_side": side,
                    "confidence": conf,
                    "padding_ratio": pad_ratio,
                })

    for item in parquet_sources:
        if isinstance(item, (str, Path)):
            table = pq.read_table(Path(item), columns=CANDIDATE_COLUMNS)
            arrays: dict[str, Any] = {
                col: table.column(col).to_numpy()
                for col in ("split", "video_id", "frame_idx", "left_valid", "right_valid", "left_confidence", "right_confidence")
            }
            for k in ("left_roi", "right_roi", "left_padding", "right_padding"):
                raw_list = table.column(k).to_pylist()
                arrays[k] = np.array(raw_list, dtype=np.int32) if raw_list else np.zeros((0, 4), dtype=np.int32)
            _process_rows(arrays)
        elif isinstance(item, Mapping):
            _process_rows(item)
        else:
            raise TypeError(f"Unsupported shard item: {type(item)}")

    return candidates


def select_visual_sample(
    candidates: Sequence[dict[str, Any]],
    *,
    seed: int = 42,
    target_counts: Sequence[tuple[str, int]] = REASON_TARGETS,
) -> list[dict[str, Any]]:
    """Deterministically select 32 unique candidate crops with balanced train/val contribution."""
    by_split: dict[str, list[dict[str, Any]]] = {"train": [], "val": []}
    for c in candidates:
        if c["split"] in by_split:
            by_split[c["split"]].append(c)

    # Deterministic canonical sort before reason ranking
    for s in ("train", "val"):
        by_split[s].sort(key=lambda c: (c["video_id"], c["frame_idx"], c["hand"]))

    rng = np.random.default_rng(seed)
    ranked: dict[str, dict[str, list[dict[str, Any]]]] = {}

    for reason, _ in target_counts:
        ranked[reason] = {}
        for s in ("train", "val"):
            items = list(by_split[s])
            if reason == "random_control":
                if items:
                    perm = rng.permutation(len(items))
                    ranked[reason][s] = [items[i] for i in perm]
                else:
                    ranked[reason][s] = []
            elif reason == "smallest_crop_side":
                items.sort(key=lambda c: (c["crop_side"], c["confidence"], c["video_id"], c["frame_idx"], c["hand"]))
                ranked[reason][s] = items
            elif reason == "largest_crop_side":
                items.sort(key=lambda c: (-c["crop_side"], -c["confidence"], c["video_id"], c["frame_idx"], c["hand"]))
                ranked[reason][s] = items
            elif reason == "lowest_confidence":
                items.sort(key=lambda c: (c["confidence"], c["crop_side"], c["video_id"], c["frame_idx"], c["hand"]))
                ranked[reason][s] = items
            elif reason == "highest_padding_ratio":
                items.sort(key=lambda c: (-c["padding_ratio"], c["crop_side"], c["video_id"], c["frame_idx"], c["hand"]))
                ranked[reason][s] = items
            else:
                raise ValueError(f"Unknown selection reason: {reason}")

    selected: list[dict[str, Any]] = []
    selected_keys: set[tuple[str, str, int, str]] = set()

    for reason, count in target_counts:
        quota_train = count // 2
        quota_val = count - quota_train

        reason_selected: list[dict[str, Any]] = []

        def _pick(pool: list[dict[str, Any]], quota: int, sel_reason: str) -> list[dict[str, Any]]:
            picked = []
            for item in pool:
                key = (item["split"], item["video_id"], item["frame_idx"], item["hand"])
                if key not in selected_keys:
                    selected_keys.add(key)
                    picked.append({**item, "selection_reason": sel_reason})
                    if len(picked) == quota:
                        break
            return picked

        from_train = _pick(ranked[reason]["train"], quota_train, reason)
        from_val = _pick(ranked[reason]["val"], quota_val, reason)
        reason_selected.extend(from_train)
        reason_selected.extend(from_val)

        shortfall = count - len(reason_selected)
        if shortfall > 0:
            if len(from_train) < quota_train:
                reason_selected.extend(_pick(ranked[reason]["val"], shortfall, reason))
            else:
                reason_selected.extend(_pick(ranked[reason]["train"], shortfall, reason))

        if len(reason_selected) < count:
            raise ValueError(
                f"Insufficient unique candidates for reason '{reason}': requested {count}, got {len(reason_selected)}"
            )
        selected.extend(reason_selected)

    return selected


def retrieve_and_validate_crops(
    selected_items: Sequence[dict[str, Any]],
    parquet_sources: Sequence[Path | str] | Sequence[Mapping[str, Any]],
) -> list[tuple[dict[str, Any], bytes, np.ndarray]]:
    """Look up JPEG bytes for selected keys, validate square dimensions and ROI equality."""
    needed_frames = {(item["split"], item["video_id"], int(item["frame_idx"])) for item in selected_items}
    retrieved: dict[tuple[str, str, int], dict[str, Any]] = {}

    for item in parquet_sources:
        if isinstance(item, (str, Path)):
            table = pq.read_table(Path(item), columns=RETRIEVAL_COLUMNS)
            p_splits = table.column("split").to_pylist()
            p_vids = [str(v).zfill(6) for v in table.column("video_id").to_pylist()]
            p_fidxs = table.column("frame_idx").to_pylist()
            for i in range(len(p_splits)):
                key = (p_splits[i], p_vids[i], int(p_fidxs[i]))
                if key in needed_frames:
                    if key in retrieved:
                        raise ValueError(f"Duplicate Parquet key for selected frame: {key}")
                    retrieved[key] = {
                        "left_valid": int(table.column("left_valid")[i].as_py()),
                        "right_valid": int(table.column("right_valid")[i].as_py()),
                        "left_jpeg": table.column("left_jpeg")[i].as_py(),
                        "right_jpeg": table.column("right_jpeg")[i].as_py(),
                        "left_roi": table.column("left_roi")[i].as_py(),
                        "right_roi": table.column("right_roi")[i].as_py(),
                    }
        elif isinstance(item, Mapping):
            p_splits = item["split"]
            p_vids = [str(v).zfill(6) for v in item["video_id"]]
            p_fidxs = item["frame_idx"]
            for i in range(len(p_splits)):
                key = (str(p_splits[i]), p_vids[i], int(p_fidxs[i]))
                if key in needed_frames:
                    if key in retrieved:
                        raise ValueError(f"Duplicate Parquet key for selected frame: {key}")
                    retrieved[key] = {
                        "left_valid": int(item["left_valid"][i]),
                        "right_valid": int(item["right_valid"][i]),
                        "left_jpeg": item["left_jpeg"][i],
                        "right_jpeg": item["right_jpeg"][i],
                        "left_roi": item["left_roi"][i],
                        "right_roi": item["right_roi"][i],
                    }
        else:
            raise TypeError(f"Unsupported shard item: {type(item)}")

    results: list[tuple[dict[str, Any], bytes, np.ndarray]] = []
    for item in selected_items:
        key = (item["split"], item["video_id"], int(item["frame_idx"]))
        if key not in retrieved:
            raise ValueError(f"Missing Parquet row for selected frame: {key}")
        row = retrieved[key]
        hand = item["hand"]
        valid = row["left_valid"] if hand == "left" else row["right_valid"]
        jpeg_bytes = row["left_jpeg"] if hand == "left" else row["right_jpeg"]
        roi = row["left_roi"] if hand == "left" else row["right_roi"]

        if not valid or jpeg_bytes is None:
            raise ValueError(
                f"Selected hand must have valid=1 and non-null JPEG: {key} hand={hand}"
            )

        arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError(f"Failed to decode JPEG bytes for {key} hand={hand}")

        h, w, _ = img_bgr.shape
        if h != w:
            raise ValueError(f"Decoded crop is not square ({w}x{h}) for {key} hand={hand}")

        roi_side = int(roi[2] - roi[0])
        if w != roi_side:
            raise ValueError(
                f"Decoded crop side {w} does not equal ROI side {roi_side} for {key} hand={hand}"
            )

        results.append((item, jpeg_bytes, img_bgr))

    return results


def create_contact_sheet(
    validated_items: Sequence[tuple[dict[str, Any], bytes, np.ndarray]],
    *,
    tile_size: int = 180,
    cols: int = 4,
) -> np.ndarray:
    """Build a labeled contact sheet grid from decoded BGR crop images."""
    n = len(validated_items)
    rows = math.ceil(n / cols) if n > 0 else 1
    label_h = 60
    cell_h = tile_size + label_h
    pad = 12

    canvas_w = pad + cols * (tile_size + pad)
    canvas_h = pad + rows * (cell_h + pad)
    canvas = np.full((canvas_h, canvas_w, 3), 30, dtype=np.uint8)

    for idx, (item, _, img_bgr) in enumerate(validated_items):
        r = idx // cols
        c = idx % cols
        x = pad + c * (tile_size + pad)
        y = pad + r * (cell_h + pad)

        thumb = cv2.resize(img_bgr, (tile_size, tile_size), interpolation=cv2.INTER_AREA)
        canvas[y : y + tile_size, x : x + tile_size] = thumb
        canvas[y + tile_size : y + cell_h, x : x + tile_size] = (45, 45, 45)

        line1 = f"#{idx+1:02d} [{item['split']}] {item['video_id']} f:{item['frame_idx']} {item['hand']}"
        line2 = f"{item['selection_reason']}"
        line3 = f"side:{item['crop_side']}px c:{item['confidence']:.2f} p:{item['padding_ratio']:.2f}"

        cv2.putText(canvas, line1, (x + 4, y + tile_size + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, line2, (x + 4, y + tile_size + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (0, 220, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, line3, (x + 4, y + tile_size + 52), cv2.FONT_HERSHEY_SIMPLEX, 0.31, (200, 200, 200), 1, cv2.LINE_AA)

    return canvas


def write_visual_sample_csv(
    items: Sequence[dict[str, Any]],
    output_path: Path,
) -> None:
    """Write reproducible visual_sample.csv with empty tag and note for manual review."""
    fieldnames = [
        "split",
        "video_id",
        "frame_idx",
        "hand",
        "selection_reason",
        "crop_side",
        "confidence",
        "padding_ratio",
        "tag",
        "note",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in items:
            writer.writerow({
                "split": item["split"],
                "video_id": item["video_id"],
                "frame_idx": item["frame_idx"],
                "hand": item["hand"],
                "selection_reason": item["selection_reason"],
                "crop_side": item["crop_side"],
                "confidence": f"{float(item['confidence']):.4f}",
                "padding_ratio": f"{float(item['padding_ratio']):.4f}",
                "tag": "",
                "note": "",
            })


def run_e026_visual_audit(config: dict[str, Any], project_root: Path) -> dict[str, Any]:
    """Execute E026 Phase C: select 32 crops, validate JPEGs, save images and contact sheet."""
    manifest_records = load_dev_manifests(config, project_root)
    allowed_vids = {s: {r.video_id for r in manifest_records[s]} for s in DEV_SPLITS}
    cache_dir = project_root / "cache" / "hf"
    u30_cfg = config["uniform30_source"]
    e021_cfg = config["e021_source"]

    u30_prefix = str(u30_cfg["artifact_path"]).rstrip("/")
    e021_prefix = str(e021_cfg["prefix"]).rstrip("/")

    # 1. Download / load Uniform-30 splits (.npz files)
    u30_splits: dict[str, dict[str, np.ndarray]] = {}
    for s in DEV_SPLITS:
        npz_file = download_file(
            repo_id=u30_cfg["repo_id"],
            repo_type="dataset",
            revision=u30_cfg["revision"],
            filename=f"{u30_prefix}/data/{s}.npz",
            cache_dir=cache_dir,
        )
        with np.load(npz_file, allow_pickle=False) as loaded:
            u30_splits[s] = {
                "video_id": loaded["video_id"],
                "frame_indices": loaded["frame_indices"],
                "valid_frame_mask": loaded["valid_frame_mask"],
            }

    # 2. Download and verify E021 Parquet shards
    parquet_paths: list[Path] = []
    e021_parts = e021_cfg.get("parts", [1, 2, 3, 4, 5, 6, 7])
    for part in sorted(map(int, e021_parts)):
        marker_path = download_file(
            repo_id=e021_cfg["repo_id"],
            repo_type="dataset",
            revision=e021_cfg["revision"],
            filename=f"{e021_prefix}/part_{part:02d}.complete.json",
            cache_dir=cache_dir,
        )
        marker_data = json.loads(marker_path.read_text(encoding="utf-8"))
        if marker_data.get("identity") != e021_cfg.get("identity"):
            raise ValueError(f"E021 identity mismatch for part {part:02d}")
        for shard_info in marker_data.get("shards", []):
            shard_name = str(shard_info["name"])
            shard_path = download_file(
                repo_id=e021_cfg["repo_id"],
                repo_type="dataset",
                revision=e021_cfg["revision"],
                filename=f"{e021_prefix}/{shard_name}",
                cache_dir=cache_dir,
            )
            if sha256(shard_path) != shard_info.get("sha256"):
                raise ValueError(f"Corrupt parquet shard checksum: {shard_name}")
            parquet_paths.append(shard_path)

    # 3. Collect aligned candidates
    candidates = collect_aligned_candidates(
        parquet_paths, u30_splits, valid_splits=DEV_SPLITS, allowed_video_ids=allowed_vids
    )

    # 4. Select deterministic sample
    seed = int(config.get("sampling", {}).get("seed", 42))
    selected_items = select_visual_sample(candidates, seed=seed)

    # 5. Retrieve JPEG payloads and validate
    validated = retrieve_and_validate_crops(selected_items, parquet_paths)

    # 6. Save outputs
    output_root = project_root / config["output"]["root"]
    output_root.mkdir(parents=True, exist_ok=True)
    crops_dir = output_root / "selected_crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    for idx, (item, jpeg_bytes, _) in enumerate(validated, start=1):
        filename = f"{idx:02d}_{item['split']}_{item['video_id']}_f{item['frame_idx']:04d}_{item['hand']}_{item['selection_reason']}.jpg"
        (crops_dir / filename).write_bytes(jpeg_bytes)

    write_visual_sample_csv(selected_items, output_root / "visual_sample.csv")

    contact_sheet = create_contact_sheet(validated)
    contact_sheet_path = output_root / "contact_sheet.jpg"
    if not cv2.imwrite(str(contact_sheet_path), contact_sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 95]):
        raise OSError(f"Failed to write contact sheet image to {contact_sheet_path}")

    provenance = {
        "e021_repo": e021_cfg["repo_id"],
        "e021_revision": e021_cfg["revision"],
        "e021_path": e021_prefix,
        "e021_identity": e021_cfg["identity"],
        "u30_repo": u30_cfg["repo_id"],
        "u30_revision": u30_cfg["revision"],
        "u30_artifact": u30_prefix,
        "train_videos": len(manifest_records["train"]),
        "val_videos": len(manifest_records["val"]),
    }

    selected_role_counts = {
        r: sum(1 for c in selected_items if c["selection_reason"] == r) for r, _ in REASON_TARGETS
    }
    split_counts = {
        "train": sum(1 for c in selected_items if c["split"] == "train"),
        "val": sum(1 for c in selected_items if c["split"] == "val"),
    }

    summary_payload = {
        "status": "completed",
        "experiment_id": config["experiment"]["id"],
        "seed": seed,
        "provenance": provenance,
        "requested_role_counts": dict(REASON_TARGETS),
        "selected_role_counts": selected_role_counts,
        "split_counts": split_counts,
        "total_selected": len(selected_items),
        "validation_checks": {
            "all_jpegs_valid": True,
            "all_crops_square": True,
            "all_roi_sides_matched": True,
            "no_duplicate_keys": True,
        },
        "output_files": {
            "manifest_csv": "visual_sample.csv",
            "contact_sheet": "contact_sheet.jpg",
            "crops_dir": "selected_crops",
            "summary_json": "summary.json",
        },
    }

    atomic_json(output_root / "summary.json", summary_payload)
    return summary_payload
