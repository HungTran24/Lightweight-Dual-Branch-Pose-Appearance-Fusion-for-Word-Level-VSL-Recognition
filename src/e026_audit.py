"""E026 Hand-RGB crop metadata and Uniform-30 alignment audit (Phases A–B)."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from .artifacts import atomic_json, sha256
from .integrations.huggingface import download_file
from .rtmw.manifest import RTMWVideoRecord, read_canonical_csv

METADATA_COLUMNS = [
    "split",
    "video_id",
    "frame_idx",
    "left_valid",
    "right_valid",
    "left_roi",
    "right_roi",
    "left_bbox",
    "right_bbox",
    "left_padding",
    "right_padding",
    "left_confidence",
    "right_confidence",
]

ALIGNMENT_COLUMNS = [
    "split",
    "video_id",
    "frame_idx",
    "left_valid",
    "right_valid",
]

DEV_SPLITS = ("train", "val")


def _compute_side_stats(
    valid: np.ndarray,
    roi: np.ndarray,
    bbox: np.ndarray,
    padding: np.ndarray,
    confidence: np.ndarray,
) -> dict[str, Any]:
    """Compute exact Phase A crop distribution metrics for one hand side."""
    total_frames = int(len(valid))
    valid_mask = valid.astype(bool)
    valid_count = int(np.sum(valid_mask))
    missing_count = total_frames - valid_count
    missing_rate = float(missing_count / total_frames) if total_frames > 0 else 0.0

    if valid_count == 0:
        return {
            "total_frames": total_frames,
            "valid_count": 0,
            "missing_count": missing_count,
            "missing_rate": missing_rate,
            "p5_crop_side": 0.0,
            "median_crop_side": 0.0,
            "p95_crop_side": 0.0,
            "p99_crop_side": 0.0,
            "fraction_lt_64": 0.0,
            "fraction_lt_96": 0.0,
            "fraction_gt_224": 0.0,
            "padding_area_ratio_mean": 0.0,
            "crops_with_padding_rate": 0.0,
            "roi_expansion_ratio_mean": 0.0,
            "roi_expansion_ratio_median": 0.0,
            "confidence_p5": 0.0,
            "confidence_median": 0.0,
        }

    valid_roi = roi[valid_mask]
    valid_bbox = bbox[valid_mask]
    valid_pad = padding[valid_mask]
    valid_conf = confidence[valid_mask]

    sides = (valid_roi[:, 2] - valid_roi[:, 0]).astype(np.float64)
    p5_side = float(np.percentile(sides, 5))
    median_side = float(np.percentile(sides, 50))
    p95_side = float(np.percentile(sides, 95))
    p99_side = float(np.percentile(sides, 99))

    fraction_lt_64 = float(np.mean(sides < 64.0))
    fraction_lt_96 = float(np.mean(sides < 96.0))
    fraction_gt_224 = float(np.mean(sides > 224.0))

    unpadded_w = np.maximum(0.0, sides - valid_pad[:, 0] - valid_pad[:, 2])
    unpadded_h = np.maximum(0.0, sides - valid_pad[:, 1] - valid_pad[:, 3])
    crop_area = sides * sides
    inside_area = unpadded_w * unpadded_h
    padding_area = np.maximum(0.0, crop_area - inside_area)
    pad_ratios = padding_area / np.maximum(crop_area, 1.0)
    padding_area_ratio_mean = float(np.mean(pad_ratios))
    crops_with_padding_rate = float(np.mean(pad_ratios > 0.0))

    bbox_w = valid_bbox[:, 2] - valid_bbox[:, 0]
    bbox_h = valid_bbox[:, 3] - valid_bbox[:, 1]

    # Valid bounding boxes must have strictly positive width and height per E021 contract
    if np.any(bbox_w <= 0.0) or np.any(bbox_h <= 0.0):
        raise ValueError("Valid hand crop has non-positive bbox dimension")

    max_bbox_side = np.maximum(bbox_w, bbox_h)
    roi_ratios = sides / max_bbox_side
    roi_expansion_ratio_mean = float(np.mean(roi_ratios))
    roi_expansion_ratio_median = float(np.median(roi_ratios))

    conf_p5 = float(np.percentile(valid_conf, 5))
    conf_median = float(np.percentile(valid_conf, 50))

    return {
        "total_frames": total_frames,
        "valid_count": valid_count,
        "missing_count": missing_count,
        "missing_rate": missing_rate,
        "p5_crop_side": p5_side,
        "median_crop_side": median_side,
        "p95_crop_side": p95_side,
        "p99_crop_side": p99_side,
        "fraction_lt_64": fraction_lt_64,
        "fraction_lt_96": fraction_lt_96,
        "fraction_gt_224": fraction_gt_224,
        "padding_area_ratio_mean": padding_area_ratio_mean,
        "crops_with_padding_rate": crops_with_padding_rate,
        "roi_expansion_ratio_mean": roi_expansion_ratio_mean,
        "roi_expansion_ratio_median": roi_expansion_ratio_median,
        "confidence_p5": conf_p5,
        "confidence_median": conf_median,
    }


def audit_crop_metadata(
    records_or_paths: Sequence[Path | str] | Sequence[Mapping[str, Any]],
    *,
    valid_splits: tuple[str, ...] = DEV_SPLITS,
    allowed_video_ids: Mapping[str, set[str]] | None = None,
) -> dict[str, Any]:
    """Extract metadata-only crop statistics across development splits."""
    accumulators: dict[str, dict[str, list[np.ndarray]]] = {
        s: {
            "left_valid": [],
            "right_valid": [],
            "left_roi": [],
            "right_roi": [],
            "left_bbox": [],
            "right_bbox": [],
            "left_padding": [],
            "right_padding": [],
            "left_confidence": [],
            "right_confidence": [],
        }
        for s in (*valid_splits, "overall")
    }

    def _ingest_dict(rows: Mapping[str, Any]) -> None:
        splits = np.asarray(rows["split"], dtype=str)
        vids = np.array([str(v).zfill(6) for v in rows["video_id"]], dtype=str)
        for s in valid_splits:
            mask = splits == s
            if allowed_video_ids is not None and s in allowed_video_ids:
                mask = mask & np.isin(vids, list(allowed_video_ids[s]))
            if not np.any(mask):
                continue
            for key in ("left_valid", "right_valid", "left_confidence", "right_confidence"):
                arr = np.asarray(rows[key])[mask]
                accumulators[s][key].append(arr)
                accumulators["overall"][key].append(arr)
            for key in ("left_roi", "right_roi", "left_bbox", "right_bbox", "left_padding", "right_padding"):
                arr = np.asarray(rows[key])[mask]
                if arr.ndim == 1 and len(arr) == 0:
                    arr = arr.reshape(0, 4)
                accumulators[s][key].append(arr)
                accumulators["overall"][key].append(arr)

    for item in records_or_paths:
        if isinstance(item, (str, Path)):
            path = Path(item)
            table = pq.read_table(path, columns=METADATA_COLUMNS)
            arrays: dict[str, Any] = {
                col: table.column(col).to_numpy()
                for col in ("split", "video_id", "frame_idx", "left_valid", "right_valid", "left_confidence", "right_confidence")
            }
            for k in ("left_roi", "right_roi", "left_bbox", "right_bbox", "left_padding", "right_padding"):
                raw_list = table.column(k).to_pylist()
                arrays[k] = np.array(raw_list, dtype=np.float32) if raw_list else np.zeros((0, 4), dtype=np.float32)
            _ingest_dict(arrays)
        elif isinstance(item, Mapping):
            _ingest_dict(item)
        else:
            raise TypeError(f"Unsupported record type: {type(item)}")

    results: dict[str, dict[str, Any]] = {}
    for s in (*valid_splits, "overall"):
        results[s] = {}
        split_data = accumulators[s]
        if not split_data["left_valid"]:
            empty_1d = np.zeros(0, dtype=np.uint8)
            empty_2d = np.zeros((0, 4), dtype=np.float32)
            empty_conf = np.zeros(0, dtype=np.float32)
            results[s]["left"] = _compute_side_stats(empty_1d, empty_2d, empty_2d, empty_2d, empty_conf)
            results[s]["right"] = _compute_side_stats(empty_1d, empty_2d, empty_2d, empty_2d, empty_conf)
            continue

        l_valid = np.concatenate(split_data["left_valid"])
        r_valid = np.concatenate(split_data["right_valid"])
        l_roi = np.concatenate(split_data["left_roi"])
        r_roi = np.concatenate(split_data["right_roi"])
        l_bbox = np.concatenate(split_data["left_bbox"])
        r_bbox = np.concatenate(split_data["right_bbox"])
        l_pad = np.concatenate(split_data["left_padding"])
        r_pad = np.concatenate(split_data["right_padding"])
        l_conf = np.concatenate(split_data["left_confidence"])
        r_conf = np.concatenate(split_data["right_confidence"])

        results[s]["left"] = _compute_side_stats(l_valid, l_roi, l_bbox, l_pad, l_conf)
        results[s]["right"] = _compute_side_stats(r_valid, r_roi, r_bbox, r_pad, r_conf)

    return results


def audit_uniform30_alignment(
    manifest_rows: Mapping[str, Sequence[Any]],
    uniform30_splits: Mapping[str, Mapping[str, np.ndarray]],
    parquet_sources: Sequence[Path | str] | Sequence[Mapping[str, Any]],
    *,
    valid_splits: tuple[str, ...] = DEV_SPLITS,
) -> dict[str, Any]:
    """Verify exact 1:1 join between Uniform-30 frame indices and E021 Parquet shards (metadata-only)."""
    for s in valid_splits:
        if s not in manifest_rows or s not in uniform30_splits:
            raise ValueError(f"Missing required split '{s}' in manifest or Uniform-30 data")
        manifest_set = {str(r.video_id).zfill(6) for r in manifest_rows[s]}
        u30_vids = [str(v).zfill(6) for v in uniform30_splits[s]["video_id"]]
        u30_set = set(u30_vids)

        if len(manifest_rows[s]) != len(manifest_set):
            raise ValueError(f"Duplicate video IDs found in manifest split '{s}'")
        if len(u30_vids) != len(u30_set):
            raise ValueError(f"Duplicate video IDs found in Uniform-30 split '{s}'")
        if manifest_set != u30_set:
            diff_m = manifest_set - u30_set
            diff_u = u30_set - manifest_set
            raise ValueError(
                f"Manifest video IDs and Uniform-30 video IDs set mismatch for split '{s}': "
                f"in manifest only={sorted(diff_m)[:5]}, in u30 only={sorted(diff_u)[:5]}"
            )

    available_keys: dict[tuple[str, str, int], tuple[int, int]] = {}

    for item in parquet_sources:
        if isinstance(item, (str, Path)):
            table = pq.read_table(Path(item), columns=ALIGNMENT_COLUMNS)
            p_splits = table.column("split").to_pylist()
            p_vids = [str(v).zfill(6) for v in table.column("video_id").to_pylist()]
            p_fidxs = table.column("frame_idx").to_pylist()
            p_lval = table.column("left_valid").to_pylist()
            p_rval = table.column("right_valid").to_pylist()

            for s, vid, fidx, lv, rv in zip(p_splits, p_vids, p_fidxs, p_lval, p_rval, strict=True):
                if s not in valid_splits:
                    continue
                key = (s, vid, int(fidx))
                if key in available_keys:
                    raise ValueError(f"Duplicate Parquet frame key: {key}")
                available_keys[key] = (int(lv), int(rv))
        elif isinstance(item, Mapping):
            splits = item["split"]
            vids = [str(v).zfill(6) for v in item["video_id"]]
            fidxs = item["frame_idx"]
            lval = item["left_valid"]
            rval = item["right_valid"]
            for s, vid, fidx, lv, rv in zip(splits, vids, fidxs, lval, rval, strict=True):
                if s not in valid_splits:
                    continue
                key = (str(s), vid, int(fidx))
                if key in available_keys:
                    raise ValueError(f"Duplicate Parquet frame key: {key}")
                available_keys[key] = (int(lv), int(rv))
        else:
            raise TypeError(f"Unsupported shard item: {type(item)}")

    matched_slots = 0
    total_slots = 0
    split_summaries: dict[str, dict[str, int]] = {}

    for s in valid_splits:
        vids = [str(v).zfill(6) for v in uniform30_splits[s]["video_id"]]
        indices = uniform30_splits[s]["frame_indices"]
        masks = uniform30_splits[s]["valid_frame_mask"]
        split_matched = 0

        for i, vid in enumerate(vids):
            vid_indices = indices[i]
            vid_masks = masks[i]
            if len(vid_indices) != 30 or len(vid_masks) != 30:
                raise ValueError(f"Expected 30 temporal slots, got {len(vid_indices)} for {vid}")
            for t in range(30):
                total_slots += 1
                f_idx = int(vid_indices[t])
                v_mask = int(vid_masks[t])
                if v_mask == 0:
                    if f_idx != -1:
                        raise ValueError(f"Padded frame must have index -1, got {f_idx} for {vid}:{t}")
                    continue
                if f_idx < 0:
                    raise ValueError(f"Valid frame must have non-negative index, got {f_idx} for {vid}:{t}")
                key = (s, vid, f_idx)
                if key not in available_keys:
                    raise ValueError(f"Missing Uniform-30 frame key in E021 Parquet: {key}")
                split_matched += 1
                matched_slots += 1

        split_summaries[s] = {"videos": len(vids), "matched_frames": split_matched}

    return {
        "status": "passed",
        "splits": split_summaries,
        "total_videos": sum(v["videos"] for v in split_summaries.values()),
        "total_uniform30_slots": total_slots,
        "matched_frames": matched_slots,
        "missing_keys": 0,
        "duplicate_keys": 0,
        "split_mismatches": 0,
        "deviation_note": "JPEG nullability check is deferred to Phase C/F selected reads per metadata-only audit contract.",
    }


def format_crop_audit_markdown(
    meta_stats: dict[str, Any],
    alignment_summary: dict[str, Any],
    provenance: dict[str, Any],
) -> str:
    """Format Phase A and Phase B audit findings into clean GitHub-flavored markdown."""
    lines = [
        "# E026 Hand-RGB Crop and Alignment Audit (Phases A–B)",
        "",
        "## Overview",
        "",
        f"- **E021 Source**: `{provenance.get('e021_repo')}` @ `{provenance.get('e021_revision')}`",
        f"- **E021 Path**: `{provenance.get('e021_path')}`",
        f"- **E021 Identity**: `{provenance.get('e021_identity')}`",
        f"- **Uniform-30 Artifact**: `{provenance.get('u30_repo')}` @ `{provenance.get('u30_revision')}` / `{provenance.get('u30_artifact')}`",
        f"- **Development Videos**: `train` ({provenance.get('train_videos', 0)}), `val` ({provenance.get('val_videos', 0)}), total = {provenance.get('train_videos', 0) + provenance.get('val_videos', 0)}",
        "- **Excluded Defect**: `019442` (retained)",
        "",
        "## Phase A: Metadata Crop Distribution",
        "",
        "| Split | Hand | Frames | Valid | Missing % | Side P5 | Median | P95 | P99 | <64px % | <96px % | >224px % | Pad Area % | Pad Rate % | ROI/BBox Med | Conf Med |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for split in (*DEV_SPLITS, "overall"):
        for side in ("left", "right"):
            s = meta_stats[split][side]
            lines.append(
                f"| {split} | {side} | {s['total_frames']} | {s['valid_count']} | "
                f"{s['missing_rate']*100:.2f}% | {s['p5_crop_side']:.1f} | {s['median_crop_side']:.1f} | "
                f"{s['p95_crop_side']:.1f} | {s['p99_crop_side']:.1f} | {s['fraction_lt_64']*100:.2f}% | "
                f"{s['fraction_lt_96']*100:.2f}% | {s['fraction_gt_224']*100:.2f}% | "
                f"{s['padding_area_ratio_mean']*100:.2f}% | {s['crops_with_padding_rate']*100:.2f}% | "
                f"{s['roi_expansion_ratio_median']:.2f} | {s['confidence_median']:.2f} |"
            )

    lines.extend(
        [
            "",
            "## Phase B: Exact Uniform-30 Key Alignment",
            "",
            f"- **Status**: `{alignment_summary['status']}`",
            f"- **Total Evaluated Videos**: `{alignment_summary['total_videos']}`",
            f"- **Total Temporal Slots (30/video)**: `{alignment_summary['total_uniform30_slots']}`",
            f"- **Successfully Matched Frames**: `{alignment_summary['matched_frames']}`",
            f"- **Missing Keys**: `{alignment_summary['missing_keys']}`",
            f"- **Duplicate Keys in Parquet**: `{alignment_summary['duplicate_keys']}`",
            f"- **Split Mismatches**: `{alignment_summary['split_mismatches']}`",
            f"- **Audit Scope Note**: `{alignment_summary['deviation_note']}`",
            "",
            "## Summary & Observations",
            "",
            "1. **Deterministic Alignment**: 100% of required Uniform-30 frame keys match E021 Parquet shards without missing or duplicate keys.",
            "2. **Measured Missing Rates**: Hand missing rates on clean development splits are recorded in the Phase A table above.",
            "3. **Next Step**: Proceed to Phase C visual sampling on aligned keys to verify crop appearance and lock resolution for Phase D.",
            "",
        ]
    )
    return "\n".join(lines)


def load_dev_manifests(
    config: dict[str, Any], project_root: Path, *, splits: tuple[str, ...] = DEV_SPLITS
) -> dict[str, list[RTMWVideoRecord]]:
    manifests = config["manifests"]
    excluded_ids = {str(x).zfill(6) for x in config.get("excluded_video_ids", [])}
    expected_counts = config.get("expected_videos", {})
    records_by_split: dict[str, list[RTMWVideoRecord]] = {}

    for s in splits:
        if s not in manifests:
            raise ValueError(f"manifests must define '{s}'")
        path = project_root / manifests[s]
        rows = [r for r in read_canonical_csv(path) if r.video_id not in excluded_ids]
        if s in expected_counts and len(rows) != int(expected_counts[s]):
            raise ValueError(
                f"Manifest count mismatch for {s}: expected {expected_counts[s]}, got {len(rows)}"
            )
        records_by_split[s] = rows

    return records_by_split


def run_e026_audit(config: dict[str, Any], project_root: Path) -> dict[str, Any]:
    """Run E026 Phase A (crop metadata audit) and Phase B (Uniform-30 alignment audit)."""
    manifest_records = load_dev_manifests(config, project_root)
    allowed_vids = {s: {r.video_id for r in manifest_records[s]} for s in DEV_SPLITS}
    cache_dir = project_root / "cache" / "hf"
    u30_cfg = config["uniform30_source"]
    e021_cfg = config["e021_source"]

    u30_prefix = str(u30_cfg["artifact_path"]).rstrip("/")
    e021_prefix = str(e021_cfg["prefix"]).rstrip("/")

    # 1. Download / load Uniform-30 splits (.npz files are compact metadata arrays)
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

    # 3. Phase A: Metadata crop audit (metadata-only projection, filtered to allowed clean video IDs)
    meta_stats = audit_crop_metadata(
        parquet_paths, valid_splits=DEV_SPLITS, allowed_video_ids=allowed_vids
    )

    # 4. Phase B: Uniform-30 alignment audit (metadata-only join by (split, video_id, frame_idx))
    alignment_summary = audit_uniform30_alignment(
        manifest_records, u30_splits, parquet_paths, valid_splits=DEV_SPLITS
    )

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

    report_md = format_crop_audit_markdown(meta_stats, alignment_summary, provenance)

    output_root = project_root / config["output"]["root"]
    output_root.mkdir(parents=True, exist_ok=True)

    result_payload = {
        "status": "completed",
        "experiment_id": config["experiment"]["id"],
        "provenance": provenance,
        "crop_metadata_stats": meta_stats,
        "alignment_audit": alignment_summary,
    }

    atomic_json(output_root / "crop_audit.json", result_payload)
    atomic_json(output_root / "summary.json", result_payload)
    (output_root / "crop_audit.md").write_text(report_md, encoding="utf-8")

    return result_payload
