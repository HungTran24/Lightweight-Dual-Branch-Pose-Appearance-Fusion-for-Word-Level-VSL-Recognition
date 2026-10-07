"""Derive immutable uniformly sampled Pose59 FeatureSplit artifacts."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from .artifacts import atomic_json, sha256, upload_with_retry
from .data import FeatureSplit, validate_split
from .frame_sampling import uniform_indices
from .integrations.huggingface import download_file, ensure_repo
from .pose_views.pose59 import validate_pose59_schema
from .rtmw.manifest import read_canonical_csv

SPLITS = ("train", "val", "test")


def _manifests(config: dict[str, Any], root: Path):
    source = config["source"]
    split_names = tuple(source["manifests"])
    rows, hashes, excluded, ids = {s: [] for s in split_names}, {}, [], set()
    excluded_ids = {str(x).zfill(6) for x in source.get("excluded_video_ids", [])}
    for split in split_names:
        path = root / source["manifests"][split]
        hashes[split] = sha256(path)
        for row in read_canonical_csv(path):
            if row.video_id in ids or row.split != split or not 0 <= row.class_id < 400:
                raise ValueError(f"Invalid or duplicate manifest row: {row.video_id}")
            ids.add(row.video_id)
            if row.video_id in excluded_ids:
                excluded.append(
                    {"video_id": row.video_id, "split": split, "num_frames": row.num_frames}
                )
            else:
                rows[split].append(row)
    if {s: len(rows[s]) for s in split_names} != {
        s: int(v) for s, v in source["expected_videos"].items()
    }:
        raise ValueError("Manifest video counts mismatch")
    if {x["video_id"] for x in excluded} != excluded_ids or len(excluded) != len(excluded_ids):
        raise ValueError("Excluded video IDs must occur exactly once")
    return rows, hashes, excluded


def _missingness(valid: np.ndarray, keypoints: np.ndarray) -> dict[str, Any]:
    valid, keypoints = valid.astype(bool), keypoints.astype(bool)
    missing = [valid & ~keypoints[:, :, a:b].any(2) for a, b in ((17, 38), (38, 59))]
    both, either = missing[0] & missing[1], missing[0] | missing[1]
    slots, affected = int(valid.sum()), either.sum(1)

    def rate(x):
        return {"count": int(x.sum()), "rate": float(x.sum() / slots) if slots else 0.0}

    max_run = 0
    for clip in either:
        run = 0
        for value in np.r_[clip, False]:
            run = run + 1 if value else 0
            max_run = max(max_run, run)
    return {
        "valid_frame_slots": slots,
        "missing_left": rate(missing[0]),
        "missing_right": rate(missing[1]),
        "missing_both": rate(both),
        "clips": len(valid),
        "clips_with_missing_hand": {
            "count": int(np.count_nonzero(affected)),
            "rate": float(np.count_nonzero(affected) / len(valid)) if len(valid) else 0.0,
        },
        "max_consecutive_missing_frames": max_run,
    }


def _audit(splits: dict[str, FeatureSplit]) -> dict[str, Any]:
    valid = np.concatenate([x.valid_frame_mask for x in splits.values()])
    masks = np.concatenate([x.keypoint_valid_mask for x in splits.values()])
    return {
        "overall": _missingness(valid, masks),
        "by_split": {
            s: _missingness(x.valid_frame_mask, x.keypoint_valid_mask) for s, x in splits.items()
        },
    }


def derive_uniform_pose59(config: dict[str, Any], project_root: Path) -> dict[str, Any]:
    source, lengths = (
        config["source"],
        tuple(sorted(map(int, config["sampling"]["sequence_lengths"]))),
    )
    split_names = tuple(source["manifests"])
    output = project_root / config["output"]["root"]
    if output.exists():
        raise FileExistsError(f"Artifact root already exists: {output}")
    rows, manifest_hashes, excluded = _manifests(config, project_root)
    work = output.parent / f".{output.name}.work"
    work.mkdir(parents=True)
    try:
        summary_path = download_file(
            repo_id=source["repo_id"],
            repo_type="dataset",
            revision=source["revision"],
            filename=f"{source['prefix'].rstrip('/')}/summary.json",
            local_dir=work,
        )
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        source_totals = source.get("source_totals", {})
        if (
            summary.get("identity") != source["identity"]
            or int(summary.get("videos", -1))
            != int(source_totals.get("videos", sum(map(len, rows.values()))))
            or int(summary.get("frames", -1))
            != int(source_totals.get("frames", source["expected_frames"]))
        ):
            raise ValueError("Source summary identity or counts mismatch")
        provenance = summary.get("provenance", {})
        if any(
            provenance.get("manifests", {}).get(s, {}).get("sha256") != manifest_hashes[s]
            for s in split_names
        ):
            raise ValueError("Source manifest checksums mismatch")
        if {
            (str(x["video_id"]), str(x["split"]), int(x["num_frames"]))
            for x in summary.get("excluded_defects", [])
            if str(x["split"]) in split_names
        } != {(x["video_id"], x["split"], int(x["num_frames"])) for x in excluded}:
            raise ValueError("Source exclusions mismatch")
        parts = provenance.get("part_artifacts", {})
        if set(parts) != {f"part_{int(x):02d}" for x in source["parts"]}:
            raise ValueError("Source part coverage mismatch")
        lookup = {r.video_id: (s, i, r) for s in split_names for i, r in enumerate(rows[s])}
        out = {
            t: {
                s: {
                    "X": np.zeros((len(rows[s]), t, 59, 3), np.float32),
                    "y": np.array([r.class_id for r in rows[s]], np.int64),
                    "video_id": np.array([r.video_id for r in rows[s]], "U6"),
                    "signer_id": np.array([r.signer_id for r in rows[s]], "U3"),
                    "frame_indices": np.full((len(rows[s]), t), -1, np.int32),
                    "valid_frame_mask": np.zeros((len(rows[s]), t), np.uint8),
                    "keypoint_valid_mask": np.zeros((len(rows[s]), t, 59), np.uint8),
                }
                for s in split_names
            }
            for t in lengths
        }
        seen, frame_total, source_video_total, source_frame_total = set(), 0, 0, 0
        for part in sorted(map(int, source["parts"])):
            key, expected = f"part_{part:02d}", parts[f"part_{part:02d}"]
            shard = download_file(
                repo_id=source["repo_id"],
                repo_type="dataset",
                revision=source["revision"],
                filename=f"{source['prefix'].rstrip('/')}/{key}.npz",
                local_dir=work,
            )
            if sha256(shard) != expected["npz_sha256"]:
                raise ValueError(f"Source checksum mismatch for {key}")
            with np.load(shard, allow_pickle=False) as loaded:
                arrays = {name: loaded[name] for name in loaded.files}
            validate_pose59_schema(arrays)
            offsets = arrays["video_frame_offsets"]
            if len(arrays["video_id"]) != int(expected["videos"]) or int(offsets[-1]) != int(
                expected["frames"]
            ):
                raise ValueError(f"Source part counts mismatch for {key}")
            source_video_total += len(arrays["video_id"])
            source_frame_total += int(offsets[-1])
            for j, raw_id in enumerate(arrays["video_id"]):
                video_id = str(raw_id)
                if video_id not in lookup:
                    continue
                if video_id in seen or str(arrays["split"][j]) != lookup[video_id][0]:
                    raise ValueError(f"Source coverage mismatch for {video_id}")
                split, index, row = lookup[video_id]
                start, stop = map(int, offsets[j : j + 2])
                if stop - start != row.num_frames:
                    raise ValueError(f"Source coverage mismatch for {video_id}")
                for t in lengths:
                    selected = uniform_indices(stop - start, t)
                    valid = selected >= 0
                    target = out[t][split]
                    target["frame_indices"][index], target["valid_frame_mask"][index] = (
                        selected,
                        valid,
                    )
                    target["X"][index, valid] = arrays["X"][start:stop][selected[valid]]
                    target["keypoint_valid_mask"][index, valid] = arrays["keypoint_valid_mask"][
                        start:stop
                    ][selected[valid]]
                seen.add(video_id)
                frame_total += stop - start
            del arrays
            shard.unlink(missing_ok=True)
        if (
            source_video_total != int(source_totals.get("videos", source_video_total))
            or source_frame_total != int(source_totals.get("frames", source_frame_total))
            or len(seen) != sum(map(len, rows.values()))
            or frame_total != source["expected_frames"]
        ):
            raise ValueError("Source coverage totals mismatch")
        output.mkdir(parents=True)
        hf = config.get("integrations", {}).get("hf_artifacts", {})
        if hf.get("enabled"):
            ensure_repo(
                repo_id=hf["repo_id"], repo_type="dataset", private=bool(hf.get("private", True))
            )
        artifacts = {}
        for t in lengths:
            identity_payload = {
                "source_revision": source["revision"],
                "source_identity": source["identity"],
                "manifests": manifest_hashes,
                "excluded_videos": excluded,
                "sequence_length": t,
            }
            identity = hashlib.sha256(
                json.dumps(identity_payload, sort_keys=True).encode()
            ).hexdigest()
            name = f"yolo-e005-pose59-uniform{t}-{identity[:16]}"
            root = output / name
            (root / "data").mkdir(parents=True)
            objects, files = {}, {}
            for split in split_names:
                obj = FeatureSplit(**out[t][split])
                validate_split(obj, num_classes=400, nodes=59)
                objects[split] = obj
                path = root / "data" / f"{split}.npz"
                np.savez_compressed(path, **out[t][split])
                files[split] = {
                    "path": f"data/{split}.npz",
                    "sha256": sha256(path),
                    "count": len(obj.X),
                }
            receipt = {
                "identity": identity,
                "identity_payload": identity_payload,
                "files": files,
                "counts": {s: len(rows[s]) for s in split_names},
                "missingness": _audit(objects),
            }
            atomic_json(root / "receipt.json", receipt)
            atomic_json(
                root / "_SUCCESS.json",
                {"identity": identity, "receipt_sha256": sha256(root / "receipt.json")},
            )
            if hf.get("enabled"):
                upload_with_retry(
                    repo_id=hf["repo_id"],
                    files={f"{name}/data/{s}.npz": root / "data" / f"{s}.npz" for s in split_names}
                    | {f"{name}/{x}": root / x for x in ("receipt.json", "_SUCCESS.json")},
                    commit_message=f"derive {name}",
                    retries=int(hf.get("max_retries", 3)),
                )
            artifacts[str(t)] = {
                "artifact": name,
                "identity": identity,
                "missingness": receipt["missingness"],
                "files": files,
            }
        result = {
            "status": "complete",
            "operation": "derive",
            "source_identity": source["identity"],
            "source_revision": source["revision"],
            "artifacts": artifacts,
        }
        atomic_json(output / "summary.json", result)
        return result
    finally:
        shutil.rmtree(work, ignore_errors=True)
