"""Build fixed-shape Pose view NPZ splits from raw RTMW shards."""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

VIEWS = ("pose42", "pose53")


def _load_view(name: str) -> Any:
    if name not in VIEWS:
        raise ValueError(f"Unsupported view: {name}")
    return importlib.import_module(f"src.pose_views.{name}")


def _read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["video_id"] = str(row["video_id"]).zfill(6)
        row["signer_id"] = str(row["signer_id"]).zfill(3)
        row["class_id"] = str(row["class_id"])
    return rows


def _shard_index(manifest_path: Path) -> dict[str, str]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {
        str(video_id).zfill(6): str(shard["shard_id"])
        for shard in manifest["shards"]
        for video_id in shard["video_ids"]
    }


def _download(repo: str, revision: str, filename: str, cache: Path) -> Path:
    from huggingface_hub import hf_hub_download

    cache.mkdir(parents=True, exist_ok=True)
    return Path(
        hf_hub_download(
            repo_id=repo,
            repo_type="dataset",
            revision=revision,
            filename=filename,
            local_dir=str(cache),
        )
    )


def build_split(
    records: list[dict[str, str]],
    shards: dict[str, Any],
    view: Any,
    *,
    frames: int,
    threshold: float,
) -> dict[str, np.ndarray]:
    """Build one [N,T,V,3] split with uniform sampling from raw shards."""
    schema = importlib.import_module("src.rtmw.schema")
    nodes = int(view.NODE_COUNT)
    arrays = {
        "X": np.zeros((len(records), frames, nodes, 3), dtype=np.float32),
        "y": np.zeros(len(records), dtype=np.int64),
        "video_id": np.zeros(len(records), dtype="U6"),
        "signer_id": np.zeros(len(records), dtype="U3"),
        "frame_indices": np.full((len(records), frames), -1, dtype=np.int32),
        "valid_frame_mask": np.zeros((len(records), frames), dtype=np.uint8),
        "keypoint_valid_mask": np.zeros((len(records), frames, nodes), dtype=np.uint8),
    }
    for index, row in enumerate(sorted(records, key=lambda r: r["video_id"])):
        shard = shards[row["video_id"]]
        if not isinstance(shard, schema.RawRTMWShard):
            shard = schema.RawRTMWShard.load_npz(shard)
            shards[row["video_id"]] = shard
        built = view.BUILD_VIEW(
            shard,
            row["video_id"],
            sequence_length=frames,
            keypoint_score_threshold=threshold,
        )
        arrays["X"][index] = built.X
        arrays["y"][index] = int(row["class_id"])
        arrays["video_id"][index] = row["video_id"]
        arrays["signer_id"][index] = row["signer_id"]
        arrays["frame_indices"][index] = built.frame_indices
        arrays["valid_frame_mask"][index] = built.valid_frame_mask
        arrays["keypoint_valid_mask"][index] = built.keypoint_valid_mask
    return arrays


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--view", choices=VIEWS, required=True)
    parser.add_argument("--frames", type=int, required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--canonical", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.0)
    args = parser.parse_args(argv)

    if args.frames <= 0:
        raise ValueError(f"frames must be positive, got {args.frames}")
    view = _load_view(args.view)
    rows = _read_manifest(args.canonical)
    index = _shard_index(args.manifest)
    sys.path.insert(0, str(Path.cwd()))

    from src.rtmw.schema import RawRTMWShard

    needed = sorted({index[row["video_id"]] for row in rows})
    raw_paths = {
        shard_id: _download(
            args.repo, args.revision, f"{args.prefix}/data/{shard_id}.npz", args.cache
        )
        for shard_id in needed
    }
    shards = {vid: raw_paths[sid] for vid, sid in index.items() if sid in raw_paths}
    loaded = {vid: RawRTMWShard.load_npz(path) for vid, path in shards.items()}

    args.output.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val"):
        records = [row for row in rows if row["split"] == split]
        arrays = build_split(records, loaded, view, frames=args.frames, threshold=args.threshold)
        np.savez_compressed(
            args.output / f"{split}.npz",
            X=arrays["X"],
            y=arrays["y"],
            video_id=arrays["video_id"],
            signer_id=arrays["signer_id"],
            frame_indices=arrays["frame_indices"],
            valid_frame_mask=arrays["valid_frame_mask"],
            keypoint_valid_mask=arrays["keypoint_valid_mask"],
        )
        print(json.dumps({"split": split, "samples": len(records), "frames": args.frames}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
