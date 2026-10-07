"""Materialize 24-frame feature splits for Uniform-24, Motion-24, and KCC-24 (Pose53 view).

Filters:
- Train signers: exclude 024, 025, 026, 027 and defect 019442 -> 14,250 clips, 400 classes
- Val signers: exclude 028 -> 1,829 clips, 400 classes
- Total classes: 400
- View: Pose53 (body + hands + face)
- Sequence length: 24 frames
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import FeatureSplit  # noqa: E402
from src.frame_sampling import kcc_indices, motion_indices, uniform_indices  # noqa: E402
from src.pose_views.pose53 import build_rtmw_pose53_view  # noqa: E402
from src.rtmw.manifest import RTMWExtractionManifest  # noqa: E402
from src.rtmw.schema import RawRTMWShard  # noqa: E402

RAW_REPO_ID = "hungchill24/vsl400-rtmw-raw"
RAW_REVISION = "5fdb1efd1296d64b3e2445924869667cdc634463"
RAW_PREFIX = "rtmw-raw-v1-69b596d53b27537a"

MANIFEST_CANDIDATES = [
    PROJECT_ROOT / "manifests/vsl400_front_400gloss/all.csv",
    Path("/content/project_ai/manifests/vsl400_front_400gloss/all.csv"),
]
MANIFEST_PATH = next(p for p in MANIFEST_CANDIDATES if p.exists())

RTMW_MANIFEST_CANDIDATES = [
    PROJECT_ROOT / "manifests/rtmw_full400/rtmw_manifest_full.json",
    Path("/content/project_ai/manifests/rtmw_full400/rtmw_manifest_full.json"),
]
RTMW_MANIFEST_PATH = next(p for p in RTMW_MANIFEST_CANDIDATES if p.exists())

EXCLUDED_TRAIN_SIGNERS = {"024", "025", "026", "027"}
EXCLUDED_VAL_SIGNERS = {"028"}
SEQUENCE_LENGTH = 24


def build_split_features(
    df: Any,
    method: str,
    raw_shards: dict[int, RawRTMWShard],
    shard_index: dict[str, int],
) -> FeatureSplit:
    views = []
    for _, row in df.iterrows():
        vid = str(row["video_id"]).zfill(6)
        s_idx = shard_index[vid]
        shard = raw_shards[s_idx]

        matches = np.flatnonzero(shard.video_id.astype(str) == vid)
        if matches.size != 1:
            raise ValueError(f"Video {vid} not found in shard {s_idx}")
        v_pos = int(matches[0])
        f_start = int(shard.video_frame_offsets[v_pos])
        f_stop = int(shard.video_frame_offsets[v_pos + 1])
        frame_count = f_stop - f_start

        frame_kpts = np.zeros((frame_count, 133, 3), dtype=np.float32)
        for t in range(frame_count):
            p_start = int(shard.frame_person_offsets[f_start + t])
            p_stop = int(shard.frame_person_offsets[f_start + t + 1])
            if p_stop > p_start:
                primary_p = p_start + int(np.argmax(shard.person_score[p_start:p_stop]))
                frame_kpts[t] = shard.keypoints[primary_p]

        sampler_fn: Any
        if method == "uniform":
            def _uni_sampler(fc: int, sl: int = SEQUENCE_LENGTH) -> np.ndarray:
                return uniform_indices(fc, sl)

            sampler_fn = _uni_sampler
        elif method == "motion":
            def _mot_sampler(fc: int, sl: int = SEQUENCE_LENGTH, kpts: np.ndarray = frame_kpts) -> np.ndarray:
                return motion_indices(kpts, sl)

            sampler_fn = _mot_sampler
        elif method == "kcc":
            def _kcc_sampler(fc: int, sl: int = SEQUENCE_LENGTH, kpts: np.ndarray = frame_kpts) -> np.ndarray:
                return kcc_indices(kpts, sl, thresh=4.0)

            sampler_fn = _kcc_sampler
        else:
            raise ValueError(f"Unknown method: {method}")

        v = build_rtmw_pose53_view(
            shard,
            vid,
            sequence_length=SEQUENCE_LENGTH,
            sampler=sampler_fn,
        )
        views.append((row, v))

    views.sort(key=lambda item: str(item[0]["video_id"]).zfill(6))
    split = FeatureSplit(
        X=np.stack([v.X for _, v in views]).astype(np.float32, copy=False),
        y=np.asarray([int(r["class_id"]) for r, _ in views], dtype=np.int64),
        video_id=np.asarray([str(r["video_id"]).zfill(6) for r, _ in views], dtype="U6"),
        signer_id=np.asarray([str(r["signer_id"]).zfill(3) for r, _ in views], dtype="U3"),
        frame_indices=np.stack([v.frame_indices for _, v in views]).astype(np.int32, copy=False),
        valid_frame_mask=np.stack([v.valid_frame_mask for _, v in views]).astype(np.uint8, copy=False),
        keypoint_valid_mask=np.stack([v.keypoint_valid_mask for _, v in views]).astype(np.uint8, copy=False),
    )
    return split


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "artifacts/frame_selection_splits")
    parser.add_argument("--cache-dir", type=Path, default=PROJECT_ROOT / "cache/hf")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    token = os.environ.get("HF_TOKEN")

    print(f"1. Loading manifests from {MANIFEST_PATH} and filtering signers...")
    all_df = pd.read_csv(MANIFEST_PATH)
    all_df["signer_id"] = all_df["signer_id"].astype(str).str.zfill(3)
    all_df["video_id"] = all_df["video_id"].astype(str).str.zfill(6)

    train_df = all_df[
        (all_df["split"] == "train")
        & (~all_df["signer_id"].isin(list(EXCLUDED_TRAIN_SIGNERS)))
        & (all_df["video_id"] != "019442")
    ].reset_index(drop=True)

    val_df = all_df[
        (all_df["split"] == "val") & (~all_df["signer_id"].isin(list(EXCLUDED_VAL_SIGNERS)))
    ].reset_index(drop=True)

    print(f"Train records: {len(train_df)} (signers: {sorted(set(train_df['signer_id']))})")
    print(f"Val records: {len(val_df)} (signers: {sorted(set(val_df['signer_id']))})")
    assert len(set(train_df["class_id"])) == 400
    assert len(set(val_df["class_id"])) == 400

    print(f"2. Mapping video IDs to shards using {RTMW_MANIFEST_PATH}...")
    manifest = RTMWExtractionManifest.load_json(RTMW_MANIFEST_PATH)
    shard_index = {
        video_id: shard.shard_index
        for shard in manifest.shards
        for video_id in shard.video_ids
    }

    needed_train_shards = {shard_index[str(v)] for v in train_df["video_id"]}
    needed_val_shards = {shard_index[str(v)] for v in val_df["video_id"]}
    all_needed_shards = sorted(needed_train_shards | needed_val_shards)
    print(f"Total shards needed: {len(all_needed_shards)}")

    print("3. Downloading and loading raw shards from HF...")
    raw_shards: dict[int, RawRTMWShard] = {}
    for s_idx in all_needed_shards:
        path = hf_hub_download(
            repo_id=RAW_REPO_ID,
            repo_type="dataset",
            filename=f"{RAW_PREFIX}/data/shard_{s_idx:04d}.npz",
            revision=RAW_REVISION,
            token=token,
            cache_dir=str(args.cache_dir),
        )
        raw_shards[s_idx] = RawRTMWShard.load_npz(path)
    print(f"Loaded {len(raw_shards)} raw shards into memory.")

    methods = ["uniform", "motion", "kcc"]

    for method in methods:
        save_dir = args.output_dir / f"{method}_24_pose53"
        train_file = save_dir / "train.npz"
        val_file = save_dir / "val.npz"
        if train_file.exists() and val_file.exists():
            print(f"Split {method}-24 (pose53) already exists at {save_dir}, skipping.")
            continue

        print(f"\n--- Materializing {method}-24 (pose53) ---")
        t0 = time.time()
        tr_split = build_split_features(train_df, method, raw_shards, shard_index)
        va_split = build_split_features(val_df, method, raw_shards, shard_index)

        save_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            train_file,
            X=tr_split.X,
            y=tr_split.y,
            video_id=tr_split.video_id,
            signer_id=tr_split.signer_id,
            frame_indices=tr_split.frame_indices,
            valid_frame_mask=tr_split.valid_frame_mask,
            keypoint_valid_mask=tr_split.keypoint_valid_mask,
        )
        np.savez_compressed(
            val_file,
            X=va_split.X,
            y=va_split.y,
            video_id=va_split.video_id,
            signer_id=va_split.signer_id,
            frame_indices=va_split.frame_indices,
            valid_frame_mask=va_split.valid_frame_mask,
            keypoint_valid_mask=va_split.keypoint_valid_mask,
        )
        print(f"Saved {method}-24 (pose53) in {time.time() - t0:.1f}s -> {save_dir}")

    print("\nAll splits ready!")


if __name__ == "__main__":
    main()
