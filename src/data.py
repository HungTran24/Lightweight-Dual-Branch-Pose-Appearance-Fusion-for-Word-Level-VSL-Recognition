from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .integrations.huggingface import download_file


@dataclass(frozen=True)
class FeatureSplit:
    X: np.ndarray
    y: np.ndarray
    video_id: np.ndarray
    signer_id: np.ndarray
    frame_indices: np.ndarray
    valid_frame_mask: np.ndarray
    keypoint_valid_mask: np.ndarray

    def take(self, count: int) -> FeatureSplit:
        if count <= 0 or count > len(self.X):
            raise ValueError(f"subset count must be between 1 and {len(self.X)}")
        return FeatureSplit(
            **{field.name: getattr(self, field.name)[:count] for field in fields(self)}
        )


def load_split(data_config: dict[str, Any], split: str, cache_dir: Path) -> FeatureSplit:
    """Download or load one published NPZ split and validate its data contract."""
    if split not in {"train", "val", "test"}:
        raise ValueError(f"Unsupported split: {split}")

    if "split_path" in data_config:
        base_path = Path(data_config["split_path"])
        if not base_path.is_absolute():
            # Relative to cache_dir parent (project root)
            candidate = cache_dir.parent / base_path
            path = candidate if candidate.exists() else base_path
        else:
            path = base_path
        path = path / f"{split}.npz"
    else:
        path = download_file(
            repo_id=data_config["repo_id"],
            repo_type="dataset",
            revision=data_config["revision"],
            filename=f"{data_config['artifact_path']}/data/{split}.npz",
            cache_dir=cache_dir,
        )

    with np.load(path, allow_pickle=False) as arrays:
        names = tuple(field.name for field in fields(FeatureSplit))
        if set(arrays.files) != set(names):
            raise ValueError(f"Unexpected NPZ fields: {sorted(arrays.files)}")
        result = FeatureSplit(**{name: arrays[name] for name in names})
    validate_split(result, num_classes=int(data_config["num_classes"]), nodes=result.X.shape[2])
    return result


def select_view(split: FeatureSplit, view: str) -> FeatureSplit:
    """Select a classifier view from the canonical Pose42, Pose53, or Pose59 layout."""
    nodes = split.X.shape[2]
    expected = {
        "pose42": 42,
        "pose53": 53,
        "pose59": 59,
        "spoter54": 54,
        "slgcn27": 27,
    }
    if view not in expected:
        raise ValueError(f"Unsupported feature view: {view}")
    if nodes == expected[view]:
        return split

    if nodes == 59 and view in {"spoter54", "slgcn27"}:
        return select_view(select_view(split, "pose53"), view)

    if nodes == 53 and view == "slgcn27":
        # Official SL-GCN sign/27 order: nose; upper body; ten points per hand.
        hand_indices = (0, 4, 5, 8, 9, 12, 13, 16, 17, 20)
        indices = (48, 0, 1, 2, 3, 4, 5, *(6 + index for index in hand_indices), *(27 + index for index in hand_indices))
        return FeatureSplit(
            X=np.ascontiguousarray(split.X[:, :, indices, :]),
            y=split.y,
            video_id=split.video_id,
            signer_id=split.signer_id,
            frame_indices=split.frame_indices,
            valid_frame_mask=split.valid_frame_mask,
            keypoint_valid_mask=np.ascontiguousarray(split.keypoint_valid_mask[:, :, indices]),
        )

    if nodes == 53 and view == "spoter54":
        # Pose53 is body(0:6), left hand(6:27), right hand(27:48), face(48:53).
        source_x = split.X
        neck = (source_x[:, :, 0, :] + source_x[:, :, 1, :]) / 2.0
        neck[:, :, 2] = np.minimum(source_x[:, :, 0, 2], source_x[:, :, 1, 2])
        result_x = np.concatenate(
            (source_x[:, :, 48:53], neck[:, :, None, :], source_x[:, :, :48]), axis=2
        )
        shoulder_mask = np.minimum(
            split.keypoint_valid_mask[:, :, 0], split.keypoint_valid_mask[:, :, 1]
        )
        result_mask = np.concatenate(
            (
                split.keypoint_valid_mask[:, :, 48:53],
                shoulder_mask[:, :, None],
                split.keypoint_valid_mask[:, :, :48],
            ),
            axis=2,
        )
        return FeatureSplit(
            X=np.ascontiguousarray(result_x),
            y=split.y,
            video_id=split.video_id,
            signer_id=split.signer_id,
            frame_indices=split.frame_indices,
            valid_frame_mask=split.valid_frame_mask,
            keypoint_valid_mask=np.ascontiguousarray(result_mask),
        )

    indices: slice | tuple[int, ...]
    if nodes == 53 and view == "pose42":
        indices = slice(6, 48)
    elif nodes == 59 and view == "pose42":
        indices = slice(17, 59)
    elif nodes == 59 and view == "pose53":
        indices = (*range(5, 11), *range(17, 59), *range(5))
    else:
        raise ValueError(f"Cannot select {view} from Pose{nodes}")

    return FeatureSplit(
        X=np.ascontiguousarray(split.X[:, :, indices, :]),
        y=split.y,
        video_id=split.video_id,
        signer_id=split.signer_id,
        frame_indices=split.frame_indices,
        valid_frame_mask=split.valid_frame_mask,
        keypoint_valid_mask=np.ascontiguousarray(split.keypoint_valid_mask[:, :, indices]),
    )


def validate_split(split: FeatureSplit, *, num_classes: int, nodes: int) -> None:
    if split.X.ndim != 4 or split.X.shape[2:] != (nodes, 3):
        raise ValueError(f"Expected X [N,T,{nodes},3], got {split.X.shape}")
    sample_count, frame_count = split.X.shape[:2]
    if split.X.dtype != np.float32 or not np.isfinite(split.X).all():
        raise ValueError("X must be finite float32")
    if np.any(split.X[..., :2] < 0.0) or np.any(split.X[..., :2] > 1.0):
        raise ValueError("XY coordinates must be normalized to [0, 1]")
    if split.y.shape != (sample_count,) or split.y.dtype != np.int64:
        raise ValueError("y must be int64 [N]")
    if np.any(split.y < 0) or np.any(split.y >= num_classes):
        raise ValueError("y contains an invalid class")
    if split.video_id.shape != (sample_count,) or split.video_id.dtype != np.dtype("U6"):
        raise ValueError("video_id must be U6 [N]")
    if split.signer_id.shape != (sample_count,) or split.signer_id.dtype != np.dtype("U3"):
        raise ValueError("signer_id must be U3 [N]")
    if (
        split.frame_indices.shape != (sample_count, frame_count)
        or split.frame_indices.dtype != np.int32
    ):
        raise ValueError("frame_indices must be int32 [N,T]")
    if (
        split.valid_frame_mask.shape != (sample_count, frame_count)
        or split.valid_frame_mask.dtype != np.uint8
    ):
        raise ValueError("valid_frame_mask must be uint8 [N,T]")
    if (
        split.keypoint_valid_mask.shape != (sample_count, frame_count, nodes)
        or split.keypoint_valid_mask.dtype != np.uint8
    ):
        raise ValueError("keypoint_valid_mask must be uint8 [N,T,V]")
    if (
        not np.isin(split.valid_frame_mask, (0, 1)).all()
        or not np.isin(split.keypoint_valid_mask, (0, 1)).all()
    ):
        raise ValueError("masks must contain only 0 or 1")
    invalid = split.valid_frame_mask == 0
    if np.any(split.frame_indices[invalid] != -1):
        raise ValueError("padded frame_indices must be -1")
    if np.any(split.X[invalid] != 0) or np.any(split.keypoint_valid_mask[invalid] != 0):
        raise ValueError("padded frames must be zero-filled")
    if np.any(split.frame_indices[~invalid] < 0):
        raise ValueError("valid frame_indices must be non-negative")


def make_loader(split: FeatureSplit, batch_size: int, *, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(split.X),
        torch.from_numpy(split.y),
        torch.from_numpy(split.valid_frame_mask),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
