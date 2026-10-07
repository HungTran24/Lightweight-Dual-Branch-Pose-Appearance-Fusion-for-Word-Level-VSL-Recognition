"""Reusable YOLO dataset, target transform, and dataloader primitives."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .transforms import apply_augmentation


def epoch_indices(
    total_samples: int,
    *,
    seed: int,
    epoch: int,
    start_sample: int = 0,
    shuffle: bool = True,
) -> np.ndarray:
    """Generate the epoch order, optionally shuffled, sliced from start_sample."""
    if total_samples < 0:
        raise ValueError(f"total_samples must be non-negative, got {total_samples}")
    if not 0 <= start_sample <= total_samples:
        raise ValueError(f"Invalid start_sample: {start_sample} for total_samples: {total_samples}")
    if not shuffle:
        return np.arange(total_samples, dtype=np.int64)[start_sample:]
    rng = np.random.default_rng(seed + epoch)
    permutation = rng.permutation(total_samples)
    return permutation[start_sample:]


class YoloDataset(Dataset[dict[str, torch.Tensor]]):
    """Read one letterboxed JPEG and its mmap-backed targets."""

    def __init__(
        self,
        root: Path,
        split: str,
        *,
        images: int,
        frames_per_video: int,
        image_size: int,
        indices: Sequence[int] | np.ndarray | None = None,
        start: int = 0,
        augmentation: dict[str, Any] | None = None,
        seed: int = 42,
        epoch: int = 0,
    ) -> None:
        self.root = root
        self.split = split
        self.total = images
        self.frames = frames_per_video
        self.image_size = image_size
        self.augmentation = augmentation
        self.seed = seed
        self.epoch = epoch
        self.boxes_path = root / f"{split}_boxes.npy"
        self.classes_path = root / f"{split}_classes.npy"
        self.person_keypoints_path = root / f"{split}_person_keypoints.npy"
        self.hand_keypoints_path = root / f"{split}_hand_keypoints.npy"
        self.boxes = self._open_array(self.boxes_path, (3, 4), np.float32)
        self.classes = self._open_array(self.classes_path, (3,), np.int8)
        self.person_keypoints = self._open_array(self.person_keypoints_path, (17, 3), np.float32)
        self.hand_keypoints = self._open_array(self.hand_keypoints_path, (2, 21, 3), np.float32)

        if indices is not None:
            arr = np.asarray(indices, dtype=np.int64)
            if arr.ndim != 1:
                raise ValueError(f"indices must be 1D, got shape {arr.shape}")
            if len(arr) > 0 and (int(arr.min()) < 0 or int(arr.max()) >= images):
                raise ValueError(f"indices out of range [0, {images})")
            if start > 0:
                if start > len(arr):
                    raise ValueError(f"Invalid start index: {start} for indices length {len(arr)}")
                arr = arr[start:]
            self.indices = arr
        else:
            if not 0 <= start <= images:
                raise ValueError(f"Invalid {split} start index: {start}")
            self.indices = np.arange(start, images, dtype=np.int64)

    def _open_array(self, path: Path, sample_shape: tuple[int, ...], dtype: Any) -> np.ndarray:
        array = np.load(path, mmap_mode="r")
        expected_shape = (self.total, *sample_shape)
        if array.shape != expected_shape:
            raise ValueError(f"{path.name} must have shape {expected_shape}, got {array.shape}")
        expected_dtype = np.dtype(dtype)
        if array.dtype != expected_dtype:
            raise ValueError(f"{path.name} must have dtype {expected_dtype}, got {array.dtype}")
        return array

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        actual = int(self.indices[index])
        video = actual // self.frames
        frame = actual % self.frames
        path = self.root / self.split / "images" / f"{video:05d}_{frame:02d}.jpg"
        image = cv2.imread(str(path))
        if image is None:
            raise FileNotFoundError(path)
        if image.shape[:2] != (self.image_size, self.image_size):
            raise ValueError(f"Unexpected image shape for {path}: {image.shape}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        boxes = self.boxes[actual].copy()
        classes = self.classes[actual].copy()
        person_keypoints = self.person_keypoints[actual].copy()
        hand_keypoints = self.hand_keypoints[actual].copy()

        if self.augmentation is not None and self.augmentation.get("enabled", False):
            image, boxes, classes, person_keypoints, hand_keypoints = apply_augmentation(
                image,
                boxes,
                classes,
                person_keypoints,
                hand_keypoints,
                image_size=self.image_size,
                seed=self.seed,
                epoch=self.epoch,
                sample_index=actual,
                config=self.augmentation,
            )

        return {
            "image": torch.from_numpy(image).permute(2, 0, 1).contiguous(),
            "boxes": torch.from_numpy(boxes),
            "classes": torch.from_numpy(classes),
            "person_keypoints": torch.from_numpy(person_keypoints),
            "hand_keypoints": torch.from_numpy(hand_keypoints),
        }


def make_targets(
    batch: dict[str, torch.Tensor],
    *,
    device: torch.device | str,
    image_size: int,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, int]]:
    """Convert raw dataset tensors into YOLO dual-head model targets."""
    target_device = torch.device(device) if isinstance(device, str) else device
    images = batch["image"].to(target_device, non_blocking=True).float().div_(255)
    boxes = batch["boxes"].to(target_device, non_blocking=True)
    classes = batch["classes"].to(target_device, non_blocking=True)
    person_keypoints = batch["person_keypoints"].to(target_device, non_blocking=True)
    hand_keypoints = batch["hand_keypoints"].to(target_device, non_blocking=True)

    def xywh(values: torch.Tensor) -> torch.Tensor:
        result = values.clone()
        result[:, 0] = (values[:, 0] + values[:, 2]) / (2 * image_size)
        result[:, 1] = (values[:, 1] + values[:, 3]) / (2 * image_size)
        result[:, 2] = (values[:, 2] - values[:, 0]) / image_size
        result[:, 3] = (values[:, 3] - values[:, 1]) / image_size
        return result

    person_batch = (classes[:, 0] == 0).nonzero(as_tuple=True)[0]
    pose_keypoints = person_keypoints[person_batch].clone()
    pose_keypoints[..., :2] /= image_size

    pose = {
        "img": images,
        "batch_idx": person_batch,
        "cls": torch.zeros(len(person_batch), dtype=torch.long, device=target_device),
        "bboxes": xywh(boxes[person_batch, 0]),
        "keypoints": pose_keypoints,
    }

    hand_valid = classes[:, 1:] >= 1
    hand_batch, hand_slot = hand_valid.nonzero(as_tuple=True)
    raw_cls = classes[hand_batch, hand_slot + 1].long()
    hand_cls = torch.where(raw_cls >= 1, raw_cls - 1, hand_slot.long())

    raw_hand_kpts = hand_keypoints[hand_batch, hand_slot].clone()
    raw_hand_kpts[..., :2] /= image_size

    hands = {
        "img": images,
        "batch_idx": hand_batch,
        "cls": hand_cls,
        "bboxes": xywh(boxes[hand_batch, hand_slot + 1]),
        "keypoints": raw_hand_kpts,
    }
    audit = {
        "person": len(person_batch),
        "left_hand": int((hands["cls"] == 0).sum()),
        "right_hand": int((hands["cls"] == 1).sum()),
    }
    return pose, hands, audit


def loader(
    dataset: YoloDataset,
    *,
    batch_size: int,
    workers: int = 0,
    pin_memory: bool = True,
    persistent_workers: bool | None = None,
) -> DataLoader[dict[str, torch.Tensor]]:
    """Build a DataLoader with shuffle=False (sample order is managed by YoloDataset)."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=pin_memory,
        persistent_workers=workers > 0 if persistent_workers is None else persistent_workers,
    )
