from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from src.yolo.dataset import (
    YoloDataset,
    epoch_indices,
    loader,
    make_targets,
)


def test_epoch_indices_deterministic_and_distinct() -> None:
    total = 100
    perm_e0_s42_a = epoch_indices(total, seed=42, epoch=0)
    perm_e0_s42_b = epoch_indices(total, seed=42, epoch=0)
    assert np.array_equal(perm_e0_s42_a, perm_e0_s42_b)
    assert len(perm_e0_s42_a) == total
    assert len(set(perm_e0_s42_a.tolist())) == total

    perm_e1_s42 = epoch_indices(total, seed=42, epoch=1)
    assert not np.array_equal(perm_e0_s42_a, perm_e1_s42)

    perm_e0_s43 = epoch_indices(total, seed=43, epoch=0)
    assert not np.array_equal(perm_e0_s42_a, perm_e0_s43)
    assert np.array_equal(epoch_indices(5, seed=42, epoch=0, shuffle=False), np.arange(5))


def test_epoch_indices_mid_epoch_resume_slice() -> None:
    total = 50
    full_perm = epoch_indices(total, seed=42, epoch=3)
    start_sample = 17
    sliced_perm = epoch_indices(total, seed=42, epoch=3, start_sample=start_sample)
    assert np.array_equal(sliced_perm, full_perm[start_sample:])
    assert len(sliced_perm) == total - start_sample

    with pytest.raises(ValueError, match="Invalid start_sample"):
        epoch_indices(total, seed=42, epoch=0, start_sample=51)


def test_yolo_dataset_reads_with_indices_and_images(tmp_path: Path) -> None:
    split = "train"
    images = 4
    frames_per_video = 2
    image_size = 64

    img_dir = tmp_path / split / "images"
    img_dir.mkdir(parents=True)

    dummy_img = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    for video in range(2):
        for frame in range(frames_per_video):
            # Write a distinguishing pixel pattern
            img = dummy_img.copy()
            img[0, 0] = [video, frame, 255]
            cv2.imwrite(str(img_dir / f"{video:05d}_{frame:02d}.jpg"), img)

    boxes = np.zeros((images, 3, 4), dtype=np.float32)
    classes = np.zeros((images, 3), dtype=np.int8)
    person_keypoints = np.zeros((images, 17, 3), dtype=np.float32)
    hand_keypoints = np.zeros((images, 2, 21, 3), dtype=np.float32)

    for i in range(images):
        boxes[i, 0] = [10.0 * i, 10.0 * i, 20.0 * i, 20.0 * i]
        classes[i] = [0, 1, -1]
        person_keypoints[i, 0] = [float(i), float(i), 1.0]
        hand_keypoints[i, 0, 0] = [float(i), float(i), 1.0]

    np.save(tmp_path / f"{split}_boxes.npy", boxes)
    np.save(tmp_path / f"{split}_classes.npy", classes)
    np.save(tmp_path / f"{split}_person_keypoints.npy", person_keypoints)
    np.save(tmp_path / f"{split}_hand_keypoints.npy", hand_keypoints)

    # 1. Test normal sequential dataset
    ds_seq = YoloDataset(
        tmp_path,
        split,
        images=images,
        frames_per_video=frames_per_video,
        image_size=image_size,
    )
    assert len(ds_seq) == 4
    item0 = ds_seq[0]
    assert item0["image"].shape == (3, image_size, image_size)
    assert item0["boxes"][0, 0].item() == 0.0
    assert item0["person_keypoints"].shape == (17, 3)
    assert item0["hand_keypoints"].shape == (2, 21, 3)

    # 2. Test dataset with permuted indices
    perm = np.array([3, 1, 2, 0], dtype=np.int64)
    ds_perm = YoloDataset(
        tmp_path,
        split,
        images=images,
        frames_per_video=frames_per_video,
        image_size=image_size,
        indices=perm,
    )
    assert len(ds_perm) == 4
    assert ds_perm[0]["boxes"][0, 0].item() == 30.0
    assert ds_perm[1]["boxes"][0, 0].item() == 10.0

    # 3. Test dataset with start resume
    ds_resume = YoloDataset(
        tmp_path,
        split,
        images=images,
        frames_per_video=frames_per_video,
        image_size=image_size,
        indices=perm,
        start=2,
    )
    assert len(ds_resume) == 2
    assert ds_resume[0]["boxes"][0, 0].item() == 20.0
    assert ds_resume[1]["boxes"][0, 0].item() == 0.0


def test_yolo_dataset_rejects_incorrect_target_shape_or_dtype(tmp_path: Path) -> None:
    arrays = {
        "train_boxes": np.zeros((2, 3, 4), dtype=np.float32),
        "train_classes": np.zeros((2, 3), dtype=np.int8),
        "train_person_keypoints": np.zeros((2, 17, 3), dtype=np.float32),
        "train_hand_keypoints": np.zeros((2, 2, 21, 3), dtype=np.float32),
    }
    for name, array in arrays.items():
        np.save(tmp_path / f"{name}.npy", array)

    arrays["train_boxes"] = np.zeros((2, 4, 4), dtype=np.float32)
    np.save(tmp_path / "train_boxes.npy", arrays["train_boxes"])
    with pytest.raises(ValueError, match="train_boxes.npy must have shape"):
        YoloDataset(tmp_path, "train", images=2, frames_per_video=1, image_size=64)

    arrays["train_boxes"] = np.zeros((2, 3, 4), dtype=np.float32)
    arrays["train_classes"] = np.zeros((2, 3), dtype=np.int64)
    np.save(tmp_path / "train_boxes.npy", arrays["train_boxes"])
    np.save(tmp_path / "train_classes.npy", arrays["train_classes"])
    with pytest.raises(ValueError, match="train_classes.npy must have dtype int8"):
        YoloDataset(tmp_path, "train", images=2, frames_per_video=1, image_size=64)


def test_loader_preserves_order(tmp_path: Path) -> None:
    split = "train"
    images = 4
    frames_per_video = 2
    image_size = 32

    img_dir = tmp_path / split / "images"
    img_dir.mkdir(parents=True)
    for video in range(2):
        for frame in range(frames_per_video):
            cv2.imwrite(
                str(img_dir / f"{video:05d}_{frame:02d}.jpg"),
                np.zeros((image_size, image_size, 3), dtype=np.uint8),
            )

    np.save(tmp_path / f"{split}_boxes.npy", np.zeros((images, 3, 4), dtype=np.float32))
    np.save(tmp_path / f"{split}_classes.npy", np.zeros((images, 3), dtype=np.int8))
    np.save(tmp_path / f"{split}_person_keypoints.npy", np.zeros((images, 17, 3), dtype=np.float32))
    np.save(tmp_path / f"{split}_hand_keypoints.npy", np.zeros((images, 2, 21, 3), dtype=np.float32))

    perm = np.array([2, 0, 3, 1], dtype=np.int64)
    dataset = YoloDataset(
        tmp_path,
        split,
        images=images,
        frames_per_video=frames_per_video,
        image_size=image_size,
        indices=perm,
    )
    dl = loader(dataset, batch_size=2, workers=0)
    assert isinstance(dl.sampler, torch.utils.data.SequentialSampler)

    batches = list(dl)
    assert len(batches) == 2
    assert batches[0]["image"].shape == (2, 3, image_size, image_size)


def test_make_targets_normalized_and_device() -> None:
    batch = {
        "image": torch.full((2, 3, 640, 640), fill_value=255, dtype=torch.uint8),
        "boxes": torch.tensor(
            [
                [[0, 0, 640, 640], [10, 20, 50, 60], [100, 120, 200, 220]],
                [[50, 50, 250, 250], [0, 0, 0, 0], [0, 0, 0, 0]],
            ],
            dtype=torch.float32,
        ),
        "classes": torch.tensor(
            [
                [0, 1, 2],
                [0, -1, -1],
            ],
            dtype=torch.int8,
        ),
        "person_keypoints": torch.zeros((2, 17, 3), dtype=torch.float32),
        "hand_keypoints": torch.zeros((2, 2, 21, 3), dtype=torch.float32),
    }
    batch["person_keypoints"][0, 0] = torch.tensor([320.0, 160.0, 2.0])
    batch["hand_keypoints"][0, 0, 0] = torch.tensor([30.0, 40.0, 2.0])

    pose, hands, audit = make_targets(batch, device=torch.device("cpu"), image_size=640)

    assert pose["img"].shape == (2, 3, 640, 640)
    assert torch.allclose(pose["img"], torch.ones_like(pose["img"]))
    assert pose["batch_idx"].tolist() == [0, 1]
    assert pose["cls"].tolist() == [0, 0]
    # xywh for first box [0, 0, 640, 640] -> [(0+640)/(2*640), (0+640)/(2*640), 640/640, 640/640] = [0.5, 0.5, 1.0, 1.0]
    assert pose["bboxes"][0].tolist() == [0.5, 0.5, 1.0, 1.0]
    assert pose["keypoints"][0, 0].tolist() == [0.5, 0.25, 2.0]

    # hands
    assert hands["batch_idx"].tolist() == [0, 0]
    assert hands["cls"].tolist() == [0, 1]  # class 1 -> 0 (left_hand), class 2 -> 1 (right_hand)
    assert hands["keypoints"].shape == (2, 21, 3)
    assert hands["keypoints"][0, 0].tolist() == [30.0 / 640, 40.0 / 640, 2.0]
    assert audit == {"person": 2, "left_hand": 1, "right_hand": 1}


def test_apply_augmentation_geometry_and_determinism() -> None:
    from src.yolo.transforms import apply_augmentation

    img = np.full((100, 100, 3), 128, dtype=np.uint8)
    boxes = np.array([
        [10.0, 10.0, 90.0, 90.0],
        [20.0, 30.0, 40.0, 50.0],
        [60.0, 70.0, 80.0, 90.0],
    ], dtype=np.float32)
    classes = np.array([0, 1, 2], dtype=np.int8)
    person_kpts = np.zeros((17, 3), dtype=np.float32)
    person_kpts[0] = [50.0, 50.0, 2.0]
    hand_kpts = np.zeros((2, 21, 3), dtype=np.float32)
    hand_kpts[0, 0] = [30.0, 40.0, 2.0]
    hand_kpts[1, 0] = [70.0, 80.0, 2.0]

    config = {
        "hsv_h": 0.01,
        "hsv_s": 0.30,
        "hsv_v": 0.20,
        "translate": 0.05,
        "scale": 0.10,
    }

    # Determinism: same seed, epoch, sample_index gives identical output
    res1 = apply_augmentation(
        img.copy(), boxes.copy(), classes.copy(), person_kpts.copy(), hand_kpts.copy(),
        image_size=100, seed=42, epoch=0, sample_index=7, config=config,
    )
    res2 = apply_augmentation(
        img.copy(), boxes.copy(), classes.copy(), person_kpts.copy(), hand_kpts.copy(),
        image_size=100, seed=42, epoch=0, sample_index=7, config=config,
    )
    np.testing.assert_array_equal(res1[0], res2[0])
    np.testing.assert_array_equal(res1[1], res2[1])
    np.testing.assert_array_equal(res1[2], res2[2])
    np.testing.assert_array_equal(res1[3], res2[3])
    np.testing.assert_array_equal(res1[4], res2[4])

    # Distinctness: different sample index gives different output
    res3 = apply_augmentation(
        img.copy(), boxes.copy(), classes.copy(), person_kpts.copy(), hand_kpts.copy(),
        image_size=100, seed=42, epoch=0, sample_index=8, config=config,
    )
    assert not np.array_equal(res1[0], res3[0])

    # No horizontal flip: left hand slot 1 remains class 1, right hand slot 2 remains class 2
    assert res1[2][1] == 1
    assert res1[2][2] == 2


def test_yolo_dataset_augmentation_and_val_unmodified(tmp_path: Path) -> None:
    split = "train"
    images = 2
    frames_per_video = 1
    image_size = 64

    img_dir = tmp_path / split / "images"
    img_dir.mkdir(parents=True)
    cv2.imwrite(str(img_dir / "00000_00.jpg"), np.full((image_size, image_size, 3), 100, dtype=np.uint8))
    cv2.imwrite(str(img_dir / "00001_00.jpg"), np.full((image_size, image_size, 3), 100, dtype=np.uint8))

    boxes = np.array([
        [[10.0, 10.0, 50.0, 50.0], [15.0, 15.0, 25.0, 25.0], [0.0, 0.0, 0.0, 0.0]],
        [[10.0, 10.0, 50.0, 50.0], [0.0, 0.0, 0.0, 0.0], [35.0, 35.0, 45.0, 45.0]],
    ], dtype=np.float32)
    classes = np.array([[0, 1, -1], [0, -1, 2]], dtype=np.int8)
    person_keypoints = np.zeros((images, 17, 3), dtype=np.float32)
    hand_keypoints = np.zeros((images, 2, 21, 3), dtype=np.float32)

    np.save(tmp_path / f"{split}_boxes.npy", boxes)
    np.save(tmp_path / f"{split}_classes.npy", classes)
    np.save(tmp_path / f"{split}_person_keypoints.npy", person_keypoints)
    np.save(tmp_path / f"{split}_hand_keypoints.npy", hand_keypoints)

    # Dataset without augmentation returns exact arrays
    ds_plain = YoloDataset(tmp_path, split, images=images, frames_per_video=frames_per_video, image_size=image_size)
    assert np.array_equal(ds_plain[0]["boxes"].numpy(), boxes[0])

    # Dataset with augmentation returns deterministic altered item
    aug_cfg = {"enabled": True, "hsv_h": 0.01, "hsv_s": 0.3, "hsv_v": 0.2, "translate": 0.1, "scale": 0.2}
    ds_aug = YoloDataset(
        tmp_path,
        split,
        images=images,
        frames_per_video=frames_per_video,
        image_size=image_size,
        augmentation=aug_cfg,
        seed=42,
        epoch=1,
    )
    item0_a = ds_aug[0]
    item0_b = ds_aug[0]
    assert torch.equal(item0_a["image"], item0_b["image"])
    assert torch.equal(item0_a["boxes"], item0_b["boxes"])
