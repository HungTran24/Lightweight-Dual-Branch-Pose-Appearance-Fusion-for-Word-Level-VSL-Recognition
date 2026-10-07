from __future__ import annotations

import numpy as np
import torch

import src.yolo as yolo_pkg
import src.yolo_data as yolo_data_shim
from src.frame_sampling.uniform import uniform_indices
from src.models.yolo_dual_head import decode
from src.yolo.selection import _identity, canonical_selection
from src.yolo.transforms import (
    _clip_box,
    _hand_box,
    _labels,
    _letterbox,
    _open_arrays,
    _visible,
)
from src.yolo_pilot import make_targets


def test_package_and_shim_exports() -> None:
    assert hasattr(yolo_pkg, "canonical_selection")
    assert hasattr(yolo_pkg, "materialize")
    for name in [
        "_clip_box",
        "_hand_box",
        "_identity",
        "_labels",
        "_letterbox",
        "_open_arrays",
        "_visible",
        "canonical_selection",
        "materialize",
    ]:
        assert hasattr(yolo_data_shim, name)


def test_letterbox_and_clip_box() -> None:
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    image, scale, pad_x, pad_y = _letterbox(frame, image_size=200)
    assert image.shape == (200, 200, 3)
    assert scale == 1.0
    assert pad_x == 0
    assert pad_y == 50

    assert _clip_box(np.array([10.0, 20.0, 50.0, 60.0]), 100, 100) == [10.0, 20.0, 50.0, 60.0]
    assert _clip_box(np.array([-5.0, -10.0, 120.0, 150.0]), 100, 100) == [0.0, 0.0, 100.0, 100.0]
    assert _clip_box(np.array([50.0, 50.0, 50.0, 50.0]), 100, 100) is None


def test_visible_and_hand_box() -> None:
    pts = np.array([[10.0, 20.0, 0.9], [5.0, 5.0, 0.1], [30.0, 40.0, 0.8]])
    vis = _visible(pts, width=100, height=100, threshold=0.5)
    assert np.array_equal(vis, np.array([True, False, True]))

    box = _hand_box(pts, width=100, height=100, threshold=0.5)
    assert box == [10.0, 20.0, 30.0, 40.0]


def test_identity_deterministic() -> None:
    config = {
        "data": {"foo": 1},
        "model": {"image_size": 640, "pose_keypoints": 53},
    }
    selected = {
        "train": [{"video_id": "000001"}],
        "val": [{"video_id": "000002"}],
    }
    hash1 = _identity(config, selected)
    hash2 = _identity(config, selected)
    assert hash1 == hash2 and len(hash1) == 64


def test_uniform_24_is_deterministic() -> None:
    assert uniform_indices(100, 24).tolist() == [round(index * 99 / 23) for index in range(24)]


def test_selection_skips_known_decode_defect(tmp_path) -> None:
    header = "video_id,split,signer_id,class_id,part_id,num_frames\n"
    (tmp_path / "train.csv").write_text(
        header + "019442,train,024,35,6,54\n" + "019443,train,024,35,6,61\n",
        encoding="utf-8",
    )
    (tmp_path / "val.csv").write_text(
        header + "003414,val,006,0,1,54\n",
        encoding="utf-8",
    )
    selected = canonical_selection(tmp_path, known_decode_defect="019442")
    assert selected["train"][0]["video_id"] == "019443"
    assert selected["val"][0]["video_id"] == "003414"


def test_target_split_preserves_pose_visibility() -> None:
    batch = {
        "image": torch.zeros((1, 3, 640, 640), dtype=torch.uint8),
        "boxes": torch.tensor([[[0, 0, 640, 640], [10, 20, 30, 40], [0, 0, 0, 0]]]),
        "classes": torch.tensor([[0, 1, -1]], dtype=torch.int8),
        "person_keypoints": torch.zeros((1, 17, 3)),
        "hand_keypoints": torch.zeros((1, 2, 21, 3)),
    }
    batch["person_keypoints"][0, 0] = torch.tensor([320, 160, 2])
    batch["hand_keypoints"][0, 0, 0] = torch.tensor([15, 25, 2])
    pose, hands, audit = make_targets(batch, device=torch.device("cpu"), image_size=640)
    assert pose["keypoints"][0, 0].tolist() == [0.5, 0.25, 2.0]
    assert hands["cls"].tolist() == [0]
    assert hands["keypoints"][0, 0].tolist() == [15.0 / 640, 25.0 / 640, 2.0]
    assert audit == {"person": 1, "left_hand": 1, "right_hand": 0}


def test_labels_and_open_arrays(tmp_path) -> None:
    class DummyShard:
        frame_person_offsets = np.array([0, 1], dtype=np.int32)
        person_score = np.array([0.95], dtype=np.float32)
        person_bbox_xyxy = np.array([[10.0, 10.0, 100.0, 100.0]], dtype=np.float32)
        keypoints = np.zeros((1, 133, 3), dtype=np.float32)

    shard = DummyShard()
    # Person keypoint 0 (nose)
    shard.keypoints[0, 0] = [20.0, 20.0, 0.9]
    # Left hand keypoints (indices 91 and 92)
    shard.keypoints[0, 91] = [30.0, 30.0, 0.9]
    shard.keypoints[0, 92] = [50.0, 60.0, 0.9]
    # Right hand keypoints (indices 112 and 113)
    shard.keypoints[0, 112] = [40.0, 40.0, 0.9]
    shard.keypoints[0, 113] = [70.0, 80.0, 0.9]

    boxes, classes, person_kpts, hand_kpts = _labels(
        shard,
        frame_position=0,
        width=200,
        height=200,
        threshold=0.3,
        pose_indices=slice(0, 17),
        hand_slices=(slice(91, 112), slice(112, 133)),
    )
    assert boxes.shape == (3, 4)
    assert classes.tolist() == [0, 1, 2]
    assert person_kpts.shape == (17, 3)
    assert person_kpts[0, 2] == 2.0
    assert hand_kpts.shape == (2, 21, 3)
    assert hand_kpts[0, 0, 2] == 2.0
    assert hand_kpts[1, 0, 2] == 2.0

    # Test open arrays
    selected = {"train": [{"video_id": "000001"}]}
    arrays = _open_arrays(tmp_path, selected, frames=2)
    assert arrays["train_boxes"].shape == (2, 3, 4)
    assert arrays["train_classes"].shape == (2, 3)
    assert arrays["train_person_keypoints"].shape == (2, 17, 3)
    assert arrays["train_hand_keypoints"].shape == (2, 2, 21, 3)


def test_decode_supports_pose_rows_and_zero_detections(monkeypatch) -> None:
    def dummy_nms(prediction, conf_thres, iou_thres, nc, max_det):
        # Return 1 detection for image 0, 0 detections for image 1
        return [
            torch.zeros((1, 6 + (17 if nc == 1 else 21) * 3)),
            torch.zeros((0, 6 + (17 if nc == 1 else 21) * 3)),
        ]

    monkeypatch.setattr("ultralytics.utils.nms.non_max_suppression", dummy_nms)

    output = {
        "person": torch.zeros((2, 57, 10)),
        "hands": torch.zeros((2, 69, 10)),
    }
    decoded = decode(output, confidence=0.25, iou=0.7, max_detections=300)
    assert len(decoded) == 2
    assert decoded[0]["person"].shape == (1, 57)
    assert decoded[0]["hands"].shape == (1, 69)
    assert decoded[1]["person"].shape == (0, 57)
    assert decoded[1]["hands"].shape == (0, 69)
