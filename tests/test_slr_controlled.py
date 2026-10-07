from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from src.config import load_config
from src.data import FeatureSplit, select_view
from src.models.slgcn_bone_controlled import BONE_PAIRS
from src.models.slgcn_bone_controlled import Model as SLGCNBoneControlled
from src.models.spoter_pose54_controlled import Model as SPOTERPose54Controlled


def _split(samples: int = 1, nodes: int = 53) -> FeatureSplit:
    rng = np.random.default_rng(7)
    frames = 6
    return FeatureSplit(
        X=rng.random((samples, frames, nodes, 3), dtype=np.float32),
        y=np.zeros(samples, dtype=np.int64),
        video_id=np.array([f"{index:06d}" for index in range(samples)], dtype="U6"),
        signer_id=np.array(["001"] * samples, dtype="U3"),
        frame_indices=np.tile(np.arange(frames, dtype=np.int32), (samples, 1)),
        valid_frame_mask=np.ones((samples, frames), dtype=np.uint8),
        keypoint_valid_mask=np.ones((samples, frames, nodes), dtype=np.uint8),
    )


def test_sl_gcn_view_matches_official_27_node_order() -> None:
    source = _split()
    source.X[0, 0, :, 0] = np.arange(53, dtype=np.float32)
    view = select_view(source, "slgcn27")
    hand = (0, 4, 5, 8, 9, 12, 13, 16, 17, 20)
    expected = np.array([48, 0, 1, 2, 3, 4, 5, *(6 + i for i in hand), *(27 + i for i in hand)])
    np.testing.assert_array_equal(view.X[0, 0, :, 0], expected)
    assert len(BONE_PAIRS) == 26
    assert view.X.shape[2] == 27


def test_spoter_controlled_forward_masks_frames_and_ignores_scores() -> None:
    model = SPOTERPose54Controlled(nodes=54, classes=5, config={"max_frames": 60, "dropout": 0.0}).eval()
    x = torch.rand(2, 60, 54, 3)
    mask = torch.ones(2, 60, dtype=torch.uint8)
    mask[1, 40:] = 0
    variant = x.clone()
    variant[1, 40:] = 0
    with torch.no_grad():
        output = model(x, mask)
        masked_output = model(variant, mask)
        score_variant = x.clone()
        score_variant[..., 2] = torch.rand_like(score_variant[..., 2])
        score_output = model(score_variant, mask)
    assert output.shape == (2, 5)
    torch.testing.assert_close(output, masked_output)
    torch.testing.assert_close(output, score_output)
    with pytest.raises(ValueError, match="at most 60"):
        model(torch.rand(1, 61, 54, 3), torch.ones(1, 61, dtype=torch.uint8))


def test_slgcn_bone_stream_is_translation_invariant() -> None:
    model = SLGCNBoneControlled(nodes=27, classes=5, config={"input_channels": 2, "groups": 16}).eval()
    # Binary-exact coordinates keep this algebraic invariance check independent
    # of random network amplification of floating-point cancellation error.
    x = torch.randint(0, 8, (2, 60, 27, 3), dtype=torch.float32) / 8
    mask = torch.ones(2, 60, dtype=torch.uint8)
    translated = x.clone()
    translated[..., :2] += torch.tensor([0.25, 0.125])
    with torch.no_grad():
        output = model(x, mask)
        translated_output = model(translated, mask)
    assert output.shape == (2, 5)
    torch.testing.assert_close(output, translated_output, atol=1e-5, rtol=1e-5)


def test_controlled_configs_load() -> None:
    root = Path(__file__).parents[1]
    for model, view in (
        ("spoter_pose54_controlled", "spoter54"),
        ("slgcn_bone_controlled", "slgcn27"),
    ):
        prefix = (
            "e034_spoter_pose54_controlled"
            if model == "spoter_pose54_controlled"
            else "e035_slgcn_bone_pose27_controlled"
        )
        path = root / "configs/experiments" / f"{prefix}_u30_s42.yaml"
        config = load_config(path)
        assert config["model"]["name"] == model
        assert config["feature"]["view"] == view
        assert config["experiment"]["seed"] == 42
        assert config["data"]["artifact_path"] == "yolo-e005-pose59-uniform30-a65d7d605573ea13"
        assert config["train"]["epochs"] == 250
        assert config["train"]["min_epochs"] == 100
        assert config["train"]["patience"] == 20
        if model == "spoter_pose54_controlled":
            assert config["model"]["max_frames"] == 30

    benchmark = load_config(root / "configs/experiments/benchmark_slr_controlled_a100.yaml")
    assert [(entry["frames"], entry["nodes"], entry["model"]["name"]) for entry in benchmark["entries"]] == [
        (30, 54, "spoter_pose54_controlled"),
        (30, 27, "slgcn_bone_controlled"),
    ]
