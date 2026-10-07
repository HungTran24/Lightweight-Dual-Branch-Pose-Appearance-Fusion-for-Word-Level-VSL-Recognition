from pathlib import Path

import torch

from src.config import load_config
from src.models.spoter import SPOTER


def test_spoter_forward_uses_frame_mask() -> None:
    model = SPOTER(nodes=54, classes=7, config={"max_frames": 30, "dropout": 0.0})
    model.eval()
    x = torch.randn(2, 30, 54, 3)
    mask = torch.ones(2, 30, dtype=torch.uint8)
    mask[1, 12:] = 0
    masked_variant = x.clone()
    masked_variant[1, 12:] = 0
    with torch.no_grad():
        output = model(x, mask)
        variant_output = model(masked_variant, mask)
    assert output.shape == (2, 7)
    assert torch.isfinite(output).all()
    torch.testing.assert_close(output, variant_output)


def test_spoter_smoke_config() -> None:
    path = (
        Path(__file__).parents[1]
        / "configs/experiments/e019_spoter_pose54_uniform30_smoke_s42.yaml"
    )
    config = load_config(path)
    assert config["feature"]["view"] == "spoter54"
    assert config["data"]["subset"] == {"train": 64, "val": 32}
    assert config["integrations"]["wandb"]["enabled"] is False
    assert config["integrations"]["hf_checkpoints"]["enabled"] is False
