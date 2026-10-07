from __future__ import annotations

import torch
from torch import nn

from src.yolo.onnx_export import RawDualHeadWrapper


class _DualHead(nn.Module):
    def forward(self, images: torch.Tensor) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        batch = len(images)
        return {
            "person": (torch.ones((batch, 58, 10)), torch.empty(0)),
            "hands": (torch.full((batch, 69, 10), 2.0), torch.empty(0)),
        }


def test_raw_dual_head_wrapper_returns_primary_head_tensors() -> None:
    person, hands = RawDualHeadWrapper(_DualHead())(torch.zeros((1, 3, 640, 640)))

    assert person.shape == (1, 58, 10)
    assert hands.shape == (1, 69, 10)
    assert float(person.mean()) == 1.0
    assert float(hands.mean()) == 2.0
