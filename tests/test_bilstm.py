from __future__ import annotations

import torch

from src.models import build_model


def test_bilstm_uses_xy_and_valid_lengths() -> None:
    model = build_model(
        {"name": "bilstm", "input_channels": 2, "hidden_size": 8},
        nodes=42,
        num_classes=4,
    ).eval()
    features = torch.rand(2, 5, 42, 3)
    changed_scores = features.clone()
    changed_scores[..., 2] = 100
    mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]])

    with torch.no_grad():
        logits = model(features, mask)
        score_logits = model(changed_scores, mask)

    assert logits.shape == (2, 4)
    torch.testing.assert_close(logits, score_logits)
