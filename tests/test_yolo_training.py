from __future__ import annotations

from typing import Any

import pytest
import torch
import torch.nn as nn

from src.yolo.training import (
    batch_loss,
    build_yolo_optimizer,
    build_yolo_scheduler,
    train_step,
    validation_loss,
    yolo_lr_factor,
)


class DummyDualHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 8, kernel_size=1)
        self.head_person = nn.Linear(8, 1)
        self.head_hands = nn.Linear(8, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        feat = self.conv(x).mean(dim=[2, 3])
        return {
            "person": self.head_person(feat),
            "hands": self.head_hands(feat),
        }


def dummy_pose_loss(pred: torch.Tensor, target: dict[str, Any]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    total = pred.sum()
    return total, {"pose_loss": total.detach()}


def dummy_hand_loss(pred: torch.Tensor, target: dict[str, Any]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    total = (pred**2).sum()
    return total, {"pose_loss": total.detach()}


def test_batch_loss_and_validation_loss() -> None:
    model = DummyDualHead()
    pose = {"img": torch.randn(2, 3, 16, 16)}
    hands = {"img": pose["img"]}

    loss = batch_loss(model, dummy_pose_loss, dummy_hand_loss, pose, hands, amp=False)
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0
    assert torch.isfinite(loss)

    val_loss = validation_loss(model, dummy_pose_loss, dummy_hand_loss, pose, hands, amp=False)
    assert isinstance(val_loss, torch.Tensor)
    assert torch.isfinite(val_loss)


def test_train_step_updates_weights() -> None:
    model = DummyDualHead()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scaler = torch.amp.GradScaler("cpu", enabled=False)

    pose = {"img": torch.randn(2, 3, 16, 16)}
    hands = {"img": pose["img"]}

    initial_weight = model.conv.weight.clone()
    loss = train_step(
        model,
        optimizer,
        scaler,
        dummy_pose_loss,
        dummy_hand_loss,
        pose,
        hands,
        amp=False,
        return_components=False,
    )
    assert torch.isfinite(loss)
    assert not torch.equal(model.conv.weight, initial_weight)


def test_discriminative_optimizer_groups_are_complete_and_ratioed() -> None:
    class GroupedModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = nn.ModuleList(
                [
                    nn.Linear(2, 2),
                    nn.ModuleDict({"person": nn.Linear(2, 1), "hands": nn.Linear(2, 1)}),
                ]
            )

    model = GroupedModel()
    optimizer = build_yolo_optimizer(
        model,
        {"learning_rate": 1e-4, "hand_learning_rate": 5e-4, "weight_decay": 0.0},
    )
    trainable = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    grouped = [id(parameter) for group in optimizer.param_groups for parameter in group["params"]]

    assert len(optimizer.param_groups) == 2
    assert set(grouped) == trainable
    assert len(grouped) == len(set(grouped))
    assert optimizer.param_groups[0]["name"] == "shared_person"
    assert optimizer.param_groups[1]["name"] == "hands"
    assert optimizer.param_groups[1]["lr"] / optimizer.param_groups[0]["lr"] == pytest.approx(5.0)


def test_warmup_cosine_lr_milestones_and_resume_state() -> None:
    parameter = nn.Parameter(torch.ones(()))
    config = {
        "epochs": 10,
        "warmup_epochs": 1,
        "warmup_start_factor": 0.1,
        "min_lr_factor": 0.1,
    }
    optimizer = torch.optim.AdamW([{"params": [parameter], "lr": 1e-4}])
    scheduler = build_yolo_scheduler(optimizer, config, steps_per_epoch=3)

    factor_args = {
        "total_steps": 30,
        "warmup_steps": 3,
        "warmup_start_factor": config["warmup_start_factor"],
        "min_factor": config["min_lr_factor"],
    }
    assert yolo_lr_factor(0, **factor_args) == pytest.approx(0.1)
    assert yolo_lr_factor(3, **factor_args) == pytest.approx(1.0)
    assert yolo_lr_factor(16, **factor_args) == pytest.approx(0.55)
    assert yolo_lr_factor(29, **factor_args) == pytest.approx(0.1)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-5)

    for _ in range(5):
        parameter.grad = torch.zeros_like(parameter)
        optimizer.step()
        scheduler.step()
    state = scheduler.state_dict()
    resumed_optimizer = torch.optim.AdamW([{"params": [nn.Parameter(torch.ones(()))], "lr": 1e-4}])
    resumed_scheduler = build_yolo_scheduler(resumed_optimizer, config, steps_per_epoch=3)
    resumed_scheduler.load_state_dict(state)
    assert resumed_scheduler.last_epoch == 5
    assert resumed_scheduler.get_last_lr() == pytest.approx(scheduler.get_last_lr())


def test_train_step_raises_on_non_finite_loss() -> None:
    model = DummyDualHead()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scaler = torch.amp.GradScaler("cpu", enabled=False)

    def nan_loss(pred: torch.Tensor, target: Any) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        value = torch.tensor(float("nan"))
        return value, {"pose_loss": value}

    pose = {"img": torch.randn(2, 3, 16, 16)}
    hands = {"img": pose["img"]}

    with pytest.raises(RuntimeError, match="Non-finite training loss at sample 42"):
        train_step(
            model,
            optimizer,
            scaler,
            nan_loss,
            dummy_hand_loss,
            pose,
            hands,
            amp=False,
            sample_index=42,
            return_components=False,
        )
