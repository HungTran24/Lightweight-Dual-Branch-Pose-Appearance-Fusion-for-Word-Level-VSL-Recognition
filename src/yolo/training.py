"""Reusable YOLO dual-head loss and training step primitives."""

from __future__ import annotations

import math
from typing import Any, Literal, overload

import torch


def build_yolo_optimizer(model: Any, train_config: dict[str, Any]) -> torch.optim.Optimizer:
    """Build E005's shared/person and hand parameter groups."""
    dual_head = model.model[-1]
    hand_parameters = [parameter for parameter in dual_head.hands.parameters() if parameter.requires_grad]
    hand_ids = {id(parameter) for parameter in hand_parameters}
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    shared_person = [parameter for parameter in trainable if id(parameter) not in hand_ids]
    if not shared_person or not hand_parameters or len(shared_person) + len(hand_parameters) != len(trainable):
        raise ValueError("YOLO optimizer groups must partition every trainable parameter")

    learning_rate = float(train_config.get("learning_rate", 1e-4))
    hand_learning_rate = float(train_config.get("hand_learning_rate", learning_rate * 5.0))
    if learning_rate <= 0 or hand_learning_rate <= 0:
        raise ValueError("YOLO learning rates must be positive")
    return torch.optim.AdamW(
        [
            {"params": shared_person, "lr": learning_rate, "name": "shared_person"},
            {"params": hand_parameters, "lr": hand_learning_rate, "name": "hands"},
        ],
        weight_decay=float(train_config.get("weight_decay", 0.0)),
    )


def yolo_lr_factor(
    step: int,
    *,
    total_steps: int,
    warmup_steps: int,
    warmup_start_factor: float = 0.1,
    min_factor: float = 0.1,
) -> float:
    """Return the per-step warmup and cosine-decay factor."""
    if total_steps <= 0 or not 0 <= warmup_steps < total_steps:
        raise ValueError("warmup_steps must be in [0, total_steps)")
    if not 0 < warmup_start_factor <= 1 or not 0 < min_factor <= 1:
        raise ValueError("warmup and minimum LR factors must be in (0, 1]")
    if step < warmup_steps:
        if warmup_steps == 0:
            return 1.0
        return warmup_start_factor + (1.0 - warmup_start_factor) * step / warmup_steps
    decay_steps = max(total_steps - warmup_steps - 1, 1)
    progress = min(max((step - warmup_steps) / decay_steps, 0.0), 1.0)
    return min_factor + (1.0 - min_factor) * 0.5 * (1.0 + math.cos(progress * math.pi))


def build_yolo_scheduler(
    optimizer: torch.optim.Optimizer,
    train_config: dict[str, Any],
    *,
    steps_per_epoch: int,
) -> Any:
    """Build E005's optimizer-step warmup/cosine scheduler."""
    if steps_per_epoch <= 0:
        raise ValueError("steps_per_epoch must be positive")
    total_epochs = int(train_config.get("epochs", 1))
    warmup_epochs = int(train_config.get("warmup_epochs", 1))
    total_steps = total_epochs * steps_per_epoch
    warmup_steps = warmup_epochs * steps_per_epoch
    warmup_start_factor = float(train_config.get("warmup_start_factor", 0.1))
    min_factor = float(train_config.get("min_lr_factor", 0.1))
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: yolo_lr_factor(
            step,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            warmup_start_factor=warmup_start_factor,
            min_factor=min_factor,
        ),
    )


def _component_telemetry(prefix: str, components: Any) -> dict[str, float]:
    """Return a small, detached view of named loss components."""
    if not isinstance(components, dict):
        return {}
    telemetry: dict[str, float] = {}
    for name, value in list(components.items())[:16]:
        if isinstance(value, torch.Tensor) and value.numel() == 1:
            telemetry[f"{prefix}/{name}"] = float(value.detach().item())
    return telemetry


def forward_loss(
    model: Any,
    pose_loss: Any,
    hand_loss: Any,
    pose: dict[str, torch.Tensor],
    hands: dict[str, torch.Tensor],
    *,
    amp: bool = False,
) -> tuple[dict[str, Any], torch.Tensor, dict[str, float]]:
    """Run both heads and return output, total loss, and bounded telemetry."""
    device_type = pose["img"].device.type
    with torch.autocast(
        device_type,
        dtype=torch.float16,
        enabled=amp and device_type == "cuda",
    ):
        output = model(pose["img"])
        person_output = output["person"]
        person_total, person_components = pose_loss(person_output, pose)
        hands_total, hand_components = hand_loss(output["hands"], hands)
        telemetry = {
            **_component_telemetry("person_loss", person_components),
            **_component_telemetry("hands_loss", hand_components),
        }
        return output, person_total.sum() + hands_total.sum(), telemetry


def batch_loss(
    model: Any,
    pose_loss: Any,
    hand_loss: Any,
    pose: dict[str, torch.Tensor],
    hands: dict[str, torch.Tensor],
    *,
    amp: bool = False,
) -> torch.Tensor:
    """Compute combined pose and hand loss under autocast."""
    return forward_loss(model, pose_loss, hand_loss, pose, hands, amp=amp)[1]


@overload
def train_step(
    model: Any,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    pose_loss: Any,
    hand_loss: Any,
    pose: dict[str, torch.Tensor],
    hands: dict[str, torch.Tensor],
    *,
    amp: bool = True,
    sample_index: int | None = None,
    return_components: Literal[False] = False,
) -> torch.Tensor: ...


@overload
def train_step(
    model: Any,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    pose_loss: Any,
    hand_loss: Any,
    pose: dict[str, torch.Tensor],
    hands: dict[str, torch.Tensor],
    *,
    amp: bool = True,
    sample_index: int | None = None,
    return_components: Literal[True],
) -> tuple[torch.Tensor, dict[str, float]]: ...


def train_step(
    model: Any,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    pose_loss: Any,
    hand_loss: Any,
    pose: dict[str, torch.Tensor],
    hands: dict[str, torch.Tensor],
    *,
    amp: bool = True,
    sample_index: int | None = None,
    return_components: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, float]]:
    """Execute one training step: zero_grad, forward, finite check, and scaler step."""
    optimizer.zero_grad(set_to_none=True)
    _, loss, components = forward_loss(model, pose_loss, hand_loss, pose, hands, amp=amp)
    if not bool(torch.isfinite(loss)):
        sample_info = f" at sample {sample_index}" if sample_index is not None else ""
        raise RuntimeError(f"Non-finite training loss{sample_info}")
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    return (loss, components) if return_components else loss


def validation_loss(
    model: Any,
    pose_loss: Any,
    hand_loss: Any,
    pose: dict[str, torch.Tensor],
    hands: dict[str, torch.Tensor],
    *,
    amp: bool = False,
) -> torch.Tensor:
    """Compute validation loss for a batch under inference mode."""
    with torch.inference_mode():
        return batch_loss(model, pose_loss, hand_loss, pose, hands, amp=amp)
