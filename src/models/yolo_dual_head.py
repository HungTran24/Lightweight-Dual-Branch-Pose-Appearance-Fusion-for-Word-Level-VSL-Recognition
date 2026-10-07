"""YOLO26 shared-trunk dual-head (person Pose26 and hands Pose26) primitives."""

from __future__ import annotations

import importlib
from typing import Any


def _transfer(source: Any, target: Any, torch: Any) -> dict[str, int]:
    source_state = source.state_dict()
    target_state = target.state_dict()
    copied = 0
    with torch.no_grad():
        for name, value in source_state.items():
            if name in target_state and target_state[name].shape == value.shape:
                target_state[name].copy_(value)
                copied += 1
    target.load_state_dict(target_state)
    return {"copied": copied, "skipped": len(source_state) - copied}


def build_model(
    torch: Any,
    ultralytics: Any,
    *,
    model_name: str,
    person_keypoints: int,
    hand_keypoints: int,
) -> tuple[Any, dict[str, Any]]:
    """Replace the stock pose head with person-pose and hand-pose Pose26 heads."""
    modules = importlib.import_module("ultralytics.nn.modules")
    model = ultralytics.YOLO(model_name).model
    source = model.model[-1]
    if source.__class__.__name__ != "Pose26":
        raise RuntimeError(f"Expected Pose26, got {source.__class__.__name__}")

    channels = tuple(branch[0].conv.in_channels for branch in source.cv2)
    person = modules.Pose26(
        nc=1,
        kpt_shape=(person_keypoints, 3),
        reg_max=source.reg_max,
        end2end=False,
        ch=channels,
    )
    hands = modules.Pose26(
        nc=2,
        kpt_shape=(hand_keypoints, 3),
        reg_max=source.reg_max,
        end2end=False,
        ch=channels,
    )
    for head in (person, hands):
        head.stride = source.stride.clone()
        head.bias_init()

    class DualHead(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.person = person
            self.hands = hands
            self.f = source.f
            self.i = source.i

        def forward(self, features: list[Any]) -> dict[str, Any]:
            return {
                "person": self.person(features),
                "hands": self.hands(features),
            }

    model.model[-1] = DualHead()
    model.requires_grad_(True)
    return model, {
        "channels": list(channels),
        "person_transfer": _transfer(source, person, torch),
        "hand_transfer": _transfer(source, hands, torch),
    }


def sync_heads(model: Any, torch: Any) -> None:
    """Move non-buffer stride/anchor tensors used by custom final heads."""
    device = next(model.parameters()).device
    model.stride = torch.as_tensor(model.stride, device=device)
    dual = model.model[-1]
    for head in (dual.person, dual.hands):
        for name in ("stride", "anchors", "strides"):
            value = getattr(head, name, None)
            if value is not None:
                setattr(head, name, torch.as_tensor(value, device=device))


def build_losses(model: Any, torch: Any) -> tuple[Any, Any]:
    """Bind stock Ultralytics pose losses to each custom Pose26 head."""
    loss_module = importlib.import_module("ultralytics.utils.loss")
    cfg = importlib.import_module("ultralytics.cfg").get_cfg(overrides=model.args)

    class Proxy(torch.nn.Module):
        def __init__(self, head: Any) -> None:
            super().__init__()
            self.model = torch.nn.ModuleList([head])
            self.args = cfg

    dual = model.model[-1]
    return (
        loss_module.PoseLoss26(Proxy(dual.person)),
        loss_module.PoseLoss26(Proxy(dual.hands)),
    )


def decode(
    output: dict[str, Any],
    *,
    confidence: float,
    iou: float,
    max_detections: int,
) -> list[dict[str, Any]]:
    """Apply NMS independently and retain person/hand namespaces with pose rows."""
    nms = importlib.import_module("ultralytics.utils.nms").non_max_suppression
    person_out = output.get("person")
    person = person_out[0] if isinstance(person_out, tuple) else person_out
    hands_out = output.get("hands")
    hands = hands_out[0] if isinstance(hands_out, tuple) else hands_out
    if person is None or hands is None:
        raise ValueError("YOLO output must contain person and hands predictions")
    person_rows = nms(
        person.detach().clone(),
        conf_thres=confidence,
        iou_thres=iou,
        nc=1,
        max_det=max_detections,
    )
    hand_rows = nms(
        hands.detach().clone(),
        conf_thres=confidence,
        iou_thres=iou,
        nc=2,
        max_det=max_detections,
    )
    return [
        {"person": person_row, "hands": hand_row}
        for person_row, hand_row in zip(person_rows, hand_rows, strict=True)
    ]
