from __future__ import annotations

from typing import Any

from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score


def new_evaluation_metrics(num_classes: int, device: Any) -> MetricCollection:
    return MetricCollection(
        {
            "top1": MulticlassAccuracy(num_classes=num_classes, top_k=1, average="micro"),
            "top5": MulticlassAccuracy(num_classes=num_classes, top_k=5, average="micro"),
            "macro_f1": MulticlassF1Score(num_classes=num_classes, average="macro"),
        }
    ).to(device)


def new_train_metric(num_classes: int, device: Any) -> MulticlassAccuracy:
    return MulticlassAccuracy(num_classes=num_classes, average="micro").to(device)


def as_floats(values: dict[str, Any]) -> dict[str, float]:
    return {name: float(value.detach().cpu().item()) for name, value in values.items()}
