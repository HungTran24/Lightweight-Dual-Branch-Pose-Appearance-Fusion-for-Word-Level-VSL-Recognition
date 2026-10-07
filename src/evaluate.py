from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from .metrics import as_floats, new_evaluation_metrics


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    num_classes: int,
    device: torch.device,
) -> tuple[dict[str, float], list[int], list[float]]:
    """Evaluate one split and return aggregate metrics plus per-sample predictions."""
    result, predictions, confidences, _, _ = evaluate_with_scores(
        model, loader, num_classes=num_classes, device=device, collect_scores=False
    )
    return result, predictions, confidences


def evaluate_with_scores(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    num_classes: int,
    device: torch.device,
    collect_scores: bool = True,
) -> tuple[dict[str, float], list[int], list[float], list[int], torch.Tensor]:
    """Evaluate once and optionally return ordered labels and dense CPU probabilities."""
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    metric = new_evaluation_metrics(num_classes, device)
    loss_total = 0.0
    sample_total = 0
    predictions: list[int] = []
    confidences: list[float] = []
    labels_all: list[int] = []
    probability_batches: list[torch.Tensor] = []
    with torch.inference_mode():
        for features, labels, frame_mask in loader:
            features = features.to(device)
            labels = labels.to(device)
            frame_mask = frame_mask.to(device)
            logits = model(features, frame_mask)
            loss_total += float(criterion(logits, labels).item()) * len(labels)
            sample_total += len(labels)
            metric.update(logits, labels)
            probabilities = logits.softmax(dim=1)
            confidence, prediction = probabilities.max(dim=1)
            predictions.extend(prediction.cpu().tolist())
            confidences.extend(confidence.cpu().tolist())
            if collect_scores:
                labels_all.extend(labels.cpu().tolist())
                probability_batches.append(probabilities.cpu())
    result: dict[str, float] = as_floats(metric.compute())
    result["loss"] = loss_total / max(sample_total, 1)
    result["samples"] = float(sample_total)
    dense = (
        torch.cat(probability_batches, dim=0)
        if probability_batches
        else torch.empty((0, num_classes), dtype=torch.float32)
    )
    return result, predictions, confidences, labels_all, dense
