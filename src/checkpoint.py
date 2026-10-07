from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import torch


def _atomic_save(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def save_best(path: Path, model: torch.nn.Module, *, epoch: int, metrics: dict[str, float]) -> None:
    _atomic_save({"model": model.state_dict(), "epoch": epoch, "metrics": metrics}, path)


def save_last(path: Path, state: dict[str, Any]) -> None:
    _atomic_save(state, path)


def load_best(path: Path, model: torch.nn.Module, device: torch.device) -> dict[str, Any]:
    state = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state["model"])
    return state


def load_last(path: Path, model: torch.nn.Module, optimizer: Any, scheduler: Any, device: torch.device) -> dict[str, Any]:
    state = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    return state
