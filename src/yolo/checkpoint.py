"""Atomic checkpoint save/load and Hugging Face integration boundary for YOLO."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import torch


def save_checkpoint(path: Path, *, state: dict[str, Any]) -> None:
    """Atomically save checkpoint dictionary to path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(state, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_checkpoint(
    path: Path,
    *,
    device: torch.device | str,
    model: torch.nn.Module,
    optimizer: Any = None,
    scaler: Any = None,
    scheduler: Any = None,
    strict: bool = True,
    expected_identity: str | None = None,
) -> dict[str, Any]:
    """Load checkpoint weights into model (and optionally optimizer/scaler)."""
    target_device = torch.device(device) if isinstance(device, str) else device
    state = torch.load(path, map_location=target_device, weights_only=True)
    if expected_identity is not None and state.get("identity") != expected_identity:
        raise RuntimeError("Checkpoint identity does not match dataset/config identity")
    model.load_state_dict(state["model"], strict=strict)
    if optimizer is not None and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    if scaler is not None and "scaler" in state:
        scaler.load_state_dict(state["scaler"])
    if scheduler is not None and state.get("scheduler") is not None:
        scheduler.load_state_dict(state["scheduler"])
    return state


def upload_checkpoint(
    config: dict[str, Any],
    output_root: Path,
    *,
    batch: int | None = None,
    include_best: bool = False,
    final: bool = False,
) -> None:
    """Upload checkpoint and summary to Hugging Face Hub if enabled."""
    from ..integrations.huggingface import ensure_repo, upload_files

    settings = config.get("integrations", {}).get("hf_checkpoints", {})
    if not isinstance(settings, dict) or not settings.get("enabled", False):
        return

    repo_id = str(settings["repo_id"])
    ensure_repo(
        repo_id=repo_id,
        repo_type="model",
        private=bool(settings.get("private", True)),
    )
    prefix = f'runs/{config["experiment"]["id"]}_s{config["experiment"]["seed"]}'
    files: dict[str, Path] = {}
    last_pt = output_root / "last.pt"
    if last_pt.is_file():
        files[f"{prefix}/last.pt"] = last_pt
    if include_best or final:
        best_pt = output_root / "best.pt"
        if best_pt.is_file():
            files[f"{prefix}/best.pt"] = best_pt
    if final:
        summary_json = output_root / "summary.json"
        if summary_json.is_file():
            files[f"{prefix}/summary.json"] = summary_json

    if not files:
        return

    commit_msg = (
        f"checkpoint: {prefix} final"
        if final
        else f"checkpoint: {prefix} batch {batch if batch is not None else 'unknown'}"
    )
    upload_files(
        repo_id=repo_id,
        repo_type="model",
        files=files,
        commit_message=commit_msg,
    )


def download_checkpoint(
    *,
    repo_id: str,
    revision: str,
    path_in_repo: str,
    cache_dir: Path,
) -> Path:
    """Download checkpoint file from Hugging Face Hub."""
    from ..integrations.huggingface import download_file

    return download_file(
        repo_id=repo_id,
        repo_type="model",
        revision=revision,
        filename=path_in_repo,
        cache_dir=cache_dir,
    )
