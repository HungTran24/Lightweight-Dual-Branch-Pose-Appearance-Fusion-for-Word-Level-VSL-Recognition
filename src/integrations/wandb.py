from __future__ import annotations

import os
from contextlib import suppress
from typing import Any


def start_run(config: dict[str, Any], environment: dict[str, Any]) -> Any | None:
    """Start W&B when enabled and configured, otherwise run without tracking."""
    settings = config["integrations"].get("wandb", {})
    if settings is False or not isinstance(settings, dict):
        return None
    if not settings.get("enabled", False) or not os.getenv("WANDB_API_KEY"):
        return None
    import wandb

    return wandb.init(
        project=settings["project"],
        group=config["experiment"]["id"],
        name=f"{config['experiment']['id']}_s{config['experiment']['seed']}",
        id=settings.get("id"),
        resume=settings.get("resume"),
        config={**config, "environment": environment},
    )


def log_metrics(run: Any | None, metrics: dict[str, float], step: int) -> None:
    if run is None:
        return
    with suppress(Exception):
        run.log(metrics, step=step)


def log_epoch(run: Any | None, metrics: dict[str, float], epoch: int) -> None:
    log_metrics(run, metrics, epoch)


def finish_run(run: Any | None, summary: dict[str, Any]) -> None:
    if run is None:
        return
    with suppress(Exception):
        run.summary.update(summary)
        run.finish()
