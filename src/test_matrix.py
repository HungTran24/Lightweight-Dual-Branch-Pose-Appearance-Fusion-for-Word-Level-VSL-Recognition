"""Run a locked set of inference-only test evaluations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import load_config, output_path
from .experiment import run_test_evaluate


def run_test_matrix(
    config: dict[str, Any], project_root: Path, *, device: str | None = None
) -> dict[str, Any]:
    """Evaluate each locked test config sequentially in one runtime."""
    output = output_path(config, project_root)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Test matrix already exists: {output}")

    results: list[dict[str, Any]] = []
    for relative_path in config["matrix"]["configs"]:
        test_config = load_config(project_root / relative_path)
        if test_config["experiment"].get("operation") != "test_evaluate":
            raise ValueError(f"Test matrix entry is not test_evaluate: {relative_path}")
        summary = run_test_evaluate(test_config, project_root, device=device)
        results.append(
            {
                "config": relative_path,
                "source_experiment": summary["source_experiment"],
                "seed": summary["seed"],
                "output": str(output_path(test_config, project_root).relative_to(project_root)),
                "test_top1": summary["test_top1"],
                "test_top5": summary["test_top5"],
                "test_macro_f1": summary["test_macro_f1"],
            }
        )

    output.mkdir(parents=True, exist_ok=True)
    summary = {
        "status": "ok",
        "experiment_id": config["experiment"]["id"],
        "operation": "test_matrix",
        "entries": results,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
