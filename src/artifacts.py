"""Small dependency-free artifact helpers shared by YOLO pipelines."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

from .integrations.huggingface import upload_files


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def upload_with_retry(
    *,
    repo_id: str,
    files: dict[str, Path],
    commit_message: str,
    retries: int = 3,
    repo_type: str = "dataset",
    backoff_s: float = 2.0,
) -> str:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            return upload_files(
                repo_id=repo_id, repo_type=repo_type, files=files, commit_message=commit_message
            )
        except Exception as exc:
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(backoff_s * (2**attempt))
    assert last_error is not None
    raise last_error
