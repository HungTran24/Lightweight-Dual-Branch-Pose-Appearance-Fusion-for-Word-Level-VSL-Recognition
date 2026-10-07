from __future__ import annotations

from pathlib import Path


def download_file(
    *,
    repo_id: str,
    repo_type: str,
    revision: str,
    filename: str,
    cache_dir: Path | None = None,
    local_dir: Path | None = None,
) -> Path:
    """Download one pinned Hub file; credentials come from the environment."""
    from huggingface_hub import hf_hub_download

    if local_dir is not None:
        return Path(
            hf_hub_download(
                repo_id=repo_id,
                repo_type=repo_type,
                revision=revision,
                filename=filename,
                local_dir=str(local_dir.resolve()),
            )
        )
    return Path(
        hf_hub_download(
            repo_id=repo_id,
            repo_type=repo_type,
            revision=revision,
            filename=filename,
            cache_dir=str(cache_dir.resolve()) if cache_dir is not None else None,
        )
    )


def ensure_repo(*, repo_id: str, repo_type: str, private: bool = True) -> None:
    from huggingface_hub import HfApi

    HfApi().create_repo(
        repo_id=repo_id,
        repo_type=repo_type,
        private=private,
        exist_ok=True,
    )


def upload_files(
    *,
    repo_id: str,
    repo_type: str,
    files: dict[str, Path],
    commit_message: str,
) -> str:
    """Upload related files in one Hub commit and return its revision."""
    from huggingface_hub import CommitOperationAdd, HfApi

    operations = [
        CommitOperationAdd(path_in_repo=remote_path, path_or_fileobj=str(local_path))
        for remote_path, local_path in files.items()
    ]
    commit = HfApi().create_commit(
        repo_id=repo_id,
        repo_type=repo_type,
        operations=operations,
        commit_message=commit_message,
    )
    return str(commit.oid)
