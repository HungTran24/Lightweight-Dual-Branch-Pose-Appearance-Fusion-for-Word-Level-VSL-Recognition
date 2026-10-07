from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

from scripts.colab_queue import build_command, main, run_queue
from src.experiment import _upload_hf


def test_experiment_upload_hf_uses_upload_with_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict[str, Any]] = []

    def mock_upload_with_retry(**kwargs: Any) -> str:
        calls.append(kwargs)
        return "commit_123"

    def mock_ensure_repo(**kwargs: Any) -> None:
        del kwargs

    monkeypatch.setattr("src.experiment.upload_with_retry", mock_upload_with_retry)
    monkeypatch.setattr("src.experiment.ensure_repo", mock_ensure_repo)

    config = {
        "experiment": {"id": "test_exp", "seed": 42},
        "integrations": {
            "hf_checkpoints": {
                "enabled": True,
                "repo_id": "test/model-checkpoints",
                "upload_every_epochs": 1,
                "private": True,
            }
        },
    }
    (tmp_path / "last.pt").touch()
    (tmp_path / "best.pt").touch()
    (tmp_path / "summary.json").touch()

    _upload_hf(config, tmp_path, epoch=1, final=True)

    assert len(calls) == 1
    call = calls[0]
    assert call["repo_id"] == "test/model-checkpoints"
    assert call["repo_type"] == "model"
    assert "runs/test_exp_s42/last.pt" in call["files"]
    assert "runs/test_exp_s42/best.pt" in call["files"]
    assert "runs/test_exp_s42/summary.json" in call["files"]
    assert call["commit_message"] == "checkpoint: runs/test_exp_s42 epoch 1"


def test_experiment_upload_hf_periodic_failure_warns_and_does_not_raise(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def mock_upload_with_retry_fail(**kwargs: Any) -> str:
        del kwargs
        raise RuntimeError("HF upload failed")

    def mock_ensure_repo(**kwargs: Any) -> None:
        del kwargs

    monkeypatch.setattr("src.experiment.upload_with_retry", mock_upload_with_retry_fail)
    monkeypatch.setattr("src.experiment.ensure_repo", mock_ensure_repo)

    config = {
        "experiment": {"id": "test_exp", "seed": 42},
        "integrations": {
            "hf_checkpoints": {
                "enabled": True,
                "repo_id": "test/model-checkpoints",
                "upload_every_epochs": 1,
                "private": True,
            }
        },
    }
    (tmp_path / "last.pt").touch()
    (tmp_path / "best.pt").touch()

    with pytest.warns(UserWarning, match="Periodic HF checkpoint upload failed"):
        _upload_hf(config, tmp_path, epoch=1, final=False)


def test_experiment_upload_hf_final_failure_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def mock_upload_with_retry_fail(**kwargs: Any) -> str:
        del kwargs
        raise RuntimeError("HF upload failed")

    def mock_ensure_repo(**kwargs: Any) -> None:
        del kwargs

    monkeypatch.setattr("src.experiment.upload_with_retry", mock_upload_with_retry_fail)
    monkeypatch.setattr("src.experiment.ensure_repo", mock_ensure_repo)

    config = {
        "experiment": {"id": "test_exp", "seed": 42},
        "integrations": {
            "hf_checkpoints": {
                "enabled": True,
                "repo_id": "test/model-checkpoints",
                "upload_every_epochs": 1,
                "private": True,
            }
        },
    }
    (tmp_path / "last.pt").touch()
    (tmp_path / "best.pt").touch()
    (tmp_path / "summary.json").touch()

    with pytest.raises(RuntimeError, match="HF upload failed"):
        _upload_hf(config, tmp_path, epoch=1, final=True)


def test_colab_queue_build_command_and_resume(tmp_path: Path) -> None:
    config_dict = {
        "experiment": {"id": "e101_uniform30", "mode": "full", "seed": 42},
        "model": {"name": "tcn"},
        "data": {"split_path": "splits/test"},
        "feature": {"view": "pose53"},
        "train": {"epochs": 1, "batch_size": 4, "patience": 0},
        "output": {"root": "artifacts/runs"},
    }
    cfg_file = tmp_path / "e101_uniform30.yaml"
    cfg_file.write_text(yaml.safe_dump(config_dict), encoding="utf-8")

    # When last.pt does not exist
    cmd = build_command(
        cfg_file,
        tmp_path,
        session_prefix="queue1-",
        gpu="A100",
        timeout=3600,
        colab_command="colab-acc2",
        python_executable="python3",
    )
    assert cmd == [
        "python3",
        str(tmp_path / "scripts" / "colab_run.py"),
        str(cfg_file),
        "--gpu",
        "A100",
        "--session",
        "queue1-e101-uniform30",
        "--timeout",
        "3600",
        "--colab-command",
        "colab-acc2",
    ]

    # When last.pt exists, --resume is added
    run_dir = tmp_path / "artifacts" / "runs" / "e101_uniform30_s42"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "last.pt").touch()

    cmd_resumed = build_command(
        cfg_file,
        tmp_path,
        session_prefix="queue1-",
        gpu="A100",
        timeout=3600,
        colab_command="colab-acc2",
        python_executable="python3",
    )
    assert cmd_resumed[-1] == "--resume"


def test_colab_queue_continues_on_failure_and_exits_1(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg1 = tmp_path / "cfg1.yaml"
    cfg2 = tmp_path / "cfg2.yaml"
    cfg3 = tmp_path / "cfg3.yaml"
    for cfg in (cfg1, cfg2, cfg3):
        cfg.touch()

    executed: list[list[str]] = []

    def mock_runner(cmd: list[str]) -> subprocess.CompletedProcess[str]:
        executed.append(cmd)
        if str(cfg2) in cmd:
            return subprocess.CompletedProcess(cmd, returncode=1)
        return subprocess.CompletedProcess(cmd, returncode=0)

    exit_code = run_queue(
        [cfg1, cfg2, cfg3],
        tmp_path,
        session_prefix="batch-",
        runner=mock_runner,
    )

    assert exit_code == 1
    assert len(executed) == 3
    captured = capsys.readouterr().out
    assert f"[FAILED] {cfg2}" in captured
    assert f"[SUCCESS] {cfg1}" in captured
    assert f"[SUCCESS] {cfg3}" in captured


def test_colab_queue_all_success_returns_0(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg1 = tmp_path / "cfg1.yaml"
    cfg2 = tmp_path / "cfg2.yaml"
    cfg1.touch()
    cfg2.touch()

    def mock_runner(cmd: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(cmd, returncode=0)

    exit_code = run_queue(
        [cfg1, cfg2],
        tmp_path,
        runner=mock_runner,
    )

    assert exit_code == 0
    captured = capsys.readouterr().out
    assert f"[SUCCESS] {cfg1}" in captured
    assert f"[SUCCESS] {cfg2}" in captured


def test_colab_queue_main_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = tmp_path / "cfg.yaml"
    cfg.touch()

    def mock_run_queue(*args: Any, **kwargs: Any) -> int:
        del args, kwargs
        return 0

    monkeypatch.setattr("scripts.colab_queue.run_queue", mock_run_queue)

    with pytest.raises(SystemExit) as exc_info:
        main([str(cfg), "--gpu", "T4"])
    assert exc_info.value.code == 0
