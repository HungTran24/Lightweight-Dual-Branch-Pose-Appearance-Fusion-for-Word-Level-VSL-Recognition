from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from ..config import load_config, output_path


def _command(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, check=check, text=True)


def remote_run_command(
    remote_root: str, remote_config: str, resume_flag: str, operation: str | None
) -> str:
    if operation == "frontend_benchmark":
        environment = "/content/project-ai-frontend-venv"
        python = f"{environment}/bin/python"
        uv_pip = f"python -m uv pip install --python {python} -q"
        mmcv_wheel = (
            "https://download.openmmlab.com/mmcv/dist/cu118/torch2.0.0/"
            "mmcv-2.0.1-cp310-cp310-manylinux1_x86_64.whl"
        )
        return (
            "set -euo pipefail; "
            "export MPLBACKEND=Agg; "
            "python -m pip install -q uv; "
            f"python -m uv venv --seed --python 3.10.21 {environment}; "
            f"{uv_pip} pip setuptools wheel; "
            f"{uv_pip} --index-url https://download.pytorch.org/whl/cu118 "
            "torch==2.0.1+cu118 torchvision==0.15.2+cu118; "
            f"{uv_pip} numpy==1.26.4 openmim==0.3.9; "
            f"{uv_pip} setuptools==60.2.0; "
            f"{uv_pip} opencv-python==4.11.0.86; "
            f"{uv_pip} --no-deps {mmcv_wheel}; "
            f"{uv_pip} --no-build-isolation "
            "mmengine==0.9.0 mmdet==3.2.0 mmpose==1.3.1; "
            f"cd {remote_root}; {uv_pip} -e '.[ml]'; "
            f"{python} scripts/run.py {remote_config} --device cuda{resume_flag}"
        )
    return (
        "set -euo pipefail; "
        f"cd {remote_root}; python -m pip install -q -e '.[ml]'; "
        f"python scripts/run.py {remote_config} --device cuda{resume_flag}"
    )


def run_on_colab(
    config_path: Path,
    project_root: Path,
    *,
    session: str,
    gpu: str,
    identity: Path,
    timeout: int,
    resume: bool,
    colab_command: str = "colab",
) -> Path:
    config_path = config_path.resolve()
    project_root = project_root.resolve()
    relative_config = config_path.relative_to(project_root)
    config = load_config(config_path)
    local_run = output_path(config, project_root)
    retrieval_runs = [local_run]
    if config.get("experiment", {}).get("operation") == "test_matrix":
        retrieval_runs.extend(
            output_path(load_config(project_root / path), project_root)
            for path in config["matrix"]["configs"]
        )
    for run in retrieval_runs:
        if run.is_dir() and any(run.iterdir()) and not resume:
            raise FileExistsError(
                f"Output directory '{run}' exists and is not empty; pass --resume or choose a new run directory"
            )
    remote_root = "/content/project_ai"
    relative_run = local_run.relative_to(project_root)
    remote_run = f"{remote_root}/{relative_run}"

    with tempfile.TemporaryDirectory(prefix="project-ai-colab-") as temporary:
        temporary_path = Path(temporary)
        ssh_config = temporary_path / "ssh_config"
        exec_script = temporary_path / "exec.py"
        host = "project-ai-runtime"
        ssh: tuple[str, ...] = ()
        created = False
        try:
            _command(colab_command, "new", "-s", session, "--gpu", gpu)
            created = True
            ssh_config.write_text(
                "\n".join(
                    [
                        f"Host {host}",
                        f"  HostName {session}",
                        "  User root",
                        f"  IdentityFile {identity.expanduser()}",
                        "  IdentitiesOnly yes",
                        "  StrictHostKeyChecking accept-new",
                        f"  UserKnownHostsFile {temporary_path / 'known_hosts'}",
                        f"  ProxyCommand {colab_command} ssh --proxy-mode -s {session} --identity {identity.expanduser()}",
                    ]
                ),
                encoding="utf-8",
            )
            ssh = ("ssh", "-F", str(ssh_config), host)
            rsync_ssh = f"ssh -F {ssh_config}"
            _command(*ssh, "mkdir", "-p", remote_root)

            # Sync project_root directly to remote_root
            _command(
                "rsync",
                "-az",
                "--exclude",
                ".env",
                "--exclude",
                ".env.*",
                "--exclude",
                ".git",
                "--exclude",
                ".venv",
                "--exclude",
                ".venv*",
                "--exclude",
                "artifacts",
                "--exclude",
                "cache",
                "--exclude",
                "runs",
                "--exclude",
                "output",
                "--exclude",
                "data",
                "--exclude",
                "wandb",
                "--exclude",
                "worktrees",
                "--exclude",
                "__pycache__",
                "-e",
                rsync_ssh,
                f"{project_root}/",
                f"{host}:{remote_root}/",
            )

            operation = config.get("experiment", {}).get("operation")
            if operation == "test_evaluate" and "local_path" in config["checkpoint"]:
                checkpoint = (project_root / config["checkpoint"]["local_path"]).resolve()
                try:
                    relative = checkpoint.relative_to(project_root)
                except ValueError as exc:
                    raise ValueError("Test checkpoint must stay below project root") from exc
                if not checkpoint.is_file():
                    raise FileNotFoundError(f"Test checkpoint is missing: {checkpoint}")
                _command(*ssh, "mkdir", "-p", str(Path(f"{remote_root}/{relative}").parent))
                _command("rsync", "-az", "-e", rsync_ssh, str(checkpoint), f"{host}:{remote_root}/{relative}")

            if operation in {"e026_rgb_train", "e028_rgb_train", "e038_rgb_bilstm_train"}:
                rgb_data = config.get("rgb_data")
                if not isinstance(rgb_data, dict):
                    raise ValueError(f"{operation} requires rgb_data")
                try:
                    local_rgb_root = (project_root / rgb_data["root"]).resolve()
                    receipt_setting = rgb_data.get("receipt")
                    local_receipt = (
                        (project_root / receipt_setting).resolve()
                        if receipt_setting
                        else (local_rgb_root.parent / "receipt.json").resolve()
                    )
                    rgb_root_relative = local_rgb_root.relative_to(project_root)
                    receipt_relative = local_receipt.relative_to(project_root)
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError("rgb_data paths must stay below project root") from exc
                if not local_rgb_root.is_dir():
                    raise FileNotFoundError("Configured RGB input root is missing")
                remote_rgb_root = f"{remote_root}/{rgb_root_relative}"
                _command(*ssh, "mkdir", "-p", remote_rgb_root)
                _command("rsync", "-az", "-e", rsync_ssh, f"{local_rgb_root}/", f"{host}:{remote_rgb_root}/")
                if local_receipt.is_file():
                    remote_receipt = f"{remote_root}/{receipt_relative}"
                    _command(*ssh, "mkdir", "-p", str(Path(remote_receipt).parent))
                    _command("rsync", "-az", "-e", rsync_ssh, str(local_receipt), f"{host}:{remote_receipt}")
            elif operation == "classifier_benchmark":
                for entry in config["entries"]:
                    checkpoint = (project_root / entry["checkpoint"]).resolve()
                    try:
                        relative = checkpoint.relative_to(project_root)
                    except ValueError as exc:
                        raise ValueError("Benchmark checkpoints must stay below project root") from exc
                    if not checkpoint.is_file():
                        raise FileNotFoundError(f"Benchmark checkpoint is missing: {checkpoint}")
                    _command(*ssh, "mkdir", "-p", str(Path(f"{remote_root}/{relative}").parent))
                    _command("rsync", "-az", "-e", rsync_ssh, str(checkpoint), f"{host}:{remote_root}/{relative}")
            elif operation == "rgb_test_evaluate":
                for branch in config.get("branches", []):
                    rgb_data = branch["rgb_data"]
                    try:
                        local_rgb_root = (project_root / rgb_data["root"]).resolve()
                        local_receipt = (project_root / rgb_data["receipt"]).resolve()
                        rgb_relative = local_rgb_root.relative_to(project_root)
                        receipt_relative = local_receipt.relative_to(project_root)
                        inputs = [
                            (project_root / path).resolve()
                            for paths in (branch["checkpoints"], branch["pose_scores"])
                            for path in paths.values()
                        ]
                        input_relatives = [path.relative_to(project_root) for path in inputs]
                    except (KeyError, TypeError, ValueError) as exc:
                        raise ValueError("RGB test inputs must stay below project root") from exc
                    if not local_rgb_root.is_dir() or not local_receipt.is_file() or not all(
                        path.is_file() for path in inputs
                    ):
                        raise FileNotFoundError("Configured RGB test input is missing")
                    _command(*ssh, "mkdir", "-p", f"{remote_root}/{rgb_relative}", str(Path(f"{remote_root}/{receipt_relative}").parent))
                    _command("rsync", "-az", "-e", rsync_ssh, f"{local_rgb_root}/", f"{host}:{remote_root}/{rgb_relative}/")
                    _command("rsync", "-az", "-e", rsync_ssh, str(local_receipt), f"{host}:{remote_root}/{receipt_relative}")
                    for input_path, relative in zip(inputs, input_relatives, strict=True):
                        _command(*ssh, "mkdir", "-p", str(Path(f"{remote_root}/{relative}").parent))
                        _command("rsync", "-az", "-e", rsync_ssh, str(input_path), f"{host}:{remote_root}/{relative}")
            elif isinstance(config.get("data"), dict) and isinstance(
                config["data"].get("split_path"), str
            ):
                try:
                    local_split_root = (project_root / config["data"]["split_path"]).resolve()
                    split_relative = local_split_root.relative_to(project_root)
                except (TypeError, ValueError) as exc:
                    raise ValueError("data.split_path must stay below project root") from exc
                if not local_split_root.is_dir():
                    raise FileNotFoundError("Configured data.split_path directory is missing")
                remote_split_root = f"{remote_root}/{split_relative}"
                _command(*ssh, "mkdir", "-p", remote_split_root)
                _command(
                    "rsync", "-az", "-e", rsync_ssh,
                    f"{local_split_root}/", f"{host}:{remote_split_root}/",
                )

            env_path = project_root / ".env"
            if env_path.exists():
                _command(
                    "rsync", "-az", "-e", rsync_ssh, str(env_path), f"{host}:{remote_root}/.env"
                )
                _command(*ssh, "chmod", "600", f"{remote_root}/.env")

            if resume and (local_run / "last.pt").exists():
                _command(*ssh, "mkdir", "-p", remote_run)
                _command("rsync", "-az", "-e", rsync_ssh, f"{local_run}/", f"{host}:{remote_run}/")

            resume_flag = " --resume" if resume else ""
            remote_config = f"{remote_root}/{relative_config}"
            command = remote_run_command(remote_root, remote_config, resume_flag, operation)
            exec_script.write_text(
                "import subprocess\n"
                f"command = {command!r}\n"
                "args = ['bash', '-lc', command]\n"
                "process = subprocess.Popen(\n"
                "    args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,\n"
                "    text=True, bufsize=1,\n"
                ")\n"
                "assert process.stdout is not None\n"
                "for line in process.stdout:\n"
                "    print(line, end='', flush=True)\n"
                "returncode = process.wait()\n"
                "if returncode != 0:\n"
                "    raise subprocess.CalledProcessError(returncode, args)\n",
                encoding="utf-8",
            )
            _command(
                colab_command,
                "exec",
                "-s",
                session,
                "-f",
                str(exec_script),
                "--timeout",
                str(timeout),
            )
            summary_check = _command(*ssh, "test", "-f", f"{remote_run}/summary.json", check=False)
            if summary_check.returncode != 0:
                raise RuntimeError(
                    f"Remote execution failed: summary not produced at {remote_run}/summary.json"
                )

            operation = config.get("experiment", {}).get("operation")
            hf_enabled = bool(
                config.get("integrations", {}).get("hf_artifacts", {}).get("enabled", False)
            )
            retrieval_excludes: list[str] = []
            if operation in {"extract", "extract_hand_rgb"}:
                retrieval_excludes.extend(["--exclude", ".video_cache_part_*"])
                if hf_enabled:
                    retrieval_excludes.extend(
                        [
                            "--exclude",
                            "part_*.npz",
                            "--exclude",
                            "part_*_crops_*.parquet",
                        ]
                    )

            for run in retrieval_runs:
                run.mkdir(parents=True, exist_ok=True)
                relative = run.relative_to(project_root)
                _command(
                    "rsync",
                    "-az",
                    *retrieval_excludes,
                    "-e",
                    rsync_ssh,
                    f"{host}:{remote_root}/{relative}/",
                    f"{run}/",
                )
        finally:
            if created:
                if ssh:
                    _command(*ssh, "rm", "-f", f"{remote_root}/.env", check=False)
                _command(colab_command, "stop", "-s", session, check=False)
    return local_run
