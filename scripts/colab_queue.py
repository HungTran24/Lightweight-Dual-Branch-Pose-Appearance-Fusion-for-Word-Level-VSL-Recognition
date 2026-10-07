from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def build_command(
    config_path: Path,
    project_root: Path,
    *,
    session_prefix: str = "project-ai-",
    gpu: str = "T4",
    timeout: int = 21600,
    colab_command: str = "colab",
    python_executable: str = sys.executable,
) -> list[str]:
    from src.config import load_config, output_path

    session = f"{session_prefix}{config_path.stem.replace('_', '-')}"
    cmd = [
        python_executable,
        str(project_root / "scripts" / "colab_run.py"),
        str(config_path),
        "--gpu",
        gpu,
        "--session",
        session,
        "--timeout",
        str(timeout),
        "--colab-command",
        colab_command,
    ]
    try:
        config = load_config(config_path)
        run_dir = output_path(config, project_root)
        if (run_dir / "last.pt").is_file():
            cmd.append("--resume")
    except Exception:
        pass
    return cmd


def run_queue(
    configs: Sequence[Path],
    project_root: Path,
    *,
    session_prefix: str = "project-ai-",
    gpu: str = "T4",
    timeout: int = 21600,
    colab_command: str = "colab",
    runner: Callable[..., subprocess.CompletedProcess[Any]] = subprocess.run,
) -> int:
    has_failure = False
    for config_path in configs:
        cmd = build_command(
            config_path,
            project_root,
            session_prefix=session_prefix,
            gpu=gpu,
            timeout=timeout,
            colab_command=colab_command,
        )
        print(f"[RUNNING] {config_path}")
        try:
            result = runner(cmd)
            if result.returncode != 0:
                print(f"[FAILED] {config_path}")
                has_failure = True
            else:
                print(f"[SUCCESS] {config_path}")
        except Exception:
            print(f"[FAILED] {config_path}")
            has_failure = True

    return 1 if has_failure else 0


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run a queue of Project AI experiments on Colab")
    parser.add_argument("configs", nargs="+", type=Path, help="Experiment YAML config files")
    parser.add_argument("--gpu", default="T4", help="Colab GPU type")
    parser.add_argument(
        "--session-prefix", default="project-ai-", help="Prefix for Colab session names"
    )
    parser.add_argument(
        "--timeout", type=int, default=21600, help="Per-experiment timeout in seconds"
    )
    parser.add_argument(
        "--colab-command",
        choices=("colab", "colab-acc2"),
        default="colab",
        help="Colab CLI binary",
    )
    args = parser.parse_args(argv)

    exit_code = run_queue(
        args.configs,
        PROJECT_ROOT,
        session_prefix=args.session_prefix,
        gpu=args.gpu,
        timeout=args.timeout,
        colab_command=args.colab_command,
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
