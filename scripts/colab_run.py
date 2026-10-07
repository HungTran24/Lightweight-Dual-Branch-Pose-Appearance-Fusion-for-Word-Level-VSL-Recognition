from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    from src.runtime.colab import run_on_colab

    parser = argparse.ArgumentParser(description="Run one Project AI experiment on Colab")
    parser.add_argument("config", type=Path)
    parser.add_argument("--gpu", default="T4")
    parser.add_argument("--session")
    parser.add_argument("--identity", type=Path, default=Path("~/.ssh/id_ed25519_colab_cli"))
    parser.add_argument("--timeout", type=int, default=21600)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--colab-command", choices=("colab", "colab-acc2"), default="colab")
    args = parser.parse_args()

    session = args.session or f"project-ai-{args.config.stem.replace('_', '-')}"
    output = run_on_colab(
        args.config,
        PROJECT_ROOT,
        session=session,
        gpu=args.gpu,
        identity=args.identity,
        timeout=args.timeout,
        resume=args.resume,
        colab_command=args.colab_command,
    )
    print(output)


if __name__ == "__main__":
    main()
