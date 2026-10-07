"""Run the Project AI YOLO dual-head capacity pilot (compatibility launcher)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    from src.config import load_config
    from src.yolo import run_yolo

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    summary = run_yolo(config, PROJECT_ROOT, device=args.device, resume=args.resume)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
