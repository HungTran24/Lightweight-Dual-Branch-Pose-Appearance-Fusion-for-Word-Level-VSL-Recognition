from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    from src.yolo.onnx_export import export_onnx

    parser = argparse.ArgumentParser(description="Export the pinned E005 dual-head YOLO to ONNX")
    parser.add_argument(
        "config",
        nargs="?",
        type=Path,
        default=Path("configs/experiments/e005_yolo_pose59_extract.yaml"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/yolo/onnx/e005_dual_head_b1_640"),
    )
    args = parser.parse_args()
    summary = export_onnx(
        args.config,
        PROJECT_ROOT,
        output_dir=PROJECT_ROOT / args.output,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
