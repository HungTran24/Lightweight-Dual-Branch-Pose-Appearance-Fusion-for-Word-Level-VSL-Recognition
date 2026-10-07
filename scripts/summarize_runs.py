from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    from src.comparison import summarize_runs

    parser = argparse.ArgumentParser(description="Summarize compatible three-seed runs")
    parser.add_argument("configs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = summarize_runs(args.configs, PROJECT_ROOT, args.output)
    print(args.output / "comparison.json")
    print(f"best: {report['best_experiment_id']}")


if __name__ == "__main__":
    main()
