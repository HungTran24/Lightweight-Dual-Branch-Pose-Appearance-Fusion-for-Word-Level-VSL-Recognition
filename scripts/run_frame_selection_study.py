"""Master Frame Selection Study Launcher (Project AI standard).

Executes:
1. Materializes clean splits for Uniform-24, Motion-24, and KCC-24 (Pose53 view).
2. Runs 2x3 matrix: TCN & ST-GCN on each of the 3 sampling methods.
3. Compiles final Top-1 accuracy comparison summary.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    from src.config import load_config
    from src.experiment import run_experiment

    parser = argparse.ArgumentParser(description="Run full Frame Selection Study")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-materialize", action="store_true")
    args = parser.parse_args()

    # Step 1: Materialize splits if not present
    if not args.skip_materialize:
        print("=== Step 1: Materializing 24-frame splits (Pose53) ===")
        import subprocess
        mat_script = PROJECT_ROOT / "scripts/materialize_frame_selection_splits.py"
        subprocess.run([sys.executable, str(mat_script)], check=True)

    # Step 2: Run 6 experiment configs
    configs = [
        "configs/experiments/frame_selection/e101_uniform24_tcn_pose53.yaml",
        "configs/experiments/frame_selection/e102_uniform24_stgcn_pose53.yaml",
        "configs/experiments/frame_selection/e103_motion24_tcn_pose53.yaml",
        "configs/experiments/frame_selection/e104_motion24_stgcn_pose53.yaml",
        "configs/experiments/frame_selection/e105_kcc24_tcn_pose53.yaml",
        "configs/experiments/frame_selection/e106_kcc24_stgcn_pose53.yaml",
    ]

    print("\n=== Step 2: Running 2x3 Evaluation Matrix ===")
    results = {}

    for cfg_rel in configs:
        cfg_path = PROJECT_ROOT / cfg_rel
        cfg = load_config(cfg_path)
        exp_id = cfg["experiment"]["id"]
        print(f"\n---> Starting experiment: {exp_id}")
        t0 = time.time()
        summary = run_experiment(cfg, PROJECT_ROOT, device=args.device, resume=False)
        dur = time.time() - t0

        results[exp_id] = {
            "experiment_id": exp_id,
            "model": cfg["model"]["name"],
            "sampling_method": exp_id.split("_")[1],  # uniform24, motion24, kcc24
            "best_validation_top1": summary["best_validation_top1"],
            "best_validation_top5": summary["best_validation_top5"],
            "best_validation_macro_f1": summary["best_validation_macro_f1"],
            "best_epoch": summary["best_epoch"],
            "epochs_trained": summary["epochs_trained"],
            "duration_sec": dur,
        }

    out_dir = PROJECT_ROOT / "artifacts/frame_selection_eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "comparison_summary.json"
    summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("\n\n=======================================================")
    print("FINAL FRAME SELECTION COMPARISON SUMMARY:")
    print("=======================================================")
    print(f"{'Experiment':<30} | {'Method':<10} | {'Model':<8} | {'Val Top-1':<10} | {'Best Epoch':<10} | {'Time':<8}")
    print("-" * 88)
    for exp_id, v in results.items():
        print(f"{exp_id:<30} | {v['sampling_method']:<10} | {v['model']:<8} | {v['best_validation_top1']*100:6.2f}%    | {v['best_epoch']:<10} | {v['duration_sec']/60:5.1f}m")
    print("=======================================================\n")


if __name__ == "__main__":
    main()
