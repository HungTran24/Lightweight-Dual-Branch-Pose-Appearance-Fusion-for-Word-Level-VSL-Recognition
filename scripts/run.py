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
    load_dotenv(PROJECT_ROOT / ".env")
    from src.config import load_config

    parser = argparse.ArgumentParser(description="Run one Project AI experiment")
    parser.add_argument("config", type=Path)
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    if config["experiment"].get("operation") == "derive_pose59":
        from src.pose59_uniform import derive_uniform_pose59

        summary = derive_uniform_pose59(config, PROJECT_ROOT)
        print(json.dumps(summary, indent=2))
        return
    if config["experiment"].get("operation") == "signer_diagnostic":
        from src.signer_diagnostic import run_signer_diagnostic

        summary = run_signer_diagnostic(config, PROJECT_ROOT, device=args.device)
        print(json.dumps(summary, indent=2))
        return
    if config["experiment"].get("operation") == "test_evaluate":
        from src.experiment import run_test_evaluate

        summary = run_test_evaluate(config, PROJECT_ROOT, device=args.device)
        print(json.dumps(summary, indent=2))
        return
    if config["experiment"].get("operation") == "test_matrix":
        from src.test_matrix import run_test_matrix

        summary = run_test_matrix(config, PROJECT_ROOT, device=args.device)
        print(json.dumps(summary, indent=2))
        return
    if config["experiment"].get("operation") == "e026_audit":
        from src.e026_audit import run_e026_audit

        summary = run_e026_audit(config, PROJECT_ROOT)
        print(json.dumps(summary, indent=2))
        return
    if config["experiment"].get("operation") == "e026_visual_audit":
        from src.e026_visual_audit import run_e026_visual_audit

        summary = run_e026_visual_audit(config, PROJECT_ROOT)
        print(json.dumps(summary, indent=2))
        return
    if config["experiment"].get("operation") == "e026_rgb_features":
        from src.e026_rgb_features import run_e026_rgb_features

        summary = run_e026_rgb_features(config, PROJECT_ROOT, device=args.device)
        print(json.dumps(summary, indent=2))
        return
    if config["experiment"].get("operation") in {"e026_rgb_train", "e026_fusion"}:
        from src.e026_rgb_fusion import run_e026_fusion, run_e026_rgb_train

        runner = (
            run_e026_rgb_train
            if config["experiment"]["operation"] == "e026_rgb_train"
            else run_e026_fusion
        )
        summary = runner(config, PROJECT_ROOT, device=args.device)
        print(json.dumps(summary, indent=2))
        return
    if config["experiment"].get("operation") == "e026_fusion_summary":
        from src.e026_rgb_fusion import run_e026_fusion_summary

        summary = run_e026_fusion_summary(config, PROJECT_ROOT)
        print(json.dumps(summary, indent=2))
        return
    if config["experiment"].get("operation") == "e027_log_fusion":
        from src.e027_log_fusion import run_e027_log_fusion

        summary = run_e027_log_fusion(config, PROJECT_ROOT)
        print(json.dumps(summary, indent=2))
        return
    if config["experiment"].get("operation") == "rgb_test_evaluate":
        from src.rgb_test_evaluate import run_rgb_test_evaluate

        summary = run_rgb_test_evaluate(config, PROJECT_ROOT, device=args.device)
        print(json.dumps(summary, indent=2))
        return
    if config["experiment"].get("operation") == "classifier_benchmark":
        from src.classifier_benchmark import run_classifier_benchmark

        summary = run_classifier_benchmark(config, PROJECT_ROOT, device=args.device)
        print(json.dumps(summary, indent=2))
        return
    if config["experiment"].get("operation") == "frontend_benchmark":
        from src.frontend_benchmark import run_frontend_benchmark

        summary = run_frontend_benchmark(config, PROJECT_ROOT, device=args.device)
        print(json.dumps(summary, indent=2))
        return
    if config["experiment"].get("operation") == "e028_rgb_features":
        from src.e028_rgb_features import run_e028_rgb_features

        summary = run_e028_rgb_features(config, PROJECT_ROOT, device=args.device)
        print(json.dumps(summary, indent=2))
        return
    if config["experiment"].get("operation") in {"e028_rgb_train", "e028_fixed_fusion"}:
        from src.e028_rgb_fusion import run_e028_fixed_fusion, run_e028_rgb_train

        if config["experiment"]["operation"] == "e028_rgb_train":
            summary = run_e028_rgb_train(config, PROJECT_ROOT, device=args.device)
        else:
            summary = run_e028_fixed_fusion(config, PROJECT_ROOT)
        print(json.dumps(summary, indent=2))
        return
    if config["experiment"].get("operation") == "e036_rgb_signer_report":
        from src.redflag_diagnostic import run_rgb_signer_report

        print(json.dumps(run_rgb_signer_report(config, PROJECT_ROOT), indent=2))
        return
    if config["experiment"].get("operation") in {"e038_rgb_frame_features", "e038_rgb_frame_features_test"}:
        from src.e038_rgb_frame_features import run_e038_rgb_frame_features

        summary = run_e038_rgb_frame_features(config, PROJECT_ROOT, device=args.device)
        print(json.dumps(summary, indent=2))
        return
    if config["experiment"].get("operation") in {"e038_rgb_bilstm_train", "e038_fixed_fusion", "e038_test_evaluate"}:
        from src.e038_rgb_frame_fusion import (
            run_e038_fixed_fusion,
            run_e038_rgb_bilstm_train,
            run_e038_test_evaluate,
        )

        operation = config["experiment"]["operation"]
        if operation == "e038_rgb_bilstm_train":
            summary = run_e038_rgb_bilstm_train(config, PROJECT_ROOT, device=args.device)
        elif operation == "e038_fixed_fusion":
            summary = run_e038_fixed_fusion(config, PROJECT_ROOT)
        else:
            summary = run_e038_test_evaluate(config, PROJECT_ROOT, device=args.device)
        print(json.dumps(summary, indent=2))
        return

    model_name = config["model"]["name"].lower()
    if model_name.startswith("yolo") or "yolo" in config:
        from src.yolo.experiment import run_yolo

        summary = run_yolo(config, PROJECT_ROOT, device=args.device, resume=args.resume)
    else:
        from src.experiment import run_experiment

        summary = run_experiment(config, PROJECT_ROOT, device=args.device, resume=args.resume)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
