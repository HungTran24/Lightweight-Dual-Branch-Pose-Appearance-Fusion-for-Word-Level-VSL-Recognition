from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def _validate_common(config: dict[str, Any]) -> None:
    experiment = config.get("experiment")
    if not isinstance(experiment, dict) or experiment.get("mode") not in {"smoke", "full"}:
        raise ValueError("experiment.mode must be smoke or full")
    if "description" in experiment and not isinstance(experiment["description"], str):
        raise ValueError("experiment.description must be a string")
    if experiment.get("operation") not in {
        "derive_pose59",
        "test_matrix",
        "e026_audit",
        "e026_visual_audit",
        "e026_rgb_features",
        "e026_rgb_train",
        "e026_fusion",
        "e026_fusion_summary",
        "e027_log_fusion",
        "e028_rgb_features",
        "e028_rgb_train",
        "e028_fixed_fusion",
        "rgb_test_evaluate",
        "classifier_benchmark",
        "frontend_benchmark",
        "e036_rgb_signer_report",
        "e038_rgb_frame_features",
        "e038_rgb_frame_features_test",
        "e038_rgb_bilstm_train",
        "e038_fixed_fusion",
        "e038_test_evaluate",
    }:
        model = config.get("model")
        if not isinstance(model, dict) or not isinstance(model.get("name"), str):
            raise ValueError("model.name must be a string")
    if not isinstance(config.get("output"), dict) or not isinstance(
        config["output"].get("root"), str
    ):
        raise ValueError("output.root must be a string")


def _validate_e036_rgb_report(config: dict[str, Any]) -> None:
    report = config.get("report")
    if (
        not isinstance(report, dict)
        or report.get("manifest") != "manifests/audit/val_clean.csv"
        or report.get("signers") != ["006", "020", "028"]
        or not isinstance(report.get("sources"), list)
        or len(report["sources"]) != 2
    ):
        raise ValueError("E036 RGB report requires locked clean validation sources and signers")
    for source in report["sources"]:
        archives = source.get("archives") if isinstance(source, dict) else None
        if (
            not isinstance(source, dict)
            or not isinstance(source.get("name"), str)
            or not isinstance(archives, dict)
            or set(archives) != {42, 43, 44}
            or not all(isinstance(path, str) and path for path in archives.values())
        ):
            raise ValueError("E036 RGB report requires three named seed score archives")


def _validate_signer_partition(data: dict[str, Any]) -> None:
    partition = data.get("signer_partition")
    if partition is None:
        return
    if not isinstance(partition, dict):
        raise ValueError("data.signer_partition must be a mapping")
    manifests = partition.get("manifests")
    train = partition.get("train")
    val = partition.get("val")
    if (
        manifests != {"train": "manifests/audit/train_clean.csv", "val": "manifests/audit/val_clean.csv"}
        or not isinstance(partition.get("excluded_video_ids"), list)
        or not isinstance(partition.get("prohibited_signers"), list)
        or not isinstance(train, dict)
        or train.get("source") != "train"
        or not isinstance(train.get("signers"), list)
        or not isinstance(train.get("expected_samples"), int)
        or not isinstance(val, dict)
        or not isinstance(val.get("sources"), dict)
        or not isinstance(val.get("expected_samples"), int)
    ):
        raise ValueError("Invalid clean signer partition contract")
    if set(val["sources"]) - {"train", "val"}:
        raise ValueError("Signer partition validation sources must be train and/or val")
    if not all(isinstance(signer, str) for signer in train["signers"]):
        raise ValueError("Signer partition train signers must be strings")
    if not all(
        isinstance(signers, list) and all(isinstance(signer, str) for signer in signers)
        for signers in val["sources"].values()
    ):
        raise ValueError("Signer partition validation signers must be string lists")


def _validate_classifier(config: dict[str, Any]) -> None:
    data = config.get("data")
    feature = config.get("feature")
    train = config.get("train")
    remote_data = isinstance(data, dict) and all(
        isinstance(data.get(name), str) for name in ("repo_id", "revision", "artifact_path")
    )
    local_data = isinstance(data, dict) and isinstance(data.get("split_path"), str)
    if not remote_data and not local_data:
        raise ValueError("data must define split_path or repo_id, revision, and artifact_path")
    assert isinstance(data, dict)
    if not isinstance(feature, dict) or feature.get("view") not in {
        "pose42",
        "pose53",
        "pose59",
        "spoter54",
        "slgcn27",
    }:
        raise ValueError("feature.view must be pose42, pose53, pose59, spoter54, or slgcn27")
    _validate_signer_partition(data)
    operation = config["experiment"].get("operation")
    if operation == "signer_diagnostic":
        diagnostic = config.get("diagnostic")
        checkpoint = config.get("checkpoint")
        if (
            not isinstance(diagnostic, dict)
            or diagnostic.get("split") not in {"train", "val"}
            or not isinstance(diagnostic.get("manifest"), str)
            or not isinstance(diagnostic.get("signers"), list)
            or not diagnostic["signers"]
            or not all(isinstance(signer, str) for signer in diagnostic["signers"])
            or not isinstance(diagnostic.get("batch_size"), int)
            or diagnostic["batch_size"] <= 0
        ):
            raise ValueError("signer_diagnostic requires split, manifest, signers, and batch_size")
        if (
            not isinstance(checkpoint, dict)
            or not all(
                isinstance(checkpoint.get(name), str) and checkpoint[name]
                for name in ("repo_id", "revision", "path", "source_experiment")
            )
        ):
            raise ValueError("signer_diagnostic requires a pinned checkpoint source")
        return
    if operation == "test_evaluate":
        checkpoint = config.get("checkpoint")
        evaluation = config.get("evaluation")
        remote_checkpoint = isinstance(checkpoint, dict) and all(
            isinstance(checkpoint.get(name), str) and checkpoint[name]
            for name in ("repo_id", "revision", "path", "source_experiment")
        )
        local_checkpoint = (
            isinstance(checkpoint, dict)
            and isinstance(checkpoint.get("source_experiment"), str)
            and isinstance(checkpoint.get("local_path"), str)
        )
        if not remote_checkpoint and not local_checkpoint:
            raise ValueError("test_evaluate requires a pinned remote or local checkpoint source")
        if (
            not isinstance(evaluation, dict)
            or evaluation.get("split") != "test"
            or not isinstance(evaluation.get("batch_size"), int)
            or evaluation["batch_size"] <= 0
        ):
            raise ValueError("test_evaluate requires evaluation.split=test and batch_size")
        return
    if not isinstance(train, dict):
        raise ValueError("train must be a mapping")
    for name in ("epochs", "batch_size"):
        if not isinstance(train.get(name), int) or train[name] <= 0:
            raise ValueError(f"train.{name} must be a positive integer")
    if not isinstance(train.get("patience"), int) or train["patience"] < 0:
        raise ValueError("train.patience must be a non-negative integer")
    if train["epochs"] < train["patience"] and train["patience"]:
        raise ValueError("train.epochs must be at least train.patience")
    if config["experiment"]["mode"] == "smoke":
        subset = data.get("subset")
        if not isinstance(subset, dict) or not all(
            isinstance(subset.get(name), int) and subset[name] > 0 for name in ("train", "val")
        ):
            raise ValueError("smoke mode requires positive data.subset.train and data.subset.val")


def _validate_classifier_benchmark(config: dict[str, Any]) -> None:
    hardware = config.get("hardware")
    protocol = config.get("protocol")
    entries = config.get("entries")
    if (
        not isinstance(hardware, dict)
        or not isinstance(protocol, dict)
        or not isinstance(entries, list)
        or not entries
    ):
        raise ValueError("classifier_benchmark requires hardware, protocol, and entries")
    kind = hardware.get("kind", "cuda")
    if kind == "cuda":
        if not isinstance(hardware.get("gpu_contains"), str) or not hardware["gpu_contains"]:
            raise ValueError("CUDA classifier_benchmark requires hardware.gpu_contains")
    elif kind == "cpu":
        if (
            not isinstance(hardware.get("cpu_contains"), str)
            or not hardware["cpu_contains"]
            or not isinstance(hardware.get("cpu_threads"), int)
            or hardware["cpu_threads"] <= 0
            or not isinstance(hardware.get("interop_threads"), int)
            or hardware["interop_threads"] <= 0
        ):
            raise ValueError(
                "CPU classifier_benchmark requires CPU identity and positive thread counts"
            )
    else:
        raise ValueError("classifier_benchmark hardware.kind must be cuda or cpu")
    for name in ("warmup_iterations", "batch1_iterations", "batch32_iterations"):
        if not isinstance(protocol.get(name), int) or protocol[name] <= 0:
            raise ValueError(f"classifier_benchmark protocol.{name} must be positive")
    if "repeats" in protocol and (type(protocol["repeats"]) is not int or protocol["repeats"] <= 0):
        raise ValueError("classifier_benchmark protocol.repeats must be a positive integer")
    if "order_seed" in protocol and type(protocol["order_seed"]) is not int:
        raise ValueError("classifier_benchmark protocol.order_seed must be an integer")
    if "allow_tf32" in protocol and type(protocol["allow_tf32"]) is not bool:
        raise ValueError("classifier_benchmark protocol.allow_tf32 must be boolean")
    if "precision" in protocol and protocol["precision"] != "float32":
        raise ValueError("classifier_benchmark supports only float32 precision")
    for name, expected in (("inputs_on_gpu", kind == "cuda"), ("inputs_on_cpu", kind == "cpu")):
        if name in protocol and (type(protocol[name]) is not bool or protocol[name] != expected):
            raise ValueError(f"classifier_benchmark protocol.{name} conflicts with hardware.kind")
    for entry in entries:
        if (
            not isinstance(entry, dict)
            or not isinstance(entry.get("checkpoint"), str)
            or not isinstance(entry.get("model"), dict)
        ):
            raise ValueError("classifier_benchmark entries require checkpoint and model")
        for name, value in (
            ("frames", entry.get("frames")),
            ("nodes", entry.get("nodes", 53)),
            (
                "input_channels",
                entry["model"].get("input_channels", entry.get("input_channels", 2)),
            ),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"classifier_benchmark entry.{name} must be a positive integer")
        declared_hash = entry.get("checkpoint_sha256")
        if declared_hash is not None and (
            not isinstance(declared_hash, str)
            or len(declared_hash) != 64
            or any(character not in "0123456789abcdefABCDEF" for character in declared_hash)
        ):
            raise ValueError("classifier_benchmark checkpoint_sha256 must be a SHA-256 hex digest")


def _validate_frontend_benchmark(config: dict[str, Any]) -> None:
    hardware = config.get("hardware")
    protocol = config.get("protocol")
    data = config.get("data")
    if not isinstance(hardware, dict) or hardware.get("kind") not in {"cpu", "cuda"}:
        raise ValueError("frontend_benchmark requires hardware.kind cpu or cuda")
    if hardware["kind"] == "cpu" and (
        not isinstance(hardware.get("cpu_threads"), int)
        or hardware["cpu_threads"] <= 0
        or not isinstance(hardware.get("interop_threads"), int)
        or hardware["interop_threads"] <= 0
    ):
        raise ValueError("CPU frontend_benchmark requires positive thread counts")
    if hardware["kind"] == "cuda" and not isinstance(hardware.get("gpu_contains"), str):
        raise ValueError("CUDA frontend_benchmark requires hardware.gpu_contains")
    if not isinstance(protocol, dict) or protocol.get("warmup_passes") != 1:
        raise ValueError("frontend_benchmark requires exactly one warmup pass")
    if (
        not isinstance(data, dict)
        or data.get("manifest") != "manifests/audit/val_clean.csv"
        or data.get("frames_per_video") != 30
        or data.get("expected_frames") != 300
        or data.get("signer_counts") != {"006": 4, "020": 3, "028": 3}
        or not isinstance(data.get("selection_seed"), int)
    ):
        raise ValueError("frontend_benchmark requires the locked clean-validation Uniform-30 sample")
    if not isinstance(config.get("backends"), list) or not config["backends"]:
        raise ValueError("frontend_benchmark requires non-empty backends")
    if "yolo_onnx" in config["backends"]:
        artifact = config.get("yolo_onnx")
        if (
            not isinstance(artifact, dict)
            or not isinstance(artifact.get("path"), str)
            or not isinstance(artifact.get("sha256"), str)
            or len(artifact["sha256"]) != 64
        ):
            raise ValueError("YOLO ONNX frontend benchmark requires a pinned path and SHA-256")


def _validate_test_matrix(config: dict[str, Any]) -> None:
    matrix = config.get("matrix")
    if (
        not isinstance(matrix, dict)
        or not isinstance(matrix.get("configs"), list)
        or not matrix["configs"]
        or not all(isinstance(path, str) and path for path in matrix["configs"])
    ):
        raise ValueError("test_matrix requires a non-empty matrix.configs list")


def _validate_pose59_derivation(config: dict[str, Any]) -> None:
    if config.get("experiment", {}).get("id") == "test_pose59_uniform60":
        if (
            config.get("source")
            != {
                "repo_id": "hungchill24/vsl400-kp-selection-features",
                "revision": "f94c83a148f01630d7091e394a461acb91bad9c9",
                "prefix": "e005_yolo_pose59_extract",
                "identity": "c6544234e818c9b7ca17a427a8cbec34979ff01af8778f6b40b49dffb2fd460d",
                "parts": [1, 2, 3, 4, 5, 6, 7],
                "manifests": {"test": "manifests/vsl400_front_400gloss/test.csv"},
                "excluded_video_ids": [],
                "expected_videos": {"test": 2508},
                "expected_frames": 154288,
                "source_totals": {"videos": 24751, "frames": 1587085},
            }
            or config.get("sampling") != {"method": "uniform", "sequence_lengths": [60]}
            or config.get("output") != {"root": "artifacts/yolo/test_pose59_uniform60"}
            or config.get("integrations") != {"hf_artifacts": {"enabled": False}}
        ):
            raise ValueError("test_pose59_uniform60 requires the locked test-only contract")
        return
    if config.get("experiment", {}).get("id") == "e029_pose59_uniform60_dev":
        if (
            config.get("experiment")
            != {
                "id": "e029_pose59_uniform60_dev",
                "operation": "derive_pose59",
                "mode": "full",
                "description": "Derive dev-only Uniform-60 Pose59 features from pinned all-frame YOLO shards.",
                "seed": 42,
            }
            or config.get("source")
            != {
                "repo_id": "hungchill24/vsl400-kp-selection-features",
                "revision": "f94c83a148f01630d7091e394a461acb91bad9c9",
                "prefix": "e005_yolo_pose59_extract",
                "identity": "c6544234e818c9b7ca17a427a8cbec34979ff01af8778f6b40b49dffb2fd460d",
                "parts": [1, 2, 3, 4, 5, 6, 7],
                "manifests": {
                    "train": "manifests/audit/train_clean.csv",
                    "val": "manifests/audit/val_clean.csv",
                },
                "excluded_video_ids": ["019442"],
                "expected_videos": {"train": 19714, "val": 2529},
                "expected_frames": 1432797,
                "source_totals": {"videos": 24751, "frames": 1587085},
            }
            or config.get("sampling")
            != {"method": "uniform", "sequence_lengths": [60]}
            or config.get("output") != {"root": "artifacts/yolo/e029_pose59_uniform60_dev"}
            or config.get("integrations") != {"hf_artifacts": {"enabled": False}}
        ):
            raise ValueError("e029_pose59_uniform60_dev requires the locked dev-only Uniform-60 contract")
        return
    source = config.get("source")
    if not isinstance(source, dict):
        raise ValueError("derive_pose59 config must define source")
    for name in ("repo_id", "revision", "prefix", "identity"):
        if not isinstance(source.get(name), str) or not source[name]:
            raise ValueError(f"derive_pose59 source.{name} must be a non-empty string")
    if (
        not isinstance(source.get("parts"), list)
        or not source["parts"]
        or not all(isinstance(part, int) and part > 0 for part in source["parts"])
    ):
        raise ValueError("derive_pose59 source.parts must be positive integers")
    manifests = source.get("manifests")
    if (
        not isinstance(manifests, dict)
        or set(manifests) != {"train", "val", "test"}
        or not all(isinstance(path, str) for path in manifests.values())
    ):
        raise ValueError("derive_pose59 source.manifests must define train, val, and test")
    expected_videos = source.get("expected_videos")
    if (
        not isinstance(expected_videos, dict)
        or set(expected_videos) != {"train", "val", "test"}
        or not all(isinstance(count, int) and count > 0 for count in expected_videos.values())
    ):
        raise ValueError("derive_pose59 source.expected_videos must define positive split counts")
    if not isinstance(source.get("expected_frames"), int) or source["expected_frames"] <= 0:
        raise ValueError("derive_pose59 source.expected_frames must be positive")
    excluded = source.get("excluded_video_ids", [])
    if not isinstance(excluded, list) or not all(
        isinstance(value, str) and value for value in excluded
    ):
        raise ValueError("derive_pose59 source.excluded_video_ids must be a list of strings")

    sampling = config.get("sampling")
    if not isinstance(sampling, dict):
        raise ValueError("derive_pose59 config must define sampling")
    lengths = sampling.get("sequence_lengths")
    if not isinstance(lengths, list) or sorted(lengths) != [16, 24, 30]:
        raise ValueError("derive_pose59 sampling.sequence_lengths must contain 16, 24, and 30")

    integrations = config.get("integrations", {})
    if not isinstance(integrations, dict):
        raise ValueError("integrations must be a mapping")
    hf = integrations.get("hf_artifacts", {})
    if not isinstance(hf, dict) or not isinstance(hf.get("enabled", False), bool):
        raise ValueError("integrations.hf_artifacts.enabled must be a boolean")
    if hf.get("enabled") and not isinstance(hf.get("repo_id"), str):
        raise ValueError("enabled HF publishing requires integrations.hf_artifacts.repo_id")


def _validate_e026_audit(config: dict[str, Any]) -> None:
    exp = config.get("experiment", {})
    if not isinstance(exp.get("id"), str) or not exp["id"].strip():
        raise ValueError("e026_audit requires a non-empty experiment.id")

    e021 = config.get("e021_source")
    if not isinstance(e021, dict):
        raise ValueError("e026_audit config must define e021_source")
    for name in ("repo_id", "revision", "prefix", "identity"):
        if not isinstance(e021.get(name), str) or not e021[name]:
            raise ValueError(f"e026_audit e021_source.{name} must be a non-empty string")
    if (
        not isinstance(e021.get("parts"), list)
        or not e021["parts"]
        or not all(isinstance(part, int) and part > 0 for part in e021["parts"])
    ):
        raise ValueError("e026_audit e021_source.parts must be positive integers")

    u30 = config.get("uniform30_source")
    if not isinstance(u30, dict):
        raise ValueError("e026_audit config must define uniform30_source")
    for name in ("repo_id", "revision", "artifact_path"):
        if not isinstance(u30.get(name), str) or not u30[name]:
            raise ValueError(f"e026_audit uniform30_source.{name} must be a non-empty string")

    manifests = config.get("manifests")
    if (
        not isinstance(manifests, dict)
        or set(manifests) != {"train", "val"}
        or not all(isinstance(path, str) and path for path in manifests.values())
    ):
        raise ValueError("e026_audit manifests must define train and val without test split")

    expected_videos = config.get("expected_videos")
    if (
        not isinstance(expected_videos, dict)
        or set(expected_videos) != {"train", "val"}
        or not all(isinstance(count, int) and count > 0 for count in expected_videos.values())
    ):
        raise ValueError("e026_audit expected_videos must define positive train and val counts")

    excluded = config.get("excluded_video_ids", [])
    if not isinstance(excluded, list) or not all(
        isinstance(value, str) and value for value in excluded
    ):
        raise ValueError("e026_audit excluded_video_ids must be a list of strings")


def _validate_e026_rgb_features(config: dict[str, Any]) -> None:
    exp = config.get("experiment", {})
    if not isinstance(exp.get("id"), str) or not exp["id"].strip():
        raise ValueError("e026_rgb_features requires a non-empty experiment.id")

    e021 = config.get("e021_source")
    if not isinstance(e021, dict):
        raise ValueError("e026_rgb_features config must define e021_source")
    for name in ("repo_id", "revision", "prefix", "identity"):
        if not isinstance(e021.get(name), str) or not e021[name]:
            raise ValueError(f"e026_rgb_features e021_source.{name} must be a non-empty string")
    if (
        not isinstance(e021.get("parts"), list)
        or not e021["parts"]
        or not all(isinstance(part, int) and part > 0 for part in e021["parts"])
    ):
        raise ValueError("e026_rgb_features e021_source.parts must be positive integers")

    for u_name in ("u24_source", "u30_source"):
        u_cfg = config.get(u_name)
        if not isinstance(u_cfg, dict):
            raise ValueError(f"e026_rgb_features config must define {u_name}")
        for name in ("repo_id", "revision", "artifact_path"):
            if not isinstance(u_cfg.get(name), str) or not u_cfg[name]:
                raise ValueError(f"e026_rgb_features {u_name}.{name} must be a non-empty string")

    manifests = config.get("manifests")
    if (
        not isinstance(manifests, dict)
        or set(manifests) not in ({"train", "val"}, {"test"})
        or not all(isinstance(path, str) and path for path in manifests.values())
    ):
        raise ValueError("e026_rgb_features manifests must define train/val or test only")

    expected_videos = config.get("expected_videos")
    if (
        not isinstance(expected_videos, dict)
        or set(expected_videos) != set(manifests)
        or not all(isinstance(count, int) and count > 0 for count in expected_videos.values())
    ):
        raise ValueError("e026_rgb_features expected_videos must match manifest splits")

    excluded = config.get("excluded_video_ids", [])
    if not isinstance(excluded, list) or not all(
        isinstance(value, str) and value for value in excluded
    ):
        raise ValueError("e026_rgb_features excluded_video_ids must be a list of strings")

    data = config.get("data")
    if not isinstance(data, dict):
        raise ValueError("e026_rgb_features config must define data mapping")
    if not isinstance(data.get("batch_size"), int) or data["batch_size"] <= 0:
        raise ValueError("e026_rgb_features data.batch_size must be a positive integer")
    if "target_size" in data and data["target_size"] != 128:
        raise ValueError("e026_rgb_features data.target_size must be 128")

    if exp.get("mode") == "smoke":
        smoke = config.get("smoke")
        if (
            not isinstance(smoke, dict)
            or not isinstance(smoke.get("train_videos"), int)
            or smoke["train_videos"] <= 0
            or not isinstance(smoke.get("val_videos"), int)
            or smoke["val_videos"] <= 0
        ):
            raise ValueError(
                "e026_rgb_features smoke mode requires positive train_videos and val_videos"
            )

    prep = config.get("preprocessing")
    if (
        not isinstance(prep, dict)
        or prep.get("backbone") != "mobilenet_v3_small"
        or prep.get("weights") != "IMAGENET1K_V1"
        or prep.get("feature_dim") != 576
        or prep.get("target_size") != 128
        or prep.get("interpolation") != "bilinear"
        or prep.get("antialias") is not True
        or prep.get("mean") != [0.485, 0.456, 0.406]
        or prep.get("std") != [0.229, 0.224, 0.225]
        or prep.get("output_dtype") != "float16"
    ):
        raise ValueError("e026_rgb_features preprocessing must match locked production values")




def _validate_e026_rgb_train_fusion(config: dict[str, Any]) -> None:
    operation = config["experiment"]["operation"]
    if operation == "e026_fusion_summary":
        runs = config.get("fusion_runs")
        if (
            not isinstance(runs, list)
            or [run.get("seed") for run in runs if isinstance(run, dict)] != [42, 43, 44]
            or not all(isinstance(run.get("root"), str) and run["root"] for run in runs)
        ):
            raise ValueError("e026_fusion_summary requires ordered seed 42/43/44 run roots")
        return

    manifests = config.get("manifests")
    experiment_id = config["experiment"].get("id", "")
    if operation == "e026_rgb_train" and experiment_id.startswith("e034_rgb_shared_hand128_"):
        try:
            _, _, _, _, view, seed_text = experiment_id.split("_")
            seed = int(seed_text.removeprefix("s"))
        except (ValueError, TypeError):
            raise ValueError("E034 Shared-hand-128 ID must encode U24/U30 and seed") from None
        expected_rgb_data = {
            "view": view,
            "root": f"artifacts/rgb/e026_rgb_union_full/{view}/data",
            "receipt": "artifacts/rgb/e026_rgb_union_full/receipt.json",
            "receipt_sha256": "b095aacfe2f21b220ae2932cc74f34e1d7e2bf05a025cf59c228ffcec4ebbab0",
            "extraction_identity": "50ad09795226a00a088797840c2b53452bbfb04638d91e257969bfaa6c0558be",
        }
        locked_train = {
            "epochs": 100,
            "min_epochs": 60,
            "patience": 15,
            "batch_size": 256,
            "optimizer": "adamw",
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "grad_clip_norm": 1.0,
        }
        if (
            view not in {"u24", "u30"}
            or seed not in {42, 43, 44}
            or config["experiment"]
            != {
                "id": experiment_id,
                "operation": "e026_rgb_train",
                "mode": "full",
                "seed": seed,
            }
            or config.get("manifests")
            != {
                "train": "manifests/audit/train_clean.csv",
                "val": "manifests/audit/val_clean.csv",
            }
            or config.get("expected_videos") != {"train": 19714, "val": 2529}
            or config.get("excluded_video_ids") != ["019442"]
            or config.get("rgb_data") != expected_rgb_data
            or config.get("model") != {"name": "rgb_shared_hand128"}
            or config.get("train") != locked_train
            or config.get("output") != {"root": f"artifacts/runs/{experiment_id}"}
            or config.get("integrations")
            != {
                "wandb": {"enabled": True, "project": "vsl400-recognition"},
                "hf_checkpoints": {"enabled": False},
            }
        ):
            raise ValueError("E034 Shared-hand-128 requires the locked U24/U30 seed contract")
        return
    if operation == "e026_rgb_train":
        rgb_data = config.get("rgb_data", {})
        train = config.get("train", {})
        locked_train = {
            "epochs": 100,
            "min_epochs": 60,
            "patience": 15,
            "batch_size": 256,
            "optimizer": "adamw",
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "grad_clip_norm": 1.0,
        }
        if (
            config["experiment"].get("seed") not in {42, 43, 44}
            or not isinstance(manifests, dict)
            or set(manifests) != {"train", "val"}
            or config.get("model", {}).get("name") != "rgb_linear"
            or not isinstance(rgb_data, dict)
            or not all(
                isinstance(rgb_data.get(name), str) and rgb_data[name]
                for name in ("root", "receipt", "receipt_sha256", "extraction_identity")
            )
            or rgb_data.get("view", "u30") not in {"u24", "u30"}
            or any(train.get(name) != value for name, value in locked_train.items())
        ):
            raise ValueError("e026_rgb_train must match the locked seed 42/43/44 pooled RGB contract")
        _validate_classifier(
            {
                **config,
                "data": {"split_path": rgb_data["root"], "num_classes": 400},
                "feature": {"view": "pose53"},
            }
        )
        return
    seed = config["experiment"].get("seed")
    pose = config.get("pose", {})
    checkpoint = pose.get("checkpoint", {}) if isinstance(pose, dict) else {}
    expected_epochs = {42: 55, 43: 57, 44: 59}
    if (
        operation != "e026_fusion"
        or seed not in expected_epochs
        or not isinstance(config.get("rgb_scores"), str)
        or f"_s{seed}/scores.npz" not in config["rgb_scores"]
        or not isinstance(pose, dict)
        or checkpoint.get("path") != f"runs/e008_tcn_pose53_uniform30_s{seed}/best.pt"
        or checkpoint.get("expected_epoch") != expected_epochs[seed]
        or not isinstance(config.get("evaluation", {}).get("batch_size"), int)
        or config["evaluation"]["batch_size"] <= 0
    ):
        raise ValueError("e026_fusion must pair locked RGB and Pose sources for seed 42/43/44")


def _validate_e027_log_fusion(config: dict[str, Any]) -> None:
    inputs = config.get("inputs")
    fusion = config.get("fusion")
    expected_inputs = [
        {
            "seed": seed,
            "rgb_scores": f"artifacts/runs/e026_rgb_linear_u30_s{seed}/scores.npz",
            "pose_scores": f"artifacts/runs/e026_fusion_u30_s{seed}/pose_scores.npz",
        }
        for seed in (42, 43, 44)
    ]
    if (
        config["experiment"].get("id") != "e027_weighted_log_prob_fusion"
        or inputs != expected_inputs
        or not isinstance(fusion, dict)
        or fusion.get("epsilon") != 1e-12
        or fusion.get("beta_grid") != [i / 10.0 for i in range(11)]
        or config.get("output", {}).get("root")
        != "reports/experiments/e027_weighted_log_prob_fusion"
    ):
        raise ValueError("e027_log_fusion requires locked validation inputs, output, and fusion grid")


def _validate_e028(config: dict[str, Any]) -> None:
    operation = config["experiment"]["operation"]
    if operation == "e028_rgb_features":
        experiment_id = config["experiment"].get("id")
        if experiment_id in {
            "e033_rgb_temporal3_u24_features_full",
            "e033_rgb_temporal3_u24_test",
        }:
            test_only = experiment_id.endswith("_test")
            if (
                config["experiment"].get("mode") != "full"
                or config.get("sampling_slots") != 24
                or config.get("u24_source")
                != {
                    "repo_id": "hungchill24/vsl400-kp-selection-features",
                    "revision": "5a0042e1daec2813f74f08fa955276c777e2b40a",
                    "artifact_path": "yolo-e005-pose59-uniform24-fbbaf7a173a950b2",
                }
                or config.get("e021_source")
                != {
                    "repo_id": "hungchill24/vsl400-kp-selection-features",
                    "revision": "7edced63aedc81908624119314dba5dc7c486ce7",
                    "prefix": "e021_yolo_pose59_bbox_handrgb_full",
                    "identity": "db7ffa2f0b434e05d024c11733663cd55389c191de9dabca3994abcc02abcc3f",
                    "parts": [1, 2, 3, 4, 5, 6, 7],
                }
                or config.get("data") != {"batch_size": 256}
                or config.get("preprocessing")
                != {
                    "backbone": "mobilenet_v3_small",
                    "weights": "IMAGENET1K_V1",
                    "feature_dim": 576,
                    "target_size": 128,
                    "interpolation": "bilinear",
                    "antialias": True,
                    "mean": [0.485, 0.456, 0.406],
                    "std": [0.229, 0.224, 0.225],
                    "output_dtype": "float16",
                }
                or config.get("manifests")
                != (
                    {"test": "manifests/vsl400_front_400gloss/test.csv"}
                    if test_only
                    else {
                        "train": "manifests/audit/train_clean.csv",
                        "val": "manifests/audit/val_clean.csv",
                    }
                )
                or config.get("expected_videos")
                != ({"test": 2508} if test_only else {"train": 19714, "val": 2529})
                or config.get("excluded_video_ids")
                != ([] if test_only else ["019442"])
                or config.get("output")
                != {
                    "root": (
                        "artifacts/rgb/e033_rgb_temporal3_u24_test"
                        if test_only
                        else "artifacts/rgb/e033_rgb_temporal3_u24_full"
                    )
                }
            ):
                raise ValueError("E033 temporal RGB extraction requires the locked U24 contract")
            return
        if experiment_id == "e028_rgb_temporal3_test":
            if (
                config.get("manifests") != {"test": "manifests/vsl400_front_400gloss/test.csv"}
                or config.get("expected_videos") != {"test": 2508}
                or config.get("excluded_video_ids") != []
                or config.get("output") != {"root": "artifacts/rgb/e028_rgb_temporal3_test"}
            ):
                raise ValueError("e028_rgb_temporal3_test requires the locked test-only contract")
            return
        manifests = config.get("manifests")
        if (
            config["experiment"].get("id") != "e028_rgb_temporal3_features_full"
            or config["experiment"].get("mode") != "full"
            or manifests != {
                "train": "manifests/audit/train_clean.csv",
                "val": "manifests/audit/val_clean.csv",
            }
            or config.get("expected_videos") != {"train": 19714, "val": 2529}
            or config.get("excluded_video_ids") != ["019442"]
            or config.get("e021_source") != {
                "repo_id": "hungchill24/vsl400-kp-selection-features",
                "revision": "7edced63aedc81908624119314dba5dc7c486ce7",
                "prefix": "e021_yolo_pose59_bbox_handrgb_full",
                "identity": "db7ffa2f0b434e05d024c11733663cd55389c191de9dabca3994abcc02abcc3f",
                "parts": [1, 2, 3, 4, 5, 6, 7],
            }
            or config.get("u30_source") != {
                "repo_id": "hungchill24/vsl400-kp-selection-features",
                "revision": "5a0042e1daec2813f74f08fa955276c777e2b40a",
                "artifact_path": "yolo-e005-pose59-uniform30-a65d7d605573ea13",
            }
            or config.get("data") != {"batch_size": 256}
            or config.get("output", {}).get("root") != "artifacts/rgb/e028_rgb_temporal3_full"
            or config.get("preprocessing") != {
                "backbone": "mobilenet_v3_small",
                "weights": "IMAGENET1K_V1",
                "feature_dim": 576,
                "target_size": 128,
                "interpolation": "bilinear",
                "antialias": True,
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
                "output_dtype": "float16",
            }
        ):
            raise ValueError("e028_rgb_features requires the locked train/val temporal RGB contract")
        return
    seed = config["experiment"].get("seed")
    if operation == "e028_rgb_train":
        if config["experiment"].get("id") == f"e035_rgb_shared_hand128_temporal3_s{seed}":
            rgb_data = config.get("rgb_data", {})
            if (
                seed not in {42, 43, 44}
                or config["experiment"]
                != {
                    "id": f"e035_rgb_shared_hand128_temporal3_s{seed}",
                    "operation": "e028_rgb_train",
                    "mode": "full",
                    "seed": seed,
                }
                or config.get("manifests")
                != {
                    "train": "manifests/audit/train_clean.csv",
                    "val": "manifests/audit/val_clean.csv",
                }
                or config.get("expected_videos") != {"train": 19714, "val": 2529}
                or config.get("excluded_video_ids") != ["019442"]
                or not isinstance(rgb_data, dict)
                or rgb_data.get("root")
                != "artifacts/rgb/e028_rgb_temporal3_full/data"
                or rgb_data.get("receipt")
                != "artifacts/rgb/e028_rgb_temporal3_full/receipt.json"
                or rgb_data.get("receipt_sha256")
                != "0ccade47118c75b49c19d209c75d19be3acd429e90afe997f3adc79222195a3a"
                or rgb_data.get("extraction_identity")
                != "9c5d05a21ade20db15ee466bbbcfb9a87bf4ac382fa387b7e8dc2759318f3c62"
                or config.get("model") != {"name": "rgb_shared_hand128_temporal3"}
                or config.get("train")
                != {
                    "epochs": 100,
                    "min_epochs": 60,
                    "patience": 15,
                    "batch_size": 256,
                    "optimizer": "adamw",
                    "learning_rate": 0.001,
                    "weight_decay": 0.0001,
                    "grad_clip_norm": 1.0,
                }
                or config.get("output")
                != {"root": f"artifacts/runs/e035_rgb_shared_hand128_temporal3_s{seed}"}
                or config.get("integrations")
                != {
                    "wandb": {"enabled": True, "project": "vsl400-recognition"},
                    "hf_checkpoints": {"enabled": False},
                }
            ):
                raise ValueError("E035 Shared-hand-128 Temporal3 requires the locked seed contract")
            return
        if config["experiment"].get("id") == f"e033_rgb_temporal3_u24_s{seed}":
            rgb_data = config.get("rgb_data", {})
            if (
                seed not in {42, 43, 44}
                or config["experiment"]
                != {
                    "id": f"e033_rgb_temporal3_u24_s{seed}",
                    "operation": "e028_rgb_train",
                    "mode": "full",
                    "seed": seed,
                }
                or config.get("manifests")
                != {
                    "train": "manifests/audit/train_clean.csv",
                    "val": "manifests/audit/val_clean.csv",
                }
                or config.get("expected_videos") != {"train": 19714, "val": 2529}
                or config.get("excluded_video_ids") != ["019442"]
                or not isinstance(rgb_data, dict)
                or rgb_data.get("root")
                != "artifacts/rgb/e033_rgb_temporal3_u24_full/data"
                or rgb_data.get("receipt")
                != "artifacts/rgb/e033_rgb_temporal3_u24_full/receipt.json"
                or rgb_data.get("sampling_slots") != 24
                or not isinstance(rgb_data.get("receipt_sha256"), str)
                or len(rgb_data["receipt_sha256"]) != 64
                or not isinstance(rgb_data.get("extraction_identity"), str)
                or len(rgb_data["extraction_identity"]) != 64
                or config.get("model") != {"name": "rgb_temporal3"}
                or config.get("train")
                != {
                    "epochs": 100,
                    "min_epochs": 60,
                    "patience": 15,
                    "batch_size": 256,
                    "optimizer": "adamw",
                    "learning_rate": 0.001,
                    "weight_decay": 0.0001,
                    "grad_clip_norm": 1.0,
                }
                or config.get("output")
                != {"root": f"artifacts/runs/e033_rgb_temporal3_u24_s{seed}"}
                or config.get("integrations")
                != {
                    "wandb": {"enabled": True, "project": "vsl400-recognition"},
                    "hf_checkpoints": {"enabled": False},
                }
            ):
                raise ValueError("E033 RGB training requires the locked U24 seed contract")
            return
        if (
            seed not in {42, 43, 44}
            or config["experiment"] != {
                "id": f"e028_rgb_temporal3_s{seed}",
                "operation": "e028_rgb_train",
                "mode": "full",
                "seed": seed,
            }
            or config.get("manifests") != {
                "train": "manifests/audit/train_clean.csv",
                "val": "manifests/audit/val_clean.csv",
            }
            or config.get("expected_videos") != {"train": 19714, "val": 2529}
            or config.get("excluded_video_ids") != ["019442"]
            or config.get("rgb_data") != {
                "root": "artifacts/rgb/e028_rgb_temporal3_full/data",
                "receipt": "artifacts/rgb/e028_rgb_temporal3_full/receipt.json",
                "receipt_sha256": "0ccade47118c75b49c19d209c75d19be3acd429e90afe997f3adc79222195a3a",
                "extraction_identity": "9c5d05a21ade20db15ee466bbbcfb9a87bf4ac382fa387b7e8dc2759318f3c62",
            }
            or config.get("model") != {"name": "rgb_temporal3"}
            or config.get("train") != {
                "epochs": 100, "min_epochs": 60, "patience": 15, "batch_size": 256,
                "optimizer": "adamw", "learning_rate": 0.001, "weight_decay": 0.0001,
                "grad_clip_norm": 1.0,
            }
            or config.get("output") != {"root": f"artifacts/runs/e028_rgb_temporal3_s{seed}"}
            or config.get("integrations") != {
                "wandb": {"enabled": False}, "hf_checkpoints": {"enabled": False}
            }
        ):
            raise ValueError("e028_rgb_train requires the locked official seed 42/43/44 contract")
        return
    if operation == "e028_fixed_fusion":
        seed = int(config.get("experiment", {}).get("seed", -1))
        exp_id = config.get("experiment", {}).get("id")
        if exp_id == f"e035_fixed_log_fusion_s{seed}":
            if (
                seed not in {42, 43, 44}
                or config["experiment"] != {
                    "id": f"e035_fixed_log_fusion_s{seed}",
                    "operation": "e028_fixed_fusion",
                    "mode": "full",
                    "seed": seed,
                }
                or config.get("candidate_scores") != f"artifacts/runs/e035_rgb_shared_hand128_temporal3_s{seed}/scores.npz"
                or config.get("baseline_rgb_scores") != f"artifacts/runs/e026_rgb_linear_u30_s{seed}/scores.npz"
                or config.get("pose_scores") != f"artifacts/runs/e026_fusion_u30_s{seed}/pose_scores.npz"
                or config.get("fusion") != {"beta": 0.5, "epsilon": 1e-12}
                or config.get("output") != {"root": f"reports/experiments/e035_rgb_shared_hand128_temporal3/seed{seed}"}
            ):
                raise ValueError("e035_fixed_fusion requires the locked official seed 42/43/44 validation contract")
            return
        if (
            seed not in {42, 43, 44}
            or config["experiment"] != {
                "id": f"e028_fixed_log_fusion_s{seed}",
                "operation": "e028_fixed_fusion",
                "mode": "full",
                "seed": seed,
            }
            or config.get("candidate_scores") != f"artifacts/runs/e028_rgb_temporal3_s{seed}/scores.npz"
            or config.get("baseline_rgb_scores") != f"artifacts/runs/e026_rgb_linear_u30_s{seed}/scores.npz"
            or config.get("pose_scores") != f"artifacts/runs/e026_fusion_u30_s{seed}/pose_scores.npz"
            or config.get("fusion") != {"beta": 0.5, "epsilon": 1e-12}
            or config.get("output") != {"root": f"reports/experiments/e028_rgb_temporal3/seed{seed}"}
        ):
            raise ValueError("e028_fixed_fusion requires the locked official seed 42/43/44 validation contract")
        return


def _validate_yolo(config: dict[str, Any]) -> None:
    if not isinstance(config.get("data"), dict):
        raise ValueError("YOLO config must define a data mapping")

    model = config.get("model")
    if (
        not isinstance(model, dict)
        or model.get("person_keypoints") != 17
        or model.get("hand_keypoints") != 21
    ):
        raise ValueError("YOLO model must define person_keypoints=17 and hand_keypoints=21")

    operation = config.get("experiment", {}).get("operation", "train")
    if operation not in {"probe", "train", "evaluate", "extract", "extract_hand_rgb"}:
        raise ValueError(
            "experiment.operation must be probe, train, evaluate, extract, or extract_hand_rgb"
        )

    if operation in {"probe", "evaluate", "extract", "extract_hand_rgb"}:
        checkpoint = config.get("checkpoint")
        if not isinstance(checkpoint, dict) or not all(
            isinstance(checkpoint.get(k), str) for k in ("repo_id", "revision", "path")
        ):
            raise ValueError(
                f"{operation.capitalize()} operation requires checkpoint mapping with repo_id, revision, and path"
            )

    if operation in {"extract", "extract_hand_rgb"}:
        data = config["data"]
        manifests = data.get("manifests")
        if (
            not isinstance(manifests, dict)
            or set(manifests) != {"train", "val", "test"}
            or not all(isinstance(path, str) for path in manifests.values())
        ):
            raise ValueError(f"{operation} operation requires train, val, and test manifest paths")
        if not isinstance(data.get("parts"), list) or not data["parts"]:
            raise ValueError(f"{operation} operation requires non-empty data.parts")
        for key in ("batch_size", "max_detections"):
            if not isinstance(data.get(key), int) or data[key] <= 0:
                raise ValueError(f"{operation} data.{key} must be a positive integer")
        for key in ("nms_confidence", "nms_iou"):
            if not 0.0 < float(data.get(key, 0.0)) <= 1.0:
                raise ValueError(f"{operation} data.{key} must be in (0, 1]")
        if "require_cuda" in data and not isinstance(data["require_cuda"], bool):
            raise ValueError(f"{operation} data.require_cuda must be a boolean")
        if "cleanup_part_cache" in data and not isinstance(data["cleanup_part_cache"], bool):
            raise ValueError(f"{operation} data.cleanup_part_cache must be a boolean")
        if "resume_from_hf" in data and not isinstance(data["resume_from_hf"], bool):
            raise ValueError(f"{operation} data.resume_from_hf must be a boolean")
        if "integrations" in config and isinstance(config["integrations"], dict):
            hf = config["integrations"].get("hf_artifacts")
            if (
                isinstance(hf, dict)
                and "resume_from_hf" in hf
                and not isinstance(hf["resume_from_hf"], bool)
            ):
                raise ValueError("integrations.hf_artifacts.resume_from_hf must be a boolean")
        if "known_decode_defects" in data:
            defects = data["known_decode_defects"]
            if not isinstance(defects, list):
                raise ValueError(f"{operation} data.known_decode_defects must be a list")
            for item in defects:
                if isinstance(item, str):
                    if not item.strip():
                        raise ValueError(
                            f"{operation} data.known_decode_defects items must be non-empty strings or dicts"
                        )
                elif isinstance(item, dict):
                    if not isinstance(item.get("video_id"), str) or not item["video_id"].strip():
                        raise ValueError(
                            f"{operation} data.known_decode_defects dict items must define video_id"
                        )
                    if "reason" in item and not isinstance(item["reason"], str):
                        raise ValueError(
                            f"{operation} data.known_decode_defects reason must be a string"
                        )
                else:
                    raise ValueError(
                        f"{operation} data.known_decode_defects items must be string or dict"
                    )
        if config["experiment"]["mode"] == "smoke":
            if "video_ids" in data:
                if (
                    not isinstance(data["video_ids"], list)
                    or not data["video_ids"]
                    or not all(isinstance(v, str) and v.strip() for v in data["video_ids"])
                ):
                    raise ValueError(
                        f"smoke {operation} data.video_ids must be a non-empty list of string IDs"
                    )
            elif "subset" in data:
                if (
                    not isinstance(data["subset"], dict)
                    or not data["subset"]
                    or not all(
                        isinstance(v, int)
                        and (v >= 0 if operation == "extract_hand_rgb" else v > 0)
                        for v in data["subset"].values()
                    )
                ):
                    raise ValueError(
                        f"smoke {operation} data.subset must be a non-empty mapping of split to "
                        f"{'non-negative' if operation == 'extract_hand_rgb' else 'positive'} integers"
                    )
            else:
                raise ValueError(f"smoke mode {operation} requires data.video_ids or data.subset")
        if operation == "extract_hand_rgb":
            hf_artifacts = config.get("integrations", {}).get("hf_artifacts")
            if isinstance(hf_artifacts, dict) and hf_artifacts.get("enabled", False):
                if config["experiment"]["mode"] == "smoke":
                    raise ValueError(
                        "smoke extract_hand_rgb does not support integrations.hf_artifacts.enabled=true"
                    )
                if (
                    not isinstance(hf_artifacts.get("repo_id"), str)
                    or not hf_artifacts["repo_id"].strip()
                ):
                    raise ValueError(
                        "enabled HF publishing requires integrations.hf_artifacts.repo_id"
                    )
            if "crops" in config:
                crops = config["crops"]
                if not isinstance(crops, dict):
                    raise ValueError("extract_hand_rgb crops must be a mapping")
                if "crop_shard_frames" in crops and (
                    not isinstance(crops["crop_shard_frames"], int)
                    or crops["crop_shard_frames"] <= 0
                ):
                    raise ValueError("crops.crop_shard_frames must be a positive integer")
                if "margin" in crops and float(crops["margin"]) <= 0:
                    raise ValueError("crops.margin must be positive")
                if "quality" in crops and (
                    not isinstance(crops["quality"], int) or not 1 <= crops["quality"] <= 100
                ):
                    raise ValueError("crops.quality must be an integer between 1 and 100")
        return

    train = config.get("train")
    if not isinstance(train, dict):
        raise ValueError("YOLO config must define a train mapping")
    if not isinstance(train.get("batch_size"), int) or train["batch_size"] <= 0:
        raise ValueError("train.batch_size must be a positive integer")
    if "epochs" in train and (not isinstance(train["epochs"], int) or train["epochs"] <= 0):
        raise ValueError("train.epochs must be a positive integer")
    if "optimizer" in train and train["optimizer"] != "adamw":
        raise ValueError("train.optimizer must be adamw")
    if "shuffle" in train and not isinstance(train["shuffle"], bool):
        raise ValueError("train.shuffle must be a boolean")
    if "augmentation" in train:
        aug = train["augmentation"]
        if not isinstance(aug, (bool, dict)):
            raise ValueError("train.augmentation must be a boolean or mapping")
        if isinstance(aug, dict):
            if "enabled" in aug and not isinstance(aug["enabled"], bool):
                raise ValueError("train.augmentation.enabled must be a boolean")
            for float_key in ("hsv_h", "hsv_s", "hsv_v", "translate", "scale"):
                if float_key in aug:
                    val = float(aug[float_key])
                    if not 0.0 <= val <= 1.0:
                        raise ValueError(f"train.augmentation.{float_key} must be in [0, 1]")
            for forbidden in (
                "flip",
                "fliplr",
                "flipud",
                "horizontal_flip",
                "mosaic",
                "mixup",
                "cutmix",
            ):
                if aug.get(forbidden):
                    raise ValueError(f"Augmentation '{forbidden}' is not permitted")
    if "discriminative_lr" in train and not isinstance(train["discriminative_lr"], bool):
        raise ValueError("train.discriminative_lr must be a boolean")
    if train.get("discriminative_lr") and "hand_learning_rate" not in train:
        raise ValueError("Discriminative YOLO training requires train.hand_learning_rate")
    for learning_rate_name in ("learning_rate", "hand_learning_rate"):
        if learning_rate_name in train and float(train[learning_rate_name]) <= 0:
            raise ValueError(f"train.{learning_rate_name} must be positive")
    if "lr_schedule" in train and train["lr_schedule"] not in {None, "warmup_cosine"}:
        raise ValueError("train.lr_schedule must be warmup_cosine")
    if train.get("lr_schedule") == "warmup_cosine":
        epochs = int(train.get("epochs", 0))
        warmup_epochs = train.get("warmup_epochs")
        if not isinstance(warmup_epochs, int) or not 0 <= warmup_epochs < epochs:
            raise ValueError("train.warmup_epochs must be in [0, train.epochs)")
        for factor_name in ("warmup_start_factor", "min_lr_factor"):
            if factor_name in train and not 0.0 < float(train[factor_name]) <= 1.0:
                raise ValueError(f"train.{factor_name} must be in (0, 1]")

    if "probe" in config:
        probe = config["probe"]
        if not isinstance(probe, dict):
            raise ValueError("probe must be a mapping")
        if "warmup_steps" in probe and (
            not isinstance(probe["warmup_steps"], int) or probe["warmup_steps"] < 0
        ):
            raise ValueError("probe.warmup_steps must be a non-negative integer")
        if "measured_steps" in probe and (
            not isinstance(probe["measured_steps"], int) or probe["measured_steps"] <= 0
        ):
            raise ValueError("probe.measured_steps must be a positive integer")

    if "evaluation" in config:
        evaluation = config["evaluation"]
        if not isinstance(evaluation, dict):
            raise ValueError("evaluation must be a mapping")
        if "batch_size" in evaluation and (
            not isinstance(evaluation["batch_size"], int) or evaluation["batch_size"] <= 0
        ):
            raise ValueError("evaluation.batch_size must be a positive integer")
        for conf_key in ("metric_nms_confidence", "nms_confidence"):
            if conf_key in evaluation and not (0.0 < float(evaluation[conf_key]) <= 1.0):
                raise ValueError(f"evaluation.{conf_key} must be in (0, 1]")
        for iou_key in ("metric_nms_iou", "nms_iou"):
            if iou_key in evaluation and not (0.0 < float(evaluation[iou_key]) <= 1.0):
                raise ValueError(f"evaluation.{iou_key} must be in (0, 1]")
        if "overlay_samples" in evaluation and (
            not isinstance(evaluation["overlay_samples"], int) or evaluation["overlay_samples"] < 0
        ):
            raise ValueError("evaluation.overlay_samples must be a non-negative integer")


def load_config(path: str | Path) -> dict[str, Any]:
    """Load and minimally validate a classifier or YOLO experiment YAML."""
    config_path = Path(path).resolve()
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Config must be a mapping")
    _validate_common(config)
    if config["experiment"].get("operation") == "derive_pose59":
        _validate_pose59_derivation(config)
        return config
    if config["experiment"].get("operation") == "test_matrix":
        _validate_test_matrix(config)
        return config
    if config["experiment"].get("operation") in {"e026_audit", "e026_visual_audit"}:
        _validate_e026_audit(config)
        return config
    if config["experiment"].get("operation") == "e026_rgb_features":
        _validate_e026_rgb_features(config)
        return config
    if config["experiment"].get("operation") in {
        "e026_rgb_train",
        "e026_fusion",
        "e026_fusion_summary",
    }:
        _validate_e026_rgb_train_fusion(config)
        return config
    if config["experiment"].get("operation") == "e027_log_fusion":
        _validate_e027_log_fusion(config)
        return config
    if config["experiment"].get("operation") in {
        "e028_rgb_features",
        "e028_rgb_train",
        "e028_fixed_fusion",
    }:
        _validate_e028(config)
        return config
    if config["experiment"].get("operation") == "rgb_test_evaluate":
        if config.get("fusion") != {"beta": 0.5, "epsilon": 1e-12} or config.get("manifests") != {
            "test": "manifests/vsl400_front_400gloss/test.csv"
        }:
            raise ValueError("rgb_test_evaluate requires locked test manifest and fixed beta")
        return config
    if config["experiment"].get("operation") == "classifier_benchmark":
        _validate_classifier_benchmark(config)
        return config
    if config["experiment"].get("operation") == "frontend_benchmark":
        _validate_frontend_benchmark(config)
        return config
    if config["experiment"].get("operation") == "e036_rgb_signer_report":
        _validate_e036_rgb_report(config)
        return config
    if config["experiment"].get("operation") in {
        "e038_rgb_frame_features",
        "e038_rgb_frame_features_test",
        "e038_rgb_bilstm_train",
        "e038_fixed_fusion",
        "e038_test_evaluate",
    }:
        return config
    model_name = config["model"]["name"].lower()
    if model_name.startswith("yolo") or "yolo" in config:
        _validate_yolo(config)
    else:
        _validate_classifier(config)
    return config


def output_path(config: dict[str, Any], project_root: Path) -> Path:
    """Resolve the run output below the configured project root."""
    output = config["output"]
    if config["experiment"].get("operation") in {
        "derive_pose59",
        "test_matrix",
        "e026_audit",
        "e026_visual_audit",
        "e026_rgb_features",
        "e026_rgb_train",
        "e026_fusion",
        "e026_fusion_summary",
        "e027_log_fusion",
        "e028_rgb_features",
        "e028_rgb_train",
        "e028_fixed_fusion",
        "rgb_test_evaluate",
        "classifier_benchmark",
        "frontend_benchmark",
        "e036_rgb_signer_report",
        "e038_rgb_frame_features",
        "e038_rgb_frame_features_test",
        "e038_rgb_bilstm_train",
        "e038_fixed_fusion",
        "e038_test_evaluate",
    }:
        return project_root / output["root"]
    model_name = config["model"]["name"].lower()
    if model_name.startswith("yolo") or "yolo" in config:
        return project_root / output["root"] / output.get("run_dir", "run")
    run_name = f"{config['experiment']['id']}_s{config['experiment']['seed']}"
    return project_root / output["root"] / run_name
