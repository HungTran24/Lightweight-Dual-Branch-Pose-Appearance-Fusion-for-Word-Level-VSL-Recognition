# Compact Pose and Hand-Appearance Fusion for Vietnamese Sign Language Recognition

Research code, experiment configurations and aggregate results for word-level
recognition on VSL400. The pipeline uses a shared dual-head YOLO26m front-end,
Pose42/53/59 temporal classifiers (BiLSTM, TCN, ST-GCN), frozen MobileNetV3-Small
hand features and score-level fusion.

## Data

VSL400 is controlled-access data. Obtain it from the dataset administrator under
the applicable Data Usage Agreement. This repository contains no videos, signer
images, manifests, features or checkpoints; results are stored as aggregate
CSV/JSON reports.

## Quick start

```bash
uv sync --locked --extra ml
uv run --extra ml pytest -q tests
uv run --extra ml python scripts/run.py configs/experiments/e008_tcn_pose53_uniform30_s42.yaml
```

The tests use synthetic inputs. Training needs the licensed inputs described in
[PROJECT.md](PROJECT.md).

## Layout

- `src/`, `scripts/`: models, data loading, training, evaluation and benchmarks.
- `configs/experiments/`: one YAML per experiment; `test/` holds frozen-test configs.
- `reports/experiments/recognition_validation/`: three-seed validation results.
- `reports/experiments/recognition_test/`: frozen-test results.
- `reports/experiments/efficiency_benchmark_rtx4060_2026-10-13/`: RTX 4060 efficiency benchmark.
