# Reproducing the experiments

Run all commands from the repository root.

## 1. Environment

```bash
uv sync --locked --extra ml
uv lock --check
uv run --extra ml pytest -q tests
uv run --extra ml python scripts/run.py --help
```

ONNX export needs `uv sync --locked --extra ml --group export`. RTMW needs a
separate OpenMMLab environment (Python 3.10, PyTorch 2.0.1, mmcv 2.0.1,
mmdet 3.2.0, mmpose 1.3.1, ultralytics 8.4.146); its setup sequence is in
`src/runtime/colab.py::remote_run_command`.

## 2. Inputs

VSL400 must be obtained from the dataset administrator. Configs pin every input
by repository, revision and path. Pose classifiers read NPZ feature splits;
extraction, RGB, fusion and diagnostics also need the manifests below at the
configured paths (ignored by Git):

| Path | Samples |
|---|---|
| `manifests/audit/train_clean.csv` | 19,715 rows, 19,714 usable videos |
| `manifests/audit/val_clean.csv` | 2,529 |
| `manifests/vsl400_front_400gloss/test.csv` | 2,508 |

Reference SHA-256 of the manifests used for the reported results:

```text
train_clean.csv  42b54dc9756f832ef8f34a59b27808d30c9e5f9bc3b88ad92d1ebeb9ef5ad7cf
val_clean.csv    1bd92e5de767b60a1ba86dd907c6db3c54c37252884b98638a8f19e33a4a73f1
test.csv         9569a1c87a0686d3aed8bd8e514285b827997dfbb8d6483e769812625c2cc89d
```

See `manifests/audit/README.md` for the label-correction protocol. To use your
own feature splits, set `data.split_path` in a copied config to an absolute
directory containing `train.npz`, `val.npz` and `test.npz`; the NPZ contract is
validated in `src/data.py`.

Configs keep the original W&B and Hugging Face upload settings for provenance.
Set `integrations.*.enabled: false` in a copied config to run without them.
Credentials, if needed, go in `.env` (see `.env.example`).

## 3. Training and test evaluation

```bash
for seed in 42 43 44; do
  uv run --extra ml python scripts/run.py "configs/experiments/e008_tcn_pose53_uniform30_s${seed}.yaml"
done
```

Use `--device cpu` for CPU and `--resume` to continue a run. Output goes to
`artifacts/runs/<experiment_id>_s<seed>/`. Test configs under
`configs/experiments/test/` evaluate pinned checkpoints; to evaluate your own,
set `checkpoint.local_path` in a copied config. Checkpoints are selected on
validation only.

`scripts/colab_run.py <config> --gpu A100` runs the same runner on Google Colab
(requires the `colab` CLI and an SSH identity).

## 4. Reported results

Config families are under `configs/experiments/`, reports under
`reports/experiments/`. All accuracy results use seeds 42/43/44 with sample SD.

| Result | Configs | Evidence |
|---|---|---|
| Pose42/53/59 at U30 | `e007`-`e014` | `recognition_validation/` |
| Pose53 U16 | `e015` TCN, `e017` ST-GCN, `e039` BiLSTM; `test/test_uniform16_matrix.yaml` | `recognition_validation/`, `recognition_test/` |
| Pose53 U24 | `e016` TCN, `e018` ST-GCN, `e022` BiLSTM | `recognition_validation/`, `recognition_test/` |
| Pose53 U60 | `e030`-`e032`; `test/test_uniform60_matrix.yaml` | `recognition_validation/`, `recognition_test/` |
| Global hand appearance and fusion | `e026_rgb_linear_u30_s*`, `e027_weighted_log_prob_fusion`, `test/e026_rgb_frozen_test.yaml` | `e026_hand_rgb_late_fusion/`, `e027_weighted_log_prob_fusion/` |
| Temporal3 appearance and fusion | `e035_rgb_shared_hand128_temporal3_s*`, `e035_fixed_log_fusion_s*`, `test/e035_rgb_shared_hand128_temporal3_frozen_test.yaml` | `e035_rgb_shared_hand128_temporal3/` |
| SPOTER / SL-GCN reference models | `e034_spoter_pose54_controlled_*`, `e035_slgcn_bone_pose27_controlled_*` | `e034_e035_slr_controlled_validation/` |
| Classifier GFLOPs and throughput | `benchmark_pose_classifiers_rtx4060_{gflops,u16,controlled_lengths}.yaml` | `efficiency_benchmark_rtx4060_2026-10-13/` |
| Front-end latency | `benchmark_frontends_pose_rtx4060.yaml` | `efficiency_benchmark_rtx4060_2026-10-13/` |
| Validation error analysis | `scripts/error_analysis.py` (E008, seed 42) | `e008_tcn_pose53_uniform30_s42/error_analysis/` |

Each report README lists selection rules, checkpoint identities and rounding.

## 5. Efficiency benchmark

```bash
uv run --extra ml python scripts/run.py configs/experiments/benchmark_pose_classifiers_rtx4060_gflops.yaml
```

Classifier protocol: FP32 without TF32, synthetic GPU-resident inputs, 50
warm-ups, five shuffled rounds; GFLOPs from PyTorch `FlopCounterMode` at batch 1.
The front-end benchmark runs in the RTMW environment on ten validation videos
(300 U30 frames) and times preprocessing, inference, NMS and Pose59 selection,
excluding decoding and classification. Absolute numbers depend on hardware and
software; the report records both.
