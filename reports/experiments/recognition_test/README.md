# Frozen recognition test results

This report records inference-only evaluation on the fixed 2,508-video test split. Checkpoints and the fusion rule were frozen before test inference; no model was retrained and no test result was used to tune β.

## Results

All values are three-seed mean ± sample SD.

| System | Frames | Top-1 | Top-5 | Macro-F1 |
|---|---:|---:|---:|---:|
| Pose TCN | 24 | 92.00 ± 0.61% | 99.28 ± 0.04% | 91.91 ± 0.61% |
| Global RGB | 24 | 69.74 ± 0.30% | 90.67 ± 0.28% | 69.89 ± 0.36% |
| Pose + global RGB, log fusion β=0.5 | 24 | 93.22 ± 0.24% | 99.46 ± 0.05% | 93.21 ± 0.17% |
| Pose TCN | 30 | 92.50 ± 0.35% | 99.38 ± 0.02% | 92.48 ± 0.37% |
| Global RGB | 30 | 70.00 ± 0.46% | 90.68 ± 0.27% | 70.17 ± 0.46% |
| Pose + global RGB, log fusion β=0.5 | 30 | **93.82 ± 0.25%** | 99.49 ± 0.05% | **93.85 ± 0.29%** |
| Temporal3 RGB | 30 | 77.39 ± 0.12% | 95.00 ± 0.29% | 77.37 ± 0.24% |
| Pose + Temporal3 RGB, log fusion β=0.5 | 30 | 93.65 ± 0.12% | **99.53 ± 0.02%** | 93.70 ± 0.10% |
| Shared-hand-128 RGB | 24 | 73.59 ± 0.59% | 93.87 ± 0.15% | 73.33 ± 0.73% |
| Pose + Shared-hand-128 RGB, log fusion β=0.5 | 24 | 93.31 ± 0.48% | 99.57 ± 0.06% | 93.30 ± 0.49% |
| Shared-hand-128 RGB | 30 | 74.24 ± 0.46% | 94.50 ± 0.07% | 74.03 ± 0.59% |
| Pose + Shared-hand-128 RGB, log fusion β=0.5 | 30 | **93.95 ± 0.44%** | 99.60 ± 0.07% | **93.98 ± 0.52%** |
| Shared-hand-128 + Temporal3 RGB | 30 | 77.78 ± 0.63% | 95.24 ± 0.24% | 77.89 ± 0.48% |
| Pose + Shared-hand-128 + Temporal3 RGB, log fusion β=0.5 | 30 | 93.91 ± 0.36% | 99.48 ± 0.08% | 94.03 ± 0.36% |
| Pose TCN | 60 | 93.16 ± 0.38% | 99.48 ± 0.07% | 93.16 ± 0.38% |
| Pose ST-GCN | 60 | 92.36 ± 0.56% | 99.19 ± 0.17% | 92.26 ± 0.60% |
| Pose BiLSTM | 60 | 65.71 ± 3.37% | 92.56 ± 1.62% | 63.81 ± 3.55% |

## Interpretation

- Global RGB fusion improves paired Pose Top-1 by approximately +1.22 pp at U24 and +1.32 pp at U30.
- Temporal3 substantially improves RGB-only U30 over global pooling: +7.39 pp Top-1, +4.32 pp Top-5, and +7.20 pp Macro-F1.
- Temporal3 fusion improves Pose U30 by approximately +1.15 pp Top-1 and reaches 99.53% Top-5, but does not improve fused Top-1 over global RGB on test. This is reported post hoc and is not used to retune or select β.
- TCN U60 improves over TCN U30 by +0.66 pp Top-1; ST-GCN U60 improves over ST-GCN U30 by +1.05 pp Top-1. TCN remains the stronger frozen-test pose classifier under this protocol.
- The highest observed frozen-test Top-1 is 93.95% for Pose U30 + Shared-hand-128 RGB. The 95% Top-1 target was not reached; every fused system exceeds 99% Top-5.

## Sources

- Historical Pose U24/U30: `artifacts/test_evaluate/e023_pose53_uniform24_30_test/summary.json`.
- Regenerated Pose score archives: `artifacts/test_evaluate/test_e016_tcn_pose53_uniform24_s{42,43,44}/scores.npz` and `test_e008_tcn_pose53_uniform30_s{42,43,44}/scores.npz`.
- Global RGB and fixed-log fusion: `artifacts/test_evaluate/e026_rgb_frozen/summary.json`.
- Temporal3 RGB and fixed-log fusion: `artifacts/test_evaluate/e028_rgb_temporal3_frozen/summary.json`.
- Shared-hand-128 RGB and fixed-log fusion: `artifacts/test_evaluate/e034_rgb_shared_hand128_frozen/summary.json`.
- Shared-hand-128 + Temporal3 RGB and fixed-log fusion: `artifacts/test_evaluate/e035_rgb_shared_hand128_temporal3_frozen/summary.json`.
- Uniform-60 TCN/ST-GCN matrix: `artifacts/test_evaluate/test_uniform60_matrix/summary.json`.
- Uniform-60 BiLSTM: `artifacts/test_evaluate/test_e032_bilstm_pose53_uniform60_s{42,43,44}/summary.json`.
- Uniform-60 test feature receipt: `artifacts/yolo/test_pose59_uniform60/yolo-e005-pose59-uniform60-5851659903dd66a8/receipt.json`.
- Global RGB feature receipt: `artifacts/rgb/e026_rgb_union_test/receipt.json`.
- Temporal3 feature receipt: `artifacts/rgb/e028_rgb_temporal3_test/receipt.json`, receipt SHA-256 `247bee88b4785f4f17a5b946794fb34fc26dc6b6c97a8ef576fc349a28d67350`.

The U24 fusion is a pre-specified test extension because β=0.5 was selected on U30 validation. It is reported without using its test result for further tuning. `test_summary.csv` is the durable committed ledger; paths under `artifacts/` are local generated evidence and are not tracked by Git.
