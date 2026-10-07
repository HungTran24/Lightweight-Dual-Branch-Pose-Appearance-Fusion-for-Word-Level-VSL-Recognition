# SLR-E034/SLR-E035 controlled architecture references

## Scope

`SLR-E034` SPOTER and `SLR-E035` SL-GCN are presentation namespaces for controlled 3-seed (`42`, `43`, `44`) architecture references on the clean front-view Uniform-30 protocol. The executed experiment IDs and artifact paths remain `e034_spoter_pose54_controlled_u30` and `e035_slgcn_bone_pose27_controlled_u30`. They are not faithful reproductions of original SPOTER, full SL-GCN/SAM-SLR, or VSL400 baselines.

Both use the pinned Pose59 Uniform-30 artifact `hungchill24/vsl400-kp-selection-features@5a0042e1daec2813f74f08fa955276c777e2b40a/yolo-e005-pose59-uniform30-a65d7d605573ea13`, clean signer-disjoint development data, and the locked frozen test split only after checkpoint selection.

## Recognition results

### 3-seed validation summary (VSL400 Clean, Uniform-30)

| Model | View | Validation Top-1 (Mean ± Std) | Top-5 | Macro-F1 | Loss |
|---|---|:---:|:---:|:---:|:---:|
| **SLR-E034 SPOTER controlled** | Pose54 | **78.59 ± 0.51 %** | 93.44 ± 0.04 % | 78.68 ± 0.43 % | 0.9452 ± 0.0143 |
| **SLR-E035 SL-GCN bone controlled** | Pose27 | **84.88 ± 0.70 %** | 96.11 ± 0.31 % | 84.94 ± 0.74 % | 0.6924 ± 0.0371 |

### 3-seed frozen-test summary (2,508 videos; Mean ± sample SD)

| Model | View | Frozen test Top-1 | Top-5 | Macro-F1 | Loss |
|---|---|:---:|:---:|:---:|:---:|
| **SLR-E034 SPOTER controlled** | Pose54 | **88.40 ± 0.60 %** | 98.67 ± 0.08 % | 88.27 ± 0.66 % | 0.3816 ± 0.0197 |
| **SLR-E035 SL-GCN bone controlled** | Pose27 | **91.79 ± 0.34 %** | 99.23 ± 0.15 % | 91.83 ± 0.30 % | 0.2860 ± 0.0059 |

### Per-seed runs

| Model | View | Seed | Best epoch | Hardware | Validation Top-1 | Top-5 | Macro-F1 | Frozen test Top-1 | Top-5 | Macro-F1 |
|---|---|:---:|:---:|---|:---:|:---:|:---:|:---:|:---:|:---:|
| SLR-E034 SPOTER | Pose54 | 42 | 46 | NVIDIA A100 | 78.17 % | 93.44 % | 78.27 % | 87.76 % | 98.76 % | 87.58 % |
| SLR-E034 SPOTER | Pose54 | 43 | 71 | Tesla T4 | 79.16 % | 93.48 % | 79.14 % | 88.96 % | 98.60 % | 88.89 % |
| SLR-E034 SPOTER | Pose54 | 44 | 53 | NVIDIA L4 | 78.45 % | 93.40 % | 78.64 % | 88.48 % | 98.64 % | 88.35 % |
| SLR-E035 SL-GCN | Pose27 | 42 | 84 | RTX PRO 6000 | 84.97 % | 96.36 % | 85.01 % | 91.75 % | 99.12 % | 91.75 % |
| SLR-E035 SL-GCN | Pose27 | 43 | 57 | NVIDIA A100 | 84.14 % | 95.77 % | 84.17 % | 91.47 % | 99.16 % | 91.58 % |
| SLR-E035 SL-GCN | Pose27 | 44 | 49 | NVIDIA A100 | 85.53 % | 96.20 % | 85.64 % | 92.15 % | 99.40 % | 92.17 % |

Frozen test s43/s44 ran once on the local NVIDIA GeForce RTX 4060 Laptop GPU with the locked checkpoints; seed 42 was not rerun.

## Complexity and classifier-only runtime

All complexity entries use B=1 Uniform-30 inputs. FLOPs are THOP estimates with `FLOPs = 2 × MACs`; functional graph/attention operators can be undercounted.

| Model | Parameters | FP32 state | Estimated FLOPs | A100 B=1 median / p95 | A100 B=32 throughput / peak VRAM | Ryzen B=1 median / p95 | Ryzen B=32 throughput |
|---|---:|---:|---:|---:|---:|---:|---:|
| SPOTER controlled | 5.950 M | 22.70 MiB | 0.1932 G | 7.135 / 7.637 ms | 4,099 samples/s / 71.16 MiB | 3.882 / 8.497 ms | 945 samples/s |
| SL-GCN bone controlled | 3.883 M | 14.81 MiB | 1.4529 G | 14.614 / 15.079 ms | 2,117 samples/s / 139.72 MiB | 16.346 / 27.368 ms | 68 samples/s |

## Sources

- SLR-E034 seeds: `artifacts/runs/e034_spoter_pose54_controlled_u30_s{42,43,44}/summary.json`
- SLR-E034 frozen test: `artifacts/test_evaluate/test_e034_spoter_pose54_controlled_u30_s{42,43,44}/summary.json`
- SLR-E035 seeds: `artifacts/runs/e035_slgcn_bone_pose27_controlled_u30_s{42,43,44}/summary.json`
- SLR-E035 frozen test: `artifacts/test_evaluate/test_e035_slgcn_bone_pose27_controlled_u30_s{42,43,44}/summary.json`
- CPU benchmark: `artifacts/benchmarks/slr_controlled_cpu_ryzen7_7735hs/summary.json`
- A100 benchmark: `artifacts/benchmarks/slr_controlled_a100_available_checkpoints/summary.json`
