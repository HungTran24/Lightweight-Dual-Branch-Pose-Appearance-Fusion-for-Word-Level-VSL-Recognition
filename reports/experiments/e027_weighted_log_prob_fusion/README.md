# E027 weighted log-probability fusion

Validation-only experiment over persisted E026 RGB and frozen E008 Pose score archives. The test split was not accessed.

## Locked protocol

```text
score = log(max(p_pose, 1e-12)) + β · log(max(p_rgb, 1e-12))
β ∈ {0.0, 0.1, ..., 1.0}
```

One β is selected across seeds 42/43/44 by mean validation Top-1, then mean Macro-F1, then the smaller β. The selected value is **β = 0.5**.

## Results

| Branch | Mean Top-1 | Mean Top-5 | Mean Macro-F1 |
|---|---:|---:|---:|
| RGB | 50.42 % | 72.00 % | 52.61 % |
| Pose | 83.30 % | 95.93 % | 83.67 % |
| E027 fused | **85.01 %** | **96.43 %** | **85.46 %** |
| Δ fused vs Pose | **+1.71 pp** | **+0.50 pp** | **+1.79 pp** |

| Seed | Pose Top-1 | Fused Top-1 | Δ Top-1 | Pose Top-5 | Fused Top-5 | Δ Top-5 |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 83.27 % | 85.01 % | +1.74 pp | 95.65 % | 96.24 % | +0.59 pp |
| 43 | 82.48 % | 84.74 % | +2.25 pp | 96.01 % | 96.32 % | +0.32 pp |
| 44 | 84.14 % | 85.29 % | +1.15 pp | 96.12 % | 96.72 % | +0.59 pp |

The official gate passes: mean Top-1 improves, while mean Macro-F1 and Top-5 do not decrease.

## Error rescue audit

Across the three seed-video evaluations, E027 changes 231 Pose errors to correct predictions and changes 101 Pose-correct predictions to errors, for a net gain of 130 correct predictions.

| Validation signer | Rescued | Harmed | Net | Share of net gain |
|---:|---:|---:|---:|---:|
| 006 | 72 | 13 | +59 | 45.4 % |
| 020 | 54 | 31 | +23 | 17.7 % |
| 028 | 105 | 57 | +48 | 36.9 % |

Unlike E026 arithmetic fusion, the gain is no longer concentrated almost entirely on signer 006. Signer 028 improves in every seed:

```text
seed 42: 67.86 % → 70.14 %  (+2.29 pp)
seed 43: 67.00 % → 69.86 %  (+2.86 pp)
seed 44: 68.57 % → 70.29 %  (+1.71 pp)
```

There are 49 unique videos rescued in at least two seeds: signer 028 contributes 24, signer 006 contributes 14, and signer 020 contributes 11. There are also 21 stable harms, including 12 from signer 028, so the difficult signer is improved but not solved.

Selected recurring Pose confusions are only partly repaired:

| True → Pose prediction | Pose errors | Rescued by E027 |
|---|---:|---:|
| Sớm → Giàu | 19 | 6 |
| Màu nâu → Thức dậy | 12 | 4 |
| Xôi → Ướt | 12 | 1 |
| Dơ → Con heo | 11 | 1 |
| Thơm → Nhạt | 10 | 2 |
| Con heo → Dơ | 10 | 0 |
| Bánh tét → Bánh chưng | 7 | 2 |
| Bánh chưng → Bánh tét | 7 | 1 |

Under this protocol, weighted log-probability fusion avoids the arithmetic fusion behavior observed in E026 and improves both rank-1 and rank-5 validation metrics. E028 separately evaluates whether coarse temporal pooling improves the RGB branch.
