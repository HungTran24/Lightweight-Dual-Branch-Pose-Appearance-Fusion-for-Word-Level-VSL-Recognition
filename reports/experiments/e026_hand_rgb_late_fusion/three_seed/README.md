# E026 three-seed validation report

## Locked protocol

- Split: clean signer-disjoint train/validation only. Test was not accessed.
- RGB input: pooled U30 hand features from extraction identity `50ad09795226a00a088797840c2b53452bbfb04638d91e257969bfaa6c0558be`.
- RGB classifier: `Linear(1152,400)`, 461 200 parameters, unchanged recipe for seeds 42/43/44.
- Pose classifier: frozen E008 Pose53 U30 TCN checkpoints paired by seed.
- Fusion: `p_fused = α · p_pose + (1 − α) · p_rgb`, with α ∈ {0.0,0.1,...,1.0}.
- Shared α selection: highest mean validation Top-1, then mean Macro-F1, then larger α.

## Training and checkpoint provenance

| Seed | RGB epochs | RGB best epoch | Pose checkpoint epoch |
|---:|---:|---:|---:|
| 42 | 82 | 67 | 55 |
| 43 | 76 | 61 | 57 |
| 44 | 71 | 56 | 59 |

Pose checkpoint revision: `7bf539b8a4c4c9a6eb09a909d441509268f351c9`.

## Selected shared fusion weight

The mean validation curve selects:

```text
α = 0.6
Pose weight = 60 %
RGB weight  = 40 %
```

The same α also maximized validation Top-1 independently for every seed. It is now frozen for any later E026 test evaluation.

## Validation results

| Seed | RGB Top-1 | Pose Top-1 | Fused Top-1 | Δ Top-1 | Pose Macro-F1 | Fused Macro-F1 | Δ Macro-F1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 50.53 % | 83.27 % | 83.95 % | +0.67 pp | 83.70 % | 84.37 % | +0.67 pp |
| 43 | 50.49 % | 82.48 % | 83.27 % | +0.79 pp | 82.94 % | 83.72 % | +0.78 pp |
| 44 | 50.22 % | 84.14 % | 84.46 % | +0.32 pp | 84.36 % | 84.68 % | +0.32 pp |
| **Mean** | **50.42 %** | **83.30 %** | **83.89 %** | **+0.59 pp** | **83.67 %** | **84.26 %** | **+0.59 pp** |

Top-1 and Macro-F1 improve for all three seeds. Seed 44 has a smaller gain than the seed-42 continuation threshold, but that threshold only controlled whether seeds 43/44 could be opened.

| Metric | Pose mean | Fused mean | Delta |
|---|---:|---:|---:|
| Top-1 | 83.30 % | 83.89 % | +0.59 pp |
| Top-5 | 95.93 % | 94.46 % | −1.46 pp |
| Macro-F1 | 83.67 % | 84.26 % | +0.59 pp |

The RGB branch provides a small, repeatable Top-1/Macro-F1 complement to Pose. The Top-5 decrease remains a calibration/ranking trade-off and is not hidden by the primary selection criterion.

Machine-readable outputs:

- `summary.json`: selected α, per-seed metrics and means.
- `mean_alpha_curve.csv`: complete mean validation curve.
