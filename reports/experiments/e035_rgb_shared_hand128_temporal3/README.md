# E035 Shared-hand-128 + Temporal3

Official three-seed U30 experiment. It applies one shared 576→128 projection to all six ordered hand/temporal tokens:

```text
[L_early, L_middle, L_late, R_early, R_middle, R_late]
shared(x) = Linear(576,128) → LayerNorm(128) → GELU → Dropout(0.3)
logits = Linear(768,400)
```

The head has 381,712 trainable parameters. The temporal RGB features are frozen E028 U30 features. Training used the clean manifests, the locked 100/60/15 AdamW recipe, seeds 42/43/44, and W&B. All three W&B runs finished successfully.

## Validation

| Metric | RGB-only | Pose-only (E008) | Pose + RGB (β = 0.5) | Delta vs Pose |
|---|---:|---:|---:|---:|
| Top-1 | **62.09 ± 0.59%** | 83.30 ± 0.83% | **86.08 ± 0.30%** | **+2.78 pp** |
| Top-5 | 80.94 ± 0.24% | 95.93 ± 0.25% | **96.93 ± 0.20%** | +1.00 pp |
| Macro-F1 | **64.06 ± 0.83%** | 83.67 ± 0.71% | **86.50 ± 0.22%** | +2.83 pp |

Per-seed validation breakdown:
- Seed 42: RGB Top-1 62.24%, Pose Top-1 83.27%, Fused Top-1 **85.77%** (+2.49 pp vs Pose)
- Seed 43: RGB Top-1 62.59%, Pose Top-1 82.48%, Fused Top-1 **86.12%** (+3.64 pp vs Pose)
- Seed 44: RGB Top-1 61.45%, Pose Top-1 84.14%, Fused Top-1 **86.36%** (+2.21 pp vs Pose)

## Frozen test

The test evaluation was pre-specified at β = 0.5 and ε = 1e-12:

`log(max(p_pose, ε)) + β · log(max(p_rgb, ε))`

| Metric | RGB-only | Pose + RGB |
|---|---:|---:|
| Top-1 | **77.78 ± 0.63%** | **93.91 ± 0.36%** |
| Top-5 | 95.24 ± 0.24% | 99.48 ± 0.08% |
| Macro-F1 | 77.89 ± 0.48% | 94.03 ± 0.36% |

The test split has 2,508 videos and was not used for training or model selection.

## Provenance

- Training configs: `configs/experiments/e035_rgb_shared_hand128_temporal3_s{42,43,44}.yaml`
- Frozen-test config: `configs/experiments/test/e035_rgb_shared_hand128_temporal3_frozen_test.yaml`
- Full feature receipt SHA256: `0ccade47118c75b49c19d209c75d19be3acd429e90afe997f3adc79222195a3a`
- Full extraction identity: `9c5d05a21ade20db15ee466bbbcfb9a87bf4ac382fa387b7e8dc2759318f3c62`
- Test receipt SHA256: `247bee88b4785f4f17a5b946794fb34fc26dc6b6c97a8ef576fc349a28d67350`
- Frozen-test summary: `artifacts/test_evaluate/e035_rgb_shared_hand128_temporal3_frozen/summary.json`

W&B run IDs:

| Seed | Run |
|---:|---|
| 42 | `4vyngk18` |
| 43 | `io8tszxm` |
| 44 | `4nc8wbqt` |
