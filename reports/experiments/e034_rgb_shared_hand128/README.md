# E034 Shared-hand-128 global hand-RGB

Official three-seed validation and frozen-test extension. The RGB backbone remains frozen MobileNetV3-Small; only the classifier head is trained. The same projection is applied independently to the ordered left and right embeddings, then the two projected embeddings are concatenated.

```text
xL, xR: 576 dimensions each
shared(x): Linear(576,128) → LayerNorm(128) → GELU → Dropout(0.3)
logits: Linear(256,400) over [shared(xL), shared(xR)]
trainable parameters: 176,912
```

The official run uses clean manifests, the pinned E026 global RGB receipt, seeds 42/43/44, and the locked 100/60/15 AdamW recipe. U24 and U30 were both trained through the repository runner. W&B was enabled for all six runs and every run finished successfully.
## Validation

| View | RGB Top-1 | RGB Top-5 | RGB Macro-F1 |
|---|---:|---:|---:|
| Shared-hand-128 U24 | 55.24 ± 0.67% | 76.84 ± 0.73% | 56.86 ± 0.92% |
| Shared-hand-128 U30 | 55.74 ± 0.40% | 76.57 ± 0.78% | 57.35 ± 0.45% |

Mean ± sample SD over seeds 42, 43, and 44. The validation split is `manifests/audit/val_clean.csv`; test was not used for training or model selection.

## Frozen test

Fusion is pre-specified at β = 0.5 with ε = 1e-12:

`log(max(p_pose, ε)) + β · log(max(p_rgb, ε))`

| View | RGB Top-1 | RGB Top-5 | RGB Macro-F1 | Pose + RGB Top-1 | Pose + RGB Top-5 | Pose + RGB Macro-F1 |
|---|---:|---:|---:|---:|---:|---:|
| Shared-hand-128 U24 | 73.59 ± 0.59% | 93.87 ± 0.15% | 73.33 ± 0.73% | 93.31 ± 0.48% | 99.57 ± 0.06% | 93.30 ± 0.49% |
| Shared-hand-128 U30 | 74.24 ± 0.46% | 94.50 ± 0.07% | 74.03 ± 0.59% | 93.95 ± 0.44% | 99.60 ± 0.07% | 93.98 ± 0.52% |

The test split is `manifests/vsl400_front_400gloss/test.csv` with 2,508 videos. This is frozen inference only and does not alter β, the checkpoint choice, or the training protocol.

## Provenance

- Global RGB receipt: `artifacts/rgb/e026_rgb_union_full/receipt.json`
- Receipt SHA256: `b095aacfe2f21b220ae2932cc74f34e1d7e2bf05a025cf59c228ffcec4ebbab0`
- Extraction identity: `50ad09795226a00a088797840c2b53452bbfb04638d91e257969bfaa6c0558be`
- Test receipt SHA256: `ecf89b3c98c37971a04363a29ed19a043cdec31c750bf3c329d1976b1b2589e1`
- Frozen-test config: `configs/experiments/test/e034_rgb_shared_hand128_frozen_test.yaml`
- Test summary: `artifacts/test_evaluate/e034_rgb_shared_hand128_frozen/summary.json`

W&B run IDs:

| View | Seed 42 | Seed 43 | Seed 44 |
|---|---|---|---|
| U24 | `ys255qa8` | `7nub5qvf` | `4e96og1u` |
| U30 | `n7sob98n` | `j5dzorit` | `mbpkbqrx` |
