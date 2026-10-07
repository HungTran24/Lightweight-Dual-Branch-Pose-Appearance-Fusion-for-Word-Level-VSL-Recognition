# E036 Redflag signer/domain diagnostic

E036 evaluates signer-associated residual shift on clean validation labels. It combines a score-only RGB retrospective with a held-out RTMW Pose53 TCN diagnostic. This is supporting evidence, not a causal claim.

## Locked protocol

- Clean labels: `manifests/audit/train_clean.csv` and `manifests/audit/val_clean.csv`.
- RTMW artifact: `hungchill24/vsl400-kp-selection-features`, revision `14ceaa42123a556bb1067a6690142536e88187a5`, artifact `rtmw-derived-view-v2-pose53-d84006088ffa9362`.
- TCN train partition: 17,062 samples from signers `001, 002, 004, 007, 008, 009, 010, 011, 013, 014, 015, 016, 018, 021, 022, 023, 024, 025`.
- TCN validation partition: 5,181 samples from controls `006, 020` and Redflag signers `026, 027, 028`.
- There is no train/validation video overlap. Signers `003, 012, 019` and videos `019442`, `024029` are excluded.
- TCN uses the existing Pose53 TCN recipe, seeds 42/43/44, and TorchMetrics Top-1, Top-5, and Macro-F1.

## RGB retrospective

Mean validation metrics over seeds 42/43/44 for the shared signer cohort:

| Model | Signer | Samples | Top-1 | Top-5 | Macro-F1 |
|---|---:|---:|---:|---:|---:|
| RGB Linear U30 | 006 | 833 | 64.83 % | 87.27 % | 64.08 % |
| RGB Linear U30 | 020 | 996 | 62.12 % | 86.48 % | 61.34 % |
| RGB Linear U30 | 028 | 700 | 16.62 % | 33.24 % | 14.32 % |
| RGB Temporal3 U30 | 006 | 833 | 76.75 % | 95.12 % | 74.70 % |
| RGB Temporal3 U30 | 020 | 996 | 70.52 % | 92.74 % | 69.71 % |
| RGB Temporal3 U30 | 028 | 700 | 23.86 % | 44.52 % | 20.72 % |

Source files: `per_seed_signer_metrics.csv`, `per_signer_seed_summary.csv`, and `summary.json`.

## RTMW Pose53 TCN results

Three-seed mean ± sample SD:

| Scope | Samples | Top-1 | Top-5 | Macro-F1 |
|---|---:|---:|---:|---:|
| Overall | 5,181 | 83.17 ± 0.27 % | 96.02 ± 0.16 % | 83.52 ± 0.16 % |
| 006 | 833 | 89.88 ± 0.49 % | 99.44 ± 0.18 % | 88.67 ± 0.36 % |
| 020 | 996 | 86.28 ± 0.32 % | 98.39 ± 0.36 % | 86.96 ± 0.20 % |
| 026 | 1,853 | 83.74 ± 0.39 % | 95.11 ± 0.22 % | 83.87 ± 0.21 % |
| 027 | 799 | 78.14 ± 0.71 % | 94.20 ± 0.26 % | 75.23 ± 0.78 % |
| 028 | 700 | 75.00 ± 0.38 % | 93.05 ± 0.30 % | 70.78 ± 0.64 % |

Seed-level overall results and exact per-signer values are in `rtmw_tcn_per_seed.csv`, `rtmw_tcn_per_signer.csv`, and `rtmw_tcn_aggregate.csv`. The three local run outputs are under `artifacts/runs/e036_redflag_rtmw_tcn_s{42,43,44}/`.

## Interpretation

Signer `028` is the lowest-performing signer in both modalities. RGB Temporal3 reaches 23.86 % Top-1 for `028`, while RTMW Pose53 TCN reaches 75.00 % across the three seeds. Relative to the mean of controls `006/020`, the `028` Top-1 gap is about 49.78 percentage points for RGB Temporal3 and 13.08 percentage points for RTMW TCN.

This supports an association between signer/cohort and residual domain shift. It also suggests that the pose representation is more robust than RGB in this setup. The modalities, models, and training partitions are not a controlled causal ablation, so the result does not establish that RTMW causes the improvement or that label quality causes the signer gap.

## Sớm/Giàu confusion audit

`Sớm` is a genuine class-level confusion signal, not a confirmed label correction. The clean manifests contain 52 train and 9 validation samples for `Sớm`, versus 54 train and 10 validation samples for `Giàu`, so the pattern is not explained by a simple class-count shortage. The audit explicitly keeps `024444` as `Giàu` because of the documented `Giàu`/`Sớm` minimal-pair ambiguity.

In the E036 RTMW TCN validation, all 9 clean `Sớm` examples are Top-1 errors for every seed. The `Sớm → Giàu` counts are 10, 9, and 8 for seeds 42, 43, and 44; the reverse `Giàu → Sớm` counts are 1, 3, and 2. RGB also contains this class-level confusion: `Sớm → Giàu` occurs 7 times in each RGB branch across the three seeds, while RGB Temporal3 has 3 reverse-direction events. However, for signer `028` specifically, neither RGB branch produces the `Sớm ↔ Giàu` pair. Its only two clean `Sớm` examples, `024445` and `024545`, are wrong for all three RGB seeds with diffuse predictions. TCN predicts those two as `Giàu` and `Bảo vệ`, respectively, for every seed.

This should be described as a recurring minimal-pair and signer/class interaction. It is not evidence that the `Sớm` labels are wrong. Raw-video or expert re-audit was not performed in E036.

## Completion and exclusions

- [x] RGB score-only per-signer Top-1, Top-5, Macro-F1, sample counts, and three-seed sample SD.
- [x] Locked clean-label RTMW partition with exact counts and overlap checks.
- [x] Full TCN runs for seeds 42, 43, and 44, plus durable per-seed and aggregate reports.
- [x] No recognition test split, raw video, RTMW extraction, Colab, W&B, HF upload, or secret inspection.
