# E008 validation error analysis

## Protocol

- Frozen checkpoint: TCN Pose53 Uniform-30 seed 42, best epoch 55.
- Data: `manifests/audit/val_clean.csv`, 2,529 videos, 400 glosses.
- Feature revision: `5a0042e1daec2813f74f08fa955276c777e2b40a`.
- Regenerated Top-1 predictions matched all 2,529 canonical prediction rows.

## Main results

- Top-1: 83.27% (2,106/2,529).
- Top-5: 95.65% (2,419/2,529).
- Top-1 errors: 423.
- Ground truth remained in Top-5 for 313/423 Top-1 errors (74.0%). This is shortlist retention, not a claim that the model learned a semantic group.

The five most recurrent unordered confusion pairs across TCN seeds 42/43/44 and ST-GCN seed 42 were:

| Pair | Total across four runs |
|---|---:|
| Giàu ↔ Sớm | 30 |
| Con heo ↔ Dơ | 25 |
| Rửa tay ↔ Thú vị | 20 |
| Bánh chưng ↔ Bánh tét | 19 |
| Xôi ↔ Ướt | 17 |

Full-frame qualitative videos for these five pairs are stored locally at `artifacts/error_analysis/confusion_pair_raw_videos/labeled_pairs/`. Each video shows both confusion directions and labels `GT`, `Pred`, signer and video ID. The ten original MP4 files remain unchanged in the parent directories. The videos are not distributed.

## Signer 028 diagnostic

For E008 seed 42:

| Signer | Samples | Top-1 | Errors |
|---|---:|---:|---:|
| 006 | 833 | 91.00% | 75 |
| 020 | 996 | 87.65% | 123 |
| 028 | 700 | 67.86% | 225 |

Signer 028 represents 27.7% of validation samples but 53.2% of Top-1 errors. Across TCN seeds 42/43/44, full validation was 83.30 ± 0.83%, validation without signer 028 was 89.23 ± 0.85%, and signer 028 was 67.81 ± 0.79%.

A matched Uniform-24 seed-42 diagnostic found that replacing YOLO pose with RTMW pose improved signer-028 Top-1 from 67.14% to 78.29% for TCN and from 70.57% to 81.00% for ST-GCN. The residual gap to non-028 remained approximately 8 to 11 percentage points. This comparison is diagnostic only because it used one seed and a shorter training gate.

Safe interpretation: signer 028 is a challenging signer/domain condition. YOLO-derived pose exhibits stronger sensitivity than RTMW-derived pose, but detector quality alone does not explain the full gap. The videos have no expert annotation, so they do not establish that signer 028 signs incorrectly.
