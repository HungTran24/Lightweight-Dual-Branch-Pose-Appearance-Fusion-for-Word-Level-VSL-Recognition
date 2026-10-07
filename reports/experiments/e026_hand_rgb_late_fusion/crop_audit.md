# E026 Hand-RGB Crop and Alignment Audit (Phases A–B)

## Overview

- **E021 Source**: `hungchill24/vsl400-kp-selection-features` @ `7edced63aedc81908624119314dba5dc7c486ce7`
- **E021 Path**: `e021_yolo_pose59_bbox_handrgb_full`
- **E021 Identity**: `db7ffa2f0b434e05d024c11733663cd55389c191de9dabca3994abcc02abcc3f`
- **Uniform-30 Artifact**: `hungchill24/vsl400-kp-selection-features` @ `5a0042e1daec2813f74f08fa955276c777e2b40a` / `yolo-e005-pose59-uniform30-a65d7d605573ea13`
- **Development Videos**: `train` (19714), `val` (2529), total = 22243
- **Excluded Defect**: `019442` (retained)

## Phase A: Metadata Crop Distribution

| Split | Hand | Frames | Valid | Missing % | Side P5 | Median | P95 | P99 | <64px % | <96px % | >224px % | Pad Area % | Pad Rate % | ROI/BBox Med | Conf Med |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| train | left | 1273293 | 1270064 | 0.25% | 73.0 | 112.0 | 151.0 | 166.0 | 1.37% | 24.56% | 0.00% | 0.01% | 0.22% | 1.31 | 0.90 |
| train | right | 1273293 | 1269785 | 0.28% | 70.0 | 108.0 | 153.0 | 170.0 | 2.19% | 32.19% | 0.00% | 0.05% | 0.80% | 1.31 | 0.88 |
| val | left | 159504 | 158925 | 0.36% | 79.0 | 123.0 | 146.0 | 161.0 | 0.72% | 15.75% | 0.00% | 0.00% | 0.01% | 1.31 | 0.89 |
| val | right | 159504 | 158657 | 0.53% | 75.0 | 117.0 | 152.0 | 165.0 | 1.02% | 24.50% | 0.00% | 0.00% | 0.03% | 1.31 | 0.88 |
| overall | left | 1432797 | 1428989 | 0.27% | 74.0 | 112.0 | 150.0 | 166.0 | 1.30% | 23.58% | 0.00% | 0.01% | 0.20% | 1.31 | 0.90 |
| overall | right | 1432797 | 1428442 | 0.30% | 71.0 | 110.0 | 152.0 | 169.0 | 2.06% | 31.34% | 0.00% | 0.04% | 0.71% | 1.31 | 0.88 |

## Phase B: Exact Uniform-30 Key Alignment

- **Status**: `passed`
- **Total Evaluated Videos**: `22243`
- **Total Temporal Slots (30/video)**: `667290`
- **Successfully Matched Frames**: `667290`
- **Missing Keys**: `0`
- **Duplicate Keys in Parquet**: `0`
- **Split Mismatches**: `0`
- **Audit Scope Note**: `JPEG nullability check is deferred to Phase C/F selected reads per metadata-only audit contract.`

## Summary & Observations

1. **Deterministic Alignment**: 100% of required Uniform-30 frame keys match E021 Parquet shards without missing or duplicate keys.
2. **Measured Missing Rates**: Hand missing rates on clean development splits are recorded in the Phase A table above.
3. **Completion**: Phases A–B passed. The completed Phase C visual audit and Phase D preprocessing lock are recorded in `visual_audit/`; subsequent union extraction is recorded in `rgb_union_full.json`.
