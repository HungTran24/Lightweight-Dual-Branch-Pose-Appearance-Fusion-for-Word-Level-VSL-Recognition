---
title: Canonical VSL400 manifest
description: Báo cáo materialize fixed canonical signer mapping cho toàn bộ 400 gloss VSL400 front_view
status: ready
update_at: 28-08-26
---

# Canonical VSL400 manifest

## Contract

- Split policy: fixed canonical signer mapping
- Hugging Face dataset: `hungchill24/vsl400-front-view`
- Hugging Face revision: `df5509963c62b2adaa18525d2990c8b30a100829`
- Input legacy index file: `vsl400_front_view_index.csv`
- Total video metadata rows: 24,753
- Total gloss: 400
- Known decode defect retained in train metadata: `019442`

## Split summary

| Split | Video | Signer | Gloss |
|---|---:|---:|---:|
| train | 19,716 | 20 | 400 |
| val | 2,529 | 3 | 400 |
| test | 2,508 | 3 | 400 |

## Validation

- Video IDs unique and six digits: PASS
- Signer partitions exact and disjoint: PASS
- Every split contains all 400 glosses: PASS
- Label IDs contiguous from 0 through 399: PASS
- `019442` retained as signer `024` in train: PASS
