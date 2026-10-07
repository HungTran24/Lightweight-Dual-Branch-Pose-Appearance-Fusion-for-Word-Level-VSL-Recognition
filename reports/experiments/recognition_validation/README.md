# Canonical recognition validation index

This is the small canonical index for completed validation evidence. It links persisted reports and artifacts rather than copying predictions, score arrays, checkpoints, or charts.

## Scope and status

- Split: clean signer-disjoint `manifests/audit/train_clean.csv` and `manifests/audit/val_clean.csv` only.
- Metrics: fractions in `validation_summary.csv` and `.json`; display percentages multiply by 100. Aggregate rows are three-seed mean and sample SD unless stated otherwise.
- `complete`: all declared three seeds are persisted. `artifact_complete`: derived feature artifact, no classifier metric.
- Historical E023/E024 Pose53 U24/U30 test already exists. New test recognition remained closed during E026–E032 validation selection; frozen RGB/U60 test was executed only after choices were locked.

## Canonical sources

- E006 Pose59 frame-length rows: `../e006_tcn_pose59_uniform/comparison.csv` and `.json`.
- E007–E014 full Uniform-30 model/view matrix: `../e006_e014_uniform30/comparison.csv` and `.json`.
- E015–E018 Pose53 frame-length rows: `../e008_e018_pose53_frame/comparison.csv` and `.json`.
- E020 SPOTER negative gate: `../e020_spoter_pose54_uniform30_s42/result.json`.
- E022 BiLSTM Pose53 Uniform-24: `../../../artifacts/runs/e022_bilstm_pose53_uniform24_s{42,43,44}/summary.json`.
- E026 RGB and arithmetic fusion: `../e026_hand_rgb_late_fusion/three_seed/summary.json`, `three_seed/README.md`, and `three_seed/mean_alpha_curve.csv`.
- E027 weighted log-probability fusion: `../e027_weighted_log_prob_fusion/summary.json`, `README.md`, and `beta_curve.csv`.
- E028 temporal-3 per-seed results: `../e028_rgb_temporal3/seed42/summary.json`, `seed43/summary.json`, and `seed44/summary.json`. The seed-42 gate failure is retained as metadata; seeds 43/44 were authorized continuations.
- E034 Shared-hand-128 global RGB: `../e034_rgb_shared_hand128/summary.json` and `README.md`; six official U24/U30 seed configs use the pinned E026 global RGB receipt.
- E035 Shared-hand-128 + Temporal3 U30: `../e035_rgb_shared_hand128_temporal3/summary.json` and `README.md`; three official seed configs use the pinned E028 Temporal3 receipt.
- E029 Uniform-60 derivation: `../../../artifacts/yolo/e029_pose59_uniform60_dev/summary.json` and nested `receipt.json`.
- E032 Uniform-60 BiLSTM: `../../../artifacts/runs/e032_bilstm_pose53_uniform60_s42/summary.json`, `_s43/summary.json`, and `_s44/summary.json`; W&B runs `58em0qzx`, `vzhdv51x`, and `y9obafpd` are finished.
- E030 Uniform-60 TCN: `../../../artifacts/runs/e030_tcn_pose53_uniform60_s42_s42/summary.json`, `_s43_s43/summary.json`, and `_s44_s44/summary.json`. These historical directories repeat the seed suffix; training is not rerun solely to rename artifacts.
- E031 completed Uniform-60 ST-GCN: `../../../artifacts/runs/e031_stgcn_pose53_uniform60_s42/summary.json`, `_s43/summary.json`, and `_s44/summary.json`. W&B runs `w1ml16w4`, `duqjovvv`, and `t6zxqftw` are finished; local required files and HF revision `13c70753983ed8ad33ca28db83ae016bc2b53c8b` are verified for all three seeds.

## Ineligible or non-recognition items

- E001 is legacy/superseded unless clean-comparable validation evidence is added.
- E004/E005 are detector work, E019 is a smoke run, E021 is feature extraction, and E025 is a train-signer diagnostic. They are not recognition rows.
- E023/E024 are historical test evaluation, outside this validation-only ledger.
- Frozen RGB test inference was run only after validation choices were locked and is reported separately at `../recognition_test/`; it is not part of this validation ledger.

## Interpretation boundaries

E027 is the best completed U30 RGB-fusion validation result. E030–E032 use the separate U60 pose data revision, so their scores must not be used as an E027 fusion baseline. E031 is complete across three seeds and improves on the E013 U30 ST-GCN reference by +1.21 pp Top-1, +0.04 pp Top-5, and +1.12 pp Macro-F1. E032 BiLSTM U60 reaches 56.24 ± 3.17% Top-1 and is less competitive than its U24/U30 variants. Frozen RGB test results are isolated in `../recognition_test/`. `validation_summary.csv` and `.json` are the durable committed ledgers; paths under `artifacts/` are local generated evidence and are not tracked by Git.
