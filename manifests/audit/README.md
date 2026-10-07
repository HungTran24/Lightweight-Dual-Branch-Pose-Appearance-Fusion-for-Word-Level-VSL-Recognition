# Recognition manifests and label corrections

VSL400 inputs are obtained from the dataset administrator under the applicable
Data Usage Agreement. This repository does not grant permission to redistribute
videos, hand crops, features or dataset-derived manifests.

Place the authorized manifests at the paths below before running extraction,
RGB training, fusion or data diagnostics. Pose classification from prepared NPZ
splits does not require the CSV manifests.

| Purpose | Required local file | Expected count |
|---|---|---:|
| Training | `manifests/audit/train_clean.csv` | 19,715 manifest rows; 19,714 usable videos |
| Validation | `manifests/audit/val_clean.csv` | 2,529 videos |
| Test | `manifests/vsl400_front_400gloss/test.csv` | 2,508 videos |
| Label-delta diagnostics | `manifests/audit/vsl400_label_corrections.csv` | 56 reviewed cases |
| Original-label ablations | `manifests/vsl400_front_400gloss/train.csv`, `val.csv` | 19,716 / 2,529 rows |

The corrected protocol remaps 47 labels (25 train, 22 validation), drops one
unusable training video, and retains eight reviewed cases. An additional training
video is excluded from feature extraction because its MP4 container cannot be
decoded. The 19,715-row training manifest therefore yields 19,714 feature samples.
These exclusions must not be applied to validation or test by changing the split.

Use the corrected train/validation manifests for the reported recognition runs.
Original-label manifests are only for the corresponding label-noise ablations.
Keep the label map, signer split and corrections consistent with the feature
receipts and config identities; do not substitute newly inferred labels for the
reported protocol.

The concrete manifests and per-video correction records are deliberately not
included here. Obtain them through an authorized channel; their filenames remain
part of the input contract. See [PROJECT.md](../../PROJECT.md) for the complete
local-input layout and experiment commands.
