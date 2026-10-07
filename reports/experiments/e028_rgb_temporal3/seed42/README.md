# E028 coarse-temporal hand-RGB, seed 42

Validation-only result. The test split was not accessed.

E028 replaces each hand's global U30 mean with three masked means over sampling slots 0–9, 10–19, and 20–29. A shared 576→64 projection processes the six hand/bin tokens before a 384→400 classifier. The head has 190,928 trainable parameters. Fusion keeps the E027 protocol fixed at β = 0.5.

| Metric | E026 RGB-U30 | E028 temporal RGB | Δ |
|---|---:|---:|---:|
| Top-1 | 50.53 % | 59.59 % | +9.05 pp |
| Top-5 | 72.12 % | 80.78 % | +8.66 pp |
| Macro-F1 | 52.75 % | 61.21 % | +8.46 pp |

| Metric | E027 baseline fusion | E028 fusion | Δ |
|---|---:|---:|---:|
| Top-1 | 85.01 % | 85.37 % | +0.36 pp |
| Top-5 | 96.24 % | 96.68 % | +0.43 pp |
| Macro-F1 | 85.51 % | 85.69 % | +0.18 pp |

The RGB representation improves strongly, confirming that global temporal averaging was a substantial branch bottleneck. The fused system also improves every metric, but Top-1 did not meet the original continuation threshold of +0.50 pp, so the seed-42 gate failed. The experiment was later explicitly continued through seeds 43/44 as an approved diagnostic; the three-seed result is summarized in the parent report.
