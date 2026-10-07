# E038 Side Study: Per-Frame Hand-RGB BiLSTM and Late Fusion

## Overview

- **Objective**: Side investigation evaluating whether frame-level temporal modeling via Bidirectional LSTM on per-frame MobileNetV3-Small embeddings outperforms segment-pooled representations (Temporal3) and global average pooling.
- **Backbone**: Frozen MobileNetV3-Small (`IMAGENET1K_V1`), extracting 576-D embeddings per valid hand crop (128x128).
- **Sequence Head**:
  - Input: Concatenated left and right hand embeddings (`[N, T, 1152]`).
  - Projection: `Linear(1152 -> 256)` + `LayerNorm` + `GELU`.
  - Temporal Backbone: 1-layer `Bidirectional LSTM` (`hidden_size=128`, output dim `256`).
  - Pooling: Masked temporal average pooling across valid frames.
  - Classifier: `Linear(256 -> 400)`.
  - Trainable parameters: **793,744**.
- **Late Fusion Protocol**: Fixed-beta log-probability fusion with Pose53-TCN:
  $$\log(\max(P_{\text{pose}}, 10^{-12})) + 0.5 \cdot \log(\max(P_{\text{rgb}}, 10^{-12}))$$

---

## 1. Development Validation Results (3-seed Mean ± Std)

| Sampling | Model | Seed 42 | Seed 43 | Seed 44 | Mean ± Std |
|---|---|---:|---:|---:|---:|
| **U30** | RGB-only (BiLSTM) | 60.58% | 61.13% | 61.01% | **60.91 ± 0.30%** |
| | Pose53-TCN | 83.27% | 84.18% | 82.44% | **83.30 ± 0.83%** |
| | **Fused (β = 0.5)** | 85.01% | 85.57% | 85.37% | **85.32 ± 0.27%** |
| **U24** | RGB-only (BiLSTM) | 60.58% | 60.26% | 60.74% | **60.52 ± 0.24%** |

---

## 2. Frozen Test Results (3-seed Mean ± Std, 2,508 Native Deaf Signers)

| Sampling | Model | Seed 42 | Seed 43 | Seed 44 | Mean ± Std | Delta vs Pose |
|---|---|---:|---:|---:|---:|---:|
| **U30** | RGB-only (BiLSTM) | 78.79% | 78.59% | 78.11% | **78.50 ± 0.35%** | — |
| | Pose53-TCN | 91.51% | 92.54% | 91.95% | **92.00 ± 0.52%** | reference |
| | **Fused (β = 0.5)** | 93.54% | 94.10% | 93.70% | **93.78 ± 0.29%** | **+1.78%** |
| **U24** | RGB-only (BiLSTM) | 77.11% | 77.59% | 76.95% | **77.22 ± 0.33%** | — |
| | Pose53-TCN | 92.70% | 91.59% | 91.71% | **92.00 ± 0.61%** | reference |
| | **Fused (β = 0.5)** | 93.62% | 93.42% | 93.18% | **93.41 ± 0.22%** | **+1.41%** |

---

## 3. Comparison Across RGB Stream Architectures (Frozen Test U30)

| Architecture | RGB Param | RGB Top-1 | Fused Top-1 | Fused Gain vs Pose |
|---|---:|---:|---:|---:|
| Global Average Pooling (E027) | **0.46M** | 59.97 ± 0.63% | 93.73 ± 0.24% | +1.73% |
| **Per-Frame BiLSTM (E038)** | **0.79M** | **78.50 ± 0.35%** | **93.78 ± 0.29%** | **+1.78%** |
| **Temporal3 (E028 / RGB-E035)** | **1.38M** | 64.95 ± 0.40% | **94.27 ± 0.36%** | **+2.27%** |

### Findings & Conclusion
1. **RGB Standalone Accuracy**: Per-frame BiLSTM sequence modeling significantly boosts standalone RGB accuracy (78.50% vs 64.95% for Temporal3 and 59.97% for Global Pooling) because the recurrent cell effectively models temporal transitions across individual hand crops.
2. **Complementary Late Fusion Capacity**: Despite higher standalone accuracy, BiLSTM late fusion gain (+1.78%) is lower than Temporal3 (+2.27%), performing similarly to simple global pooling (+1.73%).
3. **Architectural Choice for Main Paper**: Temporal3 multi-segment pooling remains the optimal RGB representation for pairing with skeleton-based Pose TCN / ST-GCN classifiers in the primary manuscript.
