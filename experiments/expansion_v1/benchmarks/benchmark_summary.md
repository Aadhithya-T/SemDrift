# SemDrift Benchmark Results: Real Dataset Expansion V1 (10,000 samples)

- **Dataset**: Real Dataset Expansion V1 (`data/real_dataset_expansion_v1/`)
- **Splits**: Train: 8,128 | Val: 544 | Test: 1328
- **Total Samples**: 10,000 samples across 33 open-source Python repositories
- **Evaluation Date**: 2026-10-12 02:11:34 UTC

## Overall Model Performance on Test Split

| Architecture | Threshold | Accuracy | Balanced Acc | Drift Precision | Drift Recall | Binary F1 | Macro F1 | ROC-AUC | PR-AUC |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Zero-Shot CodeBERT (Cosine Divergence)** | 0.39 | 46.76% | 53.00% | 43.18% | 89.66% | 58.29% | **42.36%** | - | - |
| **TF-IDF Combined (Pure Lexical)** | 0.48 | 90.06% | 89.90% | 87.34% | 88.93% | 88.13% | **89.79%** | 0.9469 | 0.9462 |
| **TF-IDF Relational (Dual Overlap)** | 0.49 | 91.87% | 91.57% | 90.49% | 89.84% | 90.16% | **91.62%** | 0.9582 | 0.9541 |
| **Dual-Encoder (CodeBERT Bi-Encoder)** | 0.50 | 97.36% | 96.98% | 98.86% | 94.74% | 96.76% | **97.27%** | - | - |
| **Joint-Encoder (Focal Loss + Cat Weighting)** | 0.50 | 84.94% | 84.91% | 80.10% | 84.75% | 82.36% | **84.61%** | - | - |
| **Joint-Encoder (Standard CrossEntropy)** | 0.50 | 84.79% | 84.94% | 79.23% | 85.84% | 82.40% | **84.50%** | - | - |

## Confusion Matrices

| Architecture | TN | FP | FN | TP | Confusion Matrix Summary |
|:---|:---:|:---:|:---:|:---:|:---|
| **Zero-Shot CodeBERT (Cosine Divergence)** | 127 | 650 | 57 | 494 | `TN=127, FP=650, FN=57, TP=494` |
| **TF-IDF Combined (Pure Lexical)** | 706 | 71 | 61 | 490 | `TN=706, FP=71, FN=61, TP=490` |
| **TF-IDF Relational (Dual Overlap)** | 725 | 52 | 56 | 495 | `TN=725, FP=52, FN=56, TP=495` |
| **Dual-Encoder (CodeBERT Bi-Encoder)** | 771 | 6 | 29 | 522 | `TN=771, FP=6, FN=29, TP=522` |
| **Joint-Encoder (Focal Loss + Cat Weighting)** | 661 | 116 | 84 | 467 | `TN=661, FP=116, FN=84, TP=467` |
| **Joint-Encoder (Standard CrossEntropy)** | 653 | 124 | 78 | 473 | `TN=653, FP=124, FN=78, TP=473` |

## Statistical Significance Tests (McNemar's Test with Continuity Correction)

| Comparison (Model A vs Model B) | N | Model A Unique Wins | Model B Unique Wins | Chi² | p-value | Significant (p<0.05) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| Joint-Encoder (Focal) vs Dual-Encoder | 1328 | 10 | 175 | 145.3838 | 0.0000e+00 | ✅ Yes |
| Joint-Encoder (Focal) vs Joint-Encoder (CE) | 1328 | 33 | 31 | 0.0156 | 9.0052e-01 | ❌ No |
| Joint-Encoder (Focal) vs TF-IDF Combined | 1328 | 68 | 136 | 22.0049 | 3.0000e-06 | ✅ Yes |
| Dual-Encoder vs TF-IDF Combined | 1328 | 111 | 14 | 73.7280 | 0.0000e+00 | ✅ Yes |
