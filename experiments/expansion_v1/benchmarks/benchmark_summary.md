# SemDrift Benchmark Results: Real Dataset Expansion V1 (10,000 samples)

- **Dataset**: Real Dataset Expansion V1 (`data/real_dataset_expansion_v1/`)
- **Splits**: Train: 8,128 | Val: 544 | Test: 1328
- **Total Samples**: 10,000 samples across 33 open-source Python repositories
- **Evaluation Date**: 2026-10-11 12:44:58 UTC

## Overall Model Performance on Test Split

| Architecture | Threshold | Accuracy | Balanced Acc | Drift Precision | Drift Recall | Binary F1 | Macro F1 | ROC-AUC | PR-AUC |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Zero-Shot CodeBERT (Cosine Divergence)** | 0.39 | 46.76% | 53.00% | 43.18% | 89.66% | 58.29% | **42.36%** | - | - |
| **TF-IDF Combined (Pure Lexical)** | 0.48 | 90.06% | 89.90% | 87.34% | 88.93% | 88.13% | **89.79%** | 0.9469 | 0.9462 |
| **TF-IDF Relational (Dual Overlap)** | 0.51 | 91.94% | 91.58% | 90.96% | 89.47% | 90.21% | **91.68%** | 0.9587 | 0.9549 |

## Confusion Matrices

| Architecture | TN | FP | FN | TP | Confusion Matrix Summary |
|:---|:---:|:---:|:---:|:---:|:---|
| **Zero-Shot CodeBERT (Cosine Divergence)** | 127 | 650 | 57 | 494 | `TN=127, FP=650, FN=57, TP=494` |
| **TF-IDF Combined (Pure Lexical)** | 706 | 71 | 61 | 490 | `TN=706, FP=71, FN=61, TP=490` |
| **TF-IDF Relational (Dual Overlap)** | 728 | 49 | 58 | 493 | `TN=728, FP=49, FN=58, TP=493` |
