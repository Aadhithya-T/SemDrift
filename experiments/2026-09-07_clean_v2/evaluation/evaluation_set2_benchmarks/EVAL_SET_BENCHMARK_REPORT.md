# SemDrift V2 Evaluation Set Benchmark Report

This report records the comparative evaluation of all 5 SemDrift model families on the evaluation set ($N = 2,000$ samples).

---

## 1. Master Performance Comparison Table

| Model Architecture | Category | Threshold (tau) | Accuracy | Balanced Acc | Drift Recall | Drift Precision | Binary F1 | Macro F1 | TN | FP | FN | TP | Confusion Matrix |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Zero-Shot CodeBERT** | Pretrained Semantic | `0.96` | 51.35% | 51.35% | 26.70% | 52.66% | 35.43% | 48.20% | 760 | 240 | 733 | 267 | `TN=760, FP=240, FN=733, TP=267` |
| **TF-IDF Combined (Pure Lexical)** | Pure Lexical Baseline | `0.53` | 73.35% | 73.35% | 58.10% | 83.60% | 68.55% | 72.72% | 886 | 114 | 419 | 581 | `TN=886, FP=114, FN=419, TP=581` |
| **TF-IDF Relational (Dual Overlap)** | Enhanced Lexical Baseline | `0.51` | 74.95% | 74.95% | 62.80% | 82.96% | 71.49% | 74.57% | 871 | 129 | 372 | 628 | `TN=871, FP=129, FN=372, TP=628` |
| **Dual-Encoder (CodeBERT)** | Fine-Tuned Bi-Encoder | `0.50` | 89.20% | 89.20% | 88.40% | 89.84% | 89.11% | 89.20% | 900 | 100 | 116 | 884 | `TN=900, FP=100, FN=116, TP=884` |
| **Joint-Encoder (CodeBERT)** | Fine-Tuned Cross-Input | `0.50` | 92.75% | 92.75% | 90.20% | 95.05% | 92.56% | 92.75% | 953 | 47 | 98 | 902 | `TN=953, FP=47, FN=98, TP=902` |

---

## 2. Model Breakdown Highlights

Detailed per-model breakdown files (mutation types, repository distributions, and severity tiers) are stored in individual subdirectories under `experiments/2026-09-07_clean_v2/evaluation/eval_set_benchmarks/`.