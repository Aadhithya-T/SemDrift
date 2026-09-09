# SemDrift V2 Eval Set Benchmark Report (N = 2,200)

This report records the comparative evaluation of all 5 SemDrift models on the newly introduced V2 evaluation set (`data/v2_real_world/evaluation/eval_set.jsonl`).
The dataset contains exactly **2,200 samples** (1,100 consistent / 1,100 inconsistent).

---

## Master Comparison Table

| Model Architecture | Category | Threshold (tau) | Accuracy | Balanced Acc | Drift Recall | Drift Precision | Binary F1 | Macro F1 | TN | FP | FN | TP | Confusion Matrix |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Zero-Shot CodeBERT** | Pretrained Semantic | `0.96` | 50.64% | 50.64% | 32.00% | 51.01% | 39.33% | 48.86% | 762 | 338 | 748 | 352 | `TN=762, FP=338, FN=748, TP=352` |
| **TF-IDF Combined** | Pure Lexical Baseline | `0.53` | 69.45% | 69.45% | 44.64% | 88.63% | 59.37% | 67.45% | 1037 | 63 | 609 | 491 | `TN=1037, FP=63, FN=609, TP=491` |
| **TF-IDF Relational** | Enhanced Lexical Baseline | `0.51` | 69.82% | 69.82% | 46.91% | 86.58% | 60.85% | 68.15% | 1020 | 80 | 584 | 516 | `TN=1020, FP=80, FN=584, TP=516` |
| **Dual-Encoder (CodeBERT)** | Fine-Tuned Bi-Encoder | `0.50` | 73.50% | 73.50% | 48.64% | 96.75% | 64.73% | 71.75% | 1082 | 18 | 565 | 535 | `TN=1082, FP=18, FN=565, TP=535` |
| **Joint-Encoder (CodeBERT)** | Fine-Tuned Cross-Input | `0.50` | 73.50% | 73.50% | 49.91% | 94.49% | 65.32% | 71.94% | 1068 | 32 | 551 | 549 | `TN=1068, FP=32, FN=551, TP=549` |

---

## Key Analytical Observations

1. **Consistent Performance Between Deep Models**: Both the Joint-Encoder and Dual-Encoder achieve **73.50% Accuracy** and **73.50% Balanced Accuracy** on this 2,200-sample test set.
2. **Drift Specificity**: Both deep models maintain very low false-positive rates on clean samples (FP = 32 for Joint, FP = 18 for Dual), yielding high precision (>94%).
3. **Lexical Baseline Competitiveness**: The TF-IDF Relational baseline reaches **69.82% Balanced Accuracy** and **68.15% Macro F1**, demonstrating strong lexical signal retention across n-grams and overlap features.
4. **Zero-Shot Baseline**: Pretrained CodeBERT cosine divergence operates at **50.64% Balanced Accuracy** (close to random binary chance), demonstrating the clear necessity of fine-tuning.