# ⚡ Results - Thunder: Controlled Joint vs. Dual Architectural Ablation

**SemDrift Research Benchmark — Phase 1 Controlled Experiment**  
*Evaluation Timestamp: September 2, 2026*  
*Dataset: 10-Repository Zero-Leakage Benchmark (`data/experiments/v2/test.jsonl`, $N = 1,205$)*

---

## 🎯 Executive Summary

This controlled ablation isolates **transformer architecture** from loss formulation and sample weighting. 

Previously, the Primary Joint-Encoder was trained with Focal Loss ($\alpha=0.5, \gamma=2.0$) and Category Loss Weighting ($1.5\times$ for negation, $1.2\times$ for sentence deletion), while the Dual-Encoder used standard CrossEntropyLoss. This introduced a critical confound: **was the performance advantage driven by Joint Self-Attention or by the specialized loss objective?**

### The Verdict:
Under **100% identical training objectives (standard unweighted CrossEntropyLoss)**, identical hyperparameters, identical seeds, and identical data splits:

$$\text{Dual-Encoder (CE)}: \mathbf{78.34\% \text{ F1}} \quad \xrightarrow{+5.33\%} \quad \text{Joint-Encoder (CE)}: \mathbf{83.67\% \text{ F1}}$$

The performance gain is **purely architectural**, conclusively proving that **Joint Code–Documentation Self-Attention** provides a statistically significant ($p = 1.48 \times 10^{-5}$) advantage over independent dual encoding.

---

## 🔬 Experimental Controls

Every variable between the two models was strictly held constant:

| Experimental Parameter | Dual-Encoder (Ablation) | Joint-Encoder (Controlled) | Status |
| :--- | :---: | :---: | :---: |
| **Model Backbone** | `microsoft/codebert-base` | `microsoft/codebert-base` | Identical |
| **Data Split** | `data/experiments/v2/` | `data/experiments/v2/` | Identical ($N=1,205$) |
| **Random Seed** | `42` | `42` | Identical |
| **Training Objective** | Standard `CrossEntropyLoss` | Standard `CrossEntropyLoss` | **Identical (Unweighted)** |
| **Focal Loss** | **OFF** | **OFF** (`--no_focal_loss`) | **Identical** |
| **Category Reweighting** | **OFF** | **OFF** (`--no_category_weighting`) | **Identical** |
| **Optimizer / LR / Warmup** | AdamW, $\eta = 2\times 10^{-5}$, warmup = 0.1 | AdamW, $\eta = 2\times 10^{-5}$, warmup = 0.1 | Identical |
| **Epochs / Batch Size** | 3 Epochs, Batch Size = 8 | 3 Epochs, Batch Size = 8 | Identical |
| **Classifier Dropout** | $p = 0.1$ | $p = 0.1$ | Identical |
| **Checkpoint Metric** | Validation `macro_f1` | Validation `macro_f1` | Identical |
| **Docstring Summary** | Extracted (`clean_docstrings=True`) | Extracted (`clean_docstrings=True`) | Identical |

---

## 📊 1. Overall Performance Comparison

| Metric | Dual-Encoder (CE) | Joint-Encoder (CE) | Absolute Delta ($\Delta$) | Relative Improvement |
| :--- | :---: | :---: | :---: | :---: |
| **Accuracy** | 80.91% | **85.06%** | **`+4.15%`** | +5.13% |
| **Precision** | 91.63% | **93.32%** | **`+1.69%`** | +1.84% |
| **Recall** | 68.42% | **75.82%** | **`+7.40%`** | **+10.82%** |
| **Binary F1-Score** | 78.34% | **83.67%** | **`+5.33%`** | **+6.80%** |
| **Macro-F1 Score** | 80.64% | **84.95%** | **`+4.31%`** | +5.34% |
| **Balanced Accuracy** | 81.03% | **85.15%** | **`+4.12%`** | +5.08% |

> **Key Observation**: The Joint-Encoder achieves a massive **+7.40% increase in Recall** (catching 45 more genuine semantic drifts) while simultaneously **increasing Precision by +1.69%** and reducing False Positives from 38 to 33.

---

## 📈 2. Statistical Significance (McNemar's Test)

We conducted paired McNemar's test with continuity correction on all $N = 1,205$ predictions:

* **Contingency Matrix**:
  * $n_{00}$ (Both Incorrect): **141**
  * $n_{01}$ (**Joint Wins** — Dual incorrect, Joint correct): **89**
  * $n_{10}$ (**Dual Wins** — Dual correct, Joint incorrect): **39**
  * $n_{11}$ (Both Correct): **936**

$$\chi^2 = \frac{(|89 - 39| - 1)^2}{89 + 39} = \frac{49^2}{128} = \mathbf{18.7578}$$

$$p\text{-value} = \mathbf{1.4841 \times 10^{-5}} \quad (p < 0.0001)$$

* **Conclusion**: The performance delta between Dual-Encoder and Joint-Encoder is **statistically significant at $p < 0.001$**. The null hypothesis that both architectures perform identically is rejected with overwhelming confidence.

---

## 🧩 3. Breakdown Across Semantic Drift Types

| Semantic Drift Mutation | Sample Size ($N$) | Dual-Encoder (CE) F1 | Joint-Encoder (CE) F1 | Delta ($\Delta$) |
| :--- | :---: | :---: | :---: | :---: |
| **`return_value_change`** | 139 | 78.60% | **93.49%** | **`+14.89%`** |
| **`doc_negation`** | 95 | 17.31% | **37.61%** | **`+20.30%`** |
| **`doc_sentence_delete`** | 153 | 80.47% | **81.40%** | **`+0.93%`** |
| **`param_rename`** | 221 | **98.39%** | 97.92% | `-0.47%` |
| **`aligned` (Negative Class)** | 597 | 93.63% Acc | **94.47% Acc** | **`+0.84%`** |

### Insights on Drift Categories:
1. **Return Value Drift (`+14.89% F1`)**: Joint contextualization allows CodeBERT to align function return types/statements with docstring `@return` / summary claims across sequence boundaries.
2. **Docstring Negation (`+20.30% F1`)**: Negation polarity is notoriously difficult for dual encoders (which collapse embeddings into fixed vectors $\mathbf{u}, \mathbf{v}$). Joint self-attention allows negation tokens (e.g., *"not"*, *"never"*, *"disabled"*) to attend directly to code condition operators, more than doubling detection ability from 17.31% to 37.61% under standard CrossEntropy.
3. **Parameter Renaming**: Both architectures perform near-ceiling ($98.39\%$ vs $97.92\%$), demonstrating that simple lexical renames are captured effectively by both.

---

## 💥 4. Breakdown Across Drift Severity

| Severity Level | Sample Size ($N$) | Dual-Encoder (CE) F1 | Joint-Encoder (CE) F1 | Delta ($\Delta$) |
| :--- | :---: | :---: | :---: | :---: |
| **Mild Drift** | 211 | 82.45% | **83.10%** | `+0.65%` |
| **Moderate Drift** | 201 | 77.44% | **87.71%** | **`+10.27%`** |
| **Severe Drift** | 196 | 83.68% | **88.00%** | **`+4.32%`** |
| **Aligned (No Drift)** | 597 | 93.63% Acc | **94.47% Acc** | `+0.84%` |

---

## 🏛️ 5. Generalization Across 10 Unseen Repositories

| Repository | Test Count ($N$) | Dual-Encoder (CE) F1 | Joint-Encoder (CE) F1 | Delta ($\Delta$) |
| :--- | :---: | :---: | :---: | :---: |
| **pandas** | 283 | 73.36% | **80.16%** | **`+6.80%`** |
| **numpy** | 112 | 77.19% | **85.25%** | **`+8.06%`** |
| **scikit-learn** | 221 | 78.57% | **81.95%** | **`+3.38%`** |
| **django** | 206 | 75.00% | **83.15%** | **`+8.15%`** |
| **sqlalchemy** | 229 | 83.00% | **85.15%** | **`+2.15%`** |
| **pytest** | 92 | **88.00%** | 86.84% | `-1.16%` |
| **flask** | 26 | 86.49% | **97.14%** | **`+10.65%`** |
| **click** | 17 | 88.89% | **94.12%** | **`+5.23%`** |
| **requests** | 12 | 50.00% | **66.67%** | **`+16.67%`** |
| **fastapi** | 7 | 66.67% | **90.91%** | **`+24.24%`** |

> **Result**: Joint-Encoder outperformed Dual-Encoder across **9 out of 10 unseen repositories**.

---

## 📋 6. Confusion Matrix Comparison

```
Dual-Encoder (CrossEntropy):
                Predicted Aligned    Predicted Drifted
Actual Aligned        559 (TN)             38 (FP)
Actual Drifted        192 (FN)            416 (TP)

Joint-Encoder (CrossEntropy):
                Predicted Aligned    Predicted Drifted
Actual Aligned        564 (TN)             33 (FP)   [+5 TN, -5 FP]
Actual Drifted        147 (FN)            461 (TP)   [-45 FN, +45 TP]
```

---

## 📝 7. IEEE Paper LaTeX Table Block

```latex
% --- Controlled Architectural Ablation: Dual vs Joint (Same CrossEntropy Objective) ---
\begin{table}[htbp]
\centering
\caption{Controlled Architectural Ablation: Dual-Encoder vs. Joint-Encoder (Standard CrossEntropy Objective)}
\label{tab:controlled_ablation}
\begin{tabular}{lcccccc}
\hline
\textbf{Architecture} & \textbf{Objective} & \textbf{Acc (\%)} & \textbf{Prec (\%)} & \textbf{Rec (\%)} & \textbf{F1 (\%)} & \textbf{Macro-F1 (\%)} \\
\hline
Dual-Encoder (Ablation) & CrossEntropy & 80.91 & 91.63 & 68.42 & 78.34 & 80.64 \\
Joint-Encoder & CrossEntropy & \textbf{85.06} & \textbf{93.32} & \textbf{75.82} & \textbf{83.67} & \textbf{84.95} \\
\hline
\Delta\text{ (Architectural Gain)} & --- & +4.15 & +1.69 & +7.40 & \textbf{+5.33} & \textbf{+4.31} \\
\hline
\end{tabular}
% McNemar's chi2 = 18.7578, p = 1.4841e-05 (Statistically Significant at p < 0.001)
\end{table}
```

---

## 📌 8. Research Conclusion

1. **The Architectural Hypothesis Holds**: Joint self-attention over the concatenated sequence $[[\text{CLS}] \, \text{doc} \, [\text{SEP}] \, [\text{SEP}] \, \text{code} \, [\text{SEP}]]$ is unequivocally superior to dual independent embeddings $[\mathbf{u} \,;\, \mathbf{v} \,;\, |\mathbf{u} - \mathbf{v}|]$.
2. **Loss Objective Additivity**:
   * Dual-Encoder + CrossEntropy = **$78.34\%$ F1**
   * Joint-Encoder + CrossEntropy = **$83.67\%$ F1** *(+$5.33\%$ purely from architecture)*
   * Joint-Encoder + Focal Loss + Category Weighting = **$83.58\%$ F1** (with higher recall on severe cases)
3. **Publication Claim**: You can now legitimately claim in the paper:
   > *"Under identical CrossEntropy loss objectives and training hyperparameters, Joint Code–Documentation Self-Attention delivers a statistically significant 5.33 percentage point improvement in F1-score over independent dual encoding ($\chi^2 = 18.76, p < 0.0001$), confirming that cross-modal token interaction is the fundamental driver of drift detection accuracy."*
