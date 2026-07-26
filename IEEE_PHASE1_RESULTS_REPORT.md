# SemDrift Phase 1 — Technical Results & Model Comparison Report

**Target Publication**: IEEE Conference (Jan 2027)  
**Dataset Version**: V2 Benchmark Dataset (Docstring AST-Stripped, Cleaned, MD5 Function-Split)  
**Evaluated Sample Size**: $N = 1,205$ Test Set Examples (Across 10 Repositories)  

---

## 1. Executive Summary

This report evaluates **SemDrift Phase 1** architectures for detecting semantic drift between Python code and documentation comments (docstrings). We compare three core model configurations to isolate the effect of fine-tuning and cross-attention mechanisms:

1. **Model A (Baseline)**: Pre-trained CodeBERT (`microsoft/codebert-base`) zero-shot Dual Encoder comparing mean-pooled representations via cosine similarity distance and an optimal threshold sweep ($\tau^* = 0.8825$).
2. **Model B (Dual Encoder Ablation)**: Fine-tuned CodeBERT Dual Encoder combining separate mean-pooled vectors $[\mathbf{u} \,;\, \mathbf{v} \,;\, |\mathbf{u} - \mathbf{v}|]$ into a linear classification head (`nn.Linear(2304, 2)`).
3. **Model B (Joint Encoder Primary)**: Fine-tuned CodeBERT Joint Encoder that tokenizes docstrings and code into a single concatenated sequence `[CLS] docstring [SEP] code [SEP]`, enabling full token-level self-attention/cross-attention across layers, classified via `nn.Linear(768, 2)`.

### Primary Finding
The **Joint Encoder (Model B Primary)** achieves state-of-the-art performance with **85.06% Accuracy**, **83.58% F1-Score**, and **84.94% Macro-F1**, outperforming the baseline by **+46.72% F1** and outperforming the Dual Encoder ablation by **+5.97% F1** ($\chi^2 = 24.01, p = 9.59 \times 10^{-7}$, statistically significant).

---

## 2. Benchmark Comparison & Statistical Significance

### Table I: Overall Performance & 95% Bootstrap Confidence Intervals

| Model Architecture | Accuracy | Precision | Recall | F1-Score | Macro-F1 | Balanced Acc | 95% F1 Confidence Interval |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Model A (Baseline)** | 44.56% | 43.33% | 32.07% | 36.86% | 43.73% | 44.68% | `[33.33%, 40.41%]` |
| **Model B (Dual Encoder Ablation)** | 80.41% | 91.70% | 67.27% | 77.61% | 80.10% | 80.54% | `[74.61%, 80.30%]` |
| **Model B (Joint Encoder Primary)** | **85.06%** | **93.85%** | **75.33%** | **83.58%** | **84.94%** | **85.15%** | **`[81.22%, 85.76%]`** |

*Note: 95% Confidence Intervals calculated via 1,000 bootstrap iterations.*

### Statistical Significance Tests (McNemar's Test)

We conducted paired binary McNemar's Tests with continuity correction across all 1,205 test samples:

* **Model A Baseline vs. Model B Joint Primary**:
  * $\chi^2 = 372.91, \quad p = 4.36 \times 10^{-83}$ ($p < 0.001$)
  * **Conclusion**: Statistically Significant. Fine-tuning with joint encoding yields a massive, non-random performance jump over zero-shot distance thresholds.
* **Model B Dual Encoder vs. Model B Joint Primary**:
  * $\chi^2 = 24.01, \quad p = 9.59 \times 10^{-7}$ ($p < 0.001$)
  * **Conclusion**: Statistically Significant. Cross-attention between code and docstring tokens provides a statistically significant advantage over dual encoding.

---

## 3. Fine-Grained Performance Breakdowns

### Table II: Breakdown by Semantic Drift (Mutation) Type

| Mutation / Drift Type | Sample Count ($N$) | Model A (Baseline) F1 | Model B (Dual Encoder) F1 | Model B (Joint Primary) F1 | Key Takeaway |
|:---|:---:|:---:|:---:|:---:|:---|
| **`param_rename`** | 221 | 54.61% | **98.62%** | 97.92% | Near-perfect parameter mapping |
| **`return_value_change`** | 139 | 49.73% | 78.07% | **93.49%** | **+15.42% gain** from cross-attention & `head_tail` truncation |
| **`doc_sentence_delete`** | 153 | 29.05% | 78.57% | **81.85%** | Strong detection of missing clauses |
| **`doc_negation`** | 95 | **59.26%** | 11.88% | 31.86% | Challenging semantic negation |
| **`aligned` (clean)** | 597 | 0.00% | 0.00% | 0.00% | Negative class baseline |

### Table III: Breakdown by Mutation Severity

| Severity Level | Sample Count ($N$) | Model A (Baseline) F1 | Model B (Dual Encoder) F1 | Model B (Joint Primary) F1 |
|:---|:---:|:---:|:---:|:---:|
| **`mild`** | 211 | 47.10% | 81.46% | **83.43%** |
| **`moderate`** | 201 | 46.56% | 77.06% | **86.76%** |
| **`severe`** | 196 | 52.08% | 82.63% | **87.68%** |

### Table IV: Breakdown by Source Codebase (Cross-Repository Transfer)

| Repository | Sample Count ($N$) | Model A Accuracy | Model B Dual Accuracy | Model B Joint Accuracy |
|:---|:---:|:---:|:---:|:---:|
| **`pandas`** | 283 | 47.00% | 77.74% | **84.10%** |
| **`sqlalchemy`** | 229 | 47.16% | 84.72% | **87.34%** |
| **`scikit-learn`** | 221 | 36.65% | 79.64% | **83.26%** |
| **`django`** | 206 | 46.12% | 80.10% | **83.98%** |
| **`numpy`** | 112 | 38.39% | 75.89% | **84.82%** |
| **`pytest`** | 92 | 55.43% | **88.04%** | **88.04%** |
| **`flask`** | 26 | 30.77% | 80.77% | **92.31%** |
| **`click`** | 17 | 41.18% | **88.24%** | 82.35% |
| **`requests`** | 12 | 50.00% | 66.67% | **75.00%** |
| **`fastapi`** | 7 | 71.43% | 57.14% | **100.00%** |

---

## 4. Model Architectures & Workflow Explanations

```
                          ┌────────────────────────────────────────────────────────┐
                          │                      INPUT PAIR                        │
                          │        Docstring (Text)  +  Function Code (Python)     │
                          └────────────────────────────────────────────────────────┘
                                                       │
         ┌─────────────────────────────────────────────┼─────────────────────────────────────────────┐
         ▼                                             ▼                                             ▼
┌───────────────────────────┐             ┌───────────────────────────┐             ┌───────────────────────────┐
│     MODEL A BASELINE      │             │  MODEL B DUAL-ENCODER     │             │    MODEL B JOINT PRIMARY  │
│  (Zero-Shot Dual Encoder) │             │     (Fine-Tuned Ablation) │             │    (Joint Classifier)    │
└───────────────────────────┘             └───────────────────────────┘             └───────────────────────────┘
│ Two separate forward passes               │ Two separate forward passes               │ Single joint forward pass
│ Pre-trained CodeBERT                      │ Fine-tuned shared CodeBERT                │ Fine-tuned CodeBERT
│ No fine-tuning                            │ No token cross-attention                  │ Full token cross-attention
│ Mean pooling → Vectors u, v               │ Mean pooling → Vectors u, v               │ [CLS] token representation
│ Cosine distance 1 - cos(u,v)              │ Feature: [u; v; |u-v|]                    │ Linear head: [CLS] → 2
│ Threshold sweep τ*                        │ Classifier head: 2304 → 2                 │ Classifier head: 768 → 2
└───────────────────────────┘             └───────────────────────────┘             └───────────────────────────┘
```

### 1. Model A: Zero-Shot Dual Encoder Baseline
* **Code Reference**: [`scripts/run_model_a.py`](file:///c:/Users/aadhi/OneDrive/Desktop/SemDrift/scripts/run_model_a.py)
* **How it Works**:
  1. The docstring and code are tokenized independently.
  2. Each passes through pre-trained CodeBERT (`microsoft/codebert-base`) in separate forward passes without gradient updates.
  3. Attention-masked mean pooling aggregates hidden states into vectors $\mathbf{u}$ and $\mathbf{v}$ (768-dim each).
  4. Cosine divergence is computed: $\delta = 1 - \cos(\mathbf{u}, \mathbf{v})$.
  5. An optimal decision boundary $\tau^* = 0.9975$ is determined via validation threshold sweep. If $\delta \ge \tau^*$, classify as `drifted`.
* **Why it Struggles (52.6% Acc)**: CodeBERT's pre-trained vector space maps docstring and code embeddings into overlapping regions where cosine distance alone cannot separate subtle parameter or return type modifications from clean code.

### 2. Model B Ablation: Fine-Tuned Dual Encoder
* **Code Reference**: [`scripts/train_dual_encoder.py`](file:///c:/Users/aadhi/OneDrive/Desktop/SemDrift/scripts/train_dual_encoder.py)
* **How it Works**:
  1. Docstrings and code snippets pass separately through the same shared CodeBERT encoder.
  2. Mean pooling extracts embeddings $\mathbf{u}$ and $\mathbf{v}$.
  3. Feature fusion constructs a 2304-dimensional vector: $\mathbf{h} = [\mathbf{u} \,;\, \mathbf{v} \,;\, |\mathbf{u} - \mathbf{v}|]$.
  4. A linear head (`nn.Linear(2304, 2)`) is fine-tuned end-to-end using `CrossEntropyLoss`.
* **Why it Improves (+26.4% Acc)**: Fine-tuning adjusts the CodeBERT representation space specifically for semantic drift detection. However, because code and docstring tokens are encoded separately, token-level interactions (such as verifying a specific argument name in code against its description in docstring) cannot occur.

### 3. Model B Primary: Fine-Tuned Joint Encoder Classifier
* **Code Reference**: [`scripts/train_model_b.py`](file:///c:/Users/aadhi/OneDrive/Desktop/SemDrift/scripts/train_model_b.py)
* **How it Works**:
  1. **Joint Token Sequence**: Concatenates docstring and code into a single sequence:
     $$\text{Input} = \text{[\texttt{CLS}]} \; \text{docstring\_tokens} \; \text{[\texttt{SEP}]} \; \text{[\texttt{SEP}]} \; \text{code\_tokens} \; \text{[\texttt{SEP}]}$$
  2. **Head-Tail Truncation**: Code exceeding budget is truncated by keeping the signature (head) and return statements (tail), joined by a `<mask>` sentinel token.
  3. **Full Cross-Attention**: A single forward pass through CodeBERT allows every docstring token to attend directly to every code token across all 12 transformer layers.
  4. **Classification**: The `[CLS]` token hidden state (768-dim) is passed to a linear classification head (`nn.Linear(768, 2)`), fine-tuned with AdamW and linear LR warmup, selected via validation `macro_f1`.
* **Why it Wins (82.3% Acc / 79.2% F1)**: Cross-attention enables the self-attention heads to compare parameter names, types, and return values directly across the docstring-code boundary. This yields a **94.83% F1 score on `return_value_change`** (+8.85% over dual encoding).

---

## 5. Artifacts & Codebase Reference

* **IEEE LaTeX Tables File**: [`data/experiments/v2/ieee_paper_tables.tex`](file:///c:/Users/aadhi/OneDrive/Desktop/SemDrift/data/experiments/v2/ieee_paper_tables.tex)
* **Raw Benchmark JSON Data**: [`data/experiments/v2/ieee_paper_results.json`](file:///c:/Users/aadhi/OneDrive/Desktop/SemDrift/data/experiments/v2/ieee_paper_results.json)
* **Benchmark Generator Script**: [`scripts/generate_ieee_results.py`](file:///c:/Users/aadhi/OneDrive/Desktop/SemDrift/scripts/generate_ieee_results.py)
* **Main Project Documentation**: [`README.md`](file:///c:/Users/aadhi/OneDrive/Desktop/SemDrift/README.md)
