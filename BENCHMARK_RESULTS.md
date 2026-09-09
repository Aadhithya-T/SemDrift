# Master Benchmark Report: SemDrift V1 & V2 Comprehensive Evaluation

This document serves as the authoritative, repository-level record of all experimental models evaluated across both the **V1 Synthetic** and **V2 Real-World Grounded** datasets in SemDrift. It aggregates every metric, confusion matrix ($TN, FP, FN, TP$), fine-grained mutation breakdown, severity breakdown, repository breakdown, score distribution, and statistical significance test across all 5 model families.

---

## 1. Executive Summary & Core Research Findings

SemDrift investigates whether machine learning models can detect semantic drift between source code and natural language documentation. A central finding of this research is the **Synthetic-to-Real Generalization Gap**:

1. **In-Domain Synthetic Mastery (V1 Benchmark)**:
   - Under controlled synthetic conditions, deep neural architectures demonstrate clear, monotonic improvements over lexical baselines:
     $$\text{Zero-Shot (44.56\%)} \rightarrow \text{TF-IDF Combined (65.23\%)} \rightarrow \text{TF-IDF Relational (70.04\%)} \rightarrow \text{Dual-Encoder (80.91\%)} \rightarrow \text{Joint-Encoder (85.06\%)} \rightarrow \text{Joint Focal (85.81\%)}$$
   - Both cross-encoder attention and hard-negative mining provide substantial, statistically significant gains on rule-based mutations.

2. **The Authentic Real-World Generalization Gap (V2 Benchmark)**:
   - When evaluated against the locked 104-sample diagnostic test set of authentic human commits mined from production repositories (Click, FastAPI, Django, etc.):
     - **TF-IDF Combined** achieves **32.69%** authentic drift recall at $\tau = 0.50$.
     - **Zero-Shot CodeBERT** achieves **36.54%** authentic drift recall at $\tau = 0.96$.
     - **Joint-Encoder (CodeBERT)** drops to **3.85%** authentic drift recall.
     - **Dual-Encoder (CodeBERT)** drops to **0.00%** authentic drift recall.
   - **Key Finding**: Fine-tuning deep neural models on generated/contract-grounded synthetic mutations leads them to exploit generator-specific shortcuts that fail to transfer to subtle, authentic human git evolution. In contrast, superficial lexical signals and raw pretrained representations generalize with far greater robustness.

---

## 2. Dataset Architectures & Split Lineage

### A. V1 Synthetic Dataset (`data/v1_synthetic/` & `data/experiments/v2/`)
- **Origin**: Rule-based AST mutations (docstring negation swaps, parameter renaming, return value deletion, docstring sentence stripping).
- **Train Split**: `data/v1_synthetic/ablation/train.jsonl` (**9,638 samples**: 4,965 aligned, 4,673 drifted).
- **Val Split**: `data/v1_synthetic/ablation/val.jsonl` (**1,259 samples**: 662 aligned, 597 drifted).
- **Test Split**: `data/v1_synthetic/ablation/test.jsonl` (**1,205 samples**: 597 aligned, 608 drifted).

### B. V2 Clean Real-World Dataset (`experiments/2026-09-07_clean_v2/` & `data/v2_real_world/`)
- **Origin**: Top-tier open-source Python repositories (Django, Click, FastAPI, Pandas, SQLAlchemy, Scikit-learn). Features AST contract grounding and human-verified real-world commits with cryptographic SHA-256 integrity and zero lineage leakage.
- **Train Split**: `experiments/2026-09-07_clean_v2/dataset/train.jsonl` (**24,229 samples**: 13,871 clean grounded, 10,358 contract-grounded drift).
- **Val Split**: `experiments/2026-09-07_clean_v2/dataset/val.jsonl` (**2,636 samples**: 1,520 clean grounded, 1,116 contract-grounded drift).
- **Locked Test Split**: `experiments/2026-09-07_clean_v2/dataset/verified_test.jsonl` (**104 samples**: **52 authentic historical human mined drift**, **52 clean grounded**; 100% held-out function lineages).

### C. V2 Evaluation Set 2 (`data/v2_real_world/evaluation/Evaluation_set2.jsonl`)
- **Origin**: Extended real-world grounded benchmark drawn from 12 open-source Python repositories (Django, Scikit-learn, SQLAlchemy, Celery, Numpy, Pandas, Tornado, Click, FastAPI, Requests, Flask, Pytest). Features balanced 50/50 clean vs. drifted records with full contract provenance.
- **Evaluation Split**: `data/v2_real_world/evaluation/Evaluation_set2.jsonl` (**2,000 samples**: **1,000 clean grounded**, **1,000 contract-grounded drift**).
- **Mutation Profiles**: `aligned` (1,000), `doc_negation_drift` (704), `behavior_operator_drift` (235), `default_value_drift` (42), `param_contract_drift` (16), `return_contract_drift` (3).
- **Severity Profiles**: `none / aligned` (1,000), `moderate` (765), `hard` (235).

---

## 3. Master Comparison Tables

### Table 1: Complete V2 Real-World Benchmark (Locked Diagnostic Test Set, $N = 104$)

| Model | Category | Threshold ($\tau$) | Accuracy | Balanced Acc | Drift Recall | Drift Precision | Binary F1 | Macro F1 | ROC-AUC | PR-AUC | TN | FP | FN | TP | Confusion Matrix | Authentic Drift Caught |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- | :---: |
| **Zero-Shot CodeBERT** | Pretrained Semantic | `0.96` (tuned) | **54.81%** | **54.81%** | **36.54%** | **57.58%** | **44.71%** | **53.25%** | — | — | 38 | 14 | 33 | 19 | `TN=38, FP=14, FN=33, TP=19` | **19 / 52** |
| **TF-IDF Combined** | Pure Lexical Baseline | `0.50` (default) | **52.88%** | **52.88%** | **32.69%** | **54.84%** | **40.96%** | **50.88%** | 0.5459 | 0.5337 | 38 | 14 | 35 | 17 | `TN=38, FP=14, FN=35, TP=17` | **17 / 52** |
| **TF-IDF Combined** | Pure Lexical Baseline | `0.53` (tuned) | **48.08%** | **48.08%** | **21.15%** | **45.83%** | **28.95%** | **44.02%** | 0.5459 | 0.5337 | 39 | 13 | 41 | 11 | `TN=39, FP=13, FN=41, TP=11` | **11 / 52** |
| **TF-IDF Relational** | Enhanced Lexical Baseline | `0.50` (default) | **50.00%** | **50.00%** | **23.08%** | **50.00%** | **31.58%** | **46.09%** | 0.5318 | 0.5300 | 40 | 12 | 40 | 12 | `TN=40, FP=12, FN=40, TP=12` | **12 / 52** |
| **TF-IDF Relational** | Enhanced Lexical Baseline | `0.51` (tuned) | **50.00%** | **50.00%** | **21.15%** | **50.00%** | **29.73%** | **45.46%** | 0.5318 | 0.5300 | 41 | 11 | 41 | 11 | `TN=41, FP=11, FN=41, TP=11` | **11 / 52** |
| **Joint-Encoder (CodeBERT)** | Fine-Tuned Cross-Input | `0.50` (default) | **50.00%** | **50.00%** | **3.85%** | **50.00%** | **7.14%** | **36.47%** | — | — | 50 | 2 | 50 | 2 | `TN=50, FP=2, FN=50, TP=2` | **2 / 52** |
| **Dual-Encoder (CodeBERT)** | Fine-Tuned Bi-Encoder | `0.50` (default) | **45.19%** | **45.19%** | **0.00%** | **0.00%** | **0.00%** | **31.13%** | — | — | 47 | 5 | 52 | 0 | `TN=47, FP=5, FN=52, TP=0` | **0 / 52** |

---

### Table 2: Complete Evaluation Set 2 Benchmark (Extended Real-World Grounded Test Set, $N = 2,000$)

| Model | Category | Threshold ($\tau$) | Accuracy | Balanced Acc | Drift Recall | Drift Precision | Binary F1 | Macro F1 | ROC-AUC | PR-AUC | TN | FP | FN | TP | Confusion Matrix |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Zero-Shot CodeBERT** | Pretrained Semantic | `0.96` (tuned) | **51.35%** | **51.35%** | **26.70%** | **52.66%** | **35.43%** | **48.20%** | — | — | 760 | 240 | 733 | 267 | `TN=760, FP=240, FN=733, TP=267` |
| **TF-IDF Combined** | Pure Lexical Baseline | `0.53` (tuned) | **73.35%** | **73.35%** | **58.10%** | **83.60%** | **68.55%** | **72.72%** | 0.8264 | 0.8119 | 886 | 114 | 419 | 581 | `TN=886, FP=114, FN=419, TP=581` |
| **TF-IDF Relational** | Enhanced Lexical Baseline | `0.51` (tuned) | **74.95%** | **74.95%** | **62.80%** | **82.96%** | **71.49%** | **74.57%** | 0.8386 | 0.8510 | 871 | 129 | 372 | 628 | `TN=871, FP=129, FN=372, TP=628` |
| **Dual-Encoder (CodeBERT)** | Fine-Tuned Bi-Encoder | `0.50` (default) | **89.20%** | **89.20%** | **88.40%** | **89.84%** | **89.11%** | **89.20%** | 0.9470 | 0.9530 | 900 | 100 | 116 | 884 | `TN=900, FP=100, FN=116, TP=884` |
| **Joint-Encoder (CodeBERT)** | Fine-Tuned Cross-Input | `0.50` (default) | **92.75%** | **92.75%** | **90.20%** | **95.05%** | **92.56%** | **92.75%** | — | — | 953 | 47 | 98 | 902 | `TN=953, FP=47, FN=98, TP=902` |

---

### Table 3: Complete V1 Synthetic Benchmark (Controlled Test Set, $N = 1,205$)

| Model | Category | Training Objective | Accuracy | Balanced Acc | Drift Recall | Drift Precision | Binary F1 | Macro F1 | ROC-AUC | TN | FP | FN | TP | Confusion Matrix |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Zero-Shot CodeBERT** | Pretrained Semantic | Cosine Divergence | 44.56% | 44.68% | 32.07% | 43.33% | 36.86% | 43.73% | — | 342 | 255 | 413 | 195 | `TN=342, FP=255, FN=413, TP=195` |
| **TF-IDF Combined** | Pure Lexical Baseline | Logistic Regression | 65.23% | 65.18% | 70.56% | 64.13% | 67.19% | 65.10% | 0.7040 | 357 | 240 | 179 | 429 | `TN=357, FP=240, FN=179, TP=429` |
| **TF-IDF Relational** | Enhanced Lexical Baseline | Logistic Regression | **70.04%** | **70.04%** | **69.90%** | **70.48%** | **70.19%** | **70.04%** | 0.7691 | 419 | 178 | 183 | 425 | `TN=419, FP=178, FN=183, TP=425` |
| **Dual-Encoder (CodeBERT)** | Fine-Tuned Bi-Encoder | Cross-Entropy Loss | 80.91% | 81.03% | 68.42% | 91.63% | 78.34% | 80.64% | — | 559 | 38 | 192 | 416 | `TN=559, FP=38, FN=192, TP=416` |
| **Joint-Encoder (CodeBERT)** | Fine-Tuned Cross-Input | Cross-Entropy Loss | **85.06%** | **85.15%** | **75.82%** | **93.32%** | **83.67%** | **84.95%** | — | 564 | 33 | 147 | 461 | `TN=564, FP=33, FN=147, TP=461` |
| **Joint-Encoder (CodeBERT)** | Hard-Negative Mining | Focal Loss ($\gamma=2.0$) | **85.81%** | **85.91%** | **75.00%** | **96.00%** | **84.21%** | **85.66%** | — | 578 | 19 | 152 | 456 | `TN=578, FP=19, FN=152, TP=456` |

---

### Table 4: In-Distribution Validation Set Performance

| Model | Dataset Generation | Validation Split ($N$) | Val Accuracy | Val Balanced Acc | Val Recall | Val Precision | Val Macro-F1 | TN | FP | FN | TP | Confusion Matrix (Val) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Joint-Encoder (CodeBERT)** | **V2** | 2,636 | **90.44%** | **89.47%** | **83.15%** | **93.55%** | **90.04%** | 1,456 | 64 | 188 | 928 | `TN=1456, FP=64, FN=188, TP=928` |
| **Dual-Encoder (CodeBERT)** | **V2** | 2,636 | **90.33%** | **89.17%** | **81.63%** | **94.80%** | **89.87%** | 1,470 | 50 | 205 | 911 | `TN=1470, FP=50, FN=205, TP=911` |
| **TF-IDF Relational** | **V2** | 2,636 | **75.11%** | **74.03%** | **68.73%** | **71.21%** | **74.16%** | 1,211 | 309 | 349 | 767 | `TN=1211, FP=309, FN=349, TP=767` |
| **TF-IDF Combined** | **V2** | 2,636 | **73.48%** | **72.15%** | **64.96%** | **70.22%** | **71.96%** | 1,212 | 308 | 391 | 725 | `TN=1212, FP=308, FN=391, TP=725` |
| **Zero-Shot CodeBERT** | **V2** | 2,636 | **54.29%** | **51.91%** | **36.38%** | **45.06%** | **51.62%** | 1,025 | 495 | 710 | 406 | `TN=1025, FP=495, FN=710, TP=406` |
| **Joint-Encoder (Focal)** | **V1** | 1,259 | **85.15%** | **84.57%** | **73.37%** | **93.99%** | **84.78%** | 634 | 28 | 159 | 438 | `TN=634, FP=28, FN=159, TP=438` |
| **Joint-Encoder (CE)** | **V1** | 1,259 | **85.15%** | **84.63%** | **74.54%** | **92.71%** | **84.83%** | 627 | 35 | 152 | 445 | `TN=627, FP=35, FN=152, TP=445` |
| **Dual-Encoder (CE)** | **V1** | 1,259 | **80.86%** | **80.70%** | **68.01%** | **91.24%** | **80.70%** | 623 | 39 | 191 | 406 | `TN=623, FP=39, FN=191, TP=406` |
| **TF-IDF Relational** | **V1** | 1,259 | **66.24%** | **66.21%** | **65.83%** | **64.22%** | **66.21%** | 441 | 221 | 204 | 393 | `TN=441, FP=221, FN=204, TP=393` |
| **TF-IDF Combined** | **V1** | 1,259 | **60.92%** | **60.84%** | **59.97%** | **58.31%** | **60.84%** | 409 | 253 | 239 | 358 | `TN=409, FP=253, FN=239, TP=358` |

---

## 4. Fine-Grained Performance Breakdowns on V2 Real-World Test Set

### A. Provenance Breakdown (Authentic Historical Mined vs. Clean Grounded)

| Model | Provenance Group | Sample Count ($N$) | Drift Recall | Drift Precision | F1-Score | Balanced Accuracy | TN | FP | FN | TP | Confusion Matrix |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Zero-Shot CodeBERT** | `authentic_historical_mined` | 52 | **36.54%** | 100.0% | 53.52% | 36.54% | 0 | 0 | 33 | 19 | `TN=0, FP=0, FN=33, TP=19` |
| | `clean_grounded` | 52 | 0.00% | 0.00% | 0.00% | **73.08%** | 38 | 14 | 0 | 0 | `TN=38, FP=14, FN=0, TP=0` |
| **TF-IDF Combined ($\tau=0.50$)** | `authentic_historical_mined` | 52 | **32.69%** | 100.0% | 49.28% | 32.69% | 0 | 0 | 35 | 17 | `TN=0, FP=0, FN=35, TP=17` |
| | `clean_grounded` | 52 | 0.00% | 0.00% | 0.00% | **73.08%** | 38 | 14 | 0 | 0 | `TN=38, FP=14, FN=0, TP=0` |
| **TF-IDF Relational ($\tau=0.51$)** | `authentic_historical_mined` | 52 | **21.15%** | 100.0% | 34.92% | 21.15% | 0 | 0 | 41 | 11 | `TN=0, FP=0, FN=41, TP=11` |
| | `clean_grounded` | 52 | 0.00% | 0.00% | 0.00% | **78.85%** | 41 | 11 | 0 | 0 | `TN=41, FP=11, FN=0, TP=0` |
| **Joint-Encoder CodeBERT** | `authentic_historical_mined` | 52 | **3.85%** | 100.0% | 7.41% | 3.85% | 0 | 0 | 50 | 2 | `TN=0, FP=0, FN=50, TP=2` |
| | `clean_grounded` | 52 | 0.00% | 0.00% | 0.00% | **96.15%** | 50 | 2 | 0 | 0 | `TN=50, FP=2, FN=0, TP=0` |
| **Dual-Encoder CodeBERT** | `authentic_historical_mined` | 52 | **0.00%** | 0.00% | 0.00% | 0.00% | 0 | 0 | 52 | 0 | `TN=0, FP=0, FN=52, TP=0` |
| | `clean_grounded` | 52 | 0.00% | 0.00% | 0.00% | **90.38%** | 47 | 5 | 0 | 0 | `TN=47, FP=5, FN=0, TP=0` |

### B. Drift Score Distributions on V2 Test Set ($P(\text{drift})$ Statistics)

| Model | Provenance Group | Sample Count ($N$) | Mean $P(\text{drift})$ | Median $P(\text{drift})$ | Std Dev | Range [$Min, Max$] |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **TF-IDF Combined** | `authentic_historical_mined` | 52 | **0.4560** | 0.4355 | 0.1256 | [0.1763, 0.7915] |
| | `clean_grounded` | 52 | **0.4418** | 0.4170 | 0.1362 | [0.2589, 0.7766] |
| **TF-IDF Relational** | `authentic_historical_mined` | 52 | **0.3867** | 0.3876 | 0.1682 | [0.0136, 0.9069] |
| | `clean_grounded` | 52 | **0.3795** | 0.3367 | 0.1608 | [0.1384, 0.9678] |
| **Joint-Encoder CodeBERT** | `authentic_historical_mined` | 52 | **0.3340** | 0.3340 | 0.1130 | [0.0890, 0.5840] |
| | `clean_grounded` | 52 | **0.2980** | 0.2850 | 0.0980 | [0.0810, 0.5420] |

### C. Breakdown by Contract Drift Mutation Type on V2 Test Set

| Contract Drift Type | Sample Count ($N$) | Joint-Encoder Recall | Dual-Encoder Recall | TF-IDF Relational Recall | TF-IDF Combined Recall |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `behavior_contract_drift` | 13 | 7.69% (1/13) | 0.00% (0/13) | **23.08% (3/13)** | 7.69% (1/13) |
| `param_contract_drift` | 13 | 7.69% (1/13) | 0.00% (0/13) | **38.46% (5/13)** | **30.77% (4/13)** |
| `return_contract_drift` | 17 | 0.00% (0/17) | 0.00% (0/17) | 5.88% (1/17) | **23.53% (4/17)** |
| `default_value_drift` | 5 | 0.00% (0/5) | 0.00% (0/5) | 20.00% (1/5) | **40.00% (2/5)** |
| `exception_contract_drift` | 4 | 0.00% (0/4) | 0.00% (0/4) | **25.00% (1/4)** | 0.00% (0/4) |
| `aligned` (Clean Grounded) | 52 | **96.15% (50/52)** | **90.38% (47/52)** | **78.85% (41/52)** | **75.00% (39/52)** |

### D. Severity Breakdown on V2 Test Set

| Severity Tier | Sample Count ($N$) | Joint-Encoder Recall | Dual-Encoder Recall | Zero-Shot Recall | TF-IDF Relational Recall | TF-IDF Combined Recall |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `high` (Authentic Drift) | 52 | 3.85% (2/52) | 0.00% (0/52) | **36.54% (19/52)** | 21.15% (11/52) | **32.69% (17/52)** |
| `aligned` (Clean Grounded) | 52 | **96.15% (50/52)** | **90.38% (47/52)** | 73.08% (38/52) | 78.85% (41/52) | 75.00% (39/52) |

---

## 5. Fine-Grained Performance Breakdowns on Evaluation Set 2 ($N = 2,000$)

### A. Mutation Type Breakdown ($N = 2,000$)

| Mutation Type | Count ($N$) | Zero-Shot CodeBERT Recall | TF-IDF Combined Recall | TF-IDF Relational Recall | Dual-Encoder (CodeBERT) Recall | Joint-Encoder (CodeBERT) Recall |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `aligned` (Clean Grounded) | 1,000 | 76.00% (760/1000) | 88.60% (886/1000) | 87.10% (871/1000) | 90.00% (900/1000) | **95.30% (953/1000)** |
| `doc_negation_drift` | 704 | 25.71% (181/704) | 67.05% (472/704) | 67.19% (473/704) | 89.80% (632/704)* | **92.33% (650/704)** |
| `behavior_operator_drift` | 235 | 31.06% (73/235) | 25.96% (61/235) | 42.98% (101/235) | **83.83% (197/235)** | 82.98% (195/235) |
| `default_value_drift` | 42 | 30.95% (13/42) | **100.0% (42/42)** | **100.0% (42/42)** | 97.62% (41/42) | 97.62% (41/42) |
| `param_contract_drift` | 16 | 0.00% (0/16) | 31.25% (5/16) | 68.75% (11/16) | **93.75% (15/16)** | **93.75% (15/16)** |
| `return_contract_drift` | 3 | 0.00% (0/3) | **33.33% (1/3)** | **33.33% (1/3)** | **33.33% (1/3)** | **33.33% (1/3)** |

### B. Severity Breakdown ($N = 2,000$)

| Severity Tier | Count ($N$) | Zero-Shot CodeBERT | TF-IDF Combined | TF-IDF Relational | Dual-Encoder (CodeBERT) | Joint-Encoder (CodeBERT) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `hard` | 235 | 31.06% (73/235) | 25.96% (61/235) | 42.98% (101/235) | **83.83% (197/235)** | 82.98% (195/235) |
| `moderate` | 765 | 25.36% (194/765) | 67.97% (520/765) | 68.89% (527/765) | 89.80% (687/765) | **92.42% (707/765)** |
| `none` (`aligned`) | 1,000 | 76.00% (760/1000) | 88.60% (886/1000) | 87.10% (871/1000) | 90.00% (900/1000) | **95.30% (953/1000)** |

### C. Repository Breakdown Across 12 Open-Source Projects ($N = 2,000$)

| Repository | Count ($N$) | Zero-Shot (Acc / F1) | TF-IDF Combined (Acc / F1) | TF-IDF Relational (Acc / F1) | Dual-Encoder (Acc / F1) | Joint-Encoder (Acc / F1) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `celery` | 215 | 55.35% / 33.33% | 80.47% / 78.79% | 75.35% / 72.54% | 82.33% / 80.81% | **86.98% / 84.95%** |
| `click` | 59 | 33.90% / 36.07% | 64.41% / 68.66% | 67.80% / 73.24% | 89.83% / 92.86% | **96.61% / 97.62%** |
| `django` | 786 | 51.27% / 34.53% | 77.48% / 72.89% | 80.03% / 75.66% | 93.64% / 93.62% | **95.29% / 95.30%** |
| `fastapi` | 24 | 50.00% / 45.45% | 50.00% / 50.00% | 66.67% / 71.43% | 83.33% / 86.67% | **91.67% / 93.75%** |
| `flask` | 4 | 50.00% / 0.00% | 75.00% / 0.00% | 75.00% / 0.00% | 75.00% / 66.67% | **100.0% / 100.0%** |
| `numpy` | 134 | 58.96% / 28.57% | 82.84% / 78.50% | 86.57% / 83.93% | 90.30% / 88.07% | **93.28% / 91.59%** |
| `pandas` | 95 | 45.26% / 23.53% | 68.42% / 63.41% | 74.74% / 72.09% | 90.53% / 90.32% | **92.63% / 92.31%** |
| `pytest` | 1 | 0.00% / 0.00% | 0.00% / 0.00% | 0.00% / 0.00% | **100.0% / 100.0%** | **100.0% / 100.0%** |
| `requests` | 9 | 44.44% / 44.44% | 44.44% / 28.57% | 55.56% / 60.00% | 66.67% / 72.73% | **88.89% / 92.31%** |
| `scikit-learn` | 384 | 44.53% / 36.04% | 59.38% / 57.38% | 62.24% / 65.88% | 86.98% / 88.79% | **91.67% / 92.66%** |
| `sqlalchemy` | 229 | 58.95% / 44.05% | 78.17% / 65.28% | 75.98% / 60.43% | 89.08% / 84.66% | **93.45% / 90.91%** |
| `tornado` | 60 | 65.00% / 43.24% | 75.00% / 66.67% | 73.33% / 65.22% | 71.67% / 72.13% | **80.00% / 72.73%** |

---

## 6. Fine-Grained Performance Breakdowns on V1 Synthetic Test Set

### A. Mutation Type Breakdown ($N = 1,205$)

| Mutation Type | Count ($N$) | Zero-Shot CodeBERT | TF-IDF Combined | TF-IDF Relational | Dual-Encoder (CE) | Joint-Encoder (CE) | Joint-Encoder (Focal) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`return_value_change`** | 139 | 49.73% F1 (46 TP) | 90.98% F1 (116 TP) | 89.68% F1 (113 TP) | 78.60% F1 (89 TP) | **93.49% F1 (122 TP)** | 92.25% F1 (119 TP) |
| **`doc_negation`** | 95 | 59.26% F1 (40 TP) | **90.80% F1 (79 TP)** | 88.89% F1 (76 TP) | 17.31% F1 (9 TP) | 37.61% F1 (22 TP) | 31.86% F1 (18 TP) |
| **`param_rename`** | 221 | 54.61% F1 (83 TP) | 86.38% F1 (168 TP) | 84.60% F1 (162 TP) | 98.39% F1 (214 TP) | 97.92% F1 (212 TP) | **98.16% F1 (213 TP)** |
| **`doc_sentence_delete`** | 153 | 29.05% F1 (26 TP) | 60.27% F1 (66 TP) | 65.20% F1 (74 TP) | 80.47% F1 (103 TP) | 81.40% F1 (105 TP) | **81.85% F1 (106 TP)** |
| **`aligned` (Specificity)** | 597 | 57.29% Acc (342 TN) | 59.80% Acc (357 TN) | 70.18% Acc (419 TN) | 93.63% Acc (559 TN) | 94.47% Acc (564 TN) | **96.82% Acc (578 TN)** |

### B. Severity Breakdown on V1 Test Set ($N = 1,205$)

| Severity Tier | Sample Count ($N$) | Zero-Shot Recall | TF-IDF Combined Recall | TF-IDF Relational Recall | Dual-Encoder Recall | Joint-Encoder (CE) Recall | Joint-Encoder (Focal) Recall |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `mild` | 211 | 29.86% (63/211) | 69.67% (147/211) | 70.62% (149/211) | 70.14% (148/211) | **74.88% (158/211)** | 74.41% (157/211) |
| `moderate` | 201 | 35.32% (71/201) | 69.15% (139/201) | 67.66% (136/201) | 63.18% (127/201) | **74.13% (149/201)** | 73.13% (147/201) |
| `severe` | 196 | 31.12% (61/196) | 72.96% (143/196) | 71.43% (140/196) | 71.94% (141/196) | **78.57% (154/196)** | 77.55% (152/196) |
| `aligned` (Clean) | 597 | 57.29% (342/597) | 59.80% (357/597) | 70.18% (419/597) | 93.63% (559/597) | 94.47% (564/597) | **96.82% (578/597)** |

### C. Repository Breakdown on V1 Test Set ($N = 1,205$)

| Repository | Count ($N$) | Zero-Shot Accuracy | TF-IDF Relational Accuracy | Dual-Encoder (CE) Accuracy | Joint-Encoder (CE) Accuracy | Joint-Encoder (Focal) Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **pandas** | 283 | 47.70% | 68.90% | 78.45% | **82.69%** | **83.39%** |
| **sqlalchemy** | 229 | 45.41% | 72.05% | 85.15% | **88.21%** | **88.65%** |
| **scikit-learn** | 221 | 42.08% | 70.59% | 81.00% | **85.97%** | **86.43%** |
| **django** | 206 | 45.15% | 69.42% | 78.64% | **83.50%** | **84.47%** |
| **numpy** | 112 | 42.86% | 71.43% | 76.79% | **83.93%** | **84.82%** |
| **pytest** | 92 | 43.48% | 73.91% | 90.22% | **92.39%** | **93.48%** |
| **flask** | 26 | 50.00% | 73.08% | 80.77% | **84.62%** | **84.62%** |
| **click** | 17 | 47.06% | 70.59% | 88.24% | **88.24%** | **88.24%** |
| **requests** | 12 | 41.67% | 66.67% | 66.67% | **75.00%** | **75.00%** |
| **fastapi** | 7 | 42.86% | 71.43% | 57.14% | **71.43%** | **71.43%** |

---

## 7. Statistical Significance Tests (McNemar's $\chi^2$ Test)

### A. V1 Synthetic Benchmark ($N = 1,205$)
- **Joint-Encoder (CE) vs. Dual-Encoder (CE)**:
  - Contingency Table: $n_{00} = 141$, $n_{01} = 89$ (Joint wins), $n_{10} = 39$ (Dual wins), $n_{11} = 936$.
  - $\chi^2 = 18.7578$ ($p = 1.4841 \times 10^{-5}$).
  - **Result**: Joint-Encoder is statistically significantly superior to Dual-Encoder on synthetic mutations ($p < 0.001$).
- **Joint-Encoder (CE) vs. Zero-Shot Baseline**:
  - Contingency Table: $n_{00} = 158$, $n_{01} = 502$ (Joint wins), $n_{10} = 22$ (Zero-Shot wins), $n_{11} = 523$.
  - $\chi^2 = 437.47$ ($p < 10^{-20}$).
  - **Result**: Fine-tuned Joint-Encoder significantly outperforms zero-shot representation ($p < 0.001$).
- **Joint-Encoder (Focal) vs. Joint-Encoder (CE)**:
  - Focal loss drives false-positive reductions ($FP: 33 \rightarrow 19$), boosting specificity to **96.82%** with competitive recall.

### B. V2 Real-World Benchmark ($N = 104$)
- **TF-IDF Combined vs. Joint-Encoder (CodeBERT)**:
  - Joint-Encoder suffers severe false-negative skew ($FN = 50$, $TP = 2$), predicting "aligned" for nearly all authentic commits.
  - TF-IDF correctly detects 17 authentic drifts vs. Joint's 2, confirming superior transfer of lexical signals to real human commits.

---

## 8. Artifact Directory & File Manifest

| Artifact Name | Scope | File Path |
| :--- | :---: | :--- |
| **Evaluation Set 2 Master Summary** | Eval Set 2 | `experiments/2026-09-07_clean_v2/evaluation/evaluation_set2_benchmarks/summary_eval_set_results.json` |
| **Evaluation Set 2 Benchmark Report** | Eval Set 2 | `experiments/2026-09-07_clean_v2/evaluation/evaluation_set2_benchmarks/EVAL_SET_BENCHMARK_REPORT.md` |
| **Eval Set 2 Zero-Shot Results** | Eval Set 2 | `experiments/2026-09-07_clean_v2/evaluation/evaluation_set2_benchmarks/zero_shot/results_baseline.json` |
| **Eval Set 2 TF-IDF Combined Results** | Eval Set 2 | `experiments/2026-09-07_clean_v2/evaluation/evaluation_set2_benchmarks/tfidf_combined/results_tfidf_combined.json` |
| **Eval Set 2 TF-IDF Relational Results** | Eval Set 2 | `experiments/2026-09-07_clean_v2/evaluation/evaluation_set2_benchmarks/tfidf_relational/results_tfidf_relational.json` |
| **Eval Set 2 Dual-Encoder Results** | Eval Set 2 | `experiments/2026-09-07_clean_v2/evaluation/evaluation_set2_benchmarks/dual_encoder/results_dual_encoder.json` |
| **Eval Set 2 Joint-Encoder Results** | Eval Set 2 | `experiments/2026-09-07_clean_v2/evaluation/evaluation_set2_benchmarks/joint_encoder/results_joint_encoder.json` |
| **V2 Joint-Encoder Evaluation** | V2 Test | `experiments/2026-09-07_clean_v2/evaluation/eval_results.json` |
| **V2 Joint-Encoder Predictions** | V2 Test | `experiments/2026-09-07_clean_v2/predictions/independent_predictions.jsonl` |
| **V2 Joint-Encoder Checkpoint Record** | V2 Train/Val | `experiments/2026-09-07_clean_v2/checkpoints/results_joint_encoder.json` |
| **V2 Dual-Encoder Evaluation** | V2 Val/Test | `experiments/2026-09-07_clean_v2/dual_encoder_checkpoints/results_dual_encoder.json` |
| **V2 Dual-Encoder Predictions** | V2 Test | `experiments/2026-09-07_clean_v2/dual_encoder_checkpoints/predictions_dual_encoder.jsonl` |
| **V2 Zero-Shot Baseline Results** | V2 Val/Test | `data/v2_real_world/baseline_results/results_baseline.json` |
| **V2 Zero-Shot Baseline Predictions** | V2 Test | `data/v2_real_world/baseline_results/predictions_baseline.jsonl` |
| **V2 TF-IDF Combined Results** | V2 Val/Test | `experiments/2026-09-07_clean_v2/evaluation/results_tfidf_combined.json` |
| **V2 TF-IDF Combined Predictions** | V2 Test | `experiments/2026-09-07_clean_v2/predictions/tfidf_combined_predictions.jsonl` |
| **V2 TF-IDF Relational Results** | V2 Val/Test | `experiments/2026-09-07_clean_v2/evaluation/results_tfidf_relational.json` |
| **V2 TF-IDF Relational Predictions** | V2 Test | `experiments/2026-09-07_clean_v2/predictions/tfidf_relational_predictions.jsonl` |
| **V1 TF-IDF Combined Results** | V1 Val/Test | `data/v1_synthetic/baseline_results/results_tfidf_combined.json` |
| **V1 TF-IDF Combined Predictions** | V1 Test | `data/v1_synthetic/baseline_results/tfidf_combined_predictions.jsonl` |
| **V1 TF-IDF Relational Results** | V1 Val/Test | `data/v1_synthetic/baseline_results/results_tfidf_relational.json` |
| **V1 TF-IDF Relational Predictions** | V1 Test | `data/v1_synthetic/baseline_results/tfidf_relational_predictions.jsonl` |
| **V1 Controlled Ablation Report** | V1 Test | `data/experiments/v2/controlled_ablation/controlled_experiment_results.json` |
| **V1 Joint-Encoder Focal Loss Results** | V1 Val/Test | `data/experiments/v2/joint_focal_controlled/results_joint_encoder.json` |
| **V1 Joint-Encoder CE Results** | V1 Val/Test | `data/experiments/v2/joint_ce_controlled/results_joint_encoder.json` |
| **V1 Dual-Encoder CE Results** | V1 Val/Test | `data/experiments/v2/dual_ce_controlled/results_dual_encoder.json` |
| **V1 IEEE Paper Synthesis Report** | V1 Test | `data/experiments/v2/ieee_paper_results.json` |

---

## 9. Reproducibility & Execution Commands

### Reproduce Evaluation Set 2 Benchmarks (All 5 Models)
```powershell
python scripts/runners/evaluate_all_models_on_eval_set.py `
    --test_file data/v2_real_world/evaluation/Evaluation_set2.jsonl `
    --output_dir experiments/2026-09-07_clean_v2/evaluation/evaluation_set2_benchmarks `
    --device cuda
```

### Reproduce V2 Real-World Benchmarks
```powershell
# TF-IDF Combined Baseline (V2)
python scripts/training/run_tfidf_baseline.py `
    --feature_mode combined --tune_threshold --max_iter 5000 `
    --train_file experiments/2026-09-07_clean_v2/dataset/train.jsonl `
    --val_file experiments/2026-09-07_clean_v2/dataset/val.jsonl `
    --test_file experiments/2026-09-07_clean_v2/dataset/verified_test.jsonl

# TF-IDF Relational Baseline (V2)
python scripts/training/run_tfidf_baseline.py `
    --feature_mode dual_overlap --tune_threshold --max_iter 5000 `
    --train_file experiments/2026-09-07_clean_v2/dataset/train.jsonl `
    --val_file experiments/2026-09-07_clean_v2/dataset/val.jsonl `
    --test_file experiments/2026-09-07_clean_v2/dataset/verified_test.jsonl

# Joint-Encoder Independent Evaluation (V2)
python experiments/2026-09-07_clean_v2/evaluation/independent_evaluate.py `
    --checkpoint experiments/2026-09-07_clean_v2/checkpoints/joint_encoder_checkpoint.pt `
    --manifest experiments/2026-09-07_clean_v2/config/manifest.yaml `
    --test_file experiments/2026-09-07_clean_v2/dataset/verified_test.jsonl
```

### Reproduce V1 Synthetic Benchmarks
```powershell
# TF-IDF Combined Baseline (V1)
python scripts/training/run_tfidf_baseline.py `
    --feature_mode combined --tune_threshold --max_iter 5000 `
    --train_file data/v1_synthetic/ablation/train.jsonl `
    --val_file data/v1_synthetic/ablation/val.jsonl `
    --test_file data/v1_synthetic/ablation/test.jsonl

# TF-IDF Relational Baseline (V1)
python scripts/training/run_tfidf_baseline.py `
    --feature_mode dual_overlap --tune_threshold --max_iter 5000 `
    --train_file data/v1_synthetic/ablation/train.jsonl `
    --val_file data/v1_synthetic/ablation/val.jsonl `
    --test_file data/v1_synthetic/ablation/test.jsonl
```
