# SemDrift — Semantic Drift Detection in Python Repositories

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.1+](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org/)
[![Transformers 4.30+](https://img.shields.io/badge/HuggingFace-Transformers-yellow.svg)](https://huggingface.co/transformers/)
[![Tree-Sitter](https://img.shields.io/badge/Tree--Sitter-Multi--Language-green.svg)](https://tree-sitter.github.io/)
[![Unit Tests](https://img.shields.io/badge/tests-164%20passed-brightgreen.svg)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**SemDrift** is an empirical research framework for investigating and detecting **semantic drift** between source code and documentation comments (docstrings) in Python repositories. It benchmarks classical lexical baselines (TF-IDF + Logistic Regression), pretrained zero-shot representations, and fine-tuned deep transformer architectures (CodeBERT bi-encoders and cross-encoders) across two strictly separated dataset generations:
1. **V1 Controlled Synthetic Dataset**: AST rule-based mutations across 10 repositories.
2. **V2 Clean Real-World Dataset**: Contract-grounded training with a cryptographically locked diagnostic test set of authentic human commits mined from open-source repositories (Click, FastAPI, Django, Pandas, SQLAlchemy, Pytest).

> **Complete Empirical Records**: For full per-model confusion matrices ($TN, FP, FN, TP$), provenance breakdowns, drift score distributions ($P(\text{drift})$), severity analyses, and McNemar test contingency tables, see the repository-level master report: [BENCHMARK_RESULTS.md](BENCHMARK_RESULTS.md).

---

## Core Empirical Findings: The Synthetic-to-Real Generalization Gap

The empirical results reveal a critical distinction between performance under controlled synthetic conditions versus real-world software evolution:

```text
Synthetic Domain (V1 Controlled):
  Lexical TF-IDF (65.23% / 70.04%) < Dual-Encoder (80.91%) < Joint-Encoder (85.06% / 85.81%)
  → Cross-modal attention and negative mining provide significant gains on synthetic rule mutations.

Authentic Real-World Domain (V2 Locked Diagnostic):
  Fine-Tuned Neural (0.00% / 3.85% Recall) < Enhanced Lexical (21.15%) < Pure Lexical (32.69%) < Zero-Shot (36.54%)
  → Fine-tuning on generated/contract-grounded drift causes neural models to exploit generator artifacts,
    leading to severe false-negative collapse on authentic human commits. Pretrained and lexical signals
    transfer with higher robustness.
```

---

## Benchmark Results

### 1. V2 Real-World Grounded Benchmark (Locked Diagnostic Test Set, $N = 104$)

Evaluated on the locked diagnostic split (`experiments/2026-09-07_clean_v2/dataset/verified_test.jsonl`), consisting of **52 authentic historical human drift commits** mined from production repositories and **52 clean grounded functions** (100% held-out function lineages; zero training/validation overlap):

| Model Architecture | Category | Threshold ($\tau$) | Accuracy | Balanced Acc | Drift Recall | Drift Precision | Binary F1 | Macro F1 | ROC-AUC | PR-AUC | Confusion Matrix | Authentic Drift Caught |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- | :---: |
| **Zero-Shot CodeBERT** | Pretrained Semantic | `0.96` (tuned) | **54.81%** | **54.81%** | **36.54%** | **57.58%** | **44.71%** | **53.25%** | — | — | `TN=38, FP=14, FN=33, TP=19` | **19 / 52** |
| **TF-IDF Combined** | Pure Lexical Baseline | `0.50` (default) | **52.88%** | **52.88%** | **32.69%** | **54.84%** | **40.96%** | **50.88%** | 0.5459 | 0.5337 | `TN=38, FP=14, FN=35, TP=17` | **17 / 52** |
| **TF-IDF Combined** | Pure Lexical Baseline | `0.53` (tuned) | **48.08%** | **48.08%** | **21.15%** | **45.83%** | **28.95%** | **44.02%** | 0.5459 | 0.5337 | `TN=39, FP=13, FN=41, TP=11` | **11 / 52** |
| **TF-IDF Relational** | Enhanced Lexical Baseline | `0.50` (default) | **50.00%** | **50.00%** | **23.08%** | **50.00%** | **31.58%** | **46.09%** | 0.5318 | 0.5300 | `TN=40, FP=12, FN=40, TP=12` | **12 / 52** |
| **TF-IDF Relational** | Enhanced Lexical Baseline | `0.51` (tuned) | **50.00%** | **50.00%** | **21.15%** | **50.00%** | **29.73%** | **45.46%** | 0.5318 | 0.5300 | `TN=41, FP=11, FN=41, TP=11` | **11 / 52** |
| **Joint-Encoder (CodeBERT)** | Fine-Tuned Cross-Input | `0.50` (default) | **50.00%** | **50.00%** | **3.85%** | **50.00%** | **7.14%** | **36.47%** | — | — | `TN=50, FP=2, FN=50, TP=2` | **2 / 52** |
| **Dual-Encoder (CodeBERT)** | Fine-Tuned Bi-Encoder | `0.50` (default) | **45.19%** | **45.19%** | **0.00%** | **0.00%** | **0.00%** | **31.13%** | — | — | `TN=47, FP=5, FN=52, TP=0` | **0 / 52** |

#### Observations on V2:
- **Lexical and Pretrained Advantage**: Pretrained CodeBERT (Zero-Shot) and pure n-gram TF-IDF detect 19 and 17 of 52 authentic drifts, respectively.
- **Deep Fine-Tuned False-Negative Collapse**: The Joint-Encoder predicts "aligned" on 50 of 52 authentic drift instances ($FN=50$), achieving high clean-grounded specificity ($TN=50/52$, 96.15%) but near-zero real drift recall (3.85%). The Dual-Encoder predicts "aligned" on all 52 authentic drift instances ($FN=52$, 0.00% recall).
- **Practical Takeaway**: Models trained on synthetic or rule-based drift patterns should not be assumed to transfer reliably to authentic human documentation drift without real-world grounded fine-tuning data.

---

### 2. V1 Controlled Synthetic Benchmark ($N = 1,205$)

Evaluated on the 10-repository synthetic test set (`data/v1_synthetic/ablation/test.jsonl`, 597 aligned + 608 drifted instances):

| Model Architecture | Category | Objective | Accuracy | Balanced Acc | Drift Recall | Drift Precision | Binary F1 | Macro F1 | ROC-AUC | Confusion Matrix |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Zero-Shot CodeBERT** | Pretrained Semantic | Cosine Divergence | 44.56% | 44.68% | 32.07% | 43.33% | 36.86% | 43.73% | — | `TN=342, FP=255, FN=413, TP=195` |
| **TF-IDF Combined** | Pure Lexical Baseline | Logistic Regression | 65.23% | 65.18% | 70.56% | 64.13% | 67.19% | 65.10% | 0.7040 | `TN=357, FP=240, FN=179, TP=429` |
| **TF-IDF Relational** | Enhanced Lexical Baseline | Logistic Regression | **70.04%** | **70.04%** | **69.90%** | **70.48%** | **70.19%** | **70.04%** | 0.7691 | `TN=419, FP=178, FN=183, TP=425` |
| **Dual-Encoder (CodeBERT)** | Fine-Tuned Bi-Encoder | Cross-Entropy Loss | 80.91% | 81.03% | 68.42% | 91.63% | 78.34% | 80.64% | — | `TN=559, FP=38, FN=192, TP=416` |
| **Joint-Encoder (CodeBERT)** | Fine-Tuned Cross-Input | Cross-Entropy Loss | **85.06%** | **85.15%** | **75.82%** | **93.32%** | **83.67%** | **84.95%** | — | `TN=564, FP=33, FN=147, TP=461` |
| **Joint-Encoder (CodeBERT)** | Hard-Negative Mining | Focal Loss ($\gamma=2.0$) | **85.81%** | **85.91%** | **75.00%** | **96.00%** | **84.21%** | **85.66%** | — | `TN=578, FP=19, FN=152, TP=456` |

#### Controlled Architectural Ablation (Dual vs. Joint Encoder under identical CE loss):
Holding backbone (`microsoft/codebert-base`), training objective (`CrossEntropyLoss`), batch size (8), learning rate ($2 \times 10^{-5}$), and seeds constant:
- **Accuracy**: $80.91\% \rightarrow 85.06\%$ ($\Delta = +4.15\%$)
- **Binary F1**: $78.34\% \rightarrow 83.67\%$ ($\Delta = +5.33\%$)
- **Drift Recall**: $68.42\% \rightarrow 75.82\%$ ($\Delta = +7.40\%$)
- **McNemar's Paired Test**: $\chi^2 = 18.7578$, $p = 1.4841 \times 10^{-5}$ ($p < 0.001$, statistically significant).
- Joint cross-modal self-attention enables token-level interactions that isolated vector embeddings cannot capture.

---

### 3. In-Distribution Validation Set Performance

Performance on held-out validation splits during training:

| Model Architecture | Dataset Split | Sample Count ($N$) | Val Accuracy | Val Balanced Acc | Val Recall | Val Precision | Val Macro-F1 | Confusion Matrix (Val) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Joint-Encoder (CodeBERT)** | **V2** | 2,636 | **90.44%** | **89.47%** | **83.15%** | **93.55%** | **90.04%** | `TN=1456, FP=64, FN=188, TP=928` |
| **Dual-Encoder (CodeBERT)** | **V2** | 2,636 | **90.33%** | **89.17%** | **81.63%** | **94.80%** | **89.87%** | `TN=1470, FP=50, FN=205, TP=911` |
| **TF-IDF Relational** | **V2** | 2,636 | **75.11%** | **74.03%** | **68.73%** | **71.21%** | **74.16%** | `TN=1211, FP=309, FN=349, TP=767` |
| **TF-IDF Combined** | **V2** | 2,636 | **73.48%** | **72.15%** | **64.96%** | **70.22%** | **71.96%** | `TN=1212, FP=308, FN=391, TP=725` |
| **Zero-Shot CodeBERT** | **V2** | 2,636 | **54.29%** | **51.91%** | **36.38%** | **45.06%** | **51.62%** | `TN=1025, FP=495, FN=710, TP=406` |
| **Joint-Encoder (Focal)** | **V1** | 1,259 | **85.15%** | **84.57%** | **73.37%** | **93.99%** | **84.78%** | `TN=634, FP=28, FN=159, TP=438` |
| **Joint-Encoder (CE)** | **V1** | 1,259 | **85.15%** | **84.63%** | **74.54%** | **92.71%** | **84.83%** | `TN=627, FP=35, FN=152, TP=445` |
| **Dual-Encoder (CE)** | **V1** | 1,259 | **80.86%** | **80.70%** | **68.01%** | **91.24%** | **80.70%** | `TN=623, FP=39, FN=191, TP=406` |
| **TF-IDF Relational** | **V1** | 1,259 | **66.24%** | **66.21%** | **65.83%** | **64.22%** | **66.21%** | `TN=441, FP=221, FN=204, TP=393` |
| **TF-IDF Combined** | **V1** | 1,259 | **60.92%** | **60.84%** | **59.97%** | **58.31%** | **60.84%** | `TN=409, FP=253, FN=239, TP=358` |

---

### 4. Fine-Grained Mutation Breakdown (V1 Synthetic Test Set, $N = 1,205$)

| Mutation Type | Count ($N$) | Zero-Shot CodeBERT | TF-IDF Combined | TF-IDF Relational | Dual-Encoder (CE) | Joint-Encoder (CE) | Joint-Encoder (Focal) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`return_value_change`** | 139 | 49.73% F1 (46 TP) | 90.98% F1 (116 TP) | 89.68% F1 (113 TP) | 78.60% F1 (89 TP) | **93.49% F1 (122 TP)** | 92.25% F1 (119 TP) |
| **`doc_negation`** | 95 | 59.26% F1 (40 TP) | **90.80% F1 (79 TP)** | 88.89% F1 (76 TP) | 17.31% F1 (9 TP) | 37.61% F1 (22 TP) | 31.86% F1 (18 TP) |
| **`param_rename`** | 221 | 54.61% F1 (83 TP) | 86.38% F1 (168 TP) | 84.60% F1 (162 TP) | 98.39% F1 (214 TP) | 97.92% F1 (212 TP) | **98.16% F1 (213 TP)** |
| **`doc_sentence_delete`** | 153 | 29.05% F1 (26 TP) | 60.27% F1 (66 TP) | 65.20% F1 (74 TP) | 80.47% F1 (103 TP) | 81.40% F1 (105 TP) | **81.85% F1 (106 TP)** |
| **`aligned` (Specificity)** | 597 | 57.29% Acc (342 TN) | 59.80% Acc (357 TN) | 70.18% Acc (419 TN) | 93.63% Acc (559 TN) | 94.47% Acc (564 TN) | **96.82% Acc (578 TN)** |

#### Key Analytical Takeaways:
1. **Docstring Negation Sensitivity**: Lexical TF-IDF baselines preserving negation tokens (`stop_words=None`) achieve ~90% F1 on negation swaps by detecting token-level polarity shifts. In contrast, neural encoders struggle with isolated negation swaps (Dual-Encoder: 17.31%, Joint-Encoder: 37.61%), as transformer embeddings smooth out single-token polarity inversions without explicit polarity objectives.
2. **Signature & Structural Drift**: Deep models excel at structural mutations such as parameter renaming (98.39% / 97.92%) and return value drift (93.49%), where cross-attention bridges AST signature definitions with textual descriptions.

---

## Dataset Architecture & Integrity Design

SemDrift structures data curation into two strictly separated generations:

```text
                    SEMDRIFT DATASET ARCHITECTURE
                               │
             ┌─────────────────┴─────────────────┐
             ▼                                   ▼
      V1 — SYNTHETIC                      V2 — REAL-WORLD
      (12,102 Total)                      (26,969 Total)
             │                                   │
      ┌──────┴──────┐                     ┌──────┴──────┐
      │             │                     │             │
    TRAIN          TEST                 TRAIN          VAL
    9,638         1,205                 24,229        2,636
    VAL: 1,259                            │             │
                                          └──────┬──────┘
                                                 ▼
                                           LOCKED TEST SET
                                           (104 SAMPLES)
                                           52 Authentic Mined
                                           52 Clean Grounded
```

### Invariants & Cryptographic Safeguards:
- **Zero Lineage Leakage**: All 104 human-verified test lineages (`repo::file_path::qualified_name`) were purged from candidate pools prior to train/val partitioning (eliminating 71 candidate samples).
- **Two-Layer Verification Gate**: In `experiments/2026-09-07_clean_v2/dataset/`, all canonical splits are cryptographically locked via `SHA256SUMS` and `manifest.yaml`. Training and evaluation scripts refuse execution if any byte has been modified.
- **Strict Decoupling**: Training pipelines accept only `--train` and `--val`. Evaluation scripts operate independently on held-out test splits without access to training loops.

---

## Model Architectures

```text
                        ┌────────────────────────────────────────────────────────┐
                        │                      INPUT PAIR                        │
                        │        Docstring (Text)  +  Function Code (Python)     │
                        └────────────────────────────────────────────────────────┘
                                                     │
         ┌───────────────────────────────────────────┼───────────────────────────────────────────┐
         ▼                                           ▼                                           ▼
┌───────────────────────────┐               ┌───────────────────────────┐               ┌───────────────────────────┐
│     LEXICAL BASELINE      │               │     DUAL-ENCODER MODEL    │               │    JOINT-ENCODER MODEL    │
│  (TF-IDF + Logistic Reg)  │               │   (Bi-Encoder Architecture)│              │  (Cross-Encoder Self-Attn)│
└───────────────────────────┘               └───────────────────────────┘               └───────────────────────────┘
│ Feature extraction:       │               │ Two independent passes:   │               │ Single joint sequence:    │
│ • combined (pure n-grams) │               │ • Pretrained CodeBERT     │               │ • [CLS] doc [SEP] code    │
│ • dual_overlap (+ overlap)│               │ • Mean pooling: u, v      │               │ • Head-tail truncation    │
│ Class weight: balanced    │               │ Feature: [u; v; |u-v|]    │               │ Head: [CLS] → Linear(768,2)│
│ stop_words: None          │               │ Head: Linear(2304, 2)     │               │ Cross-modal self-attention│
└───────────────────────────┘               └───────────────────────────┘               └───────────────────────────┘
```

1. **TF-IDF + Logistic Regression Baseline**:
   - `scripts/training/run_tfidf_baseline.py`
   - Modes: `combined` (pure lexical n-grams) and `dual_overlap` (lexical n-grams + Cosine, Jaccard, and AST signature overlap ratios).
   - Invariant: `stop_words=None` to preserve semantic negation operators (`not`, `never`, `without`).
2. **Zero-Shot Dual Encoder Baseline**:
   - `scripts/training/run_zero_shot_baseline.py`
   - Independent passes through frozen `microsoft/codebert-base`; mean-pooled cosine distance $\delta = 1 - \cos(\mathbf{u}, \mathbf{v})$.
3. **Fine-Tuned Dual Encoder**:
   - `scripts/training/train_dual_encoder.py`
   - Shared CodeBERT encoder; concatenates $[\mathbf{u} \,;\, \mathbf{v} \,;\, |\mathbf{u} - \mathbf{v}|]$ into a linear classification head.
4. **Fine-Tuned Joint Encoder**:
   - `scripts/training/train_joint_encoder.py`
   - Full bidirectional self-attention across documentation and code tokens in a single sequence with head-tail code truncation.

---

## Repository Structure

```text
SemDrift/
├── BENCHMARK_RESULTS.md              # Authoritative Master Benchmark Report (All Models, V1 & V2)
├── README.md                         # Project documentation and summary results
├── semdrift/                         # Core Python library package
│   ├── parser/                       # Code & docstring parsing (AST & Tree-Sitter)
│   ├── embedder/                     # CodeBERT tokenization & embedding logic
│   ├── comparator/                   # Similarity & distance computation
│   ├── models/                       # PyTorch architectures (Dual & Joint Encoders)
│   └── pipeline.py                   # High-level pipeline API
├── scripts/                          # Workflows & experiment orchestration
│   ├── training/                     # Model training & baseline scripts
│   │   ├── run_tfidf_baseline.py     # TF-IDF + Logistic Regression baseline runner
│   │   ├── run_zero_shot_baseline.py # Zero-shot CodeBERT baseline sweep
│   │   ├── train_dual_encoder.py     # Dual-encoder fine-tuning
│   │   └── train_joint_encoder.py    # Joint-encoder fine-tuning
│   ├── runners/                      # Automated multi-model orchestration scripts
│   ├── data_pipeline/                # Dataset extraction, mutation, splitting, & manifest locking
│   ├── analysis/                     # Benchmark analysis & LaTeX artifact generators
│   └── scan_repo.py                  # Interactive terminal CLI scanner
├── experiments/                      # Clean V2 production experiment directory
│   └── 2026-09-07_clean_v2/          # Canonical V2 dataset, checkpoints, & evaluation records
├── data/                             # Dataset storage
│   ├── v1_synthetic/                 # V1 synthetic benchmark & ablation splits
│   ├── v2_real_world/                # V2 raw repositories, mined commits, & manifests
│   └── experiments/v2/               # Historical V1 checkpoints & IEEE paper outputs
└── tests/                            # Unit test suite (164 tests passing)
```

---

## Quick Start & Reproduction

### 1. Environment Setup

```bash
git clone https://github.com/Aadhithya-T/SemDrift.git
cd SemDrift
pip install -r requirements.txt
```

### 2. Verify Cryptographic Integrity & Run Unit Tests

```bash
# Verify canonical V2 dataset checksums against locked manifest
python -c "from semdrift.data.integrity import verify_dataset_integrity; print(verify_dataset_integrity('experiments/2026-09-07_clean_v2/dataset/'))"

# Run the test suite (164 tests)
python -m pytest tests/ -k "not test_contract_rules and not test_mine_real_drift" -v
```

### 3. Reproduce V2 Real-World Benchmarks

```powershell
# 1. TF-IDF Combined Baseline (V2)
python scripts/training/run_tfidf_baseline.py `
    --feature_mode combined --tune_threshold --max_iter 5000 `
    --train_file experiments/2026-09-07_clean_v2/dataset/train.jsonl `
    --val_file experiments/2026-09-07_clean_v2/dataset/val.jsonl `
    --test_file experiments/2026-09-07_clean_v2/dataset/verified_test.jsonl

# 2. TF-IDF Relational Baseline (V2)
python scripts/training/run_tfidf_baseline.py `
    --feature_mode dual_overlap --tune_threshold --max_iter 5000 `
    --train_file experiments/2026-09-07_clean_v2/dataset/train.jsonl `
    --val_file experiments/2026-09-07_clean_v2/dataset/val.jsonl `
    --test_file experiments/2026-09-07_clean_v2/dataset/verified_test.jsonl

# 3. Joint-Encoder Independent Evaluation on Verified Test Split
python experiments/2026-09-07_clean_v2/evaluation/independent_evaluate.py `
    --checkpoint experiments/2026-09-07_clean_v2/checkpoints/joint_encoder_checkpoint.pt `
    --manifest experiments/2026-09-07_clean_v2/config/manifest.yaml `
    --test_file experiments/2026-09-07_clean_v2/dataset/verified_test.jsonl
```

### 4. Reproduce V1 Synthetic Benchmarks

```powershell
# 1. TF-IDF Combined Baseline (V1)
python scripts/training/run_tfidf_baseline.py `
    --feature_mode combined --tune_threshold --max_iter 5000 `
    --train_file data/v1_synthetic/ablation/train.jsonl `
    --val_file data/v1_synthetic/ablation/val.jsonl `
    --test_file data/v1_synthetic/ablation/test.jsonl

# 2. TF-IDF Relational Baseline (V1)
python scripts/training/run_tfidf_baseline.py `
    --feature_mode dual_overlap --tune_threshold --max_iter 5000 `
    --train_file data/v1_synthetic/ablation/train.jsonl `
    --val_file data/v1_synthetic/ablation/val.jsonl `
    --test_file data/v1_synthetic/ablation/test.jsonl
```

### 5. Interactive Terminal CLI Scanner

```bash
# Scan a package for semantic drift with interactive inspection
python scripts/scan_repo.py semdrift --threshold 0.50 --interactive

# Export scan results to JSON or Markdown
python scripts/scan_repo.py . --output markdown --output_file drift_report.md
```

---

## Citation & License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
