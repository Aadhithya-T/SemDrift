# SemDrift — Semantic Drift Detection in Python Repositories

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.1+](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org/)
[![Transformers 4.30+](https://img.shields.io/badge/HuggingFace-Transformers-yellow.svg)](https://huggingface.co/transformers/)
[![Tree-Sitter](https://img.shields.io/badge/Tree--Sitter-Multi--Language-green.svg)](https://tree-sitter.github.io/)
[![Unit Tests](https://img.shields.io/badge/tests-51%20passed-brightgreen.svg)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**SemDrift** is an automated framework for detecting **semantic drift** between source code and documentation comments (docstrings). It combines AST parsing, multi-language Tree-Sitter support, synthetic mutation injection, and deep transformer architectures (CodeBERT) to detect when code changes silently invalidate docstring contracts.

---

## 📊 Benchmark Results (IEEE Conference Benchmark — Phase 1)

All models are evaluated on the clean, zero-leakage 10-repository V2 benchmark dataset (`data/experiments/v2/test.jsonl`, $N = 1,205$). The test repositories are strictly partitioned from training and validation sets to ensure cross-repository generalization.

### 1. Overall Performance Comparison

| Model Architecture | Loss Objective | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | Macro-F1 (%) | Balanced Acc (%) | 95% F1 Conf. Interval |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Lexical Baseline (TF-IDF + LogReg)** | Logistic Loss | 58.84% | 60.26% | 54.11% | 57.02% | 58.76% | 58.91% | — |
| **Zero-Shot Baseline (Dual Encoder)** | Cosine Threshold | 44.56% | 43.33% | 32.07% | 36.86% | 43.73% | 44.68% | `[33.33, 40.41]` |
| **Fine-Tuned Dual Encoder (Ablation)** | CrossEntropy | 80.91% | 91.63% | 68.42% | 78.34% | 80.64% | 81.03% | `[74.61, 80.30]` |
| **Fine-Tuned Joint Encoder (Controlled)** | CrossEntropy | **85.06%** | **93.32%** | **75.82%** | **83.67%** | **84.95%** | **85.15%** | `[81.22, 85.76]` |
| **Fine-Tuned Joint Encoder (Loss Ablation)** | Focal Loss (Plain) | **85.81%** | **96.00%** | 75.00% | **84.21%** | **85.66%** | **85.91%** | `[81.85, 86.30]` |
| **Fine-Tuned Joint Encoder (Weighted)** | Focal + Category | **85.06%** | **93.85%** | 75.33% | 83.58% | 84.94% | 85.15% | `[81.22, 85.76]` |

---

### 2. 🔬 Controlled Architectural Ablation (Dual vs. Joint Encoder)

To isolate the impact of **transformer architecture** from loss formulation, we conducted a strictly controlled experiment holding every variable constant:
* **Backbone**: `microsoft/codebert-base`
* **Objective**: Standard, unweighted `CrossEntropyLoss` (Focal Loss = **OFF**, Category Weights = **OFF**)
* **Hyperparameters**: Epochs = 3, Batch Size = 8, LR = $2\times 10^{-5}$, Warmup = 0.1, Dropout = 0.1, Seed = 42

$$\text{Dual-Encoder (CE)}: \mathbf{78.34\% \text{ F1}} \quad \xrightarrow{\mathbf{+5.33\% \text{ F1}}} \quad \text{Joint-Encoder (CE)}: \mathbf{83.67\% \text{ F1}}$$

| Metric | Dual-Encoder (CE) | Joint-Encoder (CE) | Absolute Delta ($\Delta$) | Relative Gain |
|:---|:---:|:---:|:---:|:---:|
| **Accuracy** | 80.91% | **85.06%** | **`+4.15%`** | +5.13% |
| **Precision** | 91.63% | **93.32%** | **`+1.69%`** | +1.84% |
| **Recall** | 68.42% | **75.82%** | **`+7.40%`** | **+10.82%** |
| **Binary F1-Score** | 78.34% | **83.67%** | **`+5.33%`** | **+6.80%** |
| **Macro-F1 Score** | 80.64% | **84.95%** | **`+4.31%`** | +5.34% |
| **Balanced Accuracy** | 81.03% | **85.15%** | **`+4.12%`** | +5.08% |

#### Statistical Significance (McNemar's Paired Test with Continuity Correction):
* Contingency Table ($N = 1,205$): $n_{00}=141$, $n_{01}=89$ (Joint wins), $n_{10}=39$ (Dual wins), $n_{11}=936$.
* **Chi-Square Statistic ($\chi^2$)**: **`18.7578`**
* **$p$-value**: **`1.4841 × 10⁻⁵`** ($p < 0.001 \rightarrow$ **Statistically Significant**)
* *Conclusion*: Joint Code–Documentation Self-Attention provides an architecture-driven advantage over independent dual encoding.

*For complete ablation tables and confusion matrices, see [`Results - Thunder.md`](Results%20-%20Thunder.md).*

---

### 3. F1-Score Breakdown Across Semantic Drift Types

| Drift Type / Mutation | Sample Count ($N$) | TF-IDF Baseline | Zero-Shot Baseline | Fine-Tuned Dual (CE) | Fine-Tuned Joint (CE) | Delta ($\Delta$) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **`return_value_change`** | 139 | 62.38% | 49.73% | 78.60% | **93.49%** | **`+14.89%`** |
| **`doc_negation`** | 95 | 64.29% | 59.26% | 17.31% | **37.61%** | **`+20.30%`** |
| **`doc_sentence_delete`** | 153 | 52.17% | 29.05% | 80.47% | **81.40%** | `+0.93%` |
| **`param_rename`** | 221 | 86.08% | 54.61% | **98.39%** | 97.92% | `-0.47%` |
| **`aligned` (Specificity)** | 597 | 63.65% Acc | 57.29% Acc | 93.63% Acc | **94.47% Acc** | `+0.84%` |

#### Key Insights on Drift Types:
1. **Return Value Drift (`+14.89% F1`)**: Joint cross-modal attention enables the model to align function return signatures and exit expressions with docstring `@return` / summary claims across token boundaries.
2. **Docstring Negation (`+20.30% F1`)**: Dual encoding collapses sentence tokens into isolated vector embeddings $\mathbf{u}$ and $\mathbf{v}$, losing polarity tokens (*"not"*, *"never"*, *"disabled"*). Joint self-attention allows negation particles to directly attend to control-flow conditions.
3. **Parameter Renames**: Both deep architectures perform near-ceiling ($98.39\%$ vs $97.92\%$), demonstrating that parameter misalignment is effectively captured by both representations.

---

### 4. Cross-Repository Generalization Across 10 Unseen Repositories

Evaluated on $N = 1,205$ test instances from 10 distinct open-source projects (zero repository overlap with train/val):

| Repository | Test Count ($N$) | Dual-Encoder (CE) F1 | Joint-Encoder (CE) F1 | Delta ($\Delta$) |
|:---|:---:|:---:|:---:|:---:|
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

> **Result**: The Joint Encoder outperforms the Dual Encoder across **9 out of 10 unseen repositories**.

---

## 🏗️ Model Architectures & Workflows

```
                          ┌────────────────────────────────────────────────────────┐
                          │                      INPUT PAIR                        │
                          │        Docstring (Text)  +  Function Code (Python)     │
                          └────────────────────────────────────────────────────────┘
                                                       │
         ┌─────────────────────────────────────────────┼─────────────────────────────────────────────┐
         ▼                                             ▼                                             ▼
┌───────────────────────────┐             ┌───────────────────────────┐             ┌───────────────────────────┐
│    ZERO-SHOT BASELINE     │             │  FINE-TUNED DUAL-ENCODER  │             │  FINE-TUNED JOINT ENCODER │
│   (Dual Encoder Baseline) │             │     (Ablation Model)      │             │   (Primary Contribution)  │
└───────────────────────────┘             └───────────────────────────┘             └───────────────────────────┘
│ Two separate forward passes               │ Two separate forward passes               │ Single joint forward pass
│ Pre-trained CodeBERT                      │ Fine-tuned shared CodeBERT                │ Fine-tuned CodeBERT
│ No fine-tuning                            │ Independent encoding (isolated)           │ Joint self-attention
│ Mean pooling → Vectors u, v               │ Mean pooling → Vectors u, v               │ [CLS] token representation
│ Cosine distance 1 - cos(u,v)              │ Feature: [u; v; |u-v|]                    │ Linear head: [CLS] → 2
│ Threshold sweep τ*                        │ Classifier head: 2304 → 2                 │ Classifier head: 768 → 2
└───────────────────────────┘             └───────────────────────────┘             └───────────────────────────┘
```

### 1. Zero-Shot Dual Encoder Baseline
* **Script**: [`scripts/training/run_zero_shot_baseline.py`](scripts/training/run_zero_shot_baseline.py) (shim: [`scripts/run_zero_shot_baseline.py`](scripts/run_zero_shot_baseline.py))
* **Workflow**: Processes docstring and code in two independent forward passes through frozen `microsoft/codebert-base`. Computes attention-masked mean-pooled representations $\mathbf{u}$ and $\mathbf{v}$, computes cosine divergence $\delta = 1 - \cos(\mathbf{u}, \mathbf{v})$, and applies threshold $\tau^*$ tuned via validation sweep ($\tau^* = 0.9975$).

### 2. Fine-Tuned Dual Encoder (Ablation)
* **Script**: [`scripts/training/train_dual_encoder.py`](scripts/training/train_dual_encoder.py) (shim: [`scripts/train_dual_encoder.py`](scripts/train_dual_encoder.py))
* **Workflow**: Independently encodes docstring and code through a shared CodeBERT encoder. Constructs interaction vector $[\mathbf{u} \,;\, \mathbf{v} \,;\, |\mathbf{u} - \mathbf{v}|] \in \mathbb{R}^{2304}$, fed into `nn.Linear(2304, 2)` and trained with CrossEntropy.

### 3. Fine-Tuned Joint Encoder (Primary Contribution)
* **Script**: [`scripts/training/train_joint_encoder.py`](scripts/training/train_joint_encoder.py) (shim: [`scripts/train_joint_encoder.py`](scripts/train_joint_encoder.py))
* **Workflow**: Formulates drift detection as a single joint sequence:
  $$\text{Input} = [\text{CLS}] \;\; \text{docstring\_tokens} \;\; [\text{SEP}] \;\; [\text{SEP}] \;\; \text{code\_tokens} \;\; [\text{SEP}]$$
  * **Head-Tail Truncation**: When code exceeds token budgets, SemDrift keeps the function header/signature and terminal return statements, inserting a `[MASK]` token in between.
  * **Joint Attention**: Full bidirectional self-attention between documentation and code tokens across all 12 transformer layers.
  * **Classification Head**: `nn.Dropout(0.1)` + `nn.Linear(768, 2)` on the pooled `[CLS]` token.

---

## 📁 Repository Structure

```text
SemDrift/
├── semdrift/                         # Core Python Library Package
│   ├── parser/                       # Code & Docstring Parsing
│   │   ├── ast_parser.py             # Python AST parser & docstring stripper
│   │   ├── universal_parser.py       # Multi-language Tree-Sitter parser
│   │   ├── doc_extractor.py          # Google/NumPy/Sphinx docstring extractor
│   │   └── formatter.py              # Input sequence formatting
│   ├── embedder/                     # CodeBERT embedding module
│   │   └── embed.py                  # Tokenization & pooling logic
│   ├── comparator/                   # Similarity & distance computations
│   ├── models/                       # PyTorch Neural Architectures
│   │   ├── dual_encoder.py           # Dual-Encoder architecture
│   │   └── joint_encoder.py          # Joint-Encoder architecture & Focal Loss
│   └── pipeline.py                   # High-level pipeline API
├── scripts/                          # Workflow & Experiment Scripts
│   ├── data_pipeline/                # Extraction, Mutation, & Splitting
│   │   ├── extract_pairs.py          # AST extraction from raw repositories
│   │   ├── filter_pairs.py           # Quality & length filtering
│   │   ├── build_dataset.py          # Synthetic mutation injector
│   │   ├── convert_dataset_format.py # Format conversion
│   │   ├── split_dataset.py          # Repository-disjoint train/val/test splits
│   │   └── mine_dataset.py           # Git commit history miner (PyDriller)
│   ├── training/                     # Model Training & Baseline Scripts
│   │   ├── run_zero_shot_baseline.py # Zero-shot CodeBERT baseline sweep
│   │   ├── train_dual_encoder.py     # Dual-encoder fine-tuning
│   │   └── train_joint_encoder.py    # Joint-encoder fine-tuning
│   ├── runners/                      # Automated Orchestration Runners
│   │   ├── run_controlled_experiment.py # Python runner for Controlled Dual vs Joint
│   │   ├── run_controlled_experiment.ps1# PowerShell runner for Controlled Experiment
│   │   ├── run_controlled_experiment.bat# Batch script for Controlled Experiment
│   │   ├── retrain_all_models.ps1    # Sequential benchmark retrain (PowerShell)
│   │   ├── retrain_all_models.bat    # Sequential benchmark retrain (Batch)
│   │   ├── run_focal_ablation.ps1    # Focal Loss ablation runner (PowerShell)
│   │   └── run_focal_ablation.bat    # Focal Loss ablation runner (Batch)
│   ├── analysis/                     # Benchmark Analysis & LaTeX Generators
│   │   ├── analyze_controlled_experiment.py # Controlled ablation evaluation
│   │   ├── analyze_loss_ablation.py  # CrossEntropy vs Focal Loss ablation
│   │   ├── diagnose_negation_and_lexical.py # TF-IDF baseline & negation diagnosis
│   │   ├── analyze_truncation.py     # Head-tail truncation analysis
│   │   ├── generate_ieee_results.py  # IEEE paper JSON & LaTeX table generator
│   │   ├── inspect_diffs.py          # Mutation visualizer
│   │   └── scan_example_heavy_docs.py# Documentation style scanner
│   ├── scan_repo.py                  # Claude Code-Style Interactive Terminal CLI
│   ├── train_joint_encoder.py        # Top-level execution shim
│   ├── train_dual_encoder.py         # Top-level execution shim
│   └── run_zero_shot_baseline.py     # Top-level execution shim
├── tests/                            # Unit Test Suite (51 tests)
│   ├── test_parser.py                # AST & docstring extraction tests
│   ├── test_embedder.py              # Embedding & model loading tests
│   ├── test_comparator.py            # Comparator unit tests
│   └── test_v2_updates.py            # Head-tail truncation, doc stripping & metrics
├── data/                             # Datasets & Model Checkpoints
│   └── experiments/v2/               # 10-Repository Zero-Leakage Dataset & Runs
├── Results - Thunder.md              # Controlled Ablation Experiment Report
├── config.yaml                       # Global pipeline configuration
├── requirements.txt                  # Python dependencies
└── README.md                         # Project documentation
```

---

## ⚡ Quick Start & Reproduction Commands

### 1. Environment Setup

```bash
# Clone repository
git clone https://github.com/Aadhithya-T/SemDrift.git
cd SemDrift

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Unit Tests (51 Tests)

```bash
python -m pytest tests/
```

### 3. Automated One-Click Runners

#### A. Run Controlled Dual vs. Joint Experiment (Identical CrossEntropy Objective)
Executes Dual-Encoder, Joint-Encoder, and statistical analysis end-to-end:
```bash
# Python runner (cross-platform)
python scripts/runners/run_controlled_experiment.py --device cuda

# PowerShell runner (Windows)
./scripts/runners/run_controlled_experiment.ps1 -Device cuda

# Batch runner (Windows CMD)
scripts\runners\run_controlled_experiment.bat cuda
```

#### B. Retrain All Benchmark Models
Retrains the zero-shot baseline, fine-tuned dual encoder, fine-tuned joint encoder, and regenerates IEEE paper artifacts:
```powershell
./scripts/runners/retrain_all_models.ps1
```

#### C. Run Loss Objective Ablation (CrossEntropy vs. Focal Loss)
```powershell
./scripts/runners/run_focal_ablation.ps1 -Device cuda
```

---

### 4. Individual Training & Baseline Execution

```bash
# 1. Lexical Baseline (TF-IDF + Logistic Regression) & Negation Diagnosis
python scripts/analysis/diagnose_negation_and_lexical.py

# 2. Zero-Shot Dual Encoder Baseline (Threshold Sweep)
python scripts/training/run_zero_shot_baseline.py \
    --val data/experiments/v2/val.jsonl \
    --test data/experiments/v2/test.jsonl \
    --output_dir data/experiments/v2/baseline_results \
    --device cuda

# 3. Fine-Tuned Dual-Encoder (Ablation Model)
python scripts/training/train_dual_encoder.py \
    --train data/experiments/v2/train.jsonl \
    --val data/experiments/v2/val.jsonl \
    --test data/experiments/v2/test.jsonl \
    --device cuda --epochs 3 --batch_size 8 \
    --output_dir data/experiments/v2/dual_encoder_results/

# 4. Fine-Tuned Joint-Encoder (Primary Contribution)
python scripts/training/train_joint_encoder.py \
    --train data/experiments/v2/train.jsonl \
    --val data/experiments/v2/val.jsonl \
    --test data/experiments/v2/test.jsonl \
    --device cuda --epochs 3 --batch_size 8 \
    --code_truncation head_tail --pooling cls \
    --checkpoint_metric macro_f1 \
    --use_focal_loss --category_weighting \
    --output_dir data/experiments/v2/joint_encoder_results/
```

---

### 5. Repository CLI Scanner (Interactive Terminal Tool)

SemDrift provides an interactive terminal CLI (`scripts/scan_repo.py`) featuring rich syntax highlighting, progress bars, interactive inspection mode, and multi-format report exports:

```bash
# Scan a directory or package using the trained Joint-Encoder
python scripts/scan_repo.py semdrift --threshold 0.60

# Interactive step-through review mode
python scripts/scan_repo.py . --interactive

# Export Markdown or JSON reports
python scripts/scan_repo.py . --output markdown --output_file drift_report.md
python scripts/scan_repo.py . --output json --output_file drift_report.json

# Limit to top-K highest-probability drift candidates
python scripts/scan_repo.py . --top_k 10 --threshold 0.50
```

---

### 6. Generate IEEE Paper Artifacts & LaTeX Tables

```bash
# Generate LaTeX tables and JSON summaries for paper submission
python scripts/analysis/generate_ieee_results.py \
    --v2_dir data/experiments/v2 \
    --output_dir data/experiments/v2

# Analyze controlled architectural ablation
python scripts/analysis/analyze_controlled_experiment.py \
    --dual_preds data/experiments/v2/controlled_ablation/dual_ce/predictions_dual_encoder.jsonl \
    --joint_preds data/experiments/v2/controlled_ablation/joint_ce/predictions_joint_encoder.jsonl \
    --output_dir data/experiments/v2/controlled_ablation
```

Outputs:
* JSON Benchmark Data: [`data/experiments/v2/ieee_paper_results.json`](data/experiments/v2/ieee_paper_results.json)
* IEEEtran LaTeX Tables: [`data/experiments/v2/ieee_paper_tables.tex`](data/experiments/v2/ieee_paper_tables.tex)
* Controlled Experiment LaTeX Table: [`data/experiments/v2/controlled_ablation/controlled_experiment_table.tex`](data/experiments/v2/controlled_ablation/controlled_experiment_table.tex)

---

### 7. Rebuild Synthetic Benchmark Dataset (Optional)

```bash
# Extract function-docstring pairs from raw repositories
python scripts/data_pipeline/extract_pairs.py --repos_dir data/raw_repos --output data/experiments/v2/extracted_pairs.jsonl

# Filter invalid, trivial, or oversized pairs
python scripts/data_pipeline/filter_pairs.py --input data/experiments/v2/extracted_pairs.jsonl --output data/experiments/v2/filtered_pairs.jsonl

# Inject synthetic mutations (param rename, return type change, doc deletion, negation)
python scripts/data_pipeline/build_dataset.py --input data/experiments/v2/filtered_pairs.jsonl --output data/experiments/v2/mutated_dataset.jsonl

# Convert to standard labeled schema
python scripts/data_pipeline/convert_dataset_format.py --input data/experiments/v2/mutated_dataset.jsonl --output data/experiments/v2/semdrift_labeled.jsonl

# Partition by repository into disjoint train, val, and test splits
python scripts/data_pipeline/split_dataset.py --input data/experiments/v2/semdrift_labeled.jsonl --output_dir data/experiments/v2/
```

---

## 📜 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
