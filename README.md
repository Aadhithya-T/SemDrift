# SemDrift — Semantic Drift Detection in Python Repositories

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.1+](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org/)
[![Transformers 4.30+](https://img.shields.io/badge/HuggingFace-Transformers-yellow.svg)](https://huggingface.co/transformers/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**SemDrift** is an automated framework for detecting **semantic drift** between Python source code and documentation comments (docstrings). It combines AST parsing, synthetic mutation injection, and deep transformer architectures (CodeBERT) to detect when code changes invalidate docstring contracts.

---

## 📊 Benchmark Results (IEEE Conference Paper — Phase 1)

All models were evaluated on the clean, un-leaked V2 benchmark dataset (`data/experiments/v2/test.jsonl`, $N = 572$).

### Overall Model Performance & 95% Confidence Intervals

| Model Architecture | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | Macro-F1 (%) | Balanced Acc (%) | 95% F1 Confidence Interval |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Model A (Baseline)** | 52.62% | 51.22% | 38.04% | 43.66% | 51.39% | 52.13% | `[38.09, 48.98]` |
| **Model B (Dual Encoder)** | 79.02% | 86.11% | 67.39% | 75.61% | 78.60% | 78.63% | `[71.16, 79.68]` |
| **Model B (Joint Primary)** | **82.34%** | **91.87%** | **69.57%** | **79.18%** | **81.92%** | **81.91%** | **`[75.05, 82.93]`** |

### Statistical Significance (McNemar's Test)

* **Model A (Base) vs. Model B (Joint Primary)**: $\chi^2 = 116.10$, $p = 4.52 \times 10^{-27}$ $\rightarrow$ **Statistically Significant ($p < 0.001$)**
* **Model B (Dual) vs. Model B (Joint Primary)**: $\chi^2 = 5.31$, $p = 0.0212$ $\rightarrow$ **Statistically Significant ($p < 0.05$)**

### F1-Score Breakdown Across Semantic Drift Types

| Drift Type / Mutation | Sample Count ($N$) | Model A (Base) | Model B (Dual Encoder) | Model B (Joint Primary) |
|:---|:---:|:---:|:---:|:---:|
| **`param_rename`** | 101 | 60.69% | 97.46% | **97.98%** |
| **`return_value_change`** | 61 | 54.76% | 85.98% | **94.83%** |
| **`doc_sentence_delete`** | 52 | 61.33% | **83.15%** | 77.65% |
| **`doc_negation`** | 62 | **38.96%** | 20.29% | 20.29% |
| **`aligned`** | 296 | 0.00% | 0.00% | 0.00% |

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
* **Script**: [`scripts/run_model_a.py`](file:///c:/Users/aadhi/OneDrive/Desktop/SemDrift/scripts/run_model_a.py)
* **Workflow**: Processes docstring and code in two independent forward passes through pre-trained `microsoft/codebert-base`. Computes attention-masked mean-pooled representations $\mathbf{u}$ and $\mathbf{v}$, calculates cosine divergence $\delta = 1 - \cos(\mathbf{u}, \mathbf{v})$, and applies an optimal threshold $\tau^*$ derived via validation sweep ($\tau^* = 0.9975$).

### 2. Model B Ablation: Fine-Tuned Dual Encoder
* **Script**: [`scripts/train_dual_encoder.py`](file:///c:/Users/aadhi/OneDrive/Desktop/SemDrift/scripts/train_dual_encoder.py)
* **Workflow**: Encodes docstring and code separately through a shared CodeBERT model. Concatenates sentence representations into $[\mathbf{u} \,;\, \mathbf{v} \,;\, |\mathbf{u} - \mathbf{v}|] \in \mathbb{R}^{2304}$, fed into a linear classification head (`nn.Linear(2304, 2)`), fine-tuned end-to-end using `CrossEntropyLoss`.

### 3. Model B Primary: Fine-Tuned Joint Encoder Classifier
* **Script**: [`scripts/train_model_b.py`](file:///c:/Users/aadhi/OneDrive/Desktop/SemDrift/scripts/train_model_b.py)
* **Workflow**: Concatenates docstring and code into a **single joint token sequence**:
  $$\text{Input} = [\text{CLS}] \;\; \text{docstring\\_tokens} \;\; [\text{SEP}] \;\; [\text{SEP}] \;\; \text{code\\_tokens} \;\; [\text{SEP}]$$
  Employs a **`head_tail` truncation strategy** to preserve return statements at the end of functions. Executes a single forward pass through CodeBERT where self-attention enables **full cross-attention** between every docstring and code token. The `[CLS]` token representation is classified via `nn.Linear(768, 2)`, fine-tuned with AdamW and linear LR warmup, selected via validation `macro_f1`.

---

## 📁 Repository Structure

```text
SemDrift/
├── semdrift/                     # Core Library Package
│   ├── parser/                   # AST parsing & docstring extraction
│   ├── embedder/                 # CodeBERT embedding module
│   ├── comparator/               # Scoring & similarity functions
│   ├── models/                   # PyTorch neural network architectures
│   │   ├── dual_encoder.py       # Dual-Encoder model class
│   │   └── joint_encoder.py      # Joint-Encoder model class & Focal Loss
│   └── pipeline.py               # End-to-end integration API
├── scripts/                      # Categorized Workflow Scripts
│   ├── data_pipeline/            # Dataset extraction, filtering, mutation, & splitting
│   │   ├── extract_pairs.py
│   │   ├── filter_pairs.py
│   │   ├── build_dataset.py
│   │   ├── convert_dataset_format.py
│   │   └── split_dataset.py
│   ├── training/                 # Model training & baseline scripts
│   │   ├── run_model_a.py
│   │   ├── train_dual_encoder.py
│   │   └── train_model_b.py
│   ├── analysis/                 # Benchmark results & LaTeX table generators
│   │   ├── generate_ieee_results.py
│   │   └── scan_example_heavy_docs.py
│   ├── runners/                  # Automated sequential retraining scripts
│   │   ├── retrain_all_models.ps1
│   │   └── retrain_all_models.bat
│   └── scan_repo.py              # CLI Terminal Scanner Entrypoint
├── tests/                        # Unit test suite
├── data/                         # Datasets & Checkpoints
├── config.yaml                   # Configuration parameters
├── requirements.txt            # Dependency specifications
└── README.md                     # Project documentation
```

---

## ⚡ Quick Start & Reproduction Commands

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/Aadhithya-T/SemDrift.git
cd SemDrift

# Install dependencies
pip install -r requirements.txt
```

### 2. Unit Tests

```bash
python -m unittest discover tests/
```

### 3. Rebuild Dataset (Optional)

```bash
python scripts/data_pipeline/extract_pairs.py --repos_dir data/raw_repos --output data/experiments/v2/extracted_pairs.jsonl
python scripts/data_pipeline/filter_pairs.py --input data/experiments/v2/extracted_pairs.jsonl --output data/experiments/v2/filtered_pairs.jsonl
python scripts/data_pipeline/build_dataset.py --input data/experiments/v2/filtered_pairs.jsonl --output data/experiments/v2/mutated_dataset.jsonl
python scripts/data_pipeline/convert_dataset_format.py --input data/experiments/v2/mutated_dataset.jsonl --output data/experiments/v2/semdrift_labeled.jsonl
python scripts/data_pipeline/split_dataset.py --input data/experiments/v2/semdrift_labeled.jsonl --output_dir data/experiments/v2/
```

### 4. Train & Evaluate Models

```bash
# Model A Baseline
python scripts/training/run_model_a.py --val data/experiments/v2/val.jsonl --test data/experiments/v2/test.jsonl --output_dir data/experiments/v2/model_a_results --device cuda

# Model B Dual-Encoder Ablation
python scripts/training/train_dual_encoder.py --train data/experiments/v2/train.jsonl --val data/experiments/v2/val.jsonl --test data/experiments/v2/test.jsonl --device cuda --epochs 3 --batch_size 8 --output_dir data/experiments/v2/dual_encoder_results/

# Model B Joint-Encoder Primary (with Focal Loss & Category Loss Weighting)
python scripts/training/train_model_b.py --train data/experiments/v2/train.jsonl --val data/experiments/v2/val.jsonl --test data/experiments/v2/test.jsonl --device cuda --epochs 3 --batch_size 8 --code_truncation head_tail --pooling cls --checkpoint_metric macro_f1 --use_focal_loss --category_weighting --output_dir data/experiments/v2/joint_encoder_results/
```

### 5. Repository CLI Scanner (Inference)

```bash
# Scan a codebase using fine-tuned Model B (Joint-Encoder)
python scripts/scan_repo.py semdrift --threshold 0.60

# Interactive step-through review mode
python scripts/scan_repo.py . --interactive

# Export Markdown report
python scripts/scan_repo.py . --output markdown --output_file drift_report.md
```

### 6. Generate IEEE Paper Artifacts & LaTeX Tables

```bash
python scripts/analysis/generate_ieee_results.py --v2_dir data/experiments/v2 --output_dir data/experiments/v2
```

Outputs:
* JSON Benchmark Data: [`data/experiments/v2/ieee_paper_results.json`](file:///c:/Users/aadhi/OneDrive/Desktop/SemDrift/data/experiments/v2/ieee_paper_results.json)
* IEEEtran LaTeX Tables: [`data/experiments/v2/ieee_paper_tables.tex`](file:///c:/Users/aadhi/OneDrive/Desktop/SemDrift/data/experiments/v2/ieee_paper_tables.tex)

---

## 📜 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
