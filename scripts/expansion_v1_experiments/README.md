# SemDrift: Expansion V1 Dataset Training, Evaluation, and Testing Suite

This suite orchestrates the complete lifecycle (**train**, **tune/validate**, and **test**) for all experimental SemDrift model architectures on the **Real Dataset Expansion V1 (10,000 samples)**.

---

## 1. Dataset Overview

- **Location**: [`data/real_dataset_expansion_v1/`](file:///data/real_dataset_expansion_v1)
- **Total Samples**: **10,000** (5,000 drifted, 5,000 consistent) across 33 open-source Python repositories
- **Taxonomy**: 13 verified semantic drift archetypes
- **Splits**:
  - `splits/train.jsonl` (8,128 samples: 4,048 aligned / 4,080 drifted)
  - `splits/val.jsonl` (544 samples: 175 aligned / 369 drifted)
  - `splits/test.jsonl` (1,328 samples: 777 aligned / 551 drifted)
- **Integrity**: Authoritative `SHA256SUMS` verified.

---

## 2. Models Supported

| # | Model Key | Model Description | Objective / Loss |
|:---:|:---|:---|:---|
| 1 | `zero_shot` | Zero-Shot CodeBERT (Cosine Divergence) | Cosine Divergence on CodeBERT embeddings (Pretrained, no fine-tuning) |
| 2 | `tfidf_combined` | TF-IDF Combined (Pure Lexical Baseline) | Logistic Regression on concatenated unigram + bigram TF-IDF features |
| 3 | `tfidf_relational` | TF-IDF Relational (Dual Overlap Baseline) | Logistic Regression on dual vocabulary + cosine sim + structural overlap |
| 4 | `dual_encoder` | Dual-Encoder (CodeBERT Bi-Encoder) | Fine-tuned Bi-Encoder with classification head on `[u; v; \|u-v\|]` (CrossEntropy) |
| 5 | `joint_encoder` | Joint-Encoder (Cross-Encoder Primary) | Full cross-attention `[CLS] doc [SEP] code [SEP]` with Focal Loss ($\gamma=2.0$) + Category Weighting |
| 6 | `joint_encoder_ce` | Joint-Encoder (Cross-Encoder CE Ablation) | Controlled architecture ablation with standard CrossEntropy (Focal=OFF, CategoryWeighting=OFF) |

---

## 3. Quick Start

### Option A: Run All Models (Master Runner)

#### PowerShell (Windows):
```powershell
# Run all 6 models with GPU
.\scripts\expansion_v1_experiments\run_all_models.ps1 -Device cuda -Epochs 3 -BatchSize 8

# Fast smoke-test on a 16-sample dry-run subset
.\scripts\expansion_v1_experiments\run_all_models.ps1 -DryRun -Device cpu
```

#### Bash (Linux / Mac / HPC):
```bash
# Run all 6 models with GPU
bash scripts/expansion_v1_experiments/run_all_models.sh cuda 3 8 2e-5

# Or via Python directly
python scripts/expansion_v1_experiments/run_all_models.py --device cuda --epochs 3 --batch_size 8
```

---

### Option B: Run Individual Models

#### 1. Zero-Shot CodeBERT Baseline
```powershell
# PowerShell
.\scripts\expansion_v1_experiments\run_zero_shot.ps1 -Device cuda -BatchSize 32

# Python
python scripts/expansion_v1_experiments/run_zero_shot.py --device cuda --batch_size 32
```

#### 2. TF-IDF Combined (Pure Lexical)
```powershell
# PowerShell
.\scripts\expansion_v1_experiments\train_eval_tfidf_combined.ps1

# Python
python scripts/expansion_v1_experiments/train_eval_tfidf_combined.py
```

#### 3. TF-IDF Relational (Dual Overlap)
```powershell
# PowerShell
.\scripts\expansion_v1_experiments\train_eval_tfidf_relational.ps1

# Python
python scripts/expansion_v1_experiments/train_eval_tfidf_relational.py
```

#### 4. Dual-Encoder (Bi-Encoder)
```powershell
# PowerShell
.\scripts\expansion_v1_experiments\train_eval_dual_encoder.ps1 -Device cuda -Epochs 3 -BatchSize 8

# Python
python scripts/expansion_v1_experiments/train_eval_dual_encoder.py --device cuda --epochs 3 --batch_size 8
```

#### 5. Joint-Encoder (Primary Proposed: Focal Loss + Category Weighting)
```powershell
# PowerShell
.\scripts\expansion_v1_experiments\train_eval_joint_encoder.ps1 -Device cuda -Epochs 3 -BatchSize 8

# Python
python scripts/expansion_v1_experiments/train_eval_joint_encoder.py --device cuda --epochs 3 --batch_size 8
```

#### 6. Joint-Encoder (Controlled Ablation: Standard CrossEntropy)
```powershell
# PowerShell
.\scripts\expansion_v1_experiments\train_eval_joint_encoder_ce.ps1 -Device cuda -Epochs 3 -BatchSize 8

# Python
python scripts/expansion_v1_experiments/train_eval_joint_encoder_ce.py --device cuda --epochs 3 --batch_size 8
```

---

## 4. Standalone Test Evaluation on Saved Checkpoints

If you already have trained checkpoints and wish to evaluate or re-evaluate them on the test set:

### Dual-Encoder Checkpoint Evaluation:
```powershell
python scripts/training/evaluate_dual_encoder.py `
    --checkpoint experiments/expansion_v1/dual_encoder/dual_encoder_checkpoint.pt `
    --test_file data/real_dataset_expansion_v1/splits/test.jsonl `
    --output_results experiments/expansion_v1/dual_encoder/results_dual_encoder.json `
    --output_preds experiments/expansion_v1/dual_encoder/predictions_dual_encoder.jsonl `
    --device cuda
```

### Joint-Encoder Checkpoint Evaluation:
```powershell
python scripts/training/evaluate_joint_encoder.py `
    --checkpoint experiments/expansion_v1/joint_encoder/joint_encoder_checkpoint.pt `
    --test_file data/real_dataset_expansion_v1/splits/test.jsonl `
    --output_results experiments/expansion_v1/joint_encoder/results_joint_encoder.json `
    --output_preds experiments/expansion_v1/joint_encoder/predictions_joint_encoder.jsonl `
    --device cuda
```

---

## 5. Output Artifacts

All experiment runs save their artifacts in `experiments/expansion_v1/`:

```
experiments/expansion_v1/
├── zero_shot/
│   ├── predictions_baseline.jsonl
│   └── results_baseline.json
├── tfidf_combined/
│   ├── tfidf_combined_model.joblib
│   ├── predictions_tfidf_combined.jsonl
│   └── results_tfidf_combined.json
├── tfidf_relational/
│   ├── tfidf_relational_model.joblib
│   ├── predictions_tfidf_relational.jsonl
│   └── results_tfidf_relational.json
├── dual_encoder/
│   ├── dual_encoder_checkpoint.pt
│   ├── predictions_dual_encoder.jsonl
│   └── results_dual_encoder.json
├── joint_encoder/
│   ├── joint_encoder_checkpoint.pt
│   ├── training_run.json
│   ├── predictions_joint_encoder.jsonl
│   └── results_joint_encoder.json
├── joint_encoder_ce/
│   ├── joint_encoder_checkpoint.pt
│   ├── training_run.json
│   ├── predictions_joint_encoder.jsonl
│   └── results_joint_encoder.json
└── benchmarks/
    ├── benchmark_summary.md       <-- Unified comparative Markdown report
    ├── benchmark_table.tex        <-- Publication LaTeX table
    └── benchmark_summary.json     <-- Master machine-readable JSON + McNemar tests
```

---

## 6. Evaluation Metrics Generated

Every model execution produces:
- **Accuracy** & **Balanced Accuracy**
- **Precision**, **Recall**, and **Binary F1** (on positive class: `drifted`)
- **Macro F1** (harmonic balance across aligned and drifted classes)
- **ROC-AUC** & **PR-AUC** (where probability estimates exist)
- **Full Confusion Matrix** (`TN`, `FP`, `FN`, `TP`)
- **Fine-Grained Breakdowns**:
  - By drift type (13 archetypes)
  - By severity level
  - By source repository (33 repositories)
  - By provenance (`authentic_historical_mined`, `contract_grounded_generated`, `clean_grounded`)
- **McNemar Significance Tests** with continuity correction comparing architectures.
