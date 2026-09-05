# Real-World Documentation Drift Datasets

This directory contains real-world Python documentation-drift datasets mined from the complete Git commit histories of popular open-source repositories (**Click**, **FastAPI**, and **Django**).

---

## 📁 Dataset Organization

```
data/real_world/
├── gold_benchmark.jsonl                  # [EVALUATION] Unified Gold Benchmark (N = 76, 100% Human Verified)
├── gold_drift_positives.jsonl            # Verified Drift Positives (N = 12)
├── gold_clean_negatives.jsonl            # Verified Clean Refactorings / Negatives (N = 64)
│
├── filtered_dataset/                     # [TRAINING / WEAK SUPERVISION] Large-Scale Filtered Pool (N = 4,997)
│   ├── filtered_dataset.jsonl            # Combined 5k dataset with pseudo-labels and confidence scores
│   ├── filtered_drift_positives.jsonl    # High-confidence drift instances (N = 13)
│   ├── filtered_clean_negatives.jsonl    # High-confidence non-drift refactorings (N = 4,984)
│   ├── filtered_ambiguous.jsonl          # Low-confidence boundary cases (N = 0)
│   └── filtered_dataset_summary.json     # Metadata summary and stats
│
├── mined_candidates/                     # Raw mined candidate pools & per-repo AST contract checks
├── verified/                             # Per-repo raw human-approved drift files
└── rejected/                             # Per-repo raw human-rejected non-drift files
```

---

## 🎯 Dataset Tiers: How to Use

### 1. Gold Benchmark (`data/real_world/gold_benchmark.jsonl`)
* **Purpose**: **Zero-leakage Evaluation & Benchmarking**.
* **Total Instances**: **101** (100% human-verified by domain expert reviewer `pradeep`).
  * `label: 1` (**Drift Positive**, $N=14$, 13.86%): Function behavior drifted and directly contradicts its docstring.
  * `label: 0` (**Clean Negative**, $N=87$, 86.14%): Function underwent refactoring, type annotation additions, or internal optimization without violating docstring claims.
* **Intended Usage**:
  * Use as the primary **Real-World Test Set** to benchmark SemDrift Joint-Encoder, Dual-Encoder, and baseline models.
  * **DO NOT** train on this file — keep it strictly as an out-of-distribution real-world benchmark.

### 2. Filtered Dataset (`data/real_world/filtered_dataset/filtered_dataset.jsonl`)
* **Purpose**: **Large-Scale Semi-Supervised Pre-training & Domain Adaptation**.
* **Total Instances**: **4,997** mined Git diff pairs (Click: 188, FastAPI: 67, Django: 4,742).
* **Labeling Method**: Multi-Signal Weak Supervision & Deterministic AST Contract Rules:
  * `pseudo_label: 1` (`filtered_label: "drift"`): Verified AST contract violations (parameter, return type, exception removal).
  * `pseudo_label: 0` (`filtered_label: "no_drift"`): Pure refactorings and typing annotations.
* **Intended Usage**:
  * Fine-tuning CodeBERT on real-world Git diff patterns.
  * Data augmentation combined with synthetic mutation datasets (`data/experiments/v2/`).

---

## 🔬 How the Data was Collected

1. **Git Commit History Mining (`scripts/mine_real_drift.py`)**:
   - Traversed full Git histories across `click`, `fastapi`, and `django`.
   - Identified commits where a function's code body (`code_before` $\to$ `code_after`) changed while its docstring (`docstring_before` $\to$ `docstring_after`) remained untouched or partially updated.
   - Extracted 76,800+ raw candidate functions.

2. **Deduplication & Quality Filtering (`scripts/filter_real_drift_candidates.py`)**:
   - Filtered out trivial whitespace/comment-only edits.
   - Removed test files, migrations, mock stubs, and empty docstrings.
   - Retained 4,997 high-quality candidate pairs.

3. **Deterministic AST Contract Checking (`scripts/contract_check_candidates.py`)**:
   - Rule 1: `parameter_contract_violation` — Parameter removed or added without default while documented in docstring.
   - Rule 2: `return_contract_violation` — Return type annotation changed or contradicted docstring return description.
   - Rule 3: `raises_contract_violation` — Documented exception `raise` statement removed from code body.
   - Rule 4: `default_contract_violation` — Parameter default value altered vs docstring claim.

4. **Human Verification ($N = 76$) (`scripts/review_candidates.py`)**:
   - Top prioritized candidates from Click ($N=19$), FastAPI ($N=7$), and Django ($N=50$) were hand-labeled.

---

## 💻 Code Example: Loading in PyTorch / HuggingFace

```python
import json
from datasets import Dataset

# 1. Load Gold Benchmark for Evaluation
def load_jsonl(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

gold_data = load_jsonl("data/real_world/gold_benchmark.jsonl")
gold_dataset = Dataset.from_list(gold_data)

print(f"Loaded Gold Evaluation Set: {len(gold_dataset)} samples")
# Features available: ['function_name', 'repo', 'code_before', 'code_after', 
#                      'docstring_before', 'docstring_after', 'label', 'gold_label']

# 2. Load Filtered Dataset for Fine-Tuning
filtered_data = load_jsonl("data/real_world/filtered_dataset/filtered_dataset.jsonl")
train_dataset = Dataset.from_list([x for x in filtered_data if x["pseudo_label"] != -1])

print(f"Loaded Training/Fine-Tuning Set: {len(train_dataset)} samples")
```

---

## 📊 Summary Statistics

| Dataset | Total Samples | Positives (Drift) | Negatives (Clean) | Drift Rate |
| :--- | :---: | :---: | :---: | :---: |
| **Gold Benchmark** | **101** | 14 (13.86%) | 87 (86.14%) | 13.86% |
| **Filtered Dataset** | **4,997** | 13 (0.26%) | 4,984 (99.74%) | 0.26% |
