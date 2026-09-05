# Real-World Documentation Drift Datasets

This directory contains real-world Python documentation-drift datasets mined from the complete Git commit histories of 7 mature open-source repositories (**Django**, **SQLAlchemy**, **Pytest**, **Celery**, **Tornado**, **Click**, and **FastAPI**).

---

## 📁 Dataset Organization

```
data/real_world/
├── verified_dataset.jsonl                # [EVALUATION] Unified Verified Dataset (N = 101, 100% Human Verified)
├── verified_drift_positives.jsonl        # Verified Drift Positives (N = 14)
├── verified_clean_negatives.jsonl        # Verified Clean Refactorings / Negatives (N = 87)
│
├── filtered_dataset/                     # [TRAINING / WEAK SUPERVISION] Large-Scale Filtered Pool (N = 15,000)
│   ├── filtered_dataset.jsonl            # Stratified 15k dataset with pseudo-labels, confidence, & AST signals
│   ├── filtered_drift_positives.jsonl    # High-confidence drift instances (N = 571)
│   ├── filtered_clean_negatives.jsonl    # High-confidence non-drift refactorings (N = 14,429)
│   ├── filtered_ambiguous.jsonl          # Low-confidence boundary cases (N = 0)
│   └── filtered_dataset_summary.json     # Comprehensive metadata summary & per-repo breakdown
│
├── mined_candidates/                     # Raw mined candidate pools & per-repo AST contract checks (N = 204,885)
├── verified/                             # Per-repo raw human-approved drift files
└── rejected/                             # Per-repo raw human-rejected non-drift files
```

---

## 🎯 Dataset Tiers: How to Use

### 1. Verified Dataset (`data/real_world/verified_dataset.jsonl`)
* **Purpose**: **Zero-leakage Evaluation & Benchmarking**.
* **Total Instances**: **101** (100% human-verified by domain expert reviewers).
  * `label: 1` (**Drift Positive**, $N=14$, 13.86%): Function behavior drifted and directly contradicts its docstring.
  * `label: 0` (**Clean Negative**, $N=87$, 86.14%): Function underwent refactoring, type annotation additions, or internal optimization without violating docstring claims.
* **Intended Usage**:
  * Use as the primary **Real-World Test Set** to benchmark SemDrift Joint-Encoder, Dual-Encoder, and baseline models.
  * **DO NOT** train on this file — keep it strictly as an out-of-distribution real-world benchmark.

### 2. Filtered Dataset (`data/real_world/filtered_dataset/filtered_dataset.jsonl`)
* **Purpose**: **Large-Scale Semi-Supervised Pre-training & Domain Adaptation**.
* **Total Instances**: **15,000** mined Git diff pairs across 7 major Python repositories (Mined from a total raw pool of **204,885 candidates** / **33,525 filtered pool**).
* **Labeling Method**: Multi-Signal Weak Supervision & Deterministic AST Contract Rules:
  * `pseudo_label: 1` (`filtered_label: "drift"`, $N=571$): Verified AST contract violations (parameter, return type, exception removal).
  * `pseudo_label: 0` (`filtered_label: "no_drift"`, $N=14,429$): Pure refactorings, typing annotations, and contract-preserving optimizations.
* **Intended Usage**:
  * Fine-tuning CodeBERT on real-world Git diff patterns.
  * Data augmentation combined with synthetic mutation datasets (`data/experiments/v2/`).

---

## 🔬 How the Data was Collected

1. **Git Commit History Mining (`scripts/mine_real_drift.py`)**:
   - Traversed full Git histories across `django`, `sqlalchemy`, `pytest`, `celery`, `tornado`, `click`, and `fastapi`.
   - Identified commits where a function's code body (`code_before` $\to$ `code_after`) changed while its docstring (`docstring_before` $\to$ `docstring_after`) remained untouched or partially updated.
   - Extracted **204,885 raw candidate functions**.

2. **Deduplication & Quality Filtering (`scripts/filter_real_drift_candidates.py`)**:
   - Filtered out trivial whitespace/comment-only edits (`is_formatting_only`).
   - Removed unreferenced variable rename-only changes (`is_rename_only`).
   - Enforced minimum code length ($\ge 3$ lines) and meaningful docstring requirements ($\ge 20$ chars).
   - Retained **33,525 unique commit evolution candidates**.

3. **Deterministic AST Contract Checking (`scripts/contract_check_candidates.py`)**:
   - Rule 1: `parameter_contract_violation` — Parameter removed or added without default while documented in docstring.
   - Rule 2: `return_contract_violation` — Return type annotation changed or contradicted docstring return description.
   - Rule 3: `raises_contract_violation` — Documented exception `raise` statement removed from code body.
   - Rule 4: `default_contract_violation` — Parameter default value altered vs docstring claim.

4. **Weak Supervision & Stratified Consolidation (`scripts/consolidate_datasets.py`)**:
   - Computed multi-signal posterior drift probabilities and confidence scores.
   - Sampled a balanced, stratified $15,000$ instance training dataset representing all 7 repositories with $100\%$ positive capture ($N=571$).

---

## 💻 Code Example: Loading in PyTorch / HuggingFace

```python
import json
from datasets import Dataset

# 1. Load Verified Benchmark for Evaluation
def load_jsonl(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

verified_data = load_jsonl("data/real_world/verified_dataset.jsonl")
verified_dataset = Dataset.from_list(verified_data)

print(f"Loaded Verified Evaluation Set: {len(verified_dataset)} samples")

# 2. Load 15k Filtered Dataset for Training / Fine-Tuning
filtered_data = load_jsonl("data/real_world/filtered_dataset/filtered_dataset.jsonl")
train_dataset = Dataset.from_list([x for x in filtered_data if x["pseudo_label"] != -1])

print(f"Loaded Training/Fine-Tuning Set: {len(train_dataset)} samples")
```

---

## 📊 Summary Statistics (15,000 Sample Dataset)

| Repository | Total Samples | Drift Positives (+) | Clean Negatives (-) | Positive Rate |
| :--- | :---: | :---: | :---: | :---: |
| **Django** | 7,355 | 5 | 7,350 | 0.07% |
| **SQLAlchemy** | 4,104 | 381 | 3,723 | 9.28% |
| **Pytest** | 1,591 | 79 | 1,512 | 4.97% |
| **Celery** | 839 | 9 | 830 | 1.07% |
| **Tornado** | 729 | 76 | 653 | 10.43% |
| **Click** | 306 | 21 | 285 | 6.86% |
| **FastAPI** | 76 | 0 | 76 | 0.00% |
| **Total** | **15,000** | **571** | **14,429** | **3.81%** |
