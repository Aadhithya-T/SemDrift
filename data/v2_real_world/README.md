# SemDrift V2 — Real-World-Grounded Dataset

**Purpose**: Main training, validation, and real-world evaluation dataset combining **authentic historical Git evolution commits** and **AST contract-grounded generated drift**, evaluated against a strictly isolated **101 human-verified test set**.

---

## 🎯 Architecture & Workflow

```text
                    SEMDRIFT
                       │
             ┌─────────┴─────────┐
             │                   │
             ▼                   ▼
       V1 — SYNTHETIC      V2 — REAL-WORLD-
          1,205               GROUNDED
             │                 ~14,796
             │                   │
       Baseline/ablation    ┌────┴────┐
                            │         │
                          TRAIN      VAL
                         13,366     1,430
                            │         │
                            └────┬────┘
                                 ▼
                           TRAIN MODEL
                                 │
                                 ▼
                            FINAL MODEL
                                 │
                                 ▼
                         ┌───────────────┐
                         │ 101 VERIFIED  │
                         │   TEST ONLY   │
                         └───────────────┘
```

> **The Research Story**:
> * **V1** answers: *"Can the model learn semantic/code-documentation consistency under controlled conditions?"*
> * **V2 train/val** answers: *"Can the model learn from a much more realistic, large-scale software evolution dataset?"*
> * **101 verified test** answers: *"Does that learned model actually generalize to independently human-verified real-world drift?"*

---

## 📁 Directory Structure

```text
data/v2_real_world/
├── raw/
│   ├── repositories/                 # Directory junction / link to data/raw_repos/
│   └── historical_candidates.jsonl   # N = 2,367 authentic mined git evolution diffs
├── mined/
│   └── filtered_candidates.jsonl     # Filtered mined candidate pool
├── generated/
│   └── contract_grounded_drift.jsonl # N = 5,133 realistic AST contract-grounded drift samples
├── training/
│   ├── train.jsonl                   # N = 13,366 samples (function-lineage grouped, balanced)
│   └── val.jsonl                     # N = 1,430 samples (function-lineage grouped, balanced)
├── evaluation/
│   └── verified_test.jsonl           # N = 101 human-verified test samples (100% held out)
└── metadata/
    ├── dataset_summary.json          # Complete counts, provenance breakdown, leakage audit
    ├── drift_distribution.json       # Contract violation breakdown
    └── repository_distribution.json  # Repository distribution across train/val/test
```

---

## 🔬 Provenance Breakdown: Authentic vs. Contract-Grounded

SemDrift explicitly distinguishes authentic historical git commits from contract-grounded AST mutations:

| Component | Count ($N$) | Percentage of Drift | Description |
|:---|:---:|:---:|:---|
| **Authentic Historical Mined Drift** | **2,222** | 30.25% | Mined directly from Git commit history where code changed and docstrings fell out of sync |
| **AST Contract-Grounded Generated Drift** | **5,122** | 69.75% | Realistic synthetic mutations applied to authentic code using deterministic AST contract rules |
| **Total Usable Drift Positives** | **7,344** | 100.0% | Complete drift positive training/validation pool |
| **Historical Clean Negatives** | **7,452** | — | Clean code-docstring pairs (refactorings, optimizations, typing) |
| **Total Usable V2 Pool** | **14,796** | — | **50.36% Clean / 49.64% Drift** |

### AST Contract Violation Rules:
1. `parameter_contract_violation` ($N = 1,451$): Parameter dropped or renamed in signature while preserved in docstring.
2. `return_contract_violation` ($N = 1,900$): Return expression/type diverges from documented `@return` contract.
3. `raises_contract_violation` ($N = 1,642$): Documented exception `raise` statement deleted from code body.
4. `default_contract_violation` ($N = 1,430$): Parameter default argument altered while docstring maintains obsolete value.

---

## 🛡️ Zero-Leakage & Lineage Isolation Guarantee

1. **Robust Function Lineage Identity**:
   ```python
   lineage_key = f"{normalized_repo}::{normalized_file_path}::{function_name}"
   ```
   Omitting `lineno` ensures functions that move across lines between commits share the exact same identity.

2. **Pre-Generation Exclusion of the 101 Test Set**:
   Before generating V2 train/val splits, all 101 verified test lineages were extracted into an exclusion set. **204 candidate rows** matching these lineages were completely purged from the 15k pool.

3. **Mathematical Invariant Verification**:
   - $\text{Lineages}(\text{Train}) \cap \text{Lineages}(\text{Val}) = \emptyset$ (0 overlapping lineages)
   - $[\text{Lineages}(\text{Train}) \cup \text{Lineages}(\text{Val})] \cap \text{Lineages}(\text{Verified Test}) = \emptyset$ (0 overlapping lineages)

> [!CAUTION]
> **Strict Evaluation Rule**:
> The 101 verified test samples (`14 drift`, `87 clean`) must **NEVER** be used for training, hyperparameter tuning, threshold sweeping, or model selection. It is your final real-world evaluation test set.

---

## 💻 How to Load in Python

```python
import json

# 1. Load Training and Validation Sets
with open("data/v2_real_world/training/train.jsonl", "r", encoding="utf-8") as f:
    train_data = [json.loads(line) for line in f if line.strip()]

with open("data/v2_real_world/training/val.jsonl", "r", encoding="utf-8") as f:
    val_data = [json.loads(line) for line in f if line.strip()]

# 2. Load Final Evaluation Test Set (101 verified ground truth)
with open("data/v2_real_world/evaluation/verified_test.jsonl", "r", encoding="utf-8") as f:
    test_data = [json.loads(line) for line in f if line.strip()]

print(f"Train: {len(train_data)} | Val: {len(val_data)} | Verified Test: {len(test_data)}")
```
