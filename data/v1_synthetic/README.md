# SemDrift V1 — Synthetic Dataset (Controlled Baseline & Ablation Benchmark)

**Purpose**: Controlled experimentation to evaluate whether neural models can learn semantic code–documentation consistency under mathematically controlled, synthetic conditions.

---

## 🎯 Dataset Roles & Research Objective

> **Research Question**: *"Can the model learn semantic code-documentation consistency under controlled mutation conditions?"*

The V1 dataset serves as SemDrift's **baseline benchmark**. It isolates architectural choices (e.g., Independent Dual Encoding vs. Joint Self-Attention) by applying deterministic mutation operators to clean, aligned code–docstring pairs across 10 mature open-source repositories.

---

## 📁 Directory Structure

```text
data/v1_synthetic/
├── benchmark/
│   └── synthetic_dataset.jsonl       # N = 1,205 controlled evaluation benchmark
├── raw/
│   └── original_aligned.jsonl       # N = 597 original aligned functions (pre-mutation)
├── ablation/
│   ├── train.jsonl                   # N = 9,638 controlled training samples
│   ├── val.jsonl                     # N = 1,259 controlled validation samples
│   └── test.jsonl                    # N = 1,205 controlled test samples (mirrors benchmark)
└── metadata/
    ├── dataset_summary.json          # High-level metrics, repositories, split counts
    └── mutation_distribution.json   # Exact breakdown by mutation category
```

---

## 📊 Summary Statistics

### 1. Controlled Benchmark Pool ($N = 1,205$)

* **Aligned Clean Samples**: **597** ($49.54\%$)
* **Synthetic Drift Samples**: **608** ($50.46\%$)
* **Total Instances**: **1,205** ($100.00\%$)

### 2. Mutation Operator Distribution

| Mutation Category | Count ($N$) | Percentage | Description |
|:---|:---:|:---:|:---|
| **`param_rename`** | 221 | 36.35% | Function parameter renamed or removed in signature but preserved in docstring |
| **`doc_sentence_delete`** | 153 | 25.16% | Informative sentence stripped from docstring body while code remains unchanged |
| **`return_value_change`** | 139 | 22.86% | Return signature/expression altered to contradict docstring return specification |
| **`doc_negation`** | 95 | 15.62% | Polarity particles (*"not"*, *"never"*, *"disabled"*) inverted in docstring |
| **Total Drift** | **608** | **100.0%** | Comprehensive controlled semantic drift |

### 3. Ablation Pool Splits (Phase 1 Experiments)

* **Train Partition (`ablation/train.jsonl`)**: **9,638** instances
* **Validation Partition (`ablation/val.jsonl`)**: **1,259** instances
* **Test Partition (`ablation/test.jsonl`)**: **1,205** instances
* **Total Ablation Pool**: **12,102** instances

---

## 💻 How to Load in Python

```python
import json

# 1. Load the 1,205 Benchmark
with open("data/v1_synthetic/benchmark/synthetic_dataset.jsonl", "r", encoding="utf-8") as f:
    benchmark_data = [json.loads(line) for line in f if line.strip()]

print(f"Loaded V1 Benchmark: {len(benchmark_data)} samples")

# 2. Accessing Fields
sample = benchmark_data[0]
print("Function:", sample["function_name"])
print("Repo:", sample["repo"])
print("Label:", sample["label"])        # "aligned" or "drifted"
print("Drift Type:", sample.get("drift_type"))
```
