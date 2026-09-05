# Real-Time Balanced Python Documentation Drift Dataset (15,000 Samples)

This directory contains the balanced **15,000 real-time dataset** (50% Drift Positives, 50% Clean Negatives) tailored specifically for training neural classifiers such as CodeBERT, GraphCodeBERT, and RoBERTa.

---

## 📁 Files in this Directory

```
data/real_time_data/
├── real_time_dataset.jsonl            # Balanced 15,000 sample dataset (7,500 drift + 7,500 clean)
├── real_time_drift_positives.jsonl    # 7,500 Drift Positives (+)
├── real_time_clean_negatives.jsonl    # 7,500 Clean Negatives (-)
├── real_time_summary.json             # Complete metadata summary and per-repository breakdown
└── README.md                          # Documentation and dataset usage guide
```

---

## 📊 Summary Statistics (50/50 Class Balance)

| Repository | Total Samples | Drift Positives (+) | Clean Negatives (-) | Positive Rate | Key Domain |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Django** | 5,083 | 2,659 | 2,424 | 52.31% | Web framework, ORM, templating |
| **Pandas** | 2,648 | 1,262 | 1,386 | 47.66% | High-performance data manipulation |
| **SQLAlchemy** | 2,385 | 1,149 | 1,236 | 48.18% | Core SQL expressions, ORM, async engines |
| **Scikit-Learn** | 1,681 | 860 | 821 | 51.16% | Machine learning algorithms & estimators |
| **Pytest** | 1,285 | 596 | 689 | 46.38% | Test fixtures, AST assertions, runners |
| **NumPy** | 1,150 | 557 | 593 | 48.43% | Numerical computing, arrays, linear algebra |
| **Click** | 319 | 169 | 150 | 52.98% | CLI option decorators, type parsers |
| **Flask** | 161 | 65 | 96 | 40.37% | WSGI microframework & routing |
| **Requests** | 127 | 59 | 68 | 46.46% | HTTP client & authentication handling |
| **FastAPI** | 76 | 39 | 37 | 51.32% | ASGI routing, dependency injection |
| **Tornado** | 76 | 76 | 0 | 100.0% | Async I/O, coroutines, web servers |
| **Celery** | 9 | 9 | 0 | 100.0% | Distributed task queues, worker loops |
| **TOTAL** | **15,000** | **7,500** | **7,500** | **50.00%** | **Comprehensive Python Ecosystem** |

---

## 🔬 Drift Curation Methodology

1. **Authentic Mined Drift (Git Commits)**: Mined real-world historical commits where docstrings fell out of sync with code modifications.
2. **AST Contract Grounded Drift Mutations**: Realistic, programmatically grounded mutations applied to authentic code-docstring pairs:
   - **Parameter Contract Violations**: Parameter removed or renamed in signature while preserved in docstring.
   - **Default Value Violations**: Default argument changed (`timeout=30` $\to$ `None`) while docstring maintains obsolete default.
   - **Return Contract Violations**: Return type annotation or return expression altered while docstring claims original type.
   - **Exception Contract Violations**: Documented `raise` statements removed or unhandled exceptions introduced.
   - **Logic & Branch Divergence**: Stale docstring summaries contradicting updated boolean/conditional return logic.

---

## 💻 How to Load in Python / Hugging Face

```python
import json
from datasets import Dataset

# Load the balanced 15k dataset
with open("data/real_time_data/real_time_dataset.jsonl", "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f if line.strip()]

dataset = Dataset.from_list(data)
print(f"Loaded {len(dataset)} balanced training samples!")

# Access fields:
# - sample['function_name']
# - sample['repo']
# - sample['code_before'], sample['code_after']
# - sample['docstring_before'], sample['docstring_after']
# - sample['pseudo_label'] (1 for drift positive, 0 for clean negative)
# - sample['filtered_drift_probability']
# - sample['curation_method']
```
