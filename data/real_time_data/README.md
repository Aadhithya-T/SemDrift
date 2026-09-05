# Real-Time Python Documentation Drift Dataset (15,000 Samples)

This directory contains the unified, consolidated **15,000 real-time dataset** mined from the full Git commit histories of 7 mature Python open-source repositories (**Django**, **SQLAlchemy**, **Pytest**, **Celery**, **Tornado**, **Click**, and **FastAPI**).

---

## 📁 Files in this Directory

```
data/real_time_data/
├── real_time_dataset.jsonl            # Unified 15,000 sample dataset (571 drift positives + 14,429 clean negatives)
├── real_time_drift_positives.jsonl    # All 571 AST-verified drift positives
├── real_time_clean_negatives.jsonl    # 14,429 clean refactorings and non-drift updates
├── real_time_summary.json             # Complete metadata summary and per-repository stats
└── README.md                          # Documentation and usage guide
```

---

## 📊 Summary Statistics

| Repository | Total Samples | Drift Positives (+) | Clean Negatives (-) | Positive Rate | Key Domain |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Django** | 7,355 | 5 | 7,350 | 0.07% | Web framework, ORM, templating |
| **SQLAlchemy** | 4,104 | 381 | 3,723 | 9.28% | Core SQL expressions, ORM, async engines |
| **Pytest** | 1,591 | 79 | 1,512 | 4.97% | Test fixtures, AST assertions, runners |
| **Celery** | 839 | 9 | 830 | 1.07% | Distributed task queues, worker loops |
| **Tornado** | 729 | 76 | 653 | 10.43% | Async I/O, coroutines, web servers |
| **Click** | 306 | 21 | 285 | 6.86% | CLI option decorators, type parsers |
| **FastAPI** | 76 | 0 | 76 | 0.00% | ASGI routing, dependency injection |
| **TOTAL** | **15,000** | **571** | **14,429** | **3.81%** | **7 Major Python Ecosystems** |

---

## 💻 How to Load in Python / Hugging Face

```python
import json
from datasets import Dataset

# Load the full 15k dataset
with open("data/real_time_data/real_time_dataset.jsonl", "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f if line.strip()]

dataset = Dataset.from_list(data)
print(f"Loaded {len(dataset)} real-time samples!")

# Access fields:
# - sample['function_name']
# - sample['repo']
# - sample['code_before'], sample['code_after']
# - sample['docstring_before'], sample['docstring_after']
# - sample['filtered_label'] ('drift' or 'no_drift')
# - sample['pseudo_label'] (1 for drift, 0 for clean)
# - sample['filtered_drift_probability']
# - sample['ast_violations']
```
