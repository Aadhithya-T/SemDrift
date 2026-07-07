# SemDrift

**Semantic Drift Detection in Code Repositories**

SemDrift is a three-stage pipeline that detects semantic drift — meaningful behavioural changes in code that go beyond surface-level diffs — by combining AST parsing, transformer-based embeddings, and vector comparison.

---

## Project Structure

```
semdrift/
├── semdrift/
│   ├── __init__.py
│   ├── parser/          # AST parsing stage
│   │   └── __init__.py
│   ├── embedder/        # Transformer / embedding stage
│   │   └── __init__.py
│   ├── comparator/      # Comparison / scoring stage
│   │   └── __init__.py
│   └── pipeline.py      # Ties all 3 stages together
├── tests/
│   ├── test_parser.py
│   ├── test_embedder.py
│   └── test_comparator.py
├── data/
│   ├── raw/             # Mined git history / raw repo data
│   ├── labeled/         # Final labeled dataset
│   └── synthetic/       # Injected-drift examples
├── scripts/
│   ├── mine_dataset.py  # PyDriller mining script
│   └── run_eval.py      # Evaluation script
├── notebooks/           # Exploration, not production code
├── .gitignore
├── requirements.txt
├── README.md
└── config.yaml          # Shared thresholds, model names, paths
```

## Pipeline Overview

| Stage        | Module              | Responsibility                                    |
| ------------ | ------------------- | ------------------------------------------------- |
| **Parser**   | `semdrift.parser`   | Extract AST representations from source code      |
| **Embedder** | `semdrift.embedder` | Generate vector embeddings via transformer models  |
| **Comparator** | `semdrift.comparator` | Compare embeddings and produce drift scores    |

## Quick Start

```bash
# 1. Clone the repository
git clone <repo-url> && cd semdrift

# 2. Create a virtual environment & install dependencies
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3. Run tests
pytest tests/ -v

# 4. Mine data (once implemented)
python scripts/mine_dataset.py --repo /path/to/repo --output data/raw/

# 5. Evaluate (once implemented)
python scripts/run_eval.py --data data/labeled/ --config config.yaml
```

## Configuration

All shared settings live in [`config.yaml`](config.yaml) — model names, thresholds, paths, and evaluation metrics.

## Contributing

Each pipeline stage is owned by a different teammate:

| Stage      | Owner      |
| ---------- | ---------- |
| Parser     | Teammate 1 |
| Embedder   | Teammate 2 |
| Comparator | Teammate 3 |

Please coordinate via pull requests and keep `config.yaml` in sync.

## License

TBD
