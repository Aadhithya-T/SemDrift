# SemDrift Project Analysis

**Status date:** 2026-09-04

## Purpose
SemDrift detects semantic drift between Python function code and its docstrings: cases where documentation no longer describes the implementation. It is a research prototype combining AST extraction, synthetic mutation generation, CodeBERT models, evaluation scripts, and a repository scanner.

## Architecture and flow
1. `semdrift.parser` extracts functions with Python's `ast` parser, separates code from docstrings, records metadata, and normalizes Google, NumPy, Sphinx, or plain docstrings. A tree-sitter `UniversalParser` exists but currently supports Python only.
2. Dataset scripts in `scripts/` extract pairs, filter them, inject mutations, convert records to labels, and split by stable original-function identity. The split design keeps mutations of one function in one partition to reduce leakage.
3. CodeBERT (`microsoft/codebert-base`) provides embeddings or learned representations. The zero-shot/Model A path compares independently encoded code and documentation with cosine divergence. The dual encoder independently encodes both inputs and classifies `[u; v; |u-v|]`. The joint encoder sends one sequence through CodeBERT, preserves code tails with head-tail truncation, and classifies the `[CLS]` representation.
4. `scripts/scan_repo.py` parses documented functions, loads a joint-encoder checkpoint, predicts drift, and can provide interactive, Markdown, or JSON output.
5. Analysis scripts calculate standard metrics, drift-type/severity/repository breakdowns, bootstrap confidence intervals, McNemar tests, and IEEE LaTeX tables.

## Data and reported results
The dataset architecture is organized into two clearly separated generations:

1. **V1 — Synthetic Dataset (`data/v1_synthetic/`)**:
   - Controlled evaluation benchmark of **1,205** examples (597 aligned, 608 synthetic drift across `param_rename`, `return_value_change`, `doc_sentence_delete`, and `doc_negation`).
   - Reference aligned functions: `raw/original_aligned.jsonl` (597 functions).
   - Controlled ablation training splits: `ablation/train.jsonl` (9,638), `ablation/val.jsonl` (1,259), and `ablation/test.jsonl` (1,205).
   - Controlled ablation results on V1 (CE objective, seed=42):
     - Zero-shot dual encoder: accuracy 44.56%, F1 36.86%.
     - Fine-tuned dual encoder (CE): accuracy 80.91%, F1 78.34%.
     - Fine-tuned joint encoder (CE): accuracy 85.06%, F1 83.67%, macro-F1 84.95%.
     - Reported joint-vs-dual McNemar result: chi-square 18.76, p = 1.48e-05 (statistically significant).

2. **V2 — Real-World-Grounded Dataset (`data/v2_real_world/`)**:
   - Main training pool of **14,799** samples partitioned into `training/train.jsonl` (13,350) and `training/val.jsonl` (1,449) using qualified function-lineage grouping (`repo::file_path::qualified_function_name`, e.g. `ClassName.method`).
   - Provenance explicitly distinguished:
     - **2,222** authentic mined Git evolution commits (`drift_source: "authentic_historical_mined"`).
     - **5,122** realistic AST contract-grounded drift mutations (`drift_source: "contract_grounded_generated"`).
     - **7,455** confirmed clean negative samples.
   - Final evaluation test set: `evaluation/verified_test.jsonl` with **101** 100% human-verified samples (14 drift, 87 clean).
   - Zero-leakage guarantee: All 101 test function lineages were purged from V2 prior to training/validation generation (eliminating 201 candidate rows). Overlap between train/val and test is mathematically 0.

Datasets and metadata are managed via `scripts/data_pipeline/setup_dataset_architecture.py` and validated by `tests/test_dataset_architecture.py`.

## Current implementation status
The parser and model utility modules contain substantial implementation and are the strongest usable parts of the project. Training and evaluation scripts exist at the top level of `scripts/`, including Java extraction/parser test utilities. However, `semdrift.pipeline.Pipeline` is only a skeleton: `_parse`, `_embed`, and `_compare` raise `NotImplementedError`, and the comparator package has no visible concrete scoring implementation in its package initializer. The documented end-to-end `Pipeline` API therefore is not operational as written; practical execution currently goes through the scripts and direct parser/model APIs.

## How to run
Install dependencies with `pip install -r requirements.txt`. The intended test command is `python -m pytest tests`. Dataset rebuilding uses the extraction, filtering, mutation, conversion, and split scripts. Model training requires downloading CodeBERT and is configured for CUDA by default in the documented commands; CPU fallback exists in parts of the Python code but may be slow. Repository scanning requires a compatible joint-encoder checkpoint, for example `python scripts/scan_repo.py . --interactive`.

## Verification and risks
The documented pytest command was attempted on the current environment but could not start because the active interpreter (`C:\Python313\python.exe`) does not have `pytest` installed. No test result should therefore be treated as passing in this environment. Main risks are stale README paths, the unwired public pipeline, dependency/environment drift, GPU/checkpoint requirements, synthetic mutations that may not represent all real-world documentation errors, and the low performance on negation mutations. The benchmark numbers are repository-reported artifacts and should be reproduced after aligning scripts, dependencies, checkpoints, and dataset versions.
