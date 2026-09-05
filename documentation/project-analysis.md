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
The V2 benchmark is documented as a clean 10-repository test set with **1,205** examples. Main drift categories are `param_rename`, `return_value_change`, `doc_sentence_delete`, and `doc_negation`, with `aligned` negatives. The README reports:

- Zero-shot dual encoder: accuracy 44.56%, F1 36.86%.
- Fine-tuned dual encoder: accuracy 80.41%, F1 77.61%.
- Fine-tuned joint encoder: accuracy 85.06%, F1 83.58%, macro-F1 84.94%.
- Reported joint-vs-zero-shot McNemar result: chi-square 372.91, p = 4.36e-83.
- `doc_negation` is the weakest reported joint category (F1 31.86%), so it remains a key robustness concern.

Datasets, predictions, checkpoints/results, JSON benchmark data, and LaTeX tables are stored under `data/`, especially `data/experiments/v2/`. Raw Python and Java repositories are also present. The checked-in README and actual workspace layout differ in places: current workflow scripts are directly under `scripts/`, while the README describes subdirectories such as `scripts/training` and `scripts/data_pipeline`.

## Current implementation status
The parser and model utility modules contain substantial implementation and are the strongest usable parts of the project. Training and evaluation scripts exist at the top level of `scripts/`, including Java extraction/parser test utilities. However, `semdrift.pipeline.Pipeline` is only a skeleton: `_parse`, `_embed`, and `_compare` raise `NotImplementedError`, and the comparator package has no visible concrete scoring implementation in its package initializer. The documented end-to-end `Pipeline` API therefore is not operational as written; practical execution currently goes through the scripts and direct parser/model APIs.

## How to run
Install dependencies with `pip install -r requirements.txt`. The intended test command is `python -m pytest tests`. Dataset rebuilding uses the extraction, filtering, mutation, conversion, and split scripts. Model training requires downloading CodeBERT and is configured for CUDA by default in the documented commands; CPU fallback exists in parts of the Python code but may be slow. Repository scanning requires a compatible joint-encoder checkpoint, for example `python scripts/scan_repo.py . --interactive`.

## Verification and risks
The documented pytest command was attempted on the current environment but could not start because the active interpreter (`C:\Python313\python.exe`) does not have `pytest` installed. No test result should therefore be treated as passing in this environment. Main risks are stale README paths, the unwired public pipeline, dependency/environment drift, GPU/checkpoint requirements, synthetic mutations that may not represent all real-world documentation errors, and the low performance on negation mutations. The benchmark numbers are repository-reported artifacts and should be reproduced after aligning scripts, dependencies, checkpoints, and dataset versions.
