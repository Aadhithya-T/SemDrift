# SemDrift Expanded Benchmark Dataset (10,000+ Samples)

## Overview
The SemDrift Expanded Dataset expands the benchmark to **10,000 verified real-world code–documentation pairs** across **33 major open-source repositories**, maintaining a balanced 50/50 distribution across 13 semantic drift categories and consistent pairs.

## Dataset Structure
- `combined/expanded_final_dataset.jsonl` (.csv): Complete combined benchmark (**10,000 samples**).
- `drift_samples/drift_samples.jsonl` (.csv): Expansion semantic drift positive samples (**5,000 samples**).
- `consistent_samples/consistent_samples.jsonl` (.csv): Expansion consistent samples (**5,000 samples**).
- `splits/train.jsonl` (.csv): Repository-isolated training partition.
- `splits/val.jsonl` (.csv): Validation partition (held-out repositories).
- `splits/test.jsonl` (.csv): Evaluation partition (strictly held-out test repositories).
- `audits/quality_audit_report.json`: Verification and leakage audit results.

## Key Properties
- **Exact Sample Count**: 10,000 (50% Drift / 50% Consistent).
- **Zero Leakage**: 100% Repository-isolated train/val/test partitions.
- **Zero Duplicate Overlap**: Full deduplication against the frozen baseline.
- **Expanded Taxonomy**: 13 semantic drift archetypes (parameter constraints, default values, return semantics, exception contracts, mutation behaviors, async execution, etc.).
