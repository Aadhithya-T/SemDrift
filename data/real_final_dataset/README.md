# SemDrift Real-World Dataset

This dataset contains **5,200** high-quality, verified code-documentation semantic drift and consistency samples collected across **19** major open-source Python repositories.

## Structure

- `drift_samples/drift_samples.jsonl`: **2,600** verified semantic drift positive samples (`drift_label: 1`).
- `consistent_samples/consistent_samples.jsonl`: **2,600** verified consistent samples (`drift_label: 0`).
- `splits/train.jsonl`: **4,606** samples.
- `splits/val.jsonl`: **393** samples.
- `splits/test.jsonl`: **201** samples.
- `real_final_dataset.jsonl`: Complete combined dataset.
- `dataset_summary.json`: Detailed statistical summary and class balances.
- `drift_distribution.json`: Breakdown of semantic drift archetypes.
- `repository_distribution.json`: Sample counts per repository.

## Schema

```json
{
  "id": "semdrift_drift_000123",
  "repository": "psf/requests",
  "commit_hash": "a8f39b1e",
  "filepath": "requests/sessions.py",
  "function_name": "send",
  "code": "def send(self, request, stream=False, timeout=None, verify=True): ...",
  "docstring": "Sends a given PreparedRequest. Returns Response object.",
  "drift_label": 1,
  "drift_type": "parameter_missing",
  "drift_explanation": "Parameter(s) ['stream', 'verify'] added to function signature but absent from docstring specification.",
  "violation_details": {
    "undocumented_params": ["stream", "verify"],
    "stale_params": [],
    "return_type_mismatch": false
  },
  "split": "train"
}
```

## Class Distribution

- **Semantic Drift Positives**: 2,600 (50.0%)
- **Consistent Negatives**: 2,600 (50.0%)
- **Repository Isolation**: Training, validation, and testing partitions are partitioned strictly at the repository boundary to prevent data leakage and ensure authentic real-world generalization.
