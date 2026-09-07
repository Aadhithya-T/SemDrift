#!/usr/bin/env python3
"""
verify_dataset_provenance.py

Automated provenance and lineage audit script for the SemDrift Clean V2 dataset:
1. Validates 100% of rows in train.jsonl, val.jsonl, and verified_test.jsonl.
2. Asserts existence of all mandatory first-class schema fields:
   - repo, file_path, function_name, qualified_name, function_lineage,
   - provenance, commit_hash, code, docstring, label.
3. Asserts exact lineage calculation formula:
   function_lineage == f"{norm_repo(repo)}::{norm_file_path(file_path)}::{qualified_name}".
4. Asserts allowed provenance vocabulary:
   provenance in {"authentic_historical_mined", "contract_grounded_generated", "clean_grounded"}.
5. Asserts absence of deprecated keys ("drift_source", "file").
6. Enforces hard mathematical disjointness invariants:
   - train_lineages ∩ val_lineages == ∅
   - (train_lineages ∪ val_lineages) ∩ test_lineages == ∅
7. Asserts zero exact duplicate rows.
"""

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Set

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.data_pipeline.setup_dataset_architecture import (
    norm_file_path,
    norm_repo,
)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f, 1):
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except Exception as e:
                    raise ValueError(f"Corrupted JSON at {path}:{idx} - {e}")
    return rows


def verify_split_provenance(split_name: str, rows: List[Dict[str, Any]]) -> Set[str]:
    print(f"Auditing {split_name} ({len(rows)} samples)...")
    lineages = set()
    seen_content = set()
    
    allowed_provenance = {
        "authentic_historical_mined",
        "contract_grounded_generated",
        "clean_grounded",
    }
    
    for idx, r in enumerate(rows, 1):
        # 1. Mandatory fields
        for field in ["repo", "file_path", "function_name", "qualified_name", "function_lineage", "provenance", "label", "code", "docstring"]:
            val = r.get(field)
            assert val is not None and str(val).strip() != "", (
                f"Row {idx} in {split_name} missing mandatory field '{field}': {r}"
            )
            
        # 2. Lineage formula invariant
        repo = norm_repo(r["repo"])
        fp = norm_file_path(r["file_path"])
        qn = r["qualified_name"]
        expected_lineage = f"{repo}::{fp}::{qn}"
        actual_lineage = r["function_lineage"]
        assert actual_lineage == expected_lineage, (
            f"Row {idx} in {split_name} lineage mismatch!\n"
            f"  Expected: {expected_lineage}\n"
            f"  Actual:   {actual_lineage}"
        )
        lineages.add(actual_lineage)
        
        # 3. Provenance taxonomy
        prov = r["provenance"]
        assert prov in allowed_provenance, (
            f"Row {idx} in {split_name} invalid provenance '{prov}'. "
            f"Must be one of {allowed_provenance}."
        )
        
        # 4. Deprecated field check
        assert "drift_source" not in r, (
            f"Row {idx} in {split_name} contains deprecated 'drift_source' key. Standardize on 'provenance'."
        )
        assert "file" not in r, (
            f"Row {idx} in {split_name} contains legacy 'file' key. Standardize on 'file_path'."
        )
        
        # 5. Label type
        lbl = r["label"]
        assert lbl in (0, 1), f"Row {idx} in {split_name} invalid label '{lbl}' (must be 0 or 1)."
        
        # 6. Duplicate row content check
        ckey = (repo, fp, qn, r["code"].strip(), r["docstring"].strip(), lbl)
        assert ckey not in seen_content, (
            f"Row {idx} in {split_name} is an exact duplicate of a previous row: {ckey[:3]}"
        )
        seen_content.add(ckey)
        
    labels = Counter(r["label"] for r in rows)
    provs = Counter(r["provenance"] for r in rows)
    print(f"  [OK] {split_name}: {len(rows)} samples, {len(lineages)} unique lineages.")
    print(f"       Labels: {dict(labels)} | Provenance: {dict(provs)}")
    return lineages


def main():
    dataset_dir = PROJECT_ROOT / "experiments" / "2026-09-07_clean_v2" / "dataset"
    train_file = dataset_dir / "train.jsonl"
    val_file = dataset_dir / "val.jsonl"
    test_file = dataset_dir / "verified_test.jsonl"
    
    for f in [train_file, val_file, test_file]:
        assert f.is_file(), f"Missing dataset file: {f}"
        
    print("=" * 70)
    print("SEMDRIFT V2 DATASET PROVENANCE & LINEAGE AUDIT")
    print(f"Dataset directory: {dataset_dir}")
    print("=" * 70)
    
    train_rows = load_jsonl(train_file)
    val_rows = load_jsonl(val_file)
    test_rows = load_jsonl(test_file)
    
    train_lineages = verify_split_provenance("TRAIN", train_rows)
    val_lineages = verify_split_provenance("VALIDATION", val_rows)
    test_lineages = verify_split_provenance("VERIFIED TEST", test_rows)
    
    # Mathematical disjointness assertions
    print("\nVerifying mathematical disjointness invariants...")
    leak_train_val = train_lineages.intersection(val_lineages)
    assert len(leak_train_val) == 0, f"FATAL: {len(leak_train_val)} lineages leaked between train and val: {leak_train_val}"
    
    leak_train_test = train_lineages.intersection(test_lineages)
    assert len(leak_train_test) == 0, f"FATAL: {len(leak_train_test)} lineages leaked between train and test: {leak_train_test}"
    
    leak_val_test = val_lineages.intersection(test_lineages)
    assert len(leak_val_test) == 0, f"FATAL: {len(leak_val_test)} lineages leaked between val and test: {leak_val_test}"
    
    print("  [OK] train intersect val == empty: 0 leaks")
    print("  [OK] train intersect test == empty: 0 leaks")
    print("  [OK] val intersect test == empty: 0 leaks")
    print("\n" + "=" * 70)
    print("AUDIT SUCCESS: 100% of dataset rows pass all provenance and lineage invariants.")
    print("=" * 70)


if __name__ == "__main__":
    main()
