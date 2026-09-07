#!/usr/bin/env python3
"""
split_clean_v2_dataset.py

Performs zero-leakage, qualified-lineage-grouped train/val splitting of the
clean V2 regenerated dataset:
1. Loads the 101-instance verified test anchor from experiments/2026-09-07_clean_v2/dataset/verified_test.jsonl
2. Constructs the locked Exclusion Set of 101 test lineages
3. Loads regenerated mutated dataset (experiments/2026-09-07_clean_v2/dataset/mutated_dataset.jsonl)
4. Purges all candidate samples sharing a qualified function lineage with test set
5. Groups surviving clean pool by qualified function lineage
6. Partitions into 90% train / 10% val via deterministic SHA256 hashing on lineage key
7. Verifies hard mathematical invariants:
   - len(verified_test) == 101
   - len(train) > 0, len(val) > 0
   - train_lineages ∩ val_lineages == ∅
   - (train_lineages ∪ val_lineages) ∩ test_lineages == ∅
8. Writes canonical splits to experiments/2026-09-07_clean_v2/dataset/
9. Synchronizes exact byte copies to data/v2_real_world/training/
10. Computes and saves SHA256SUMS for train.jsonl, val.jsonl, verified_test.jsonl
"""

import hashlib
import json
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.data_pipeline.setup_dataset_architecture import (
    get_function_lineage,
    resolve_qualified_name,
)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def split_clean_v2():
    exp_dir = PROJECT_ROOT / "experiments" / "2026-09-07_clean_v2" / "dataset"
    test_file = exp_dir / "verified_test.jsonl"
    mutated_file = exp_dir / "mutated_dataset.jsonl"
    
    assert test_file.is_file(), f"Missing verified test anchor: {test_file}"
    assert mutated_file.is_file(), f"Missing mutated dataset: {mutated_file}"
    
    # 1. Load and verify 101 test anchor
    test_rows = load_jsonl(test_file)
    assert len(test_rows) == 101, f"Expected exactly 101 test rows, got {len(test_rows)}"
    test_labels = Counter(r.get("label") for r in test_rows)
    assert test_labels[1] == 14, f"Expected 14 drift test samples, got {test_labels[1]}"
    assert test_labels[0] == 87, f"Expected 87 clean test samples, got {test_labels[0]}"
    
    for r in test_rows:
        if not r.get("qualified_name"):
            r["qualified_name"] = resolve_qualified_name(r)
            r["qualified_function_name"] = r["qualified_name"]
            
    test_lineages: Set[str] = {get_function_lineage(r) for r in test_rows}
    assert len(test_lineages) == 101, f"Expected 101 unique test lineages, got {len(test_lineages)}"
    print(f"[OK] Locked 101 test lineages from {test_file}")
    
    # 2. Load regenerated mutated dataset
    all_mutated = load_jsonl(mutated_file)
    print(f"[OK] Loaded {len(all_mutated)} candidates from {mutated_file}")
    
    # 3. Purge test lineages (Zero-Leakage Enforcement)
    clean_pool: List[Dict[str, Any]] = []
    purged_count = 0
    purged_labels = Counter()
    
    for r in all_mutated:
        if not r.get("qualified_name"):
            r["qualified_name"] = resolve_qualified_name(r)
            r["qualified_function_name"] = r["qualified_name"]
        lin = get_function_lineage(r)
        if lin in test_lineages:
            purged_count += 1
            purged_labels[r.get("label")] += 1
        else:
            clean_pool.append(r)
            
    print(f"[OK] Zero-Leakage Purge: Removed {purged_count} candidate rows sharing qualified lineage with 101 test set.")
    print(f"     Purged breakdown: {dict(purged_labels)}")
    print(f"[OK] Usable clean pool size: {len(clean_pool)}")
    
    # 4. Group by qualified function lineage
    lineage_groups = defaultdict(list)
    for r in clean_pool:
        lin = get_function_lineage(r)
        lineage_groups[lin].append(r)
        
    print(f"[OK] Grouped into {len(lineage_groups)} unique qualified function lineages.")
    
    # 5. Deterministic SHA-256 Splitting (90% Train, 10% Val)
    train_rows: List[Dict[str, Any]] = []
    val_rows: List[Dict[str, Any]] = []
    
    for lin, rows in lineage_groups.items():
        h = int(hashlib.sha256(lin.encode("utf-8")).hexdigest(), 16)
        if (h % 1000) < 900:
            train_rows.extend(rows)
        else:
            val_rows.extend(rows)
            
    # 6. Hard Invariant Assertions
    assert len(train_rows) > 0, "Train rows must be > 0"
    assert len(val_rows) > 0, "Val rows must be > 0"
    assert len(train_rows) + len(val_rows) == len(clean_pool)
    
    train_lineages = {get_function_lineage(r) for r in train_rows}
    val_lineages = {get_function_lineage(r) for r in val_rows}
    
    train_val_overlap = train_lineages.intersection(val_lineages)
    assert len(train_val_overlap) == 0, f"FATAL: Train/Val lineage leak detected: {train_val_overlap}"
    
    train_test_overlap = train_lineages.intersection(test_lineages)
    assert len(train_test_overlap) == 0, f"FATAL: Train/Test lineage leak detected: {train_test_overlap}"
    
    val_test_overlap = val_lineages.intersection(test_lineages)
    assert len(val_test_overlap) == 0, f"FATAL: Val/Test lineage leak detected: {val_test_overlap}"
    
    print(f"[OK] Mathematical Invariants Verified:")
    print(f"     Train lineages: {len(train_lineages)}")
    print(f"     Val lineages:   {len(val_lineages)}")
    print(f"     Test lineages:  {len(test_lineages)}")
    print(f"     train intersect val == empty: PASSED (0 leaks)")
    print(f"     train intersect test == empty: PASSED (0 leaks)")
    print(f"     val intersect test == empty: PASSED (0 leaks)")
    
    train_labels = Counter(r.get("label") for r in train_rows)
    val_labels = Counter(r.get("label") for r in val_rows)
    print(f"[OK] Train split: {len(train_rows)} samples (label 0: {train_labels[0]}, label 1: {train_labels[1]})")
    print(f"[OK] Val split:   {len(val_rows)} samples (label 0: {val_labels[0]}, label 1: {val_labels[1]})")
    
    # 7. Write to canonical snapshot
    canonical_train = exp_dir / "train.jsonl"
    canonical_val = exp_dir / "val.jsonl"
    write_jsonl(canonical_train, train_rows)
    write_jsonl(canonical_val, val_rows)
    print(f"[OK] Wrote canonical train to {canonical_train}")
    print(f"[OK] Wrote canonical val to {canonical_val}")
    
    # 8. Synchronize to data/v2_real_world/training/ as exact byte copies
    working_dir = PROJECT_ROOT / "data" / "v2_real_world" / "training"
    working_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(canonical_train, working_dir / "train.jsonl")
    shutil.copyfile(canonical_val, working_dir / "val.jsonl")
    print(f"[OK] Copied exact bytes to {working_dir}")
    
    # Also ensure data/v2_real_world/evaluation/verified_test.jsonl matches canonical
    eval_dir = PROJECT_ROOT / "data" / "v2_real_world" / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(test_file, eval_dir / "verified_test.jsonl")
    
    # 9. Compute SHA256SUMS
    sha_train = sha256_file(canonical_train)
    sha_val = sha256_file(canonical_val)
    sha_test = sha256_file(test_file)
    
    sums_file = exp_dir / "SHA256SUMS"
    with sums_file.open("w", encoding="utf-8") as f:
        f.write(f"{sha_train}  train.jsonl\n")
        f.write(f"{sha_val}  val.jsonl\n")
        f.write(f"{sha_test}  verified_test.jsonl\n")
        
    print(f"[OK] Wrote SHA256SUMS to {sums_file}:")
    print(f"     train.jsonl:         {sha_train}")
    print(f"     val.jsonl:           {sha_val}")
    print(f"     verified_test.jsonl: {sha_test}")
    
    return {
        "train_samples": len(train_rows),
        "val_samples": len(val_rows),
        "test_samples": len(test_rows),
        "purged_count": purged_count,
        "train_labels": dict(train_labels),
        "val_labels": dict(val_labels),
        "sha256": {
            "train": sha_train,
            "val": sha_val,
            "test": sha_test
        }
    }


if __name__ == "__main__":
    res = split_clean_v2()
    print("\nDataset split completed successfully:")
    print(json.dumps(res, indent=2))
