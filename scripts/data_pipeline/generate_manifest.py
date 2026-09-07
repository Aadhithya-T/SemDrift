#!/usr/bin/env python3
"""
scripts/data_pipeline/generate_manifest.py

Strict, assertion-heavy generation of the immutable dataset manifest and SHA256SUMS.
Enforces structural correctness BEFORE computing or writing cryptographic hashes:
  1. Schema validation (explicit provenance, function_lineage, qualified_name, no drift_source)
  2. Duplicate detection (zero duplicate rows)
  3. Lineage validation (formula: repo::normalized_file::qualified_name)
  4. Disjointness validation (train ∩ val = ∅, train ∩ test = ∅, val ∩ test = ∅)
  5. Provenance distribution validation
  6. ONLY THEN generates SHA256SUMS and manifest.yaml
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Set

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.data_pipeline.setup_dataset_architecture import norm_repo, norm_file_path, get_function_lineage
from semdrift.data.integrity import compute_sha256, verify_dataset_integrity, DatasetIntegrityError

CANONICAL_DIR = PROJECT_ROOT / "experiments" / "2026-09-07_clean_v2" / "dataset"
CONFIG_DIR = PROJECT_ROOT / "experiments" / "2026-09-07_clean_v2" / "config"
MANIFEST_PATH = CONFIG_DIR / "manifest.yaml"
SHA256SUMS_PATH = CANONICAL_DIR / "SHA256SUMS"

REQUIRED_FIELDS = {
    "repo", "file_path", "function_name", "qualified_name",
    "function_lineage", "provenance", "commit_hash",
    "code", "docstring", "label", "mutation_type", "severity"
}

ALLOWED_PROVENANCE = {
    "authentic_historical_mined",
    "contract_grounded_generated",
    "clean_grounded"
}


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Canonical dataset file missing: {path}")
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f, 1):
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise ValueError(f"Corrupt JSON at {path}:{idx}: {e}")
    return rows


def validate_split(name: str, rows: List[Dict[str, Any]]) -> Set[str]:
    """Validate schema, provenance, duplicates, and compute lineage set."""
    print(f"  Validating {name} ({len(rows)} samples)...")
    lineages: Set[str] = set()
    seen_content: Set[tuple] = set()
    seen_keys: Set[tuple] = set()
    
    for idx, r in enumerate(rows):
        # 1. Schema validation
        missing = REQUIRED_FIELDS - set(r.keys())
        if missing:
            raise ValueError(f"{name} row {idx} missing required fields: {missing}")
        if "drift_source" in r:
            raise ValueError(f"{name} row {idx} contains forbidden field 'drift_source'")
            
        # 2. Provenance validation
        prov = r["provenance"]
        if prov not in ALLOWED_PROVENANCE:
            raise ValueError(f"{name} row {idx} has invalid provenance: '{prov}'")
            
        # 3. Lineage formula validation
        expected_lineage = f"{norm_repo(r['repo'])}::{norm_file_path(r['file_path'])}::{r['qualified_name']}"
        if r["function_lineage"] != expected_lineage:
            raise ValueError(
                f"{name} row {idx} lineage mismatch:\n"
                f"  Recorded: {r['function_lineage']}\n"
                f"  Expected: {expected_lineage}"
            )
        lineages.add(r["function_lineage"])
        
        # 4. Exact duplicate row validation within function instance
        content_key = (r["repo"], norm_file_path(r["file_path"]), r["qualified_name"], r["code"].strip(), r["docstring"].strip(), r["label"])
        if content_key in seen_content:
            raise ValueError(f"{name} contains exact duplicate row at row {idx}: {content_key[:3]}")
        seen_content.add(content_key)
        
    return lineages


def generate_manifest_and_sums() -> Dict[str, Any]:
    print("=" * 70)
    print("CANONICAL DATASET STRUCTURAL VALIDATION & MANIFEST GENERATION")
    print(f"Canonical Path: {CANONICAL_DIR}")
    print("=" * 70)
    
    train_file = CANONICAL_DIR / "train.jsonl"
    val_file = CANONICAL_DIR / "val.jsonl"
    test_file = CANONICAL_DIR / "verified_test.jsonl"
    
    train_rows = load_jsonl(train_file)
    val_rows = load_jsonl(val_file)
    test_rows = load_jsonl(test_file)
    
    # ------------------------------------------------------------------
    # 1. Structural Validation
    # ------------------------------------------------------------------
    print("\nPhase 1: Structural & Schema Validation")
    train_lineages = validate_split("TRAIN", train_rows)
    val_lineages = validate_split("VAL", val_rows)
    test_lineages = validate_split("TEST", test_rows)
    
    # ------------------------------------------------------------------
    # 2. Lineage Disjointness Validation
    # ------------------------------------------------------------------
    print("\nPhase 2: Disjointness Invariants")
    leak_train_val = train_lineages.intersection(val_lineages)
    leak_train_test = train_lineages.intersection(test_lineages)
    leak_val_test = val_lineages.intersection(test_lineages)
    
    if leak_train_val:
        raise ValueError(f"FATAL: Train ∩ Val lineage leak detected: {leak_train_val}")
    if leak_train_test:
        raise ValueError(f"FATAL: Train ∩ Test lineage leak detected: {leak_train_test}")
    if leak_val_test:
        raise ValueError(f"FATAL: Val ∩ Test lineage leak detected: {leak_val_test}")
        
    print(f"  [OK] Train intersect Val  == empty ({len(train_lineages)} train, {len(val_lineages)} val)")
    print(f"  [OK] Train intersect Test == empty ({len(train_lineages)} train, {len(test_lineages)} test)")
    print(f"  [OK] Val intersect Test   == empty ({len(val_lineages)} val, {len(test_lineages)} test)")
    
    # ------------------------------------------------------------------
    # 3. Counts & Distributions
    # ------------------------------------------------------------------
    print("\nPhase 3: Sample Counts & Label Distribution")
    train_labels = Counter(r["label"] for r in train_rows)
    val_labels = Counter(r["label"] for r in val_rows)
    test_labels = Counter(r["label"] for r in test_rows)
    
    train_prov = Counter(r["provenance"] for r in train_rows)
    val_prov = Counter(r["provenance"] for r in val_rows)
    test_prov = Counter(r["provenance"] for r in test_rows)
    
    print(f"  Train: {len(train_rows)} (aligned: {train_labels[0]}, drifted: {train_labels[1]})")
    print(f"  Val:   {len(val_rows)} (aligned: {val_labels[0]}, drifted: {val_labels[1]})")
    print(f"  Test:  {len(test_rows)} (aligned: {test_labels[0]}, drifted: {test_labels[1]})")
    
    # Assert exact canonical target numbers
    assert len(train_rows) == 24229, f"Expected 24,229 train rows, got {len(train_rows)}"
    assert len(val_rows) == 2636, f"Expected 2,636 val rows, got {len(val_rows)}"
    assert len(test_rows) == 104, f"Expected 104 test rows, got {len(test_rows)}"
    assert test_labels[1] == 52, f"Expected 52 test drift positives, got {test_labels[1]}"
    assert test_labels[0] == 52, f"Expected 52 test clean negatives, got {test_labels[0]}"
    assert test_prov["authentic_historical_mined"] == 52, "Expected 52 authentic historical test samples"
    
    # ------------------------------------------------------------------
    # 4. Cryptographic Hashing & SHA256SUMS Generation
    # ------------------------------------------------------------------
    print("\nPhase 4: Cryptographic Hashing (Layer A)")
    sha_train = compute_sha256(train_file)
    sha_val = compute_sha256(val_file)
    sha_test = compute_sha256(test_file)
    
    # Write SHA256SUMS
    with SHA256SUMS_PATH.open("w", encoding="utf-8") as f:
        f.write(f"{sha_train}  train.jsonl\n")
        f.write(f"{sha_val}  val.jsonl\n")
        f.write(f"{sha_test}  verified_test.jsonl\n")
        
    sha_sums = compute_sha256(SHA256SUMS_PATH)
    print(f"  [OK] Generated {SHA256SUMS_PATH}")
    print(f"       train.jsonl:         {sha_train}")
    print(f"       val.jsonl:           {sha_val}")
    print(f"       verified_test.jsonl: {sha_test}")
    print(f"       SHA256SUMS:          {sha_sums}")
    
    # ------------------------------------------------------------------
    # 5. Manifest Generation (Layer B)
    # ------------------------------------------------------------------
    print("\nPhase 5: Manifest Generation (Layer B)")
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    
    manifest_data = {
        "experiment_id": "2026-09-07_clean_v2",
        "experiment_name": "V2 Clean Baseline — Production Training Configuration",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "disclaimer": (
            "New experimental generation with AST, mutation, lineage, and truncation corrections. "
            "Features a balanced diagnostic test set (104 samples: 52 authentic historical drift, 52 clean). "
            "Cryptographically locked with two-layer integrity verification. "
            "Pre-training manifest is immutable."
        ),
        "dataset": {
            "canonical_path": "experiments/2026-09-07_clean_v2/dataset/",
            "test_type": "balanced_diagnostic",
            "train_samples": len(train_rows),
            "val_samples": len(val_rows),
            "test_samples": len(test_rows),
            "train_labels": {
                "aligned": train_labels[0],
                "drifted": train_labels[1],
            },
            "val_labels": {
                "aligned": val_labels[0],
                "drifted": val_labels[1],
            },
            "test_labels": {
                "aligned": test_labels[0],
                "drifted": test_labels[1],
            },
            "provenance_distribution": {
                "train": {
                    "clean_grounded": train_prov.get("clean_grounded", 0),
                    "contract_grounded_generated": train_prov.get("contract_grounded_generated", 0),
                    "authentic_historical_mined": train_prov.get("authentic_historical_mined", 0),
                },
                "val": {
                    "clean_grounded": val_prov.get("clean_grounded", 0),
                    "contract_grounded_generated": val_prov.get("contract_grounded_generated", 0),
                    "authentic_historical_mined": val_prov.get("authentic_historical_mined", 0),
                },
                "test": {
                    "clean_grounded": test_prov.get("clean_grounded", 0),
                    "authentic_historical_mined": test_prov.get("authentic_historical_mined", 0),
                    "contract_grounded_generated": test_prov.get("contract_grounded_generated", 0),
                },
            },
            "zero_leakage_audit": {
                "verified_test_exclusion_set": len(test_lineages),
                "purged_leaked_candidates": 71,
                "train_lineages": len(train_lineages),
                "val_lineages": len(val_lineages),
                "test_lineages": len(test_lineages),
                "train_val_overlap": 0,
                "train_test_overlap": 0,
                "val_test_overlap": 0,
                "duplicate_rows": 0,
            },
            "checksums": {
                "train_sha256": sha_train,
                "val_sha256": sha_val,
                "verified_test_sha256": sha_test,
                "sha256sums_sha256": sha_sums,
            },
        },
        "training_hyperparameters": {
            "base_model": "microsoft/codebert-base",
            "architecture": "joint_encoder",
            "pooling": "mean",
            "code_truncation": "head_tail",
            "doc_max_tokens": 96,
            "max_length": 512,
            "epochs": 3,
            "batch_size": 8,
            "learning_rate": 2.0e-5,
            "weight_decay": 0.01,
            "warmup_ratio": 0.1,
            "use_focal_loss": True,
            "focal_alpha": 0.5,
            "focal_gamma": 2.0,
            "category_weighting": True,
            "checkpoint_metric": "macro_f1",
            "bias_collapse_guard": "ratio > 0.88 skips checkpoint save",
            "seed": 42,
            "device": "cuda",
            "zero_checkpoint_reuse": True,
        }
    }
    
    with MANIFEST_PATH.open("w", encoding="utf-8") as f:
        yaml.dump(manifest_data, f, sort_keys=False, default_flow_style=False)
        
    print(f"  [OK] Generated immutable manifest at {MANIFEST_PATH}")
    
    # ------------------------------------------------------------------
    # 6. Verification of the Generated Manifest
    # ------------------------------------------------------------------
    print("\nPhase 6: Cryptographic Verification Gate Test")
    verified = verify_dataset_integrity(CANONICAL_DIR, manifest_path=MANIFEST_PATH)
    print(f"  [OK] verify_dataset_integrity PASSED for all {len(verified)} files:")
    for fn, h in verified.items():
        print(f"       {fn}: {h}")
        
    print("\n" + "=" * 70)
    print("MANIFEST & SHA256SUMS REGENERATION COMPLETE & VERIFIED")
    print("=" * 70)
    return manifest_data


if __name__ == "__main__":
    generate_manifest_and_sums()
