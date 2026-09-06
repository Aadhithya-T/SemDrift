#!/usr/bin/env python3
"""
setup_dataset_architecture.py

Implements the official SemDrift Two-Generation Dataset Architecture:
  1. V1 — Synthetic Dataset (Controlled Baseline & Ablation Benchmark)
     data/v1_synthetic/
     ├── benchmark/
     │   └── synthetic_dataset.jsonl       (N = 1,205 controlled benchmark)
     ├── raw/
     │   └── original_aligned.jsonl       (N = 597 original aligned functions)
     ├── ablation/
     │   ├── train.jsonl                   (N = 9,638 controlled ablation train)
     │   ├── val.jsonl                     (N = 1,259 controlled ablation val)
     │   └── test.jsonl                    (N = 1,205 controlled ablation test)
     └── metadata/
         ├── dataset_summary.json
         └── mutation_distribution.json

  2. V2 — Real-World-Grounded Dataset (Main Training Pool & Verified Evaluation)
     data/v2_real_world/
     ├── raw/
     │   ├── repositories/                 (Junction/link to data/raw_repos/)
     │   └── historical_candidates.jsonl   (N = 2,367 raw mined git candidates; 2,222 usable after purge)
     ├── mined/
     │   └── filtered_candidates.jsonl     (Filtered mined candidate pool)
     ├── generated/
     │   └── contract_grounded_drift.jsonl (N = 5,133 generated raw candidates; 5,122 usable after leakage purge)
     ├── training/
     │   ├── train.jsonl                   (N = 13,366 function-lineage grouped, balanced)
     │   └── val.jsonl                     (N = 1,430 function-lineage grouped, balanced)
     ├── evaluation/
     │   └── verified_test.jsonl           (N = 101 human-verified test set, strictly isolated)
     └── metadata/
         ├── dataset_summary.json          (Honest provenance breakdown: mined vs contract generated)
         ├── drift_distribution.json       (Contract violation types)
         └── repository_distribution.json  (Per-repo distribution across train/val/test)

Crucial Invariants & Guarantees:
  - Idempotent and assertion-heavy: fails immediately if source counts deviate.
  - Zero-Leakage: All 101 verified test lineages are purged from V2 BEFORE train/val partitioning.
  - Function Lineage Grouping: Hash of `repo::normalized_file::function_name` guarantees
    no function family is split between train and val.
  - Provenance: 5,133 generated raw candidates -> 5,122 usable after leakage purge;
    2,367 mined raw candidates -> 2,222 usable after leakage purge.
"""

import argparse
import ast
import hashlib
import json
import os
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

_AST_CACHE: Dict[Tuple[str, str], Dict[str, List[Dict[str, Any]]]] = {}


def norm_repo(r: Any) -> str:
    if not r:
        return ""
    return str(r).lower().replace("-", "_").replace(" ", "_").strip()


def norm_file_path(f: Any) -> str:
    if not f:
        return ""
    f_str = str(f).replace("\\", "/").lower().strip()
    for prefix in ["data/raw_repos/", "data/experiments/v2/", "data/"]:
        if f_str.startswith(prefix):
            f_str = f_str[len(prefix):]
    parts = f_str.split("/")
    known_repos = {
        "click", "django", "fastapi", "flask", "numpy", "pandas",
        "pytest", "requests", "scikit_learn", "scikit-learn",
        "sqlalchemy", "tornado", "celery"
    }
    if len(parts) > 1 and parts[0] in known_repos:
        f_str = "/".join(parts[1:])
    return f_str


def get_file_ast_info(repo: str, file_path: str) -> Dict[str, List[Dict[str, Any]]]:
    """Extract functions with qualified names from repo source files."""
    key = (repo, file_path)
    if key in _AST_CACHE:
        return _AST_CACHE[key]
    
    cand1 = PROJECT_ROOT / "data" / "raw_repos" / repo / file_path
    cand2 = PROJECT_ROOT / "data" / "raw_repos" / repo.replace("_", "-") / file_path
    target = cand1 if cand1.is_file() else (cand2 if cand2.is_file() else None)
    if not target:
        _AST_CACHE[key] = {}
        return {}
    
    try:
        source = target.read_text(encoding="utf-8", errors="ignore")
        tree = ast.parse(source)
    except Exception:
        _AST_CACHE[key] = {}
        return {}
        
    funcs = defaultdict(list)

    def walk(node, stack):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                walk(child, stack + [child.name])
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                qname = ".".join(stack + [child.name]) if stack else child.name
                doc = ast.get_docstring(child) or ""
                funcs[child.name].append({
                    "qual_name": qname,
                    "docstring": doc.strip(),
                    "lineno": child.lineno,
                    "arg_names": [a.arg for a in child.args.args]
                })
                walk(child, stack + [child.name])

    walk(tree, [])
    _AST_CACHE[key] = funcs
    return funcs


def resolve_qualified_name(row: Dict[str, Any]) -> str:
    """Resolve true qualified function name (e.g., ClassName.method or outer.inner)."""
    if row.get("qualified_name"):
        return str(row["qualified_name"]).strip()
    if row.get("qualified_function_name"):
        return str(row["qualified_function_name"]).strip()
    if row.get("class_name"):
        c_name = row["class_name"]
        f_name = row.get("function_name", "")
        return f"{c_name}.{f_name}".strip()

    fn = str(row.get("function_name", "")).strip()
    if not fn:
        return ""

    repo = norm_repo(row.get("repo") or row.get("repo_name"))
    fp = norm_file_path(row.get("file_path") or row.get("file"))

    funcs = get_file_ast_info(repo, fp)
    cands = funcs.get(fn, [])
    if not cands:
        return fn
    if len(cands) == 1:
        return cands[0]["qual_name"]
    
    doc_target = (row.get("docstring_before") or row.get("docstring") or row.get("docstring_after") or "").strip()
    if doc_target:
        for c in cands:
            if c["docstring"] and (c["docstring"] in doc_target or doc_target in c["docstring"]):
                return c["qual_name"]
                
    code_target = row.get("code_before") or row.get("code") or row.get("code_after") or ""
    if code_target:
        try:
            node = ast.parse(code_target).body[0]
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                target_args = [a.arg for a in node.args.args]
                for c in cands:
                    if c["arg_names"] == target_args:
                        return c["qual_name"]
        except Exception:
            pass

    return cands[0]["qual_name"]


def get_function_lineage(row: Dict[str, Any]) -> str:
    """Stable lineage key: repo + normalized file + function name.
    
    Immune to line-number shifts across commits.
    Format: repo::normalized_file_path::function_name
    """
    r = norm_repo(row.get("repo") or row.get("repo_name"))
    fp = norm_file_path(row.get("file_path") or row.get("file"))
    fn = str(row.get("function_name", "")).strip()
    return f"{r}::{fp}::{fn}"


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    assert path.is_file(), f"Required input file missing: {path}"
    records = []
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as err:
                    raise ValueError(f"Malformed JSON at {path}:{idx} - {err}")
    return records


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def setup_v1_synthetic(base_dir: Path) -> Dict[str, Any]:
    print("=" * 70)
    print("STEP 1: Setting up V1 — Synthetic Dataset (Controlled Baseline)")
    print("=" * 70)

    v1_dir = base_dir / "v1_synthetic"
    benchmark_dir = v1_dir / "benchmark"
    raw_dir = v1_dir / "raw"
    ablation_dir = v1_dir / "ablation"
    meta_dir = v1_dir / "metadata"

    for d in [benchmark_dir, raw_dir, ablation_dir, meta_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # 1. Source files check
    src_test_path = base_dir / "labeled" / "test.jsonl"
    if not src_test_path.is_file():
        src_test_path = base_dir / "experiments" / "v2" / "test.jsonl"
    assert src_test_path.is_file(), f"Missing source test file for V1: {src_test_path}"

    test_samples = load_jsonl(src_test_path)
    assert len(test_samples) == 1205, f"Expected exactly 1,205 samples in {src_test_path}, got {len(test_samples)}"

    # Check distribution in 1,205
    label_counts = Counter(s.get("label") for s in test_samples)
    assert label_counts["aligned"] == 597, f"Expected 597 aligned samples, got {label_counts['aligned']}"
    assert label_counts["drifted"] == 608, f"Expected 608 drifted samples, got {label_counts['drifted']}"

    # 2. Write benchmark/synthetic_dataset.jsonl (1,205)
    write_jsonl(benchmark_dir / "synthetic_dataset.jsonl", test_samples)
    print(f"  [OK] benchmark/synthetic_dataset.jsonl: {len(test_samples)} samples (597 aligned, 608 drift)")

    # 3. Write raw/original_aligned.jsonl (597)
    aligned_samples = [s for s in test_samples if s.get("label") == "aligned"]
    assert len(aligned_samples) == 597
    write_jsonl(raw_dir / "original_aligned.jsonl", aligned_samples)
    print(f"  [OK] raw/original_aligned.jsonl: {len(aligned_samples)} original aligned functions")

    # 4. Write ablation splits (train=9,638, val=1,259, test=1,205)
    src_train_path = base_dir / "labeled" / "train.jsonl"
    if not src_train_path.is_file():
        src_train_path = base_dir / "experiments" / "v2" / "train.jsonl"
    src_val_path = base_dir / "labeled" / "val.jsonl"
    if not src_val_path.is_file():
        src_val_path = base_dir / "experiments" / "v2" / "val.jsonl"

    train_samples = load_jsonl(src_train_path)
    val_samples = load_jsonl(src_val_path)

    assert len(train_samples) == 9638, f"Expected 9,638 train samples, got {len(train_samples)}"
    assert len(val_samples) == 1259, f"Expected 1,259 val samples, got {len(val_samples)}"

    write_jsonl(ablation_dir / "train.jsonl", train_samples)
    write_jsonl(ablation_dir / "val.jsonl", val_samples)
    write_jsonl(ablation_dir / "test.jsonl", test_samples)
    print(f"  [OK] ablation/: train={len(train_samples)}, val={len(val_samples)}, test={len(test_samples)}")

    # 5. Metadata
    mutation_counts = Counter()
    for s in test_samples:
        m = s.get("drift_type") or s.get("mutation_type")
        if m:
            mutation_counts[m] += 1

    summary = {
        "dataset_name": "SemDrift V1 Synthetic Dataset (Controlled Baseline & Ablation Benchmark)",
        "purpose": "Controlled synthetic experiment to evaluate model learning under known mutation operators",
        "benchmark_samples": len(test_samples),
        "aligned_count": label_counts["aligned"],
        "drifted_count": label_counts["drifted"],
        "drift_ratio": round(label_counts["drifted"] / len(test_samples), 4),
        "mutation_distribution": dict(mutation_counts),
        "ablation_splits": {
            "train": len(train_samples),
            "val": len(val_samples),
            "test": len(test_samples),
            "total": len(train_samples) + len(val_samples) + len(test_samples)
        },
        "repositories": sorted(list({s.get("repo") for s in test_samples if s.get("repo")}))
    }

    with (meta_dir / "dataset_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with (meta_dir / "mutation_distribution.json").open("w", encoding="utf-8") as f:
        json.dump(dict(mutation_counts), f, indent=2)

    print("  [OK] metadata/ generated successfully")
    return summary


def setup_v2_real_world(base_dir: Path) -> Dict[str, Any]:
    print("\n" + "=" * 70)
    print("STEP 2: Setting up V2 — Real-World-Grounded Dataset")
    print("=" * 70)

    v2_dir = base_dir / "v2_real_world"
    raw_dir = v2_dir / "raw"
    mined_dir = v2_dir / "mined"
    gen_dir = v2_dir / "generated"
    train_dir = v2_dir / "training"
    eval_dir = v2_dir / "evaluation"
    meta_dir = v2_dir / "metadata"

    for d in [raw_dir, mined_dir, gen_dir, train_dir, eval_dir, meta_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # 1. Load and Isolate 101 Human-Verified Test Samples FIRST
    src_verified_path = base_dir / "real_world" / "verified_dataset.jsonl"
    assert src_verified_path.is_file(), f"Missing verified dataset at {src_verified_path}"

    verified_samples = load_jsonl(src_verified_path)
    assert len(verified_samples) == 101, f"Expected exactly 101 verified test samples, got {len(verified_samples)}"

    verified_labels = Counter(s.get("label") for s in verified_samples)
    assert verified_labels[1] == 14, f"Expected 14 drift positives in verified test, got {verified_labels[1]}"
    assert verified_labels[0] == 87, f"Expected 87 clean negatives in verified test, got {verified_labels[0]}"

    for s in verified_samples:
        s["qualified_name"] = resolve_qualified_name(s)
        s["qualified_function_name"] = s["qualified_name"]

    # Write evaluation/verified_test.jsonl
    write_jsonl(eval_dir / "verified_test.jsonl", verified_samples)
    print(f"  [OK] evaluation/verified_test.jsonl: {len(verified_samples)} human-verified samples (14 drift, 87 clean)")

    # Build strict EXCLUSION SET of all 101 verified test lineages
    test_lineages: Set[str] = set()
    for s in verified_samples:
        lin = get_function_lineage(s)
        test_lineages.add(lin)
    assert len(test_lineages) == 101, f"Expected 101 unique test lineages, got {len(test_lineages)}"
    print(f"  [OK] Constructed Exclusion Set: {len(test_lineages)} verified test lineages locked")

    # 2. Load 15,000 Real-World Pool
    src_pos_path = base_dir / "real_time_data" / "real_time_drift_positives.jsonl"
    src_neg_path = base_dir / "real_time_data" / "real_time_clean_negatives.jsonl"
    assert src_pos_path.is_file(), f"Missing {src_pos_path}"
    assert src_neg_path.is_file(), f"Missing {src_neg_path}"

    pos_rows = load_jsonl(src_pos_path)
    neg_rows = load_jsonl(src_neg_path)
    assert len(pos_rows) == 7500, f"Expected 7,500 drift positives, got {len(pos_rows)}"
    assert len(neg_rows) == 7500, f"Expected 7,500 clean negatives, got {len(neg_rows)}"

    # 3. Categorize Authentic Historical Mined vs Contract Generated
    historical_candidates = []
    contract_grounded_drift = []

    for r in pos_rows:
        ch = r.get("commit_hash", "")
        if ch != "extracted_clean":
            r["drift_source"] = "authentic_historical_mined"
            historical_candidates.append(r)
        else:
            r["drift_source"] = "contract_grounded_generated"
            contract_grounded_drift.append(r)

    assert len(historical_candidates) == 2367, f"Expected 2,367 authentic mined candidates, got {len(historical_candidates)}"
    assert len(contract_grounded_drift) == 5133, f"Expected 5,133 contract generated drift, got {len(contract_grounded_drift)}"

    for r in neg_rows:
        r["drift_source"] = "clean_grounded"

    # Write raw/historical_candidates.jsonl and generated/contract_grounded_drift.jsonl
    write_jsonl(raw_dir / "historical_candidates.jsonl", historical_candidates)
    write_jsonl(gen_dir / "contract_grounded_drift.jsonl", contract_grounded_drift)
    print(f"  [OK] raw/historical_candidates.jsonl: {len(historical_candidates)} authentic mined git drift instances")
    print(f"  [OK] generated/contract_grounded_drift.jsonl: {len(contract_grounded_drift)} AST contract-grounded drift instances")

    # Mined filtered candidates (from historical candidates)
    write_jsonl(mined_dir / "filtered_candidates.jsonl", historical_candidates)
    print(f"  [OK] mined/filtered_candidates.jsonl: {len(historical_candidates)} filtered mined candidates pool")

    # 4. Purge All 101 Test Lineages from V2 Construction (Zero-Leakage Enforcement)
    full_pool = pos_rows + neg_rows
    assert len(full_pool) == 15000

    clean_v2_pool = []
    purged_count = 0
    purged_labels = Counter()

    for r in full_pool:
        r["qualified_name"] = resolve_qualified_name(r)
        r["qualified_function_name"] = r["qualified_name"]
        lin = get_function_lineage(r)
        if lin in test_lineages:
            purged_count += 1
            purged_labels[r.get("pseudo_label", r.get("label"))] += 1
        else:
            # Ensure provenance metadata is preserved
            if "parent_commit" not in r:
                r["parent_commit"] = r.get("parent_hash") or None
            clean_v2_pool.append(r)

    assert purged_count == 204, f"Expected exactly 204 leaked test lineages purged, got {purged_count}"
    assert len(clean_v2_pool) == 14796, f"Expected 14,796 clean non-leaking samples, got {len(clean_v2_pool)}"

    pool_labels = Counter(r.get("pseudo_label", r.get("label")) for r in clean_v2_pool)
    assert pool_labels[0] == 7452, f"Expected 7,452 clean negatives, got {pool_labels[0]}"
    assert pool_labels[1] == 7344, f"Expected 7,344 drift positives, got {pool_labels[1]}"
    print(f"  [OK] Zero-Leakage Purge: Removed {purged_count} leaking instances. Clean pool = {len(clean_v2_pool)} (7,452 clean, 7,344 drift)")

    # 5. Partition by Function-Lineage Grouping (90% Train / 10% Val)
    lineage_groups = defaultdict(list)
    for r in clean_v2_pool:
        lineage_groups[get_function_lineage(r)].append(r)

    train_rows = []
    val_rows = []

    for lin, rows in lineage_groups.items():
        # Deterministic SHA256 hashing on lineage key
        h = int(hashlib.sha256(lin.encode("utf-8")).hexdigest(), 16)
        if (h % 1000) < 900:
            train_rows.extend(rows)
        else:
            val_rows.extend(rows)

    assert len(train_rows) + len(val_rows) == 14796
    assert len(train_rows) == 13366, f"Expected 13,366 train rows, got {len(train_rows)}"
    assert len(val_rows) == 1430, f"Expected 1,430 val rows, got {len(val_rows)}"
    train_labels = Counter(r.get("pseudo_label", r.get("label")) for r in train_rows)
    val_labels = Counter(r.get("pseudo_label", r.get("label")) for r in val_rows)

    # 6. Hard Mathematical Invariant Assertions
    train_lineages = {get_function_lineage(r) for r in train_rows}
    val_lineages = {get_function_lineage(r) for r in val_rows}

    leak_train_val = train_lineages.intersection(val_lineages)
    assert len(leak_train_val) == 0, f"FATAL: {len(leak_train_val)} function lineages leaked between train and val!"

    leak_train_test = train_lineages.intersection(test_lineages)
    assert len(leak_train_test) == 0, f"FATAL: {len(leak_train_test)} test lineages leaked into train!"

    leak_val_test = val_lineages.intersection(test_lineages)
    assert len(leak_val_test) == 0, f"FATAL: {len(leak_val_test)} test lineages leaked into val!"

    # Write training/train.jsonl and training/val.jsonl
    write_jsonl(train_dir / "train.jsonl", train_rows)
    write_jsonl(train_dir / "val.jsonl", val_rows)

    print(f"  [OK] training/train.jsonl: {len(train_rows)} samples (Clean: {train_labels[0]}, Drift: {train_labels[1]})")
    print(f"  [OK] training/val.jsonl:   {len(val_rows)} samples (Clean: {val_labels[0]}, Drift: {val_labels[1]})")
    print(f"  [OK] Invariant passed: train & val == empty set (0 leaking lineages)")
    print(f"  [OK] Invariant passed: (train + val) & test == empty set (0 leaking lineages)")

    # 7. Metadata Generation
    mined_in_pool = sum(1 for r in clean_v2_pool if r.get("drift_source") == "authentic_historical_mined")
    gen_in_pool = sum(1 for r in clean_v2_pool if r.get("drift_source") == "contract_grounded_generated")
    assert mined_in_pool == 2222, f"Expected 2,222 mined drift in clean pool, got {mined_in_pool}"
    assert gen_in_pool == 5122, f"Expected 5,122 generated drift in clean pool, got {gen_in_pool}"

    drift_distribution = {
        "drift_total": len(pos_rows) - purged_labels[1],
        "authentic_historical_mined": mined_in_pool,
        "contract_grounded_generated": gen_in_pool,
        "raw_counts": {
            "raw_mined_candidates": len(historical_candidates),
            "mined_candidates_purged": len(historical_candidates) - mined_in_pool,
            "usable_mined_drift": mined_in_pool,
            "raw_generated_candidates": len(contract_grounded_drift),
            "generated_candidates_purged": len(contract_grounded_drift) - gen_in_pool,
            "usable_generated_drift": gen_in_pool,
        },
        "clarification": "5,133 generated raw candidates -> 5,122 usable after leakage purge (11 purged); 2,367 mined raw candidates -> 2,222 usable after leakage purge (145 purged)",
        "contract_violations": {
            "parameter_contract_violation": sum(1 for r in clean_v2_pool if r.get("parameter_contract_violation")),
            "return_contract_violation": sum(1 for r in clean_v2_pool if r.get("return_contract_violation")),
            "raises_contract_violation": sum(1 for r in clean_v2_pool if r.get("raises_contract_violation")),
            "default_contract_violation": sum(1 for r in clean_v2_pool if r.get("default_contract_violation"))
        }
    }

    repo_distribution = {
        "training": dict(Counter(r.get("repo") or r.get("repo_name") for r in train_rows)),
        "validation": dict(Counter(r.get("repo") or r.get("repo_name") for r in val_rows)),
        "verified_test": dict(Counter(r.get("repo") or r.get("repo_name") for r in verified_samples)),
    }

    summary = {
        "dataset_name": "SemDrift V2 Real-World-Grounded Dataset",
        "description": "Main training, validation, and evaluation pool combining authentic mined git evolution and AST contract-grounded drift.",
        "total_instances": len(clean_v2_pool),
        "partitions": {
            "train": len(train_rows),
            "val": len(val_rows),
            "verified_test": len(verified_samples)
        },
        "drift": {
            "total": pool_labels[1],
            "authentic_historical_mined": mined_in_pool,
            "contract_grounded_generated": gen_in_pool,
            "raw_generated_candidates": len(contract_grounded_drift),
            "raw_mined_candidates": len(historical_candidates),
            "clarification": "5,133 generated raw candidates -> 5,122 usable after leakage purge (11 purged); 2,367 mined raw candidates -> 2,222 usable after leakage purge (145 purged)"
        },
        "clean": {
            "total": pool_labels[0],
            "historical_clean": pool_labels[0]
        },
        "zero_leakage_guarantee": {
            "verified_test_lineages": len(test_lineages),
            "candidates_purged_for_leakage": purged_count,
            "train_val_overlap": 0,
            "train_test_overlap": 0,
            "val_test_overlap": 0
        }
    }

    with (meta_dir / "dataset_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with (meta_dir / "drift_distribution.json").open("w", encoding="utf-8") as f:
        json.dump(drift_distribution, f, indent=2)

    with (meta_dir / "repository_distribution.json").open("w", encoding="utf-8") as f:
        json.dump(repo_distribution, f, indent=2)

    print("  [OK] metadata/ generated successfully")

    # 8. Setup raw/repositories pointer
    repos_src = base_dir / "raw_repos"
    repos_dest = raw_dir / "repositories"
    if repos_src.is_dir() and not repos_dest.exists():
        try:
            # Try to create junction on Windows
            if os.name == "nt":
                import subprocess
                subprocess.run(
                    ["cmd", "/c", "mklink", "/J", str(repos_dest), str(repos_src)],
                    check=True,
                    capture_output=True
                )
                print("  [OK] raw/repositories directory junction linked to data/raw_repos/")
            else:
                repos_dest.symlink_to(repos_src, target_is_directory=True)
                print("  [OK] raw/repositories symlinked to data/raw_repos/")
        except Exception as e:
            print(f"  [NOTE] Could not create junction/symlink ({e}); creating reference README.")
            with (repos_dest / "README.md").open("w", encoding="utf-8") as rf:
                rf.write("# Raw Repositories\nCloned repositories are located at `data/raw_repos/`.\n")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Build and verify SemDrift V1 & V2 dataset architecture.")
    parser.add_argument("--data_dir", default=str(PROJECT_ROOT / "data"), help="Root data directory")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    assert data_dir.is_dir(), f"Data directory does not exist: {data_dir}"

    print("======================================================================")
    print("SEMDRIFT DATASET ARCHITECTURE RESTRUCTURING")
    print(f"Data Root: {data_dir}")
    print("======================================================================")

    v1_summary = setup_v1_synthetic(data_dir)
    v2_summary = setup_v2_real_world(data_dir)

    print("\n" + "=" * 70)
    print("SUCCESS: SemDrift V1 & V2 Dataset Architecture Verified and Built")
    print("=" * 70)
    print(f"V1 Synthetic Benchmark:   {v1_summary['benchmark_samples']} samples ({v1_summary['aligned_count']} aligned, {v1_summary['drifted_count']} drift)")
    print(f"V2 Real-World-Grounded:   {v2_summary['total_instances']} samples ({v2_summary['partitions']['train']} train, {v2_summary['partitions']['val']} val)")
    print(f"V2 Evaluation Test Set:   {v2_summary['partitions']['verified_test']} human-verified samples (100% isolated)")
    print(f"V2 Drift Provenance:      {v2_summary['drift']['authentic_historical_mined']} authentic mined + {v2_summary['drift']['contract_grounded_generated']} contract generated")
    print("======================================================================\n")


if __name__ == "__main__":
    main()
