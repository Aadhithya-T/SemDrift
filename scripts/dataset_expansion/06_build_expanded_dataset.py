"""
SemDrift Dataset Expansion - Step 6: Dataset Combination & Split Partitioner
Merges the frozen baseline (5,200 samples) with newly verified expansion samples
to produce the benchmark dataset containing 10,000+ verified real-world samples.
Enforces strict repository-isolated train/val/test partitions with zero data leakage.
"""

import os
import sys
import json
import random
import csv
from pathlib import Path
from typing import List, Dict, Any, Set

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

def main():
    base_dir = Path(__file__).resolve().parents[2]
    baseline_file = base_dir / "data" / "real_final_dataset" / "real_final_dataset.jsonl"
    
    exp_drift_file = base_dir / "data" / "real_dataset_expansion_v1" / "drift_samples" / "drift_samples.jsonl"
    exp_clean_file = base_dir / "data" / "real_dataset_expansion_v1" / "consistent_samples" / "consistent_samples.jsonl"

    out_combined_dir = base_dir / "data" / "real_dataset_expansion_v1" / "combined"
    out_splits_dir = base_dir / "data" / "real_dataset_expansion_v1" / "splits"

    out_combined_dir.mkdir(parents=True, exist_ok=True)
    out_splits_dir.mkdir(parents=True, exist_ok=True)

    print("[*] Loading frozen baseline dataset...", flush=True)
    baseline_records: List[Dict[str, Any]] = []
    if baseline_file.exists():
        with open(baseline_file, "r", encoding="utf-8") as f:
            baseline_records = [json.loads(line) for line in f if line.strip()]
    print(f"[+] Loaded {len(baseline_records):,} baseline records.", flush=True)

    print("[*] Loading new expansion records...", flush=True)
    exp_drift_records: List[Dict[str, Any]] = []
    exp_clean_records: List[Dict[str, Any]] = []

    if exp_drift_file.exists():
        with open(exp_drift_file, "r", encoding="utf-8") as f:
            exp_drift_records = [json.loads(line) for line in f if line.strip()]

    if exp_clean_file.exists():
        with open(exp_clean_file, "r", encoding="utf-8") as f:
            exp_clean_records = [json.loads(line) for line in f if line.strip()]

    print(f"[+] Loaded {len(exp_drift_records):,} new DRIFT records and {len(exp_clean_records):,} new CONSISTENT records.", flush=True)

    import hashlib
    def compute_hash(code: str, docstring: str) -> str:
        norm_code = "".join(code.split())
        norm_doc = "".join(docstring.split())
        return hashlib.sha256((norm_code + "||" + norm_doc).encode("utf-8")).hexdigest()

    # Determine required counts to achieve >= 5,000 Drift and >= 5,000 Consistent with 100% unique hashes
    selected_drift: List[Dict[str, Any]] = []
    seen_drift_hashes: Set[str] = set()

    for r in baseline_records:
        if r.get("drift_label") == 1:
            h = compute_hash(r.get("code", ""), r.get("docstring", ""))
            if h not in seen_drift_hashes:
                seen_drift_hashes.add(h)
                selected_drift.append(r)

    for r in exp_drift_records:
        if len(selected_drift) >= 5000:
            break
        h = compute_hash(r.get("code", ""), r.get("docstring", ""))
        if h not in seen_drift_hashes:
            seen_drift_hashes.add(h)
            selected_drift.append(r)

    selected_clean: List[Dict[str, Any]] = []
    seen_clean_hashes: Set[str] = set()

    for r in baseline_records:
        if r.get("drift_label") == 0:
            h = compute_hash(r.get("code", ""), r.get("docstring", ""))
            if h not in seen_clean_hashes and h not in seen_drift_hashes:
                seen_clean_hashes.add(h)
                selected_clean.append(r)

    for r in exp_clean_records:
        if len(selected_clean) >= 5000:
            break
        h = compute_hash(r.get("code", ""), r.get("docstring", ""))
        if h not in seen_clean_hashes and h not in seen_drift_hashes:
            seen_clean_hashes.add(h)
            selected_clean.append(r)

    print(f"[*] Final Unique DRIFT samples: {len(selected_drift):,}", flush=True)
    print(f"[*] Final Unique CLEAN samples: {len(selected_clean):,}", flush=True)

    combined_all = selected_drift + selected_clean
    random.shuffle(combined_all)

    print(f"\n[+] Total Combined Benchmark Dataset: {len(combined_all):,} samples", flush=True)

    # Repository Isolation Partitioner
    all_repos = sorted(list(set(r["repository"] for r in combined_all)))
    print(f"[*] Repositories across combined dataset ({len(all_repos)} total): {all_repos}", flush=True)

    # Dedicate 5 test repos, 4 val repos, and remaining ~26 train repos
    random.seed(RANDOM_SEED)
    shuffled_repos = list(all_repos)
    random.shuffle(shuffled_repos)

    test_repos = set(shuffled_repos[:5])
    val_repos = set(shuffled_repos[5:9])
    train_repos = set(shuffled_repos[9:])

    print(f"    - Test Repositories ({len(test_repos)}): {sorted(list(test_repos))}", flush=True)
    print(f"    - Val Repositories   ({len(val_repos)}): {sorted(list(val_repos))}", flush=True)
    print(f"    - Train Repositories ({len(train_repos)}): {len(train_repos)} repositories", flush=True)

    train_split, val_split, test_split = [], [], []

    for r in combined_all:
        rep = r["repository"]
        if rep in test_repos:
            r["split"] = "test"
            test_split.append(r)
        elif rep in val_repos:
            r["split"] = "val"
            val_split.append(r)
        else:
            r["split"] = "train"
            train_split.append(r)

    print(f"[+] Partitions: Train={len(train_split):,}, Val={len(val_split):,}, Test={len(test_split):,}", flush=True)

    # Export combined and splits
    combined_jsonl = out_combined_dir / "expanded_final_dataset.jsonl"
    with open(combined_jsonl, "w", encoding="utf-8") as f:
        for r in combined_all:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(out_splits_dir / "train.jsonl", "w", encoding="utf-8") as f:
        for r in train_split:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(out_splits_dir / "val.jsonl", "w", encoding="utf-8") as f:
        for r in val_split:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(out_splits_dir / "test.jsonl", "w", encoding="utf-8") as f:
        for r in test_split:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # CSV Exports
    fieldnames = ["id", "repository", "commit_hash", "filepath", "function_name", "code", "docstring", "drift_label", "drift_type", "drift_explanation", "split"]

    def write_csv(data, path):
        with open(path, "w", encoding="utf-8", newline="") as cf:
            writer = csv.writer(cf, quoting=csv.QUOTE_ALL, escapechar="\\", doublequote=True)
            writer.writerow(fieldnames)
            for r in data:
                row = [str(r.get(k, "")) for k in fieldnames]
                writer.writerow(row)

    write_csv(combined_all, out_combined_dir / "expanded_final_dataset.csv")
    write_csv(train_split, out_splits_dir / "train.csv")
    write_csv(val_split, out_splits_dir / "val.csv")
    write_csv(test_split, out_splits_dir / "test.csv")

    # Metrics Summary
    drift_types = {}
    repo_dist = {}
    for r in combined_all:
        dt = r.get("drift_type", "none")
        drift_types[dt] = drift_types.get(dt, 0) + 1
        rp = r.get("repository", "unknown")
        repo_dist[rp] = repo_dist.get(rp, 0) + 1

    summary = {
        "dataset_name": "SemDrift Expanded Real-World Benchmark Dataset (10K+)",
        "total_samples": len(combined_all),
        "total_drift_samples": len([r for r in combined_all if r["drift_label"] == 1]),
        "total_consistent_samples": len([r for r in combined_all if r["drift_label"] == 0]),
        "drift_ratio": round(len([r for r in combined_all if r["drift_label"] == 1]) / len(combined_all), 4),
        "baseline_samples_included": len(baseline_records),
        "expansion_samples_added": len(combined_all) - len(baseline_records),
        "repositories_count": len(all_repos),
        "splits": {
            "train": len(train_split),
            "val": len(val_split),
            "test": len(test_split)
        },
        "drift_type_distribution": drift_types,
        "repository_distribution": repo_dist
    }

    with open(base_dir / "data" / "real_dataset_expansion_v1" / "dataset_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[+] Expanded dataset summary saved to dataset_summary.json", flush=True)

if __name__ == "__main__":
    main()
