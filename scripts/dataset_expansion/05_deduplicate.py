"""
SemDrift Dataset Expansion - Step 5: Deduplication Engine
Deduplicates expansion candidates against baseline dataset and within the expansion set itself.
Guarantees zero overlap with the frozen baseline dataset.
"""

import os
import sys
import json
import hashlib
import csv
from pathlib import Path
from typing import Set, Dict, Any, List

def compute_hash(code: str, docstring: str) -> str:
    norm_code = "".join(code.split())
    norm_doc = "".join(docstring.split())
    return hashlib.sha256((norm_code + "||" + norm_doc).encode("utf-8")).hexdigest()

def main():
    base_dir = Path(__file__).resolve().parents[2]
    baseline_file = base_dir / "data" / "real_final_dataset" / "real_final_dataset.jsonl"
    
    val_drift_file = base_dir / "data" / "real_dataset_expansion_v1" / "evidence" / "validated_drift.jsonl"
    val_clean_file = base_dir / "data" / "real_dataset_expansion_v1" / "evidence" / "validated_consistent.jsonl"

    out_drift_dir = base_dir / "data" / "real_dataset_expansion_v1" / "drift_samples"
    out_clean_dir = base_dir / "data" / "real_dataset_expansion_v1" / "consistent_samples"

    out_drift_dir.mkdir(parents=True, exist_ok=True)
    out_clean_dir.mkdir(parents=True, exist_ok=True)

    print("[*] Loading baseline fingerprint cache for cross-dataset deduplication...", flush=True)
    baseline_hashes: Set[str] = set()
    baseline_symbols: Set[str] = set()

    if baseline_file.exists():
        with open(baseline_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                h = compute_hash(rec.get("code", ""), rec.get("docstring", ""))
                baseline_hashes.add(h)
                sym = f"{rec.get('repository')}::{rec.get('filepath')}::{rec.get('function_name')}"
                baseline_symbols.add(sym)
        print(f"[+] Loaded {len(baseline_hashes):,} baseline hashes from frozen dataset.", flush=True)

    # 1. Deduplicate Drift Samples
    print("[*] Deduplicating expansion DRIFT samples...", flush=True)
    dedup_drift: List[Dict[str, Any]] = []
    seen_drift_hashes: Set[str] = set()

    if val_drift_file.exists():
        with open(val_drift_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                h = compute_hash(rec["code"], rec["docstring"])
                if h in baseline_hashes or h in seen_drift_hashes:
                    continue
                seen_drift_hashes.add(h)
                dedup_drift.append(rec)

    print(f"[+] Clean, non-duplicate expansion DRIFT samples: {len(dedup_drift):,}", flush=True)

    # 2. Deduplicate Consistent Samples
    print("[*] Deduplicating expansion CONSISTENT samples...", flush=True)
    dedup_clean: List[Dict[str, Any]] = []
    seen_clean_hashes: Set[str] = set()

    if val_clean_file.exists():
        with open(val_clean_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                h = compute_hash(rec["code"], rec["docstring"])
                if h in baseline_hashes or h in seen_clean_hashes:
                    continue
                seen_clean_hashes.add(h)
                dedup_clean.append(rec)

    print(f"[+] Clean, non-duplicate expansion CONSISTENT samples: {len(dedup_clean):,}", flush=True)

    # Assign clean sequential expansion IDs
    for i, r in enumerate(dedup_drift, 1):
        r["id"] = f"semdrift_exp_drift_{i:06d}"
    for i, r in enumerate(dedup_clean, 1):
        r["id"] = f"semdrift_exp_clean_{i:06d}"

    # Write separate expansion datasets
    with open(out_drift_dir / "drift_samples.jsonl", "w", encoding="utf-8") as f:
        for r in dedup_drift:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(out_clean_dir / "consistent_samples.jsonl", "w", encoding="utf-8") as f:
        for r in dedup_clean:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    fieldnames = ["id", "repository", "url", "license", "commit_hash", "filepath", "function_name", "code", "docstring", "drift_label", "drift_type", "drift_explanation", "validation_method", "confidence", "audit_status"]

    def write_csv(data, path):
        with open(path, "w", encoding="utf-8", newline="") as cf:
            writer = csv.writer(cf, quoting=csv.QUOTE_ALL, escapechar="\\", doublequote=True)
            writer.writerow(fieldnames)
            for r in data:
                row = [str(r.get(k, "")) for k in fieldnames]
                writer.writerow(row)

    write_csv(dedup_drift, out_drift_dir / "drift_samples.csv")
    write_csv(dedup_clean, out_clean_dir / "consistent_samples.csv")

    print(f"[+] Successfully saved deduplicated datasets in {out_drift_dir} and {out_clean_dir}", flush=True)

if __name__ == "__main__":
    main()
