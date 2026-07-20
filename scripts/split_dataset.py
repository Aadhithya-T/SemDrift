"""
split_dataset.py

Splits SemDrift's labeled, mutated dataset into train/val/test files
without leaking near-duplicate mutations of the same original function
across splits.

Key idea: the split is a deterministic hash of the ORIGINAL function's
identity (not the mutated row). So:
  - Every mutation of the same function always lands in the same split.
  - The split is stable across runs -- you can re-run this script after
    collecting more data, and old rows will always resolve to the same
    split they were assigned to before.
  - Safe to run incrementally: already-written rows are skipped, only
    new rows get appended.

Usage:
    python split_dataset.py --input data/labeled/semdrift_labeled.jsonl --output_dir data/

Input row schema (one JSON object per line):
    {
        "function_id": "...",   # preferred stable identifier
        "repo": "...",
        "file": "...",
        "lineno": ...,
        "function_name": "...",
        "code": "...",
        "docstring": "...",
        "label": 0 or 1,
        "mutation_type": "param_rename" | "return_value_change" | "doc_negation" | "doc_sentence_delete" | null,
        "severity": "mild" | "moderate" | "severe" | null
    }
"""

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict


def get_function_key(row: dict) -> str:
    """Stable identity for the ORIGINAL function, independent of mutation.

    Prefers function_id if present. Falls back to repo+file+lineno,
    which identifies the source location regardless of which mutation
    was applied to it.
    """
    if row.get("function_id"):
        return str(row["function_id"])
    return f"{row.get('repo')}::{row.get('file')}::{row.get('lineno')}"


def get_split(function_key: str, train: float = 0.8, val: float = 0.1) -> str:
    """Deterministically map a function key to train/val/test.

    Uses an md5 hash so the same function_key ALWAYS resolves to the
    same split, run after run, batch after batch.
    """
    h = int(hashlib.md5(function_key.encode("utf-8")).hexdigest(), 16)
    frac = h / 16 ** 32
    if frac < train:
        return "train"
    elif frac < train + val:
        return "val"
    return "test"


def row_signature(row: dict) -> str:
    """Unique signature for a single row, used to detect rows that have
    already been written in a previous run (so re-running is safe)."""
    key = get_function_key(row)
    return f"{key}::{row.get('drift_type')}::{row.get('severity')}::{row.get('label')}"


def load_existing_signatures(path: str) -> set:
    sigs = set()
    if not os.path.exists(path):
        return sigs
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            sigs.add(row_signature(row))
    return sigs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to labeled dataset jsonl")
    parser.add_argument("--output_dir", default="data", help="Directory to write train/val/test.jsonl")
    parser.add_argument("--train_frac", type=float, default=0.8)
    parser.add_argument("--val_frac", type=float, default=0.1)
    args = parser.parse_args()

    out_paths = {
        "train": os.path.join(args.output_dir, "train.jsonl"),
        "val": os.path.join(args.output_dir, "val.jsonl"),
        "test": os.path.join(args.output_dir, "test.jsonl"),
    }
    os.makedirs(args.output_dir, exist_ok=True)

    # Load signatures of rows already written in previous runs, per split,
    # so re-running this script on an updated input file only appends new rows.
    existing_sigs = {split: load_existing_signatures(p) for split, p in out_paths.items()}
    all_existing_sigs = set().union(*existing_sigs.values())

    # Read + group input rows
    rows = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    new_rows_by_split = defaultdict(list)
    skipped = 0
    for row in rows:
        sig = row_signature(row)
        if sig in all_existing_sigs:
            skipped += 1
            continue
        key = get_function_key(row)
        split = get_split(key, args.train_frac, args.val_frac)
        new_rows_by_split[split].append(row)

    # Append new rows to the appropriate files
    for split, path in out_paths.items():
        if not new_rows_by_split[split]:
            continue
        with open(path, "a", encoding="utf-8") as f:
            for row in new_rows_by_split[split]:
                f.write(json.dumps(row) + "\n")

    # ---- Report ----
    print(f"Read {len(rows)} rows from {args.input}")
    print(f"Skipped {skipped} rows already present in output files")
    for split in ("train", "val", "test"):
        print(f"Appended {len(new_rows_by_split[split])} new rows to {out_paths[split]}")

    print("\n--- Post-write split totals & balance check ---")
    for split, path in out_paths.items():
        if not os.path.exists(path):
            continue
        mutation_counter = Counter()
        severity_counter = Counter()
        repo_counter = Counter()
        label_counter = Counter()
        total = 0
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                total += 1
                mutation_counter[row.get("drift_type")] += 1
                severity_counter[row.get("severity")] += 1
                repo_counter[row.get("repo")] += 1
                label_counter[row.get("label")] += 1
        print(f"\n{split.upper()} ({path}) -- {total} rows")
        print(f"  label:      {dict(label_counter)}")
        print(f"  drift_type: {dict(mutation_counter)}")
        print(f"  severity:   {dict(severity_counter)}")
        print(f"  repo:       {dict(repo_counter)}")


if __name__ == "__main__":
    main()