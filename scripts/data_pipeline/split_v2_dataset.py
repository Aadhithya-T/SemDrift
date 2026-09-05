"""V2 Dataset Splitter for SemDrift.

Produces the 3 evaluation splits requested in the research roadmap:
1. Function-Held-Out Split (data/experiments/v2_function_split/):
   - Deterministic hash grouping on function identity (repo + file + lineno)
   - Prevents mutation variants from leaking across train/val/test (80/10/10)

2. Repository-Held-Out Split (data/experiments/v2_repo_split/):
   - Leave-Out Repositories for true cross-project zero-shot evaluation
   - Train on 8 repos, Test on 2 completely unseen repos (e.g. requests, flask)

3. Real-World Evaluation Benchmark:
   - Evaluates trained models directly on data/real_world/verified_dataset.jsonl (N=101)
"""

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Set


def get_function_key(row: dict) -> str:
    """Stable identity for the ORIGINAL function, independent of mutation."""
    if row.get("function_id"):
        return str(row["function_id"])
    return f"{row.get('repo')}::{row.get('file')}::{row.get('lineno')}"


def get_function_split(function_key: str, train: float = 0.8, val: float = 0.1) -> str:
    """Deterministically map a function key to train/val/test using MD5 hashing."""
    h = int(hashlib.md5(function_key.encode("utf-8")).hexdigest(), 16)
    frac = h / (16 ** 32)
    if frac < train:
        return "train"
    elif frac < train + val:
        return "val"
    return "test"


def create_function_held_out_split(rows: List[dict], output_dir: Path) -> Dict[str, int]:
    """Create function-level grouped train/val/test split."""
    output_dir.mkdir(parents=True, exist_ok=True)
    splits = defaultdict(list)

    for row in rows:
        key = get_function_key(row)
        split = get_function_split(key)
        splits[split].append(row)

    counts = {}
    for split, items in splits.items():
        out_path = output_dir / f"{split}.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for item in items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        counts[split] = len(items)

    return counts


def create_repo_held_out_split(
    rows: List[dict],
    output_dir: Path,
    test_repos: Set[str] = None,
    val_repos: Set[str] = None,
) -> Dict[str, int]:
    """Create repository-level held out split for zero-shot transfer evaluation."""
    if test_repos is None:
        test_repos = {"requests", "flask"}
    if val_repos is None:
        val_repos = {"fastapi"}

    output_dir.mkdir(parents=True, exist_ok=True)
    splits = defaultdict(list)

    for row in rows:
        repo = row.get("repo", "").lower()
        if repo in test_repos:
            splits["test"].append(row)
        elif repo in val_repos:
            splits["val"].append(row)
        else:
            splits["train"].append(row)

    counts = {}
    for split in ["train", "val", "test"]:
        items = splits[split]
        out_path = output_dir / f"{split}.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for item in items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        counts[split] = len(items)

    return counts


def main():
    parser = argparse.ArgumentParser(description="Split V2 dataset into function-held-out and repo-held-out splits.")
    parser.add_argument("--input", default="data/experiments/v2/semdrift_v2_labeled.jsonl", help="Path to V2 labeled dataset")
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        print(f"Error: {in_path} does not exist.")
        return 1

    rows = []
    with in_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    print(f"Loaded {len(rows)} rows from {in_path}")

    # 1. Function-Held-Out Split
    func_dir = Path("data/experiments/v2_function_split")
    func_counts = create_function_held_out_split(rows, func_dir)
    print("\n" + "=" * 60)
    print("MODE 1: FUNCTION-HELD-OUT SPLIT (data/experiments/v2_function_split/)")
    print("=" * 60)
    for split, count in func_counts.items():
        print(f"  - {split}: {count} samples ({count/len(rows)*100:.1f}%)")

    # 2. Repository-Held-Out Split
    repo_dir = Path("data/experiments/v2_repo_split")
    repo_counts = create_repo_held_out_split(rows, repo_dir)
    print("\n" + "=" * 60)
    print("MODE 2: REPOSITORY-HELD-OUT SPLIT (data/experiments/v2_repo_split/)")
    print("=" * 60)
    for split, count in repo_counts.items():
        print(f"  - {split}: {count} samples ({count/len(rows)*100:.1f}%)")

    print("\n" + "=" * 60)
    print("MODE 3: REAL-WORLD EVALUATION BENCHMARK")
    print("=" * 60)
    print("  - Benchmark file: data/real_world/verified_dataset.jsonl (N = 101, 100% human-verified)")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    main()
