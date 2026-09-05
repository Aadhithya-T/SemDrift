"""Consolidates mined real-world datasets into unified benchmark & training files.

Outputs:
1. Gold Benchmark (Evaluation Test Set):
   - data/real_world/gold_benchmark.jsonl (N=76, 100% human-verified ground truth)
   - data/real_world/gold_drift_positives.jsonl (N=12)
   - data/real_world/gold_clean_negatives.jsonl (N=64)

2. Filtered Dataset (Weakly Supervised Large-Scale Pool):
   - data/real_world/filtered_dataset/filtered_dataset.jsonl (N=4,997 combined)
   - data/real_world/filtered_dataset/filtered_drift_positives.jsonl (N=13)
   - data/real_world/filtered_dataset/filtered_clean_negatives.jsonl (N=4,984)
   - data/real_world/filtered_dataset/filtered_ambiguous.jsonl (N=0)
   - data/real_world/filtered_dataset/filtered_dataset_summary.json
"""

import json
import shutil
from pathlib import Path
from typing import List, Dict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REAL_WORLD_DIR = PROJECT_ROOT / "data" / "real_world"
VERIFIED_DIR = REAL_WORLD_DIR / "verified"
REJECTED_DIR = REAL_WORLD_DIR / "rejected"
CANDIDATES_DIR = REAL_WORLD_DIR / "mined_candidates"
FILTERED_DATASET_DIR = REAL_WORLD_DIR / "filtered_dataset"
OLD_SILVER_DIR = REAL_WORLD_DIR / "silver"


def load_jsonl(path: Path) -> List[Dict]:
    records = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                try:
                    records.append(json.loads(s))
                except json.JSONDecodeError:
                    pass
    return records


def write_jsonl(path: Path, records: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def consolidate_verified_dataset() -> None:
    repos = ["click", "fastapi", "django"]
    positives = []
    negatives = []
    all_verified = []

    for repo in repos:
        v_file = VERIFIED_DIR / f"{repo}_verified.jsonl"
        r_file = REJECTED_DIR / f"{repo}_rejected.jsonl"

        v_items = load_jsonl(v_file)
        for item in v_items:
            item["repo"] = repo
            item["verified_label"] = "drift"
            item["label"] = 1
            positives.append(item)
            all_verified.append(item)

        r_items = load_jsonl(r_file)
        for item in r_items:
            item["repo"] = repo
            item["verified_label"] = "no_drift"
            item["label"] = 0
            negatives.append(item)
            all_verified.append(item)

    # Clean up old gold_benchmark files if present
    for old_name in ["gold_benchmark.jsonl", "gold_drift_positives.jsonl", "gold_clean_negatives.jsonl"]:
        old_p = REAL_WORLD_DIR / old_name
        if old_p.exists():
            old_p.unlink()

    # Export consolidated files
    write_jsonl(REAL_WORLD_DIR / "verified_dataset.jsonl", all_verified)
    write_jsonl(REAL_WORLD_DIR / "verified_drift_positives.jsonl", positives)
    write_jsonl(REAL_WORLD_DIR / "verified_clean_negatives.jsonl", negatives)

    print("=" * 60)
    print("VERIFIED DATASET CONSOLIDATION")
    print("=" * 60)
    print(f"Total Verified Dataset instances: {len(all_verified)}")
    print(f"  - Verified Drift Positives (+): {len(positives)}")
    print(f"  - Verified Clean Negatives (-): {len(negatives)}")
    print(f"Exported to: {REAL_WORLD_DIR / 'verified_dataset.jsonl'}")


def consolidate_filtered_dataset() -> None:
    FILTERED_DATASET_DIR.mkdir(parents=True, exist_ok=True)

    # Import from generate_filtered_dataset / contract checks
    from generate_filtered_dataset import compute_silver_probability

    repos = ["click", "fastapi", "django"]
    all_candidates = []
    positives = []
    negatives = []
    ambiguous = []
    repo_stats = {}

    for repo in repos:
        path = CANDIDATES_DIR / f"{repo}_contract_checked.jsonl"
        if not path.is_file():
            path = CANDIDATES_DIR / f"{repo}_scored.jsonl"
        if not path.is_file():
            continue

        items = load_jsonl(path)
        p_cnt, n_cnt, a_cnt = 0, 0, 0

        for candidate in items:
            candidate["repo"] = repo
            p_drift, conf, signals = compute_silver_probability(candidate)
            candidate["filtered_drift_probability"] = p_drift
            candidate["filtered_confidence"] = conf
            candidate["filtered_signals"] = signals

            if p_drift >= 0.60:
                candidate["filtered_label"] = "drift"
                candidate["pseudo_label"] = 1
                positives.append(candidate)
                p_cnt += 1
            elif p_drift <= 0.25:
                candidate["filtered_label"] = "no_drift"
                candidate["pseudo_label"] = 0
                negatives.append(candidate)
                n_cnt += 1
            else:
                candidate["filtered_label"] = "ambiguous"
                candidate["pseudo_label"] = -1
                ambiguous.append(candidate)
                a_cnt += 1

            all_candidates.append(candidate)

        repo_stats[repo] = {
            "total": len(items),
            "positives": p_cnt,
            "negatives": n_cnt,
            "ambiguous": a_cnt,
        }

    # Sort
    positives.sort(key=lambda c: c["filtered_drift_probability"], reverse=True)
    negatives.sort(key=lambda c: c["filtered_drift_probability"])

    # Export consolidated files
    write_jsonl(FILTERED_DATASET_DIR / "filtered_dataset.jsonl", all_candidates)
    write_jsonl(FILTERED_DATASET_DIR / "filtered_drift_positives.jsonl", positives)
    write_jsonl(FILTERED_DATASET_DIR / "filtered_clean_negatives.jsonl", negatives)
    write_jsonl(FILTERED_DATASET_DIR / "filtered_ambiguous.jsonl", ambiguous)

    summary = {
        "dataset_name": "SemDrift Filtered Real-World Dataset",
        "total_instances": len(all_candidates),
        "filtered_positives_count": len(positives),
        "filtered_negatives_count": len(negatives),
        "ambiguous_filtered_count": len(ambiguous),
        "repo_breakdown": repo_stats,
    }

    with (FILTERED_DATASET_DIR / "filtered_dataset_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Clean old silver directory if present
    if OLD_SILVER_DIR.exists():
        shutil.rmtree(OLD_SILVER_DIR, ignore_errors=True)

    print("\n" + "=" * 60)
    print("FILTERED DATASET CONSOLIDATION")
    print("=" * 60)
    print(f"Total Filtered Dataset instances: {len(all_candidates)}")
    print(f"  - Filtered Drift Positives (+):  {len(positives)}")
    print(f"  - Filtered Clean Negatives (-):  {len(negatives)}")
    print(f"  - Ambiguous Instances:           {len(ambiguous)}")
    print(f"Exported to: {FILTERED_DATASET_DIR / 'filtered_dataset.jsonl'}")


def main():
    consolidate_verified_dataset()
    consolidate_filtered_dataset()


if __name__ == "__main__":
    main()
