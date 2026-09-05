"""Automated pipeline runner for filtering, scoring, contract-checking and consolidating newly mined repositories."""

import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

CANDIDATES_DIR = PROJECT_ROOT / "data" / "real_world" / "mined_candidates"


def run_cmd(cmd_list, desc):
    print(f"--> {desc}...")
    res = subprocess.run(cmd_list, cwd=str(PROJECT_ROOT), capture_output=True, text=True)
    if res.returncode != 0:
        print(f"Warning: {desc} failed: {res.stderr[:200]}", file=sys.stderr)
    return res.returncode == 0


def process_repo(repo_name):
    raw_path = CANDIDATES_DIR / f"{repo_name}_candidates.jsonl"
    filtered_path = CANDIDATES_DIR / f"{repo_name}_filtered.jsonl"
    scored_path = CANDIDATES_DIR / f"{repo_name}_scored.jsonl"
    contract_path = CANDIDATES_DIR / f"{repo_name}_contract_checked.jsonl"

    if not raw_path.exists():
        print(f"Skipping {repo_name}: {raw_path} not found.")
        return

    # 1. Filter candidates (retain all non-trivial deduplicated pairs)
    run_cmd(
        [sys.executable, "scripts/filter_real_drift_candidates.py", "--repo", repo_name, "--max-output", "100000"],
        f"Filtering candidates for {repo_name}",
    )

    # 2. Score priority
    run_cmd(
        [sys.executable, "scripts/score_review_priority.py", "--repo", repo_name],
        f"Scoring priority for {repo_name}",
    )

    # 3. Contract check
    run_cmd(
        [sys.executable, "scripts/contract_check_candidates.py", "--repo", repo_name],
        f"AST Contract Checking for {repo_name}",
    )


def main():
    repos = ["pytest", "sqlalchemy", "tornado", "celery", "django", "click", "fastapi"]
    for repo in repos:
        process_repo(repo)

    # Consolidate all into unified 15k dataset
    run_cmd([sys.executable, "scripts/consolidate_datasets.py"], "Consolidating all datasets into unified filtered_dataset")


if __name__ == "__main__":
    main()
