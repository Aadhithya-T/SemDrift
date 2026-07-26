#!/usr/bin/env python3
"""
mine_dataset.py — Mine commit history from git repositories using PyDriller.

Usage:
    python scripts/mine_dataset.py --repo <path_or_url> --output data/raw/

This script walks through the commit history of a repository, extracts
modified files, and stores raw diff / AST data for downstream processing.
"""

import argparse
import json
import os
from pathlib import Path

# PyDriller will be imported at runtime — listed in requirements.txt.
# from pydriller import Repository


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mine commit history for semantic drift analysis."
    )
    parser.add_argument(
        "--repo",
        type=str,
        required=True,
        help="Path or URL of the git repository to mine.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/raw/",
        help="Output directory for mined data.",
    )
    parser.add_argument(
        "--max-commits",
        type=int,
        default=None,
        help="Maximum number of commits to process (None = all).",
    )
    return parser.parse_args()


def mine(repo_path: str, output_dir: str, max_commits: int | None = None):
    """Mine commit data from the given repository.

    Parameters
    ----------
    repo_path : str
        Local path or remote URL of the repository.
    output_dir : str
        Directory where mined JSON records are saved.
    max_commits : int | None
        Cap on number of commits to process.
    """
    os.makedirs(output_dir, exist_ok=True)

    # TODO: Uncomment once pydriller is installed and implement extraction logic.
    # for commit in Repository(repo_path).traverse_commits():
    #     record = {
    #         "hash": commit.hash,
    #         "msg": commit.msg,
    #         "author": commit.author.name,
    #         "date": commit.author_date.isoformat(),
    #         "modifications": [],
    #     }
    #     for mod in commit.modified_files:
    #         record["modifications"].append({
    #             "filename": mod.filename,
    #             "change_type": mod.change_type.name,
    #             "diff": mod.diff,
    #             "source_code": mod.source_code,
    #             "source_code_before": mod.source_code_before,
    #         })
    #     out_path = Path(output_dir) / f"{commit.hash}.json"
    #     with open(out_path, "w") as f:
    #         json.dump(record, f, indent=2)
    print(f"[mine_dataset] Mining not yet implemented for: {repo_path}")


if __name__ == "__main__":
    args = parse_args()
    mine(args.repo, args.output, args.max_commits)
