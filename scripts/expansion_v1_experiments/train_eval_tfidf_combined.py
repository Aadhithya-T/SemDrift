#!/usr/bin/env python3
"""
scripts/expansion_v1_experiments/train_eval_tfidf_combined.py

Trains, tunes, and evaluates TF-IDF Combined (Pure Lexical Baseline) on Expansion V1 dataset (10,000 samples).
Trains on train split (8,128 samples), tunes threshold on val split (544 samples), and evaluates on test split (1,328 samples).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def parse_args():
    parser = argparse.ArgumentParser(description="TF-IDF Combined Baseline on Expansion V1 Dataset")
    parser.add_argument(
        "--train_file",
        default="data/real_dataset_expansion_v1/splits/train.jsonl",
        help="Path to training JSONL split",
    )
    parser.add_argument(
        "--val_file",
        default="data/real_dataset_expansion_v1/splits/val.jsonl",
        help="Path to validation JSONL split",
    )
    parser.add_argument(
        "--test_file",
        default="data/real_dataset_expansion_v1/splits/test.jsonl",
        help="Path to test JSONL split",
    )
    parser.add_argument(
        "--output_dir",
        default="experiments/expansion_v1/tfidf_combined",
        help="Directory to save model, predictions, and results",
    )
    parser.add_argument(
        "--max_iter",
        type=int,
        default=2000,
        help="Maximum solver iterations",
    )
    parser.add_argument(
        "--C",
        type=float,
        default=1.0,
        help="Inverse regularization strength",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = PROJECT_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts/training/run_tfidf_baseline.py"),
        "--feature_mode", "combined",
        "--model_type", "logistic",
        "--train_file", str(PROJECT_ROOT / args.train_file),
        "--val_file", str(PROJECT_ROOT / args.val_file),
        "--test_file", str(PROJECT_ROOT / args.test_file),
        "--tune_threshold",
        "--max_iter", str(args.max_iter),
        "--C", str(args.C),
        "--output_dir", str(out_dir),
        "--output_results", str(out_dir / "results_tfidf_combined.json"),
        "--output_preds", str(out_dir / "predictions_tfidf_combined.jsonl"),
        "--save_model", str(out_dir / "tfidf_combined_model.joblib"),
    ]

    print("=" * 70)
    print("Training TF-IDF Combined (Pure Lexical) on Expansion V1 (10K)")
    print(f"Train Set : {args.train_file}")
    print(f"Val Set   : {args.val_file}")
    print(f"Test Set  : {args.test_file}")
    print(f"Output Dir: {args.output_dir}")
    print("=" * 70)

    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    if proc.returncode != 0:
        print(f"[-] Execution failed with exit code {proc.returncode}")
        sys.exit(proc.returncode)

    print(f"\n[+] TF-IDF Combined run complete. Results saved in {out_dir}")


if __name__ == "__main__":
    main()
