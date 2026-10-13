#!/usr/bin/env python3
"""
scripts/expansion_v1_experiments/run_zero_shot.py

Evaluates Zero-Shot CodeBERT (Model A baseline) on Expansion V1 dataset (10,000 samples).
Sweeps decision threshold on validation set (544 samples) and evaluates on test set (1,328 samples).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def parse_args():
    parser = argparse.ArgumentParser(description="Zero-Shot CodeBERT on Expansion V1 Dataset")
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
        default="experiments/expansion_v1/zero_shot",
        help="Directory to save predictions and results",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for embedding generation",
    )
    parser.add_argument(
        "--device",
        default="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") != "" else "cpu",
        help="Execution device (cuda/cpu)",
    )
    parser.add_argument(
        "--metric",
        choices=["macro_f1", "balanced_accuracy", "f1", "accuracy"],
        default="macro_f1",
        help="Validation threshold sweep criterion",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = PROJECT_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts/training/run_zero_shot_baseline.py"),
        "--dataset_generation", "v2",
        "--val", str(PROJECT_ROOT / args.val_file),
        "--test", str(PROJECT_ROOT / args.test_file),
        "--output_dir", str(out_dir),
        "--batch_size", str(args.batch_size),
        "--device", str(args.device),
        "--metric", str(args.metric),
        "--no_clean_docstrings",
    ]

    print("=" * 70)
    print("Running Zero-Shot CodeBERT on Expansion V1 (10K)")
    print(f"Validation Set: {args.val_file}")
    print(f"Test Set      : {args.test_file}")
    print(f"Output Dir    : {args.output_dir}")
    print(f"Device        : {args.device}")
    print("=" * 70)

    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    if proc.returncode != 0:
        print(f"[-] Execution failed with exit code {proc.returncode}")
        sys.exit(proc.returncode)

    print(f"\n[+] Zero-Shot baseline run complete. Results saved in {out_dir}")


if __name__ == "__main__":
    main()
