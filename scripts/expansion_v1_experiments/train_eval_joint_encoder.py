#!/usr/bin/env python3
"""
scripts/expansion_v1_experiments/train_eval_joint_encoder.py

Fine-tunes and evaluates the Fine-Tuned Joint-Encoder (Primary Contribution)
with Focal Loss (gamma=2.0) and Category Weighting on Expansion V1 dataset (10,000 samples).
Trains on train split (8,128 samples), validates per-epoch on val split (544 samples) to pick best checkpoint (Macro-F1),
and performs test evaluation on test split (1,328 samples).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def parse_args():
    parser = argparse.ArgumentParser(description="Joint-Encoder (Focal) Training & Evaluation on Expansion V1 Dataset")
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
        default="experiments/expansion_v1/joint_encoder",
        help="Directory to save checkpoints, predictions, and results",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for training and evaluation",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-5,
        help="Learning rate",
    )
    parser.add_argument(
        "--focal_gamma",
        type=float,
        default=2.0,
        help="Gamma parameter for Focal Loss",
    )
    parser.add_argument(
        "--focal_alpha",
        type=float,
        default=0.5,
        help="Alpha parameter for Focal Loss",
    )
    parser.add_argument(
        "--code_truncation",
        choices=["head", "tail", "head_tail"],
        default="head_tail",
        help="Truncation strategy for code segment",
    )
    parser.add_argument(
        "--pooling",
        choices=["cls", "mean"],
        default="cls",
        help="Pooling strategy",
    )
    parser.add_argument(
        "--checkpoint_metric",
        choices=["macro_f1", "balanced_accuracy", "f1", "accuracy"],
        default="macro_f1",
        help="Checkpoint selection metric on validation set",
    )
    parser.add_argument(
        "--device",
        default="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") != "" else "cpu",
        help="Execution device (cuda/cpu)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Quick smoke test on 16 samples",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = PROJECT_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts/training/train_joint_encoder.py"),
        "--dataset_generation", "v2",
        "--train", str(PROJECT_ROOT / args.train_file),
        "--val", str(PROJECT_ROOT / args.val_file),
        "--test", str(PROJECT_ROOT / args.test_file),
        "--output_dir", str(out_dir),
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--lr", str(args.lr),
        "--use_focal_loss",
        "--focal_gamma", str(args.focal_gamma),
        "--focal_alpha", str(args.focal_alpha),
        "--category_weighting",
        "--code_truncation", str(args.code_truncation),
        "--pooling", str(args.pooling),
        "--checkpoint_metric", str(args.checkpoint_metric),
        "--device", str(args.device),
        "--seed", str(args.seed),
        "--no_clean_docstrings",
    ]
    if args.dry_run:
        cmd.append("--dry_run")

    print("=" * 70)
    print("Fine-Tuning Joint-Encoder (Focal Loss & Category Weighting) on Expansion V1 (10K)")
    print(f"Train Set : {args.train_file}")
    print(f"Val Set   : {args.val_file}")
    print(f"Test Set  : {args.test_file}")
    print(f"Output Dir: {args.output_dir}")
    print(f"Epochs    : {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Objective : Focal Loss (gamma={args.focal_gamma}, alpha={args.focal_alpha}) + Category Weighting")
    print(f"Device    : {args.device}")
    print(f"Dry Run   : {args.dry_run}")
    print("=" * 70)

    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    if proc.returncode != 0:
        print(f"[-] Training failed with exit code {proc.returncode}")
        sys.exit(proc.returncode)

    print(f"\n[+] Joint-Encoder run complete. Results saved in {out_dir}")


if __name__ == "__main__":
    main()
