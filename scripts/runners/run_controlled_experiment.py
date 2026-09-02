#!/usr/bin/env python3
"""
scripts/runners/run_controlled_experiment.py — Controlled Joint vs. Dual Architectural Experiment.

Orchestrates the controlled experiment:
  1. Dual-Encoder  + CrossEntropy (dropout=0.1, lr=2e-5, epochs=3, seed=42)
  2. Joint-Encoder + CrossEntropy (dropout=0.1, lr=2e-5, epochs=3, seed=42, no_focal_loss, no_category_weighting)
  3. Analysis & significance testing (analyze_controlled_experiment.py)

Can print commands without executing via `--only_print`.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description="Run Controlled Dual vs Joint Experiment")
    parser.add_argument("--device", default="cuda", help="Execution device (cuda/cpu)")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate before classifier head")
    parser.add_argument("--checkpoint_metric", default="macro_f1", help="Validation metric for checkpoint selection")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--base_dir", default="data/experiments/v2", help="Base dataset directory")
    parser.add_argument("--only_print", action="store_true", help="Print commands without executing them")
    args = parser.parse_args()

    train_path = os.path.join(args.base_dir, "train.jsonl")
    val_path = os.path.join(args.base_dir, "val.jsonl")
    test_path = os.path.join(args.base_dir, "test.jsonl")

    ablation_dir = os.path.join(args.base_dir, "controlled_ablation")
    dual_out = os.path.join(ablation_dir, "dual_ce")
    joint_out = os.path.join(ablation_dir, "joint_ce")

    dual_cmd = [
        sys.executable,
        os.path.join("scripts", "training", "train_dual_encoder.py"),
        "--train", train_path,
        "--val", val_path,
        "--test", test_path,
        "--device", args.device,
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--lr", str(args.lr),
        "--dropout", str(args.dropout),
        "--checkpoint_metric", args.checkpoint_metric,
        "--seed", str(args.seed),
        "--output_dir", dual_out,
    ]

    joint_cmd = [
        sys.executable,
        os.path.join("scripts", "training", "train_joint_encoder.py"),
        "--train", train_path,
        "--val", val_path,
        "--test", test_path,
        "--device", args.device,
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--lr", str(args.lr),
        "--dropout", str(args.dropout),
        "--code_truncation", "head_tail",
        "--pooling", "cls",
        "--checkpoint_metric", args.checkpoint_metric,
        "--no_focal_loss",
        "--no_category_weighting",
        "--seed", str(args.seed),
        "--output_dir", joint_out,
    ]

    analyze_cmd = [
        sys.executable,
        os.path.join("scripts", "analysis", "analyze_controlled_experiment.py"),
        "--dual_preds", os.path.join(dual_out, "predictions_dual_encoder.jsonl"),
        "--joint_preds", os.path.join(joint_out, "predictions_joint_encoder.jsonl"),
        "--output_dir", ablation_dir,
    ]

    print("=" * 80)
    print("  CONTROLLED DUAL vs. JOINT ARCHITECTURAL EXPERIMENT")
    print("=" * 80)
    print(f"Backbone          : microsoft/codebert-base")
    print(f"Device            : {args.device}")
    print(f"Epochs            : {args.epochs}")
    print(f"Batch Size        : {args.batch_size}")
    print(f"Learning Rate     : {args.lr}")
    print(f"Dropout           : {args.dropout}")
    print(f"Checkpoint Metric : {args.checkpoint_metric}")
    print(f"Random Seed       : {args.seed}")
    print(f"Dual Objective    : Standard CrossEntropyLoss")
    print(f"Joint Objective   : Standard CrossEntropyLoss (Focal=OFF, CategoryWeighting=OFF)")
    print("-" * 80)

    if args.only_print:
        print("\n[Command 1/3 - Train Dual-CE]:")
        print(" ".join(dual_cmd))
        print("\n[Command 2/3 - Train Joint-CE]:")
        print(" ".join(joint_cmd))
        print("\n[Command 3/3 - Analyze Results]:")
        print(" ".join(analyze_cmd))
        return

    # 1. Run Dual-Encoder
    print("\n[1/3] Training Dual-Encoder + CrossEntropy...", flush=True)
    subprocess.run(dual_cmd, check=True)

    # 2. Run Joint-Encoder
    print("\n[2/3] Training Joint-Encoder + CrossEntropy...", flush=True)
    subprocess.run(joint_cmd, check=True)

    # 3. Analyze
    print("\n[3/3] Analyzing Controlled Results...", flush=True)
    subprocess.run(analyze_cmd, check=True)

    print("\nControlled experiment pipeline complete!", flush=True)


if __name__ == "__main__":
    main()
