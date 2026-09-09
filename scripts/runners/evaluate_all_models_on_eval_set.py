#!/usr/bin/env python3
"""
scripts/runners/evaluate_all_models_on_eval_set.py — Benchmark All 5 Models on V2 Eval Set.

Orchestrates the evaluation of all 5 SemDrift model families against the 2,200-sample
V2 evaluation set (data/v2_real_world/evaluation/eval_set.jsonl):
  1. Zero-Shot Baseline (Pretrained CodeBERT + Cosine Divergence)
  2. Pure Lexical Baseline (TF-IDF Combined + Logistic Regression)
  3. Enhanced Lexical Baseline (TF-IDF Relational / Dual Overlap + Logistic Regression)
  4. Fine-Tuned Dual-Encoder (CodeBERT Bi-Encoder Checkpoint)
  5. Fine-Tuned Joint-Encoder (CodeBERT Cross-Input Self-Attention Checkpoint)

Generates per-model prediction JSONLs, individual result summaries, a master
summary JSON, and a formatted comparison Markdown report.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


ALL_MODELS = [
    "zero_shot",
    "tfidf_combined",
    "tfidf_relational",
    "dual_encoder",
    "joint_encoder",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate All 5 SemDrift Models on V2 Eval Set")
    default_test = "data/v2_real_world/evaluation/Evaluation_set2.jsonl"

    parser.add_argument(
        "--test_file",
        default=default_test,
        help="Path to the test set JSONL (default: data/v2_real_world/evaluation/Evaluation_set2.jsonl)",
    )
    parser.add_argument(
        "--val_file",
        default="experiments/2026-09-07_clean_v2/dataset/val.jsonl",
        help="Path to validation set for threshold tuning (default: experiments/2026-09-07_clean_v2/dataset/val.jsonl)",
    )
    parser.add_argument(
        "--train_file",
        default="experiments/2026-09-07_clean_v2/dataset/train.jsonl",
        help="Path to training set for TF-IDF fitting (default: experiments/2026-09-07_clean_v2/dataset/train.jsonl)",
    )
    parser.add_argument(
        "--output_dir",
        default="experiments/2026-09-07_clean_v2/evaluation/evaluation_set2_benchmarks",
        help="Root output directory for evaluation records and predictions",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=ALL_MODELS,
        default=ALL_MODELS,
        help="List of models to evaluate (default: all 5 models)",
    )
    parser.add_argument(
        "--device",
        default="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") != "" else "cpu",
        help="Execution device (cuda/cpu)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for neural model inference",
    )
    parser.add_argument(
        "--max_iter",
        type=int,
        default=5000,
        help="Max L-BFGS iterations for TF-IDF baselines",
    )
    return parser.parse_args()


def run_command(cmd: List[str], desc: str) -> None:
    """Execute a subprocess command with streaming stdout."""
    print(f"\n>>> Running: {desc}")
    print(f"    Command: {' '.join(cmd)}\n")
    start_time = time.time()
    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    elapsed = time.time() - start_time
    if proc.returncode != 0:
        print(f"[-] ERROR: {desc} failed with return code {proc.returncode} ({elapsed:.1f}s)")
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")
    print(f"[+] COMPLETED: {desc} ({elapsed:.1f}s)")


def evaluate_zero_shot(args, out_dir: Path) -> Path:
    results_file = out_dir / "zero_shot" / "results_baseline.json"
    cmd = [
        sys.executable,
        "scripts/training/run_zero_shot_baseline.py",
        "--dataset_generation", "v2",
        "--val", str(args.val_file),
        "--test", str(args.test_file),
        "--output_dir", str(out_dir / "zero_shot"),
        "--device", str(args.device),
        "--batch_size", str(args.batch_size),
    ]
    run_command(cmd, "1/5 Zero-Shot CodeBERT Baseline")
    return results_file


def evaluate_tfidf_combined(args, out_dir: Path) -> Path:
    results_file = out_dir / "tfidf_combined" / "results_tfidf_combined.json"
    cmd = [
        sys.executable,
        "scripts/training/run_tfidf_baseline.py",
        "--feature_mode", "combined",
        "--tune_threshold",
        "--max_iter", str(args.max_iter),
        "--train_file", str(args.train_file),
        "--val_file", str(args.val_file),
        "--test_file", str(args.test_file),
        "--output_dir", str(out_dir / "tfidf_combined"),
    ]
    run_command(cmd, "2/5 Pure Lexical Baseline (TF-IDF Combined)")
    return results_file


def evaluate_tfidf_relational(args, out_dir: Path) -> Path:
    results_file = out_dir / "tfidf_relational" / "results_tfidf_dual_overlap.json"
    if not results_file.is_file():
        results_file = out_dir / "tfidf_relational" / "results_tfidf_relational.json"
    cmd = [
        sys.executable,
        "scripts/training/run_tfidf_baseline.py",
        "--feature_mode", "dual_overlap",
        "--tune_threshold",
        "--max_iter", str(args.max_iter),
        "--train_file", str(args.train_file),
        "--val_file", str(args.val_file),
        "--test_file", str(args.test_file),
        "--output_dir", str(out_dir / "tfidf_relational"),
    ]
    run_command(cmd, "3/5 Enhanced Lexical Baseline (TF-IDF Relational / Dual Overlap)")
    if (out_dir / "tfidf_relational" / "results_tfidf_dual_overlap.json").is_file():
        return out_dir / "tfidf_relational" / "results_tfidf_dual_overlap.json"
    return results_file


def evaluate_dual_encoder(args, out_dir: Path) -> Path:
    results_file = out_dir / "dual_encoder" / "results_dual_encoder.json"
    ckpt_file = PROJECT_ROOT / "experiments/2026-09-07_clean_v2/dual_encoder_checkpoints/dual_encoder_checkpoint.pt"
    if not ckpt_file.is_file():
        raise FileNotFoundError(f"Dual-Encoder checkpoint not found at: {ckpt_file}")

    cmd = [
        sys.executable,
        "scripts/training/evaluate_dual_encoder.py",
        "--checkpoint", str(ckpt_file),
        "--test_file", str(args.test_file),
        "--output_results", str(results_file),
        "--output_preds", str(out_dir / "dual_encoder" / "predictions_dual_encoder.jsonl"),
        "--device", str(args.device),
        "--batch_size", str(args.batch_size),
    ]
    run_command(cmd, "4/5 Fine-Tuned Dual-Encoder Checkpoint")
    return results_file


def evaluate_joint_encoder(args, out_dir: Path) -> Path:
    results_file = out_dir / "joint_encoder" / "results_joint_encoder.json"
    ckpt_file = PROJECT_ROOT / "experiments/2026-09-07_clean_v2/checkpoints/joint_encoder_checkpoint.pt"
    if not ckpt_file.is_file():
        raise FileNotFoundError(f"Joint-Encoder checkpoint not found at: {ckpt_file}")

    cmd = [
        sys.executable,
        "experiments/2026-09-07_clean_v2/evaluation/independent_evaluate.py",
        "--checkpoint", str(ckpt_file),
        "--test_file", str(args.test_file),
        "--skip_integrity",
        "--output_results", str(results_file),
        "--output_preds", str(out_dir / "joint_encoder" / "predictions_joint_encoder.jsonl"),
        "--device", str(args.device),
        "--batch_size", str(args.batch_size),
    ]
    run_command(cmd, "5/5 Fine-Tuned Joint-Encoder Checkpoint")
    return results_file


def extract_standard_metrics(model_id: str, results_path: Path) -> Dict[str, Any]:
    """Extract uniform metrics from different model result formats."""
    if not results_path.is_file():
        return {"model_id": model_id, "status": "failed", "error": "file_not_found"}

    with results_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # Resolve overall metrics block
    m = data.get("overall_metrics") or data.get("test_overall") or {}
    cm = data.get("confusion_matrix") or {}
    tn = m.get("tn", cm.get("tn", 0))
    fp = m.get("fp", cm.get("fp", 0))
    fn = m.get("fn", cm.get("fn", 0))
    tp = m.get("tp", cm.get("tp", 0))

    threshold = (
        data.get("optimal_threshold")
        or data.get("threshold")
        or data.get("threshold_tuning", {}).get("selected_threshold")
        or 0.50
    )

    return {
        "model_id": model_id,
        "threshold": round(float(threshold), 4),
        "accuracy": m.get("accuracy", 0.0),
        "balanced_accuracy": m.get("balanced_accuracy", 0.0),
        "recall": m.get("recall", 0.0),
        "precision": m.get("precision", 0.0),
        "f1": m.get("f1", 0.0),
        "macro_f1": m.get("macro_f1", 0.0),
        "roc_auc": m.get("roc_auc"),
        "pr_auc": m.get("pr_auc"),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "confusion_matrix": f"TN={tn}, FP={fp}, FN={fn}, TP={tp}",
        "raw_data": data,
    }


def generate_markdown_report(metrics_list: List[Dict[str, Any]], test_samples: int, output_path: Path) -> str:
    """Generate Markdown benchmark table report."""
    display_names = {
        "zero_shot": "Zero-Shot CodeBERT",
        "tfidf_combined": "TF-IDF Combined (Pure Lexical)",
        "tfidf_relational": "TF-IDF Relational (Dual Overlap)",
        "dual_encoder": "Dual-Encoder (CodeBERT)",
        "joint_encoder": "Joint-Encoder (CodeBERT)",
    }

    categories = {
        "zero_shot": "Pretrained Semantic",
        "tfidf_combined": "Pure Lexical Baseline",
        "tfidf_relational": "Enhanced Lexical Baseline",
        "dual_encoder": "Fine-Tuned Bi-Encoder",
        "joint_encoder": "Fine-Tuned Cross-Input",
    }

    lines = [
        f"# SemDrift V2 Evaluation Set Benchmark Report",
        f"",
        f"This report records the comparative evaluation of all 5 SemDrift model families on the evaluation set ($N = {test_samples:,}$ samples).",
        f"",
        f"---",
        f"",
        f"## 1. Master Performance Comparison Table",
        f"",
        f"| Model Architecture | Category | Threshold (tau) | Accuracy | Balanced Acc | Drift Recall | Drift Precision | Binary F1 | Macro F1 | TN | FP | FN | TP | Confusion Matrix |",
        f"| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |",
    ]

    for m in metrics_list:
        mid = m["model_id"]
        dname = display_names.get(mid, mid)
        cat = categories.get(mid, "Baseline")
        tau = f"{m['threshold']:.2f}"
        acc = f"{m['accuracy'] * 100:.2f}%" if m['accuracy'] <= 1.0 else f"{m['accuracy']:.2f}%"
        bal = f"{m['balanced_accuracy'] * 100:.2f}%" if m['balanced_accuracy'] <= 1.0 else f"{m['balanced_accuracy']:.2f}%"
        rec = f"{m['recall'] * 100:.2f}%" if m['recall'] <= 1.0 else f"{m['recall']:.2f}%"
        prec = f"{m['precision'] * 100:.2f}%" if m['precision'] <= 1.0 else f"{m['precision']:.2f}%"
        f1 = f"{m['f1'] * 100:.2f}%" if m['f1'] <= 1.0 else f"{m['f1']:.2f}%"
        mf1 = f"{m['macro_f1'] * 100:.2f}%" if m['macro_f1'] <= 1.0 else f"{m['macro_f1']:.2f}%"
        tn, fp, fn, tp = m["tn"], m["fp"], m["fn"], m["tp"]
        cm_str = f"`TN={tn}, FP={fp}, FN={fn}, TP={tp}`"

        lines.append(
            f"| **{dname}** | {cat} | `{tau}` | {acc} | {bal} | {rec} | {prec} | {f1} | {mf1} | {tn} | {fp} | {fn} | {tp} | {cm_str} |"
        )

    lines.extend([
        f"",
        f"---",
        f"",
        f"## 2. Model Breakdown Highlights",
        f"",
        f"Detailed per-model breakdown files (mutation types, repository distributions, and severity tiers) are stored in individual subdirectories under `experiments/2026-09-07_clean_v2/evaluation/eval_set_benchmarks/`.",
    ])

    report_text = "\n".join(lines)
    output_path.write_text(report_text, encoding="utf-8")
    return report_text


def main():
    args = parse_args()
    out_dir = PROJECT_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    test_file_path = PROJECT_ROOT / args.test_file
    if not test_file_path.is_file():
        raise FileNotFoundError(f"Test set file not found: {test_file_path}")

    # Count test samples
    with test_file_path.open("r", encoding="utf-8") as f:
        test_samples = sum(1 for line in f if line.strip())

    print("=" * 80)
    print("SEMDRIFT V2 EVAL SET BENCHMARK RUNNER (ALL 5 MODELS)")
    print(f"Test Set    : {test_file_path} (N={test_samples:,})")
    print(f"Val Set     : {PROJECT_ROOT / args.val_file}")
    print(f"Train Set   : {PROJECT_ROOT / args.train_file}")
    print(f"Output Dir  : {out_dir}")
    print(f"Device      : {args.device}")
    print(f"Models      : {', '.join(args.models)}")
    print("=" * 80)

    results_files = {}

    for model_id in args.models:
        print(f"\n{'#' * 80}")
        print(f"STARTING EVALUATION: {model_id.upper()}")
        print(f"{'#' * 80}")
        if model_id == "zero_shot":
            results_files[model_id] = evaluate_zero_shot(args, out_dir)
        elif model_id == "tfidf_combined":
            results_files[model_id] = evaluate_tfidf_combined(args, out_dir)
        elif model_id == "tfidf_relational":
            results_files[model_id] = evaluate_tfidf_relational(args, out_dir)
        elif model_id == "dual_encoder":
            results_files[model_id] = evaluate_dual_encoder(args, out_dir)
        elif model_id == "joint_encoder":
            results_files[model_id] = evaluate_joint_encoder(args, out_dir)

    # Collect and compile master summary
    print("\n" + "=" * 80)
    print("COMPILING MASTER BENCHMARK COMPARISON REPORT")
    print("=" * 80)

    summary_list = []
    clean_summary_for_json = {}

    for model_id in args.models:
        rf = results_files.get(model_id)
        if rf and rf.is_file():
            metrics = extract_standard_metrics(model_id, rf)
            summary_list.append(metrics)
            clean_summary_for_json[model_id] = {k: v for k, v in metrics.items() if k != "raw_data"}

    # Save summary JSON
    summary_json_path = out_dir / "summary_eval_set_results.json"
    with summary_json_path.open("w", encoding="utf-8") as f:
        json.dump(clean_summary_for_json, f, indent=2)
    print(f"[OK] Saved summary JSON: {summary_json_path}")

    # Save Markdown report
    report_md_path = out_dir / "EVAL_SET_BENCHMARK_REPORT.md"
    report_md = generate_markdown_report(summary_list, test_samples, report_md_path)
    print(f"[OK] Saved Markdown Report: {report_md_path}\n")

    # Print markdown table to stdout
    print(report_md)
    print("\n" + "=" * 80)
    print("ALL 5 MODELS SUCCESSFULLY EVALUATED ON EVAL SET!")
    print("=" * 80)


if __name__ == "__main__":
    main()
