#!/usr/bin/env python3
"""
scripts/training/evaluate_sliding_window_joint.py — Phase 1 Sliding-Window Evaluation & Threshold Sweep.

Evaluates an existing Joint-Encoder checkpoint (e.g. joint_doc192) using sliding-window
inference on the validation split. Compares:
1. Exact single-window baseline (Window 0, identical to original make_collate_fn)
2. Mean-probability aggregation
3. Max-probability aggregation

Performs a full threshold sweep (tau in [0.10, 0.90]) strictly on validation data,
measures end-to-end latency stratified by window count K, and locks the optimal configuration.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
)
from transformers import AutoTokenizer

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from semdrift.models.joint_encoder import JointEncoderModel, SemDriftDataset
from semdrift.models.sliding_window import (
    slice_docstring_windows,
    assemble_joint_sequence,
)

DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def calculate_metrics(y_true: list[int], y_pred: list[int], y_probs: list[float] = None) -> dict:
    """Calculate standard classification metrics."""
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    _, _, macro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    roc_auc = None
    pr_auc = None
    if y_probs is not None and len(set(y_true)) > 1:
        try:
            roc_auc = float(roc_auc_score(y_true, y_probs))
            pr_auc = float(average_precision_score(y_true, y_probs))
        except Exception:
            pass

    return {
        "accuracy": round(float(acc), 4),
        "balanced_accuracy": round(float(bal_acc), 4),
        "precision": round(float(p), 4),
        "recall": round(float(r), 4),
        "f1": round(float(f1), 4),
        "macro_f1": round(float(macro_f1), 4),
        "roc_auc": round(roc_auc, 4) if roc_auc is not None else None,
        "pr_auc": round(pr_auc, 4) if pr_auc is not None else None,
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
        "confusion_matrix": f"TN={cm[0, 0]}, FP={cm[0, 1]}, FN={cm[1, 0]}, TP={cm[1, 1]}",
        "count": len(y_true),
    }


def warmup_gpu(model: torch.nn.Module, device: str, num_batches: int = 10, batch_size: int = 16):
    """Run warm-up forward passes to initialize CUDA kernels."""
    if device != "cuda" and not device.startswith("cuda"):
        return
    print(f"Executing {num_batches} CUDA warm-up forward passes...")
    dummy_input = {
        "input_ids": torch.randint(0, 1000, (batch_size, 512), dtype=torch.long, device=device),
        "attention_mask": torch.ones((batch_size, 512), dtype=torch.long, device=device),
    }
    with torch.no_grad():
        for _ in range(num_batches):
            _ = model(dummy_input)
    torch.cuda.synchronize()
    print("[OK] CUDA warm-up complete.")


def run_sliding_window_inference(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    records: list[dict],
    device: str,
    max_length: int = 512,
    window_size: int = 192,
    stride: int = 128,
    code_truncation: str = "head_tail",
) -> Tuple[list[dict], dict]:
    """Run sliding-window inference with exact per-function end-to-end latency measurement.

    For each function:
    - Generates all K windows and joint sequences.
    - Batches its K windows into a single forward pass through the model.
    - Aggregates window probabilities (Single, Mean, Max).
    - Measures exact end-to-end time (tokenization + slicing + assembly + GPU forward + aggregation).

    Returns:
        results: list of per-sample results containing probabilities, window counts, latencies.
        latency_stats: stratified latency statistics by window count K.
    """
    model.eval()
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 1

    results = []
    strata_latencies = defaultdict(list)

    total_samples = len(records)
    print(f"Running sliding-window inference on {total_samples} validation samples...")

    for i, rec in enumerate(records):
        doc = rec.get("docstring", "")
        code = rec.get("code", "")

        if device.startswith("cuda"):
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        # 1. Tokenization & Window Slicing
        # Suppress length warning by using truncation=False and encoding cleanly
        doc_ids = tokenizer.encode(doc, add_special_tokens=False, verbose=False)
        code_ids = tokenizer.encode(code, add_special_tokens=False, verbose=False)

        windows = slice_docstring_windows(doc_ids, window_size=window_size, stride=stride)
        k = len(windows)

        # 2. Sequence Assembly
        func_sequences = []
        for win in windows:
            seq = assemble_joint_sequence(
                doc_window_ids=win,
                code_ids=code_ids,
                tokenizer=tokenizer,
                max_length=max_length,
                code_truncation=code_truncation,
            )
            func_sequences.append(seq)

        # 3. Batch Tensor Creation (K windows batched together)
        max_func_len = max(len(s) for s in func_sequences)
        padded_ids = []
        padded_masks = []
        for s in func_sequences:
            pad_len = max_func_len - len(s)
            padded_ids.append(s + [pad_token_id] * pad_len)
            padded_masks.append([1] * len(s) + [0] * pad_len)

        tensor_inputs = {
            "input_ids": torch.tensor(padded_ids, dtype=torch.long, device=device),
            "attention_mask": torch.tensor(padded_masks, dtype=torch.long, device=device),
        }

        # 4. Model Forward Pass
        with torch.no_grad():
            logits = model(tensor_inputs)
            func_probs = F.softmax(logits, dim=1)[:, 1].cpu().tolist()  # P(drifted) for all K windows

        # 5. Probability Aggregation
        p_single = float(func_probs[0])
        p_mean = float(np.mean(func_probs))
        p_max = float(np.max(func_probs))

        if device.startswith("cuda"):
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        func_duration_ms = (t1 - t0) * 1000.0

        res = {
            "record": rec,
            "label": rec["label_int"],
            "window_count": k,
            "window_probs": [round(p, 6) for p in func_probs],
            "p_single": round(p_single, 6),
            "p_mean": round(p_mean, 6),
            "p_max": round(p_max, 6),
            "latency_ms": round(func_duration_ms, 2),
        }
        results.append(res)

        # Stratify latency by window count
        stratum = f"K={k}" if k <= 3 else "K>=4"
        strata_latencies[stratum].append(func_duration_ms)
        strata_latencies["Overall"].append(func_duration_ms)

        if (i + 1) % 50 == 0 or (i + 1 == total_samples):
            print(f"  Processed {i + 1:3d} / {total_samples} samples...")

    # Compute stratified latency metrics
    latency_summary = {}
    for stratum, vals in sorted(strata_latencies.items()):
        arr = np.array(vals)
        latency_summary[stratum] = {
            "count": len(arr),
            "mean_ms": round(float(np.mean(arr)), 2),
            "median_ms": round(float(np.median(arr)), 2),
            "p95_ms": round(float(np.percentile(arr, 95)), 2),
            "min_ms": round(float(np.min(arr)), 2),
            "max_ms": round(float(np.max(arr)), 2),
        }

    return results, latency_summary


def sweep_thresholds(
    y_true: list[int],
    y_probs: list[float],
    thresholds: list[float],
) -> Tuple[list[dict], dict]:
    """Sweep decision thresholds and find optimal threshold by Macro-F1."""
    sweep_results = []
    best_entry = None
    best_macro_f1 = -1.0
    best_bal_acc = -1.0

    for tau in thresholds:
        tau_val = round(float(tau), 4)
        y_pred = [1 if p >= tau_val else 0 for p in y_probs]
        metrics = calculate_metrics(y_true, y_pred, y_probs)
        metrics["threshold"] = tau_val
        sweep_results.append(metrics)

        # Select by Macro-F1 with balanced accuracy as tie-breaker
        if (metrics["macro_f1"] > best_macro_f1) or (
            metrics["macro_f1"] == best_macro_f1 and metrics["balanced_accuracy"] > best_bal_acc
        ):
            best_macro_f1 = metrics["macro_f1"]
            best_bal_acc = metrics["balanced_accuracy"]
            best_entry = metrics

    return sweep_results, best_entry


def parse_args():
    parser = argparse.ArgumentParser(
        description="Phase 1: Sliding-Window Joint-Encoder Validation Sweep & Latency Benchmark"
    )
    parser.add_argument(
        "--checkpoint",
        default="experiments/expansion_v1/joint_doc192/joint_encoder_checkpoint.pt",
        help="Path to joint_doc192 checkpoint (.pt)",
    )
    parser.add_argument(
        "--val_file",
        default="data/real_dataset_expansion_v1/splits/val.jsonl",
        help="Path to validation JSONL split",
    )
    parser.add_argument(
        "--output_dir",
        default="experiments/expansion_v1/joint_sliding_window",
        help="Output directory to save results, sweep data, and locked config",
    )
    parser.add_argument(
        "--device",
        default=DEFAULT_DEVICE,
        help="Execution device (cuda/cpu)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size of functions for inference",
    )
    parser.add_argument(
        "--model_name",
        default="microsoft/codebert-base",
        help="Pretrained backbone name",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum total sequence length",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=192,
        help="Documentation window size in tokens",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=128,
        help="Documentation window stride in tokens",
    )
    parser.add_argument(
        "--code_truncation",
        choices=["head", "tail", "head_tail"],
        default="head_tail",
        help="Truncation strategy for code segment",
    )
    parser.add_argument(
        "--smoke_test",
        action="store_true",
        help="Run smoke test on first 16 validation samples only",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = PROJECT_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = PROJECT_ROOT / args.checkpoint
    val_path = PROJECT_ROOT / args.val_file

    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    if not val_path.is_file():
        raise FileNotFoundError(f"Validation dataset not found: {val_path}")

    print("=" * 75)
    print("PHASE 1: SLIDING-WINDOW JOINT-ENCODER VALIDATION EVALUATION")
    print("=" * 75)
    print(f"Checkpoint   : {ckpt_path}")
    print(f"Validation   : {val_path}")
    print(f"Device       : {args.device}")
    print(f"Batch Size   : {args.batch_size}")
    print(f"Window / Stride : {args.window_size} / {args.stride}")
    print(f"Max Length   : {args.max_length}")
    print(f"Output Dir   : {out_dir}")
    print(f"Smoke Test   : {args.smoke_test}")
    print("=" * 75)

    # 1. Load Dataset (clean_docs=False to preserve raw multi-paragraph documentation)
    print(f"Loading validation dataset from {val_path} (clean_docs=False)...")
    dataset = SemDriftDataset(str(val_path), clean_docs=False)
    records = dataset.records
    if args.smoke_test:
        records = records[:16]
        print(f"[Smoke Test] Subsetting to {len(records)} samples.")
    else:
        print(f"[OK] Loaded {len(records)} validation samples.")

    # 2. Instantiate Model and Load Checkpoint
    print(f"Instantiating JointEncoderModel ({args.model_name})...")
    model = JointEncoderModel(
        model_name=args.model_name,
        pooling="cls",
        num_labels=2,
        dropout=0.1,
    )
    print(f"Loading checkpoint weights from {ckpt_path}...")
    state_dict = torch.load(ckpt_path, map_location=args.device)
    model.load_state_dict(state_dict)
    model.to(args.device)
    model.eval()
    print("[OK] Model weights successfully restored to evaluation mode.")

    # 3. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # 4. CUDA Warmup
    warmup_gpu(model, args.device, num_batches=10, batch_size=args.batch_size)

    # 5. Run Sliding-Window Inference & Latency Tracking
    results, latency_summary = run_sliding_window_inference(
        model=model,
        tokenizer=tokenizer,
        records=records,
        device=args.device,
        max_length=args.max_length,
        window_size=args.window_size,
        stride=args.stride,
        code_truncation=args.code_truncation,
    )

    y_true = [r["label"] for r in results]
    probs_single = [r["p_single"] for r in results]
    probs_mean = [r["p_mean"] for r in results]
    probs_max = [r["p_max"] for r in results]

    # 6. Window Count Distribution
    window_counts = [r["window_count"] for r in results]
    win_dist = defaultdict(int)
    for k in window_counts:
        win_dist[k] += 1

    print("\n" + "-" * 75)
    print("DOCUMENTATION WINDOW DISTRIBUTION:")
    for k in sorted(win_dist.keys()):
        count = win_dist[k]
        pct = (count / len(records)) * 100
        print(f"  K = {k} window(s) : {count:4d} samples ({pct:5.1f}%)")
    print("-" * 75)

    # 7. Latency Summary Table
    print("\n" + "-" * 75)
    print("END-TO-END LATENCY PROFILING (ms / sample):")
    print(f"  {'Stratum':<10} {'Count':<8} {'Mean (ms)':<12} {'Median (ms)':<14} {'P95 (ms)':<12}")
    for stratum, s_vals in sorted(latency_summary.items()):
        print(f"  {stratum:<10} {s_vals['count']:<8} {s_vals['mean_ms']:<12.2f} {s_vals['median_ms']:<14.2f} {s_vals['p95_ms']:<12.2f}")
    print("-" * 75)

    # 8. Threshold Sweeps (tau in [0.10, 0.90], step 0.02)
    thresholds = [round(0.10 + 0.02 * step, 4) for step in range(41)]  # 0.10 to 0.90
    print("\nSweeping decision thresholds on validation set (tau in [0.10, 0.90], step 0.02)...")

    sweep_single, best_single = sweep_thresholds(y_true, probs_single, thresholds)
    sweep_mean, best_mean = sweep_thresholds(y_true, probs_mean, thresholds)
    sweep_max, best_max = sweep_thresholds(y_true, probs_max, thresholds)

    # Default tau = 0.50 metrics for comparison
    def get_tau_050(sweep_list):
        for entry in sweep_list:
            if abs(entry["threshold"] - 0.50) < 1e-4:
                return entry
        return sweep_list[len(sweep_list) // 2]

    tau050_single = get_tau_050(sweep_single)
    tau050_mean = get_tau_050(sweep_mean)
    tau050_max = get_tau_050(sweep_max)

    print("\n" + "=" * 75)
    print("VALIDATION PERFORMANCE SUMMARY (N = 544)")
    print("=" * 75)
    print(f"{'Configuration':<30} {'Threshold':<11} {'Macro-F1':<11} {'Bal Acc':<11} {'Recall':<10} {'Precision':<11} {'Confusion Matrix'}")
    print("-" * 75)
    print(f"{'Single Baseline (Default)':<30} {tau050_single['threshold']:<11.2f} {tau050_single['macro_f1']:<11.4f} {tau050_single['balanced_accuracy']:<11.4f} {tau050_single['recall']:<10.4f} {tau050_single['precision']:<11.4f} {tau050_single['confusion_matrix']}")
    print(f"{'Single Baseline (Tuned)':<30} {best_single['threshold']:<11.2f} {best_single['macro_f1']:<11.4f} {best_single['balanced_accuracy']:<11.4f} {best_single['recall']:<10.4f} {best_single['precision']:<11.4f} {best_single['confusion_matrix']}")
    print("-" * 75)
    print(f"{'Sliding Mean (Default)':<30} {tau050_mean['threshold']:<11.2f} {tau050_mean['macro_f1']:<11.4f} {tau050_mean['balanced_accuracy']:<11.4f} {tau050_mean['recall']:<10.4f} {tau050_mean['precision']:<11.4f} {tau050_mean['confusion_matrix']}")
    print(f"{'Sliding Mean (Tuned)':<30} {best_mean['threshold']:<11.2f} {best_mean['macro_f1']:<11.4f} {best_mean['balanced_accuracy']:<11.4f} {best_mean['recall']:<10.4f} {best_mean['precision']:<11.4f} {best_mean['confusion_matrix']}")
    print("-" * 75)
    print(f"{'Sliding Max (Default)':<30} {tau050_max['threshold']:<11.2f} {tau050_max['macro_f1']:<11.4f} {tau050_max['balanced_accuracy']:<11.4f} {tau050_max['recall']:<10.4f} {tau050_max['precision']:<11.4f} {tau050_max['confusion_matrix']}")
    print(f"{'Sliding Max (Tuned)':<30} {best_max['threshold']:<11.2f} {best_max['macro_f1']:<11.4f} {best_max['balanced_accuracy']:<11.4f} {best_max['recall']:<10.4f} {best_max['precision']:<11.4f} {best_max['confusion_matrix']}")
    print("=" * 75)

    # 9. Determine Winning Strategy for Phase 2 Locking
    candidates = [
        ("single_baseline", best_single, probs_single),
        ("sliding_mean", best_mean, probs_mean),
        ("sliding_max", best_max, probs_max),
    ]
    # Sort candidates by Macro-F1 descending, then balanced accuracy descending
    candidates.sort(key=lambda c: (c[1]["macro_f1"], c[1]["balanced_accuracy"]), reverse=True)
    winning_strategy, winning_metrics, winning_probs = candidates[0]

    print(f"\n[WINNER] Selected Configuration for Phase 2 Locking:")
    print(f"  Strategy         : {winning_strategy}")
    print(f"  Locked Threshold : {winning_metrics['threshold']:.2f}")
    print(f"  Val Macro-F1     : {winning_metrics['macro_f1']:.4f}")
    print(f"  Val Balanced Acc : {winning_metrics['balanced_accuracy']:.4f}")
    print(f"  Val Confusion    : {winning_metrics['confusion_matrix']}")

    # 10. Save Outputs
    # A. Locked Configuration
    locked_config = {
        "status": "LOCKED_FROM_PHASE_1_VALIDATION",
        "checkpoint": str(ckpt_path),
        "validation_dataset": str(val_path),
        "sample_count": len(records),
        "selected_strategy": winning_strategy,
        "locked_threshold": winning_metrics["threshold"],
        "validation_metrics": winning_metrics,
        "single_baseline_comparison": {
            "default_tau_050": tau050_single,
            "tuned_val": best_single,
        },
        "sliding_mean_comparison": {
            "default_tau_050": tau050_mean,
            "tuned_val": best_mean,
        },
        "sliding_max_comparison": {
            "default_tau_050": tau050_max,
            "tuned_val": best_max,
        },
        "window_distribution": dict(win_dist),
        "latency_summary": latency_summary,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    locked_config_path = out_dir / "selected_config.json"
    with locked_config_path.open("w", encoding="utf-8") as f:
        json.dump(locked_config, f, indent=2)
    print(f"[OK] Saved locked configuration to {locked_config_path}")

    # B. Complete Threshold Sweep
    sweep_export = {
        "thresholds_evaluated": thresholds,
        "single_baseline_sweep": sweep_single,
        "sliding_mean_sweep": sweep_mean,
        "sliding_max_sweep": sweep_max,
    }
    sweep_path = out_dir / "threshold_sweep_validation.json"
    with sweep_path.open("w", encoding="utf-8") as f:
        json.dump(sweep_export, f, indent=2)
    print(f"[OK] Saved complete threshold sweep to {sweep_path}")

    # C. Per-Function Predictions Export
    preds_path = out_dir / "validation_predictions.jsonl"
    with preds_path.open("w", encoding="utf-8") as f:
        for r, w_prob in zip(results, winning_probs):
            pred_int = 1 if w_prob >= winning_metrics["threshold"] else 0
            out_rec = dict(r["record"])
            out_rec["window_count"] = r["window_count"]
            out_rec["window_probs"] = r["window_probs"]
            out_rec["p_single"] = r["p_single"]
            out_rec["p_mean"] = r["p_mean"]
            out_rec["p_max"] = r["p_max"]
            out_rec["selected_strategy"] = winning_strategy
            out_rec["selected_prob"] = w_prob
            out_rec["locked_threshold"] = winning_metrics["threshold"]
            out_rec["predicted_label"] = "drifted" if pred_int == 1 else "aligned"
            out_rec["predicted_label_int"] = pred_int
            out_rec["latency_ms"] = r["latency_ms"]
            f.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
    print(f"[OK] Saved per-function predictions to {preds_path}")


if __name__ == "__main__":
    main()
