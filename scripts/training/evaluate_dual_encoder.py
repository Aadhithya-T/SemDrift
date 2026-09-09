#!/usr/bin/env python3
"""
scripts/training/evaluate_dual_encoder.py — Independent Evaluation for Dual-Encoder Checkpoint.

Evaluates a saved Dual-Encoder checkpoint on any specified test set (e.g. eval_set.jsonl
or verified_test.jsonl) without requiring retraining.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)
from transformers import AutoTokenizer

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from semdrift.models.dual_encoder import (
    DualEncoderModel,
    DualEncoderDataset,
    make_collate_fn,
)

DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def calculate_metrics(y_true: list, y_pred: list, y_probs: list = None) -> dict:
    """Calculate standard SemDrift classification metrics."""
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


def evaluate_breakdowns(y_true: list, y_pred: list, metas: list) -> dict:
    """Evaluate fine-grained performance across sub-categories."""
    breakdowns = {
        "by_drift_type": defaultdict(lambda: {"yt": [], "yp": []}),
        "by_severity": defaultdict(lambda: {"yt": [], "yp": []}),
        "by_repo": defaultdict(lambda: {"yt": [], "yp": []}),
        "by_provenance": defaultdict(lambda: {"yt": [], "yp": []}),
    }

    for yt, yp, meta in zip(y_true, y_pred, metas):
        dt = meta.get("drift_type", "unknown")
        sev = meta.get("severity", "unknown")
        repo = meta.get("repo", "unknown")
        prov = meta.get("provenance", "unknown")

        breakdowns["by_drift_type"][dt]["yt"].append(yt)
        breakdowns["by_drift_type"][dt]["yp"].append(yp)

        breakdowns["by_severity"][sev]["yt"].append(yt)
        breakdowns["by_severity"][sev]["yp"].append(yp)

        breakdowns["by_repo"][repo]["yt"].append(yt)
        breakdowns["by_repo"][repo]["yp"].append(yp)

        breakdowns["by_provenance"][prov]["yt"].append(yt)
        breakdowns["by_provenance"][prov]["yp"].append(yp)

    res = {}
    for cat, groups in breakdowns.items():
        res[cat] = {}
        for k, v in sorted(groups.items()):
            if len(v["yt"]) > 0:
                res[cat][k] = calculate_metrics(v["yt"], v["yp"])

    return res


def compute_score_distributions(scores: list, groups: list) -> dict:
    grouped = defaultdict(list)
    for s, g in zip(scores, groups):
        grouped[g].append(s)

    stats = {}
    for g, vals in sorted(grouped.items()):
        arr = np.array(vals)
        stats[g] = {
            "count": len(arr),
            "mean": round(float(np.mean(arr)), 4),
            "median": round(float(np.median(arr)), 4),
            "std": round(float(np.std(arr)), 4),
            "min": round(float(np.min(arr)), 4),
            "max": round(float(np.max(arr)), 4),
        }
    return stats


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Fine-Tuned Dual-Encoder Checkpoint")
    parser.add_argument(
        "--checkpoint",
        default="experiments/2026-09-07_clean_v2/dual_encoder_checkpoints/dual_encoder_checkpoint.pt",
        help="Path to dual-encoder PyTorch checkpoint",
    )
    default_test = "data/v2_real_world/evaluation/Evaluation_set2.jsonl"

    parser.add_argument(
        "--test_file",
        default=default_test,
        help="Path to test JSONL file (default: data/v2_real_world/evaluation/Evaluation_set2.jsonl)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.50,
        help="Classification threshold for drift prediction",
    )
    parser.add_argument(
        "--output_results",
        default="experiments/2026-09-07_clean_v2/evaluation/results_dual_encoder.json",
        help="Path to save evaluation summary JSON",
    )
    parser.add_argument(
        "--output_preds",
        default="experiments/2026-09-07_clean_v2/predictions/predictions_dual_encoder.jsonl",
        help="Path to save per-instance predictions JSONL",
    )
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for DataLoader")
    parser.add_argument("--max_length", type=int, default=512, help="Max sequence length")
    parser.add_argument("--device", default=DEFAULT_DEVICE, help="Execution device (cuda/cpu)")
    parser.add_argument("--model_name", default="microsoft/codebert-base", help="Pretrained model identifier")
    return parser.parse_args()


def main():
    args = parse_args()

    ckpt_path = PROJECT_ROOT / args.checkpoint
    test_path = PROJECT_ROOT / args.test_file
    out_results_path = PROJECT_ROOT / args.output_results
    out_preds_path = PROJECT_ROOT / args.output_preds

    assert ckpt_path.is_file(), f"Checkpoint not found at: {ckpt_path}"
    assert test_path.is_file(), f"Test file not found at: {test_path}"

    print("=" * 70)
    print("SEMDRIFT DUAL-ENCODER INDEPENDENT EVALUATION")
    print(f"Checkpoint : {ckpt_path}")
    print(f"Test Set   : {test_path}")
    print(f"Device     : {args.device}")
    print(f"Threshold  : {args.threshold:.4f}")
    print("=" * 70)

    # 1. Load Dataset
    print(f"Loading test dataset from {test_path}...")
    test_dataset = DualEncoderDataset(str(test_path), clean_docs=False)
    print(f"[OK] Loaded {len(test_dataset)} test samples.")

    # 2. Instantiate Model and Load Weights
    print("Instantiating DualEncoderModel...")
    model = DualEncoderModel(
        model_name=args.model_name,
        variant="variant_2",
        dropout=0.1,
    )
    print(f"Loading checkpoint weights from {ckpt_path}...")
    state_dict = torch.load(ckpt_path, map_location=args.device)
    model.load_state_dict(state_dict)
    model.to(args.device)
    model.eval()
    print("[OK] Model restored to evaluation mode.")

    # 3. DataLoader
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    collate_fn = make_collate_fn(tokenizer, max_length=args.max_length)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # 4. Inference Loop
    print(f"Running inference on {len(test_dataset)} samples...")
    all_labels = []
    all_preds = []
    all_probs = []
    all_metas = []

    with torch.no_grad():
        for code_inputs, doc_inputs, labels, metas in test_loader:
            code_inputs = {k: v.to(args.device) for k, v in code_inputs.items()}
            doc_inputs = {k: v.to(args.device) for k, v in doc_inputs.items()}

            outputs = model(code_inputs, doc_inputs)
            logits = outputs[0] if isinstance(outputs, tuple) else outputs
            probs = F.softmax(logits, dim=1)[:, 1]  # P(drifted)
            preds = (probs >= args.threshold).long()

            all_labels.extend(labels.tolist())
            all_preds.extend(preds.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())
            all_metas.extend(metas)

    # 5. Metrics & Breakdowns
    overall_metrics = calculate_metrics(all_labels, all_preds, all_probs)
    breakdowns = evaluate_breakdowns(all_labels, all_preds, all_metas)

    provenances = [m.get("provenance", "unknown") for m in all_metas]
    score_distributions = compute_score_distributions(all_probs, provenances)

    # 6. Console Report
    print("\n" + "=" * 70)
    print("FINAL TEST RESULTS (Dual-Encoder)")
    print("=" * 70)
    print(f"Accuracy         : {overall_metrics['accuracy']:.4f}")
    print(f"Balanced Accuracy: {overall_metrics['balanced_accuracy']:.4f}")
    print(f"Drift Recall     : {overall_metrics['recall']:.4f}")
    print(f"Drift Precision  : {overall_metrics['precision']:.4f}")
    print(f"Binary F1        : {overall_metrics['f1']:.4f}")
    print(f"Macro F1         : {overall_metrics['macro_f1']:.4f}")
    print(f"Confusion Matrix : {overall_metrics['confusion_matrix']}")
    print("=" * 70)

    # 7. Save outputs
    out_preds_path.parent.mkdir(parents=True, exist_ok=True)
    with out_preds_path.open("w", encoding="utf-8") as f:
        for rec, prob, pred in zip(test_dataset.records, all_probs, all_preds):
            out = dict(rec)
            out["predicted_label"] = "drifted" if pred == 1 else "aligned"
            out["predicted_label_int"] = pred
            out["probability_drift"] = round(prob, 6)
            out["threshold_applied"] = args.threshold
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
    print(f"[OK] Saved predictions to {out_preds_path}")

    eval_record = {
        "evaluation_name": "Dual-Encoder Independent Evaluation",
        "checkpoint_evaluated": str(ckpt_path),
        "test_dataset": str(test_path),
        "test_samples": len(test_dataset),
        "device": args.device,
        "threshold": args.threshold,
        "overall_metrics": overall_metrics,
        "confusion_matrix": {
            "tn": overall_metrics["tn"],
            "fp": overall_metrics["fp"],
            "fn": overall_metrics["fn"],
            "tp": overall_metrics["tp"],
        },
        "score_distributions": score_distributions,
        "breakdowns": breakdowns,
    }

    out_results_path.parent.mkdir(parents=True, exist_ok=True)
    with out_results_path.open("w", encoding="utf-8") as f:
        json.dump(eval_record, f, indent=2)
    print(f"[OK] Saved evaluation summary to {out_results_path}")


if __name__ == "__main__":
    main()
