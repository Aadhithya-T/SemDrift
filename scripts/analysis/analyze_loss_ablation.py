#!/usr/bin/env python3
"""
scripts/analysis/analyze_loss_ablation.py — Controlled Loss Ablation Analysis (Joint-CE vs. Joint-Focal).

Evaluates the isolated impact of the loss function on the Joint-Encoder architecture:
  Joint + CrossEntropy (Unweighted)
         vs.
  Joint + Focal Loss (Unweighted, Category Weighting = OFF)

Both models hold all other variables constant:
  - CodeBERT backbone
  - Random seed = 42
  - LR = 2e-5, epochs = 3, batch size = 8, dropout = 0.1
  - Test benchmark (N = 1,205)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

from scipy.stats import chi2
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def load_jsonl_predictions(path: str) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def compute_metrics(y_true: list[str], y_pred: list[str]) -> dict:
    y_b_true = [1 if label == "drifted" else 0 for label in y_true]
    y_b_pred = [1 if label == "drifted" else 0 for label in y_pred]

    acc = float(accuracy_score(y_b_true, y_b_pred))
    p, r, f1, _ = precision_recall_fscore_support(y_b_true, y_b_pred, average="binary", zero_division=0)
    _, _, macro_f1, _ = precision_recall_fscore_support(y_b_true, y_b_pred, average="macro", zero_division=0)
    bal_acc = float(balanced_accuracy_score(y_b_true, y_b_pred))

    cm = confusion_matrix(y_b_true, y_b_pred)
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    return {
        "accuracy": round(acc * 100, 2),
        "precision": round(float(p) * 100, 2),
        "recall": round(float(r) * 100, 2),
        "f1": round(float(f1) * 100, 2),
        "macro_f1": round(float(macro_f1) * 100, 2),
        "balanced_accuracy": round(bal_acc * 100, 2),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "count": len(y_true),
    }


def mcnemar_test(y_true: list[str], y_pred_ce: list[str], y_pred_focal: list[str]) -> dict:
    """Computes McNemar's Chi-squared test with continuity correction."""
    n00 = 0
    n01 = 0  # CE incorrect, Focal correct (Focal wins)
    n10 = 0  # CE correct, Focal incorrect (CE wins)
    n11 = 0

    for t, p1, p2 in zip(y_true, y_pred_ce, y_pred_focal):
        c1 = (p1 == t)
        c2 = (p2 == t)
        if not c1 and not c2:
            n00 += 1
        elif not c1 and c2:
            n01 += 1
        elif c1 and not c2:
            n10 += 1
        else:
            n11 += 1

    b = n01
    c = n10
    if b + c == 0:
        stat = 0.0
        p_val = 1.0
    else:
        stat = float(((abs(b - c) - 1.0) ** 2) / (b + c))
        p_val = float(chi2.sf(stat, df=1))

    return {
        "n00": n00,
        "n01_focal_wins": n01,
        "n10_ce_wins": n10,
        "n11": n11,
        "statistic": round(stat, 4),
        "p_value": p_val,
        "p_value_formatted": f"{p_val:.4e}" if p_val < 0.0001 else f"{p_val:.4f}",
        "is_statistically_significant": p_val < 0.05,
    }


def compute_category_breakdown(records: list[dict]) -> dict[str, dict]:
    by_type = defaultdict(lambda: {"true": [], "pred": []})
    for r in records:
        dt = r.get("drift_type") or ("aligned" if r.get("label") == 0 or r.get("label") == "aligned" else "unknown")
        true_label = "drifted" if r.get("label") in (1, "drifted", "1") else "aligned"
        pred_label = r.get("predicted_label", "aligned")
        by_type[dt]["true"].append(true_label)
        by_type[dt]["pred"].append(pred_label)

    results = {}
    for dt, data in sorted(by_type.items()):
        yt = data["true"]
        yp = data["pred"]
        acc = accuracy_score(yt, yp) * 100
        p, r, f1, _ = precision_recall_fscore_support(
            [1 if l == "drifted" else 0 for l in yt],
            [1 if l == "drifted" else 0 for l in yp],
            average="binary",
            zero_division=0,
        )
        results[dt] = {
            "count": len(yt),
            "accuracy": round(acc, 2),
            "f1": round(float(f1) * 100, 2),
            "precision": round(float(p) * 100, 2),
            "recall": round(float(r) * 100, 2),
        }
    return results


def main():
    parser = argparse.ArgumentParser(description="Analyze Loss Ablation: Joint-CE vs. Joint-Focal")
    parser.add_argument(
        "--ce_preds",
        default="data/experiments/v2/joint_ce_controlled/predictions_joint_encoder.jsonl",
        help="Path to Joint-CE predictions JSONL",
    )
    parser.add_argument(
        "--focal_preds",
        default="data/experiments/v2/joint_focal_controlled/predictions_joint_encoder.jsonl",
        help="Path to Joint-Focal predictions JSONL",
    )
    parser.add_argument(
        "--output_dir",
        default="data/experiments/v2/loss_ablation",
        help="Output directory for reports",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if not os.path.exists(args.ce_preds):
        print(f"Error: Joint-CE predictions not found at {args.ce_preds}")
        sys.exit(1)

    if not os.path.exists(args.focal_preds):
        print(f"Error: Joint-Focal predictions not found at {args.focal_preds}")
        print("Please run the Joint-Focal training command first.")
        sys.exit(1)

    ce_recs = load_jsonl_predictions(args.ce_preds)
    focal_recs = load_jsonl_predictions(args.focal_preds)

    assert len(ce_recs) == len(focal_recs), f"Sample count mismatch: {len(ce_recs)} vs {len(focal_recs)}"

    y_true_ce = ["drifted" if r.get("label") in (1, "drifted", "1") else "aligned" for r in ce_recs]
    y_pred_ce = [r.get("predicted_label", "aligned") for r in ce_recs]

    y_true_focal = ["drifted" if r.get("label") in (1, "drifted", "1") else "aligned" for r in focal_recs]
    y_pred_focal = [r.get("predicted_label", "aligned") for r in focal_recs]

    assert y_true_ce == y_true_focal, "Ground truth labels do not match between runs!"

    ce_metrics = compute_metrics(y_true_ce, y_pred_ce)
    focal_metrics = compute_metrics(y_true_focal, y_pred_focal)

    mcnemar = mcnemar_test(y_true_ce, y_pred_ce, y_pred_focal)
    ce_breakdown = compute_category_breakdown(ce_recs)
    focal_breakdown = compute_category_breakdown(focal_recs)

    print("\n" + "=" * 80)
    print(f"{'CONTROLLED LOSS ABLATION: JOINT-CE VS. JOINT-FOCAL':^80}")
    print("=" * 80)
    print(f"{'Metric':<24} | {'Joint-CE':<16} | {'Joint-Focal (Plain)':<20} | {'Delta':<10}")
    print("-" * 80)

    keys = [
        ("Accuracy", "accuracy"),
        ("Precision", "precision"),
        ("Recall", "recall"),
        ("Binary F1-Score", "f1"),
        ("Macro-F1 Score", "macro_f1"),
        ("Balanced Accuracy", "balanced_accuracy"),
    ]
    for name, key in keys:
        c = ce_metrics[key]
        f = focal_metrics[key]
        diff = f - c
        diff_str = f"+{diff:.2f}%" if diff > 0 else f"{diff:.2f}%"
        print(f"{name:<24} | {c:>14.2f}% | {f:>18.2f}% | {diff_str:>9}")

    print("-" * 80)
    print(f"McNemar Test: chi2 = {mcnemar['statistic']:.4f}, p = {mcnemar['p_value_formatted']} (Significant: {mcnemar['is_statistically_significant']})")
    print("=" * 80)

    print("\n--- Drift Type Breakdown (F1-Score) ---")
    print(f"{'Drift Type':<24} | {'Sample N':<10} | {'Joint-CE F1':<14} | {'Joint-Focal F1':<16} | {'Delta':<10}")
    print("-" * 80)
    for dt in sorted(set(list(ce_breakdown.keys()) + list(focal_breakdown.keys()))):
        c_info = ce_breakdown.get(dt, {"count": 0, "f1": 0.0})
        f_info = focal_breakdown.get(dt, {"count": 0, "f1": 0.0})
        cnt = max(c_info["count"], f_info["count"])
        diff = f_info["f1"] - c_info["f1"]
        diff_str = f"+{diff:.2f}%" if diff > 0 else f"{diff:.2f}%"
        print(f"{dt:<24} | {cnt:<10} | {c_info['f1']:>12.2f}% | {f_info['f1']:>14.2f}% | {diff_str:>9}")

    # Write summary JSON
    results_json = {
        "joint_ce_metrics": ce_metrics,
        "joint_focal_metrics": focal_metrics,
        "delta": {k: round(focal_metrics[k] - ce_metrics[k], 2) for k in ["accuracy", "precision", "recall", "f1", "macro_f1", "balanced_accuracy"]},
        "mcnemar": mcnemar,
        "ce_breakdown": ce_breakdown,
        "focal_breakdown": focal_breakdown,
    }
    json_path = os.path.join(args.output_dir, "loss_ablation_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_json, f, indent=2)

    # Write markdown summary
    md_path = os.path.join(args.output_dir, "loss_ablation_summary.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# 🧪 Controlled Loss Ablation: Joint-CE vs. Joint-Focal (No Category Weights)\n\n")
        f.write(f"- **Joint-CE Binary F1**: {ce_metrics['f1']}%\n")
        f.write(f"- **Joint-Focal Binary F1**: {focal_metrics['f1']}%\n")
        delta_f1 = focal_metrics['f1'] - ce_metrics['f1']
        f.write(f"- **Delta F1**: {delta_f1:+.2f}%\n")
        f.write(f"- **McNemar p-value**: {mcnemar['p_value_formatted']}\n\n")
        f.write("See console or `loss_ablation_results.json` for full breakdown.\n")

    print(f"\nSaved analysis to: {json_path} and {md_path}\n")


if __name__ == "__main__":
    main()
