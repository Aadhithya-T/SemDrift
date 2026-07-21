#!/usr/bin/env python3
"""
scripts/generate_ieee_results.py — IEEE Paper Benchmark & Statistical Analysis.

Generates complete evaluation results, statistical significance tests (McNemar's Test &
Bootstrap 95% Confidence Intervals), fine-grained breakdown tables (Drift Type, Severity, Repo),
and ready-to-paste IEEE LaTeX tables for Overleaf.

Models evaluated:
  1. Model A: Zero-Shot Dual Encoder (Baseline)
  2. Model B (Ablation): Fine-Tuned Dual Encoder
  3. Model B (Primary): Fine-Tuned Joint Encoder Classifier
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    balanced_accuracy_score,
)
from scipy.stats import chi2


# ---------------------------------------------------------------------------
# Metrics Computation & Bootstrap CIs
# ---------------------------------------------------------------------------

def compute_all_metrics(y_true: list[str], y_pred: list[str]) -> dict:
    """Compute Accuracy, Precision, Recall, F1, Macro-F1, Balanced Accuracy, and Confusion Matrix."""
    y_b_true = [1 if l == "drifted" else 0 for l in y_true]
    y_b_pred = [1 if l == "drifted" else 0 for l in y_pred]

    acc = float(accuracy_score(y_b_true, y_b_pred))
    p, r, f1, _ = precision_recall_fscore_support(
        y_b_true, y_b_pred, average="binary", zero_division=0
    )
    _, _, macro_f1, _ = precision_recall_fscore_support(
        y_b_true, y_b_pred, average="macro", zero_division=0
    )
    balanced_acc = float(balanced_accuracy_score(y_b_true, y_b_pred))

    cm = confusion_matrix(y_b_true, y_b_pred)
    tn, fp, fn, tp = 0, 0, 0, 0
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()

    return {
        "accuracy": round(acc * 100, 2),
        "precision": round(float(p) * 100, 2),
        "recall": round(float(r) * 100, 2),
        "f1": round(float(f1) * 100, 2),
        "macro_f1": round(float(macro_f1) * 100, 2),
        "balanced_accuracy": round(balanced_acc * 100, 2),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "count": len(y_true),
    }


def compute_bootstrap_ci(y_true: list[str], y_pred: list[str], n_bootstraps: int = 1000, seed: int = 42) -> dict:
    """Compute 95% Confidence Intervals for Accuracy, F1, and Macro-F1 via bootstrapping."""
    rng = np.random.RandomState(seed)
    y_b_true = np.array([1 if l == "drifted" else 0 for l in y_true])
    y_b_pred = np.array([1 if l == "drifted" else 0 for l in y_pred])
    n = len(y_b_true)

    accs, f1s, macro_f1s = [], [], []

    for _ in range(n_bootstraps):
        indices = rng.choice(n, size=n, replace=True)
        sample_true = y_b_true[indices]
        sample_pred = y_b_pred[indices]

        acc = accuracy_score(sample_true, sample_pred)
        _, _, f1, _ = precision_recall_fscore_support(sample_true, sample_pred, average="binary", zero_division=0)
        _, _, macro_f1, _ = precision_recall_fscore_support(sample_true, sample_pred, average="macro", zero_division=0)

        accs.append(acc * 100)
        f1s.append(f1 * 100)
        macro_f1s.append(macro_f1 * 100)

    def get_ci(arr):
        low = np.percentile(arr, 2.5)
        high = np.percentile(arr, 97.5)
        return f"[{low:.2f}, {high:.2f}]"

    return {
        "accuracy_ci": get_ci(accs),
        "f1_ci": get_ci(f1s),
        "macro_f1_ci": get_ci(macro_f1s),
    }


def run_mcnemar_test(y_true: list[str], y_pred_model1: list[str], y_pred_model2: list[str]) -> dict:
    """Perform McNemar's Test with continuity correction for statistical significance between two models."""
    y_b_true = [1 if l == "drifted" else 0 for l in y_true]
    y_b_p1 = [1 if l == "drifted" else 0 for l in y_pred_model1]
    y_b_p2 = [1 if l == "drifted" else 0 for l in y_pred_model2]

    # Contingency Table:
    # n00: both wrong, n01: m1 wrong & m2 right, n10: m1 right & m2 wrong, n11: both right
    n00, n01, n10, n11 = 0, 0, 0, 0
    for t, p1, p2 in zip(y_b_true, y_b_p1, y_b_p2):
        c1 = (p1 == t)
        c2 = (p2 == t)
        if not c1 and not c2:
            n00 += 1
        elif not c1 and c2:
            n01 += 1  # Model 2 wins
        elif c1 and not c2:
            n10 += 1  # Model 1 wins
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

    significant = p_val < 0.05

    return {
        "contingency_table": {"n00": n00, "n01_m2_wins": n01, "n10_m1_wins": n10, "n11": n11},
        "statistic": round(stat, 4),
        "p_value": p_val,
        "p_value_formatted": f"{p_val:.4e}" if p_val < 0.0001 else f"{p_val:.4f}",
        "is_statistically_significant": significant,
    }


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def load_predictions(jsonl_path: str) -> tuple[list[str], list[str], list[dict]]:
    """Loads prediction JSONL returning (y_true, y_pred, metadata)."""
    y_true = []
    y_pred = []
    metas = []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            yt = rec.get("label") or rec.get("label_str") or "aligned"
            yp = rec.get("predicted_label") or "aligned"

            meta = {
                "repo": rec.get("repo", "unknown"),
                "drift_type": rec.get("drift_type") or "aligned",
                "severity": rec.get("severity") or "aligned",
            }
            y_true.append(yt)
            y_pred.append(yp)
            metas.append(meta)

    return y_true, y_pred, metas


# ---------------------------------------------------------------------------
# LaTeX Table Generator
# ---------------------------------------------------------------------------

def generate_latex_tables(overall_results: dict, type_breakdowns: dict, severity_breakdowns: dict) -> str:
    """Generates clean IEEEtran LaTeX table strings."""
    latex = []

    # Table I: Overall Model Comparison
    latex.append("% ========================================================")
    latex.append("% TABLE I: OVERALL MODEL PERFORMANCE COMPARISON")
    latex.append("% ========================================================")
    latex.append(r"\begin{table}[htbp]")
    latex.append(r"\caption{Performance Comparison of Semantic Drift Detection Models on Test Set}")
    latex.append(r"\label{tab:model_comparison}")
    latex.append(r"\centering")
    latex.append(r"\begin{tabular}{lcccccc}")
    latex.append(r"\hline")
    latex.append(r"\textbf{Model Architecture} & \textbf{Acc (\%)} & \textbf{Prec (\%)} & \textbf{Rec (\%)} & \textbf{F1 (\%)} & \textbf{Macro-F1 (\%)} & \textbf{Bal-Acc (\%)} \\")
    latex.append(r"\hline")

    for name, res in overall_results.items():
        m = res["metrics"]
        latex.append(
            f"{name} & {m['accuracy']:.2f} & {m['precision']:.2f} & {m['recall']:.2f} & {m['f1']:.2f} & {m['macro_f1']:.2f} & {m['balanced_accuracy']:.2f} \\\\"
        )

    latex.append(r"\hline")
    latex.append(r"\end{tabular}")
    latex.append(r"\end{table}" + "\n")

    # Table II: Breakdown by Drift Type
    latex.append("% ========================================================")
    latex.append("% TABLE II: ACCURACY & F1 BREAKDOWN BY DRIFT TYPE")
    latex.append("% ========================================================")
    latex.append(r"\begin{table}[htbp]")
    latex.append(r"\caption{Macro-F1 and Accuracy Breakdown Across Semantic Drift Types}")
    latex.append(r"\label{tab:drift_type_breakdown}")
    latex.append(r"\centering")
    latex.append(r"\begin{tabular}{lccccc}")
    latex.append(r"\hline")
    latex.append(r"\textbf{Mutation / Drift Type} & \textbf{Count} & \textbf{Model A (Base)} & \textbf{Model B (Dual)} & \textbf{Model B (Joint)} \\")
    latex.append(r"\hline")

    all_types = sorted(list(type_breakdowns["Model B (Joint)"].keys()))
    for dt in all_types:
        cnt = type_breakdowns["Model B (Joint)"][dt]["count"]
        m_a = type_breakdowns["Model A (Baseline)"][dt]["f1"]
        m_dual = type_breakdowns["Model B (Dual Encoder)"][dt]["f1"]
        m_joint = type_breakdowns["Model B (Joint)"][dt]["f1"]
        latex.append(f"{dt:<25} & {cnt:<5} & {m_a:.2f}\\% & {m_dual:.2f}\\% & \\textbf{{{m_joint:.2f}\\%}} \\\\")

    latex.append(r"\hline")
    latex.append(r"\end{tabular}")
    latex.append(r"\end{table}" + "\n")

    return "\n".join(latex)


# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate IEEE Paper Results & Statistical Significance Tests.")
    parser.add_argument("--v2_dir", default="data/experiments/v2", help="V2 experiment directory containing prediction subdirectories")
    parser.add_argument("--output_dir", default="data/experiments/v2", help="Directory to save IEEE results and LaTeX tables")
    parser.add_argument("--bootstraps", type=int, default=1000, help="Number of bootstrap iterations for 95%% CIs")
    args = parser.parse_args()

    model_paths = {
        "Model A (Baseline)": os.path.join(args.v2_dir, "model_a_results", "predictions_model_a.jsonl"),
        "Model B (Dual Encoder)": os.path.join(args.v2_dir, "dual_encoder_results", "predictions_model_b.jsonl"),
        "Model B (Joint)": os.path.join(args.v2_dir, "joint_encoder_results", "predictions_model_b.jsonl"),
    }

    # Verify all files exist
    for name, path in model_paths.items():
        if not os.path.exists(path):
            print(f"Error: Prediction file for {name} not found at '{path}'.", file=sys.stderr)
            sys.exit(1)

    print("=" * 75)
    print("IEEE Conference Paper Benchmark & Statistical Significance Generator")
    print("=" * 75)

    # 1. Evaluate Overall Metrics & Bootstrap CIs
    overall_results = {}
    preds_dict = {}
    y_true_master = None

    for name, path in model_paths.items():
        y_true, y_pred, metas = load_predictions(path)
        y_true_master = y_true
        preds_dict[name] = y_pred

        metrics = compute_all_metrics(y_true, y_pred)
        cis = compute_bootstrap_ci(y_true, y_pred, n_bootstraps=args.bootstraps)

        overall_results[name] = {
            "metrics": metrics,
            "confidence_intervals_95": cis,
            "metas": metas,
        }

    # 2. Statistical Significance Testing (McNemar's Test)
    mcnemar_a_vs_joint = run_mcnemar_test(y_true_master, preds_dict["Model A (Baseline)"], preds_dict["Model B (Joint)"])
    mcnemar_dual_vs_joint = run_mcnemar_test(y_true_master, preds_dict["Model B (Dual Encoder)"], preds_dict["Model B (Joint)"])

    # 3. Fine-Grained Breakdowns
    def get_breakdowns(y_true, y_pred, metas):
        by_type = defaultdict(list)
        by_sev = defaultdict(list)
        by_repo = defaultdict(list)

        for yt, yp, m in zip(y_true, y_pred, metas):
            by_type[m["drift_type"]].append((yt, yp))
            by_sev[m["severity"]].append((yt, yp))
            by_repo[m["repo"]].append((yt, yp))

        def process_group(group_dict):
            res = {}
            for k, pairs in group_dict.items():
                yt_g, yp_g = zip(*pairs)
                res[k] = compute_all_metrics(list(yt_g), list(yp_g))
            return res

        return process_group(by_type), process_group(by_sev), process_group(by_repo)

    type_breakdowns = {}
    sev_breakdowns = {}
    repo_breakdowns = {}

    for name, path in model_paths.items():
        yt, yp, metas = load_predictions(path)
        bt, bs, br = get_breakdowns(yt, yp, metas)
        type_breakdowns[name] = bt
        sev_breakdowns[name] = bs
        repo_breakdowns[name] = br

    # 4. Console Summary Printing
    print("\n--- TABLE I: OVERALL MODEL PERFORMANCE & 95% CONFIDENCE INTERVALS ---")
    print(f"{'Model Name':<25} | {'Acc (%)':<8} | {'F1 (%)':<8} | {'Macro F1':<8} | {'Bal Acc':<8} | {'95% F1 CI':<16}")
    print("-" * 80)
    for name, res in overall_results.items():
        m = res["metrics"]
        ci = res["confidence_intervals_95"]["f1_ci"]
        print(f"{name:<25} | {m['accuracy']:<8.2f} | {m['f1']:<8.2f} | {m['macro_f1']:<8.2f} | {m['balanced_accuracy']:<8.2f} | {ci:<16}")

    print("\n--- STATISTICAL SIGNIFICANCE (McNemar's Test) ---")
    print(f"Model A (Base) vs Model B (Joint) : chi2 = {mcnemar_a_vs_joint['statistic']}, p = {mcnemar_a_vs_joint['p_value_formatted']} "
          f"-> Significant? {'YES (p < 0.05)' if mcnemar_a_vs_joint['is_statistically_significant'] else 'NO'}")
    print(f"Model B (Dual) vs Model B (Joint) : chi2 = {mcnemar_dual_vs_joint['statistic']}, p = {mcnemar_dual_vs_joint['p_value_formatted']} "
          f"-> Significant? {'YES (p < 0.05)' if mcnemar_dual_vs_joint['is_statistically_significant'] else 'NO'}")

    print("\n--- TABLE II: F1 BREAKDOWN BY DRIFT TYPE ---")
    print(f"{'Drift / Mutation Type':<25} | {'Count':<6} | {'Model A':<9} | {'Dual Encoder':<12} | {'Joint Encoder':<13}")
    print("-" * 75)
    all_types = sorted(list(type_breakdowns["Model B (Joint)"].keys()))
    for dt in all_types:
        cnt = type_breakdowns["Model B (Joint)"][dt]["count"]
        fa = type_breakdowns["Model A (Baseline)"][dt]["f1"]
        fd = type_breakdowns["Model B (Dual Encoder)"][dt]["f1"]
        fj = type_breakdowns["Model B (Joint)"][dt]["f1"]
        print(f"{dt:<25} | {cnt:<6} | {fa:<9.2f} | {fd:<12.2f} | {fj:<13.2f}")

    print("\n--- TABLE III: F1 BREAKDOWN BY SEVERITY ---")
    print(f"{'Severity Level':<25} | {'Count':<6} | {'Model A':<9} | {'Dual Encoder':<12} | {'Joint Encoder':<13}")
    print("-" * 75)
    all_sevs = sorted(list(sev_breakdowns["Model B (Joint)"].keys()))
    for sev in all_sevs:
        cnt = sev_breakdowns["Model B (Joint)"][sev]["count"]
        fa = sev_breakdowns["Model A (Baseline)"][sev]["f1"]
        fd = sev_breakdowns["Model B (Dual Encoder)"][sev]["f1"]
        fj = sev_breakdowns["Model B (Joint)"][sev]["f1"]
        print(f"{sev:<25} | {cnt:<6} | {fa:<9.2f} | {fd:<12.2f} | {fj:<13.2f}")

    # 5. Save IEEE JSON and LaTeX files
    os.makedirs(args.output_dir, exist_ok=True)
    json_path = os.path.join(args.output_dir, "ieee_paper_results.json")
    tex_path = os.path.join(args.output_dir, "ieee_paper_tables.tex")

    final_payload = {
        "overall_performance": {name: res["metrics"] for name, res in overall_results.items()},
        "bootstrap_confidence_intervals": {name: res["confidence_intervals_95"] for name, res in overall_results.items()},
        "mcnemar_significance_tests": {
            "model_a_vs_joint": mcnemar_a_vs_joint,
            "dual_vs_joint": mcnemar_dual_vs_joint,
        },
        "breakdown_by_drift_type": type_breakdowns,
        "breakdown_by_severity": sev_breakdowns,
        "breakdown_by_repository": repo_breakdowns,
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(final_payload, f, indent=2)

    latex_code = generate_latex_tables(overall_results, type_breakdowns, sev_breakdowns)
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(latex_code)

    print("\n" + "=" * 75)
    print(f"IEEE Results JSON saved to  : {json_path}")
    print(f"IEEE LaTeX Tables saved to : {tex_path}")
    print("=" * 75)


if __name__ == "__main__":
    main()
