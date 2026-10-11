#!/usr/bin/env python3
"""
scripts/expansion_v1_experiments/run_all_models.py

Master Orchestrator to Train, Tune, and Evaluate ALL SemDrift Models on Expansion V1 Dataset (10K samples).

Model Architectures:
  1. zero_shot          : Pretrained CodeBERT + Cosine Divergence (Model A Baseline)
  2. tfidf_combined      : Pure Lexical TF-IDF + Logistic Regression
  3. tfidf_relational    : Enhanced Relational TF-IDF (Dual Overlap) + Logistic Regression
  4. dual_encoder       : Fine-Tuned CodeBERT Bi-Encoder (Ablation Model)
  5. joint_encoder      : Fine-Tuned CodeBERT Cross-Encoder + Focal Loss + Category Weighting (Primary Model)
  6. joint_encoder_ce   : Fine-Tuned CodeBERT Cross-Encoder + Standard CrossEntropy (Controlled Architecture Ablation)

Produces:
  - Individual checkpoints, predictions JSONLs, and per-model results JSONs
  - Unified comparative benchmark report (Markdown, LaTeX, and JSON) in experiments/expansion_v1/benchmarks/
  - McNemar's statistical significance tests across architectures
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

AVAILABLE_MODELS = [
    "zero_shot",
    "tfidf_combined",
    "tfidf_relational",
    "dual_encoder",
    "joint_encoder",
    "joint_encoder_ce",
]

MODEL_DISPLAY_NAMES = {
    "zero_shot": "Zero-Shot CodeBERT (Cosine Divergence)",
    "tfidf_combined": "TF-IDF Combined (Pure Lexical)",
    "tfidf_relational": "TF-IDF Relational (Dual Overlap)",
    "dual_encoder": "Dual-Encoder (CodeBERT Bi-Encoder)",
    "joint_encoder": "Joint-Encoder (Focal Loss + Cat Weighting)",
    "joint_encoder_ce": "Joint-Encoder (Standard CrossEntropy)",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Master Orchestration for All Models on Expansion V1 Dataset (10K)"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=AVAILABLE_MODELS,
        default=AVAILABLE_MODELS,
        help="Models to execute (default: all 6 models)",
    )
    parser.add_argument(
        "--train_file",
        default="data/real_dataset_expansion_v1/splits/train.jsonl",
        help="Path to training set",
    )
    parser.add_argument(
        "--val_file",
        default="data/real_dataset_expansion_v1/splits/val.jsonl",
        help="Path to validation set",
    )
    parser.add_argument(
        "--test_file",
        default="data/real_dataset_expansion_v1/splits/test.jsonl",
        help="Path to test set",
    )
    parser.add_argument(
        "--output_dir",
        default="experiments/expansion_v1",
        help="Root output directory for all model results",
    )
    parser.add_argument(
        "--device",
        default="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") != "" else "cpu",
        help="Execution device (cuda/cpu)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Training epochs for neural models",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for neural model training/inference",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-5,
        help="Learning rate for neural models",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Run fast smoke-test subset (16 samples per split) across all pipelines",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip models that already have completed test results",
    )
    return parser.parse_args()


def run_command(cmd: List[str], desc: str) -> float:
    """Execute command with timing and status reporting."""
    print(f"\n======================================================================")
    print(f">>> START: {desc}")
    print(f"    Command: {' '.join(cmd)}")
    print(f"======================================================================\n", flush=True)

    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    elapsed = time.time() - t0

    if proc.returncode != 0:
        print(f"\n[-] FAILED: {desc} (Exit code {proc.returncode}, elapsed: {elapsed:.1f}s)", flush=True)
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")

    print(f"\n[+] COMPLETED: {desc} in {elapsed:.1f}s", flush=True)
    return elapsed


def extract_standard_metrics(model_key: str, results_file: Path) -> Dict[str, Any]:
    """Parse standardized metrics dict from model results JSON."""
    if not results_file.is_file():
        return {
            "model_key": model_key,
            "display_name": MODEL_DISPLAY_NAMES.get(model_key, model_key),
            "status": "missing_results",
        }

    with results_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # Resolve metrics block across various formats
    m = (
        data.get("test_metrics")
        or data.get("test_overall")
        or data.get("overall_metrics")
        or {}
    )

    threshold = (
        data.get("optimal_threshold")
        or data.get("threshold")
        or data.get("threshold_tuning", {}).get("selected_threshold")
        or 0.50
    )

    cm = m.get("confusion_matrix")
    if isinstance(cm, dict):
        tn, fp, fn, tp = cm.get("tn", 0), cm.get("fp", 0), cm.get("fn", 0), cm.get("tp", 0)
    else:
        tn = m.get("tn", 0)
        fp = m.get("fp", 0)
        fn = m.get("fn", 0)
        tp = m.get("tp", 0)

    acc = m.get("accuracy", 0.0)
    bal_acc = m.get("balanced_accuracy", 0.0)
    prec = m.get("precision", 0.0)
    rec = m.get("recall", 0.0)
    f1 = m.get("f1", 0.0)
    macro_f1 = m.get("macro_f1", 0.0)
    roc_auc = m.get("roc_auc")
    pr_auc = m.get("pr_auc")

    return {
        "model_key": model_key,
        "display_name": MODEL_DISPLAY_NAMES.get(model_key, model_key),
        "status": "success",
        "threshold": round(float(threshold), 4),
        "accuracy": round(float(acc), 4),
        "balanced_accuracy": round(float(bal_acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1": round(float(f1), 4),
        "macro_f1": round(float(macro_f1), 4),
        "roc_auc": round(float(roc_auc), 4) if roc_auc is not None else None,
        "pr_auc": round(float(pr_auc), 4) if pr_auc is not None else None,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "confusion_matrix": f"TN={tn}, FP={fp}, FN={fn}, TP={tp}",
        "raw_results_file": str(results_file),
    }


def compute_mcnemar_test(preds_a_path: Path, preds_b_path: Path) -> Optional[Dict[str, Any]]:
    """Compute McNemar's test with continuity correction between two sets of model predictions."""
    if not preds_a_path.is_file() or not preds_b_path.is_file():
        return None

    def load_preds(path: Path) -> List[Tuple[int, int]]:
        res = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                y_true = 1 if (d.get("drift_label") == 1 or d.get("label") in (1, "drifted")) else 0
                pred = d.get("predicted_label_int")
                if pred is None:
                    p_str = d.get("predicted_label")
                    pred = 1 if p_str in (1, "drifted", "1") else 0
                res.append((y_true, int(pred)))
        return res

    pairs_a = load_preds(preds_a_path)
    pairs_b = load_preds(preds_b_path)

    if len(pairs_a) != len(pairs_b) or len(pairs_a) == 0:
        return None

    # Contingency table:
    # b: A correct, B incorrect
    # c: A incorrect, B correct
    b = 0
    c = 0
    both_correct = 0
    both_wrong = 0

    for (yt_a, yp_a), (yt_b, yp_b) in zip(pairs_a, pairs_b):
        a_corr = (yt_a == yp_a)
        b_corr = (yt_b == yp_b)
        if a_corr and b_corr:
            both_correct += 1
        elif not a_corr and not b_corr:
            both_wrong += 1
        elif a_corr and not b_corr:
            b += 1
        elif not a_corr and b_corr:
            c += 1

    # Edwards' continuity correction: (|b - c| - 1)^2 / (b + c)
    discordant = b + c
    if discordant == 0:
        chi2 = 0.0
        p_val = 1.0
    else:
        chi2 = (abs(b - c) - 1.0) ** 2 / float(discordant)
        # Approximate 1-df chi-square p-value
        p_val = math.erfc(math.sqrt(chi2) / math.sqrt(2.0))

    return {
        "n_samples": len(pairs_a),
        "both_correct": both_correct,
        "both_incorrect": both_wrong,
        "model_a_only_correct": b,
        "model_b_only_correct": c,
        "discordant_pairs": discordant,
        "chi2": round(float(chi2), 4),
        "p_value": round(float(p_val), 6),
        "significant_at_05": bool(p_val < 0.05),
        "significant_at_01": bool(p_val < 0.01),
    }


def generate_benchmark_reports(
    metrics_list: List[Dict[str, Any]],
    significance_tests: Dict[str, Any],
    benchmarks_dir: Path,
    test_samples: int,
) -> None:
    """Generate Markdown, LaTeX, and JSON benchmark reports."""
    benchmarks_dir.mkdir(parents=True, exist_ok=True)

    # 1. Markdown Summary
    md_lines = [
        "# SemDrift Benchmark Results: Real Dataset Expansion V1 (10,000 samples)\n",
        f"- **Dataset**: Real Dataset Expansion V1 (`data/real_dataset_expansion_v1/`)",
        f"- **Splits**: Train: 8,128 | Val: 544 | Test: {test_samples}",
        f"- **Total Samples**: 10,000 samples across 33 open-source Python repositories",
        f"- **Evaluation Date**: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}\n",
        "## Overall Model Performance on Test Split\n",
        "| Architecture | Threshold | Accuracy | Balanced Acc | Drift Precision | Drift Recall | Binary F1 | Macro F1 | ROC-AUC | PR-AUC |",
        "|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]

    for m in metrics_list:
        if m.get("status") != "success":
            md_lines.append(f"| **{m['display_name']}** | *N/A* | *Failed/Skipped* | - | - | - | - | - | - | - |")
            continue

        roc_str = f"{m['roc_auc']:.4f}" if m.get("roc_auc") is not None else "-"
        pr_str = f"{m['pr_auc']:.4f}" if m.get("pr_auc") is not None else "-"
        md_lines.append(
            f"| **{m['display_name']}** | {m['threshold']:.2f} | "
            f"{m['accuracy']*100:.2f}% | {m['balanced_accuracy']*100:.2f}% | "
            f"{m['precision']*100:.2f}% | {m['recall']*100:.2f}% | "
            f"{m['f1']*100:.2f}% | **{m['macro_f1']*100:.2f}%** | "
            f"{roc_str} | {pr_str} |"
        )

    md_lines.append("\n## Confusion Matrices\n")
    md_lines.append("| Architecture | TN | FP | FN | TP | Confusion Matrix Summary |")
    md_lines.append("|:---|:---:|:---:|:---:|:---:|:---|")
    for m in metrics_list:
        if m.get("status") == "success":
            md_lines.append(
                f"| **{m['display_name']}** | {m['tn']} | {m['fp']} | {m['fn']} | {m['tp']} | `{m['confusion_matrix']}` |"
            )

    if significance_tests:
        md_lines.append("\n## Statistical Significance Tests (McNemar's Test with Continuity Correction)\n")
        md_lines.append("| Comparison (Model A vs Model B) | N | Model A Unique Wins | Model B Unique Wins | Chi² | p-value | Significant (p<0.05) |")
        md_lines.append("|:---|:---:|:---:|:---:|:---:|:---:|:---:|")
        for comp_name, test in significance_tests.items():
            sig_mark = "✅ Yes" if test["significant_at_05"] else "❌ No"
            md_lines.append(
                f"| {comp_name} | {test['n_samples']} | {test['model_a_only_correct']} | {test['model_b_only_correct']} | {test['chi2']:.4f} | {test['p_value']:.4e} | {sig_mark} |"
            )

    md_report_path = benchmarks_dir / "benchmark_summary.md"
    with md_report_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")
    print(f"\n[OK] Markdown benchmark summary saved to: {md_report_path}")

    # 2. LaTeX Table
    tex_lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Comprehensive Benchmark Results on SemDrift Expansion V1 Dataset (10,000 samples).}",
        r"\label{tab:expansion_v1_benchmark}",
        r"\small",
        r"\begin{tabular}{lcccccccc}",
        r"\toprule",
        r"\textbf{Architecture} & \textbf{Acc} & \textbf{BalAcc} & \textbf{Prec} & \textbf{Rec} & \textbf{F1} & \textbf{Macro-F1} & \textbf{ROC-AUC} & \textbf{PR-AUC} \\",
        r"\midrule",
    ]
    for m in metrics_list:
        if m.get("status") == "success":
            roc_str = f"{m['roc_auc']:.3f}" if m.get("roc_auc") is not None else "--"
            pr_str = f"{m['pr_auc']:.3f}" if m.get("pr_auc") is not None else "--"
            tex_lines.append(
                f"{m['display_name']} & {m['accuracy']*100:.1f}\\% & {m['balanced_accuracy']*100:.1f}\\% & "
                f"{m['precision']*100:.1f}\\% & {m['recall']*100:.1f}\\% & {m['f1']*100:.1f}\\% & "
                f"\\textbf{{{m['macro_f1']*100:.1f}\\%}} & {roc_str} & {pr_str} \\\\"
            )
    tex_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ])
    tex_path = benchmarks_dir / "benchmark_table.tex"
    with tex_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(tex_lines) + "\n")
    print(f"[OK] LaTeX benchmark table saved to: {tex_path}")

    # 3. Master JSON
    master_record = {
        "dataset": "Real Dataset Expansion V1",
        "dataset_path": "data/real_dataset_expansion_v1",
        "test_samples": test_samples,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "models": metrics_list,
        "significance_tests": significance_tests,
    }
    json_path = benchmarks_dir / "benchmark_summary.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(master_record, f, indent=2)
    print(f"[OK] Master benchmark JSON saved to: {json_path}")


def main():
    args = parse_args()
    root_out = PROJECT_ROOT / args.output_dir
    root_out.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("SEMDRIFT MASTER BENCHMARK ORCHESTRATOR — EXPANSION V1 DATASET (10K)")
    print("=" * 80)
    print(f"Models to Run : {', '.join(args.models)}")
    print(f"Train File    : {args.train_file}")
    print(f"Val File      : {args.val_file}")
    print(f"Test File     : {args.test_file}")
    print(f"Root Output   : {args.output_dir}")
    print(f"Device        : {args.device}")
    print(f"Epochs        : {args.epochs}")
    print(f"Batch Size    : {args.batch_size}")
    print(f"Learning Rate : {args.lr}")
    print(f"Dry Run Mode  : {args.dry_run}")
    print(f"Skip Existing : {args.skip_existing}")
    print("=" * 80)

    # 1. Zero-Shot
    if "zero_shot" in args.models:
        zs_out = root_out / "zero_shot"
        zs_res = zs_out / "results_baseline.json"
        if args.skip_existing and zs_res.is_file():
            print(f"[SKIP] Zero-Shot results already exist at {zs_res}")
        else:
            cmd = [
                sys.executable,
                "scripts/expansion_v1_experiments/run_zero_shot.py",
                "--val_file", args.val_file,
                "--test_file", args.test_file,
                "--output_dir", str(zs_out),
                "--device", args.device,
                "--batch_size", str(args.batch_size * 2),
            ]
            run_command(cmd, "1/6 Zero-Shot CodeBERT Baseline")

    # 2. TF-IDF Combined
    if "tfidf_combined" in args.models:
        tc_out = root_out / "tfidf_combined"
        tc_res = tc_out / "results_tfidf_combined.json"
        if args.skip_existing and tc_res.is_file():
            print(f"[SKIP] TF-IDF Combined results already exist at {tc_res}")
        else:
            cmd = [
                sys.executable,
                "scripts/expansion_v1_experiments/train_eval_tfidf_combined.py",
                "--train_file", args.train_file,
                "--val_file", args.val_file,
                "--test_file", args.test_file,
                "--output_dir", str(tc_out),
            ]
            run_command(cmd, "2/6 TF-IDF Combined (Pure Lexical Baseline)")

    # 3. TF-IDF Relational
    if "tfidf_relational" in args.models:
        tr_out = root_out / "tfidf_relational"
        tr_res = tr_out / "results_tfidf_relational.json"
        if args.skip_existing and tr_res.is_file():
            print(f"[SKIP] TF-IDF Relational results already exist at {tr_res}")
        else:
            cmd = [
                sys.executable,
                "scripts/expansion_v1_experiments/train_eval_tfidf_relational.py",
                "--train_file", args.train_file,
                "--val_file", args.val_file,
                "--test_file", args.test_file,
                "--output_dir", str(tr_out),
            ]
            run_command(cmd, "3/6 TF-IDF Relational (Enhanced Lexical Baseline)")

    # 4. Dual-Encoder
    if "dual_encoder" in args.models:
        de_out = root_out / "dual_encoder"
        de_res = de_out / "results_dual_encoder.json"
        if args.skip_existing and de_res.is_file():
            print(f"[SKIP] Dual-Encoder results already exist at {de_res}")
        else:
            cmd = [
                sys.executable,
                "scripts/expansion_v1_experiments/train_eval_dual_encoder.py",
                "--train_file", args.train_file,
                "--val_file", args.val_file,
                "--test_file", args.test_file,
                "--output_dir", str(de_out),
                "--device", args.device,
                "--epochs", str(args.epochs),
                "--batch_size", str(args.batch_size),
                "--lr", str(args.lr),
            ]
            if args.dry_run:
                cmd.append("--dry_run")
            run_command(cmd, "4/6 Fine-Tuned Dual-Encoder (Bi-Encoder)")

    # 5. Joint-Encoder (Focal)
    if "joint_encoder" in args.models:
        je_out = root_out / "joint_encoder"
        je_res = je_out / "results_joint_encoder.json"
        if args.skip_existing and je_res.is_file():
            print(f"[SKIP] Joint-Encoder (Focal) results already exist at {je_res}")
        else:
            cmd = [
                sys.executable,
                "scripts/expansion_v1_experiments/train_eval_joint_encoder.py",
                "--train_file", args.train_file,
                "--val_file", args.val_file,
                "--test_file", args.test_file,
                "--output_dir", str(je_out),
                "--device", args.device,
                "--epochs", str(args.epochs),
                "--batch_size", str(args.batch_size),
                "--lr", str(args.lr),
            ]
            if args.dry_run:
                cmd.append("--dry_run")
            run_command(cmd, "5/6 Fine-Tuned Joint-Encoder (Focal + Category Weighting)")

    # 6. Joint-Encoder (CE Ablation)
    if "joint_encoder_ce" in args.models:
        je_ce_out = root_out / "joint_encoder_ce"
        je_ce_res = je_ce_out / "results_joint_encoder.json"
        if args.skip_existing and je_ce_res.is_file():
            print(f"[SKIP] Joint-Encoder (CE) results already exist at {je_ce_res}")
        else:
            cmd = [
                sys.executable,
                "scripts/expansion_v1_experiments/train_eval_joint_encoder_ce.py",
                "--train_file", args.train_file,
                "--val_file", args.val_file,
                "--test_file", args.test_file,
                "--output_dir", str(je_ce_out),
                "--device", args.device,
                "--epochs", str(args.epochs),
                "--batch_size", str(args.batch_size),
                "--lr", str(args.lr),
            ]
            if args.dry_run:
                cmd.append("--dry_run")
            run_command(cmd, "6/6 Fine-Tuned Joint-Encoder (Standard CrossEntropy Ablation)")

    # ---------------------------------------------------------------------------
    # Aggregate Metrics & Compute Significance
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("Aggregating Evaluation Metrics & Generating Benchmark Reports...")
    print("=" * 80)

    result_files_map = {
        "zero_shot": root_out / "zero_shot" / "results_baseline.json",
        "tfidf_combined": root_out / "tfidf_combined" / "results_tfidf_combined.json",
        "tfidf_relational": root_out / "tfidf_relational" / "results_tfidf_relational.json",
        "dual_encoder": root_out / "dual_encoder" / "results_dual_encoder.json",
        "joint_encoder": root_out / "joint_encoder" / "results_joint_encoder.json",
        "joint_encoder_ce": root_out / "joint_encoder_ce" / "results_joint_encoder.json",
    }

    prediction_files_map = {
        "zero_shot": root_out / "zero_shot" / "predictions_baseline.jsonl",
        "tfidf_combined": root_out / "tfidf_combined" / "predictions_tfidf_combined.jsonl",
        "tfidf_relational": root_out / "tfidf_relational" / "predictions_tfidf_relational.jsonl",
        "dual_encoder": root_out / "dual_encoder" / "predictions_dual_encoder.jsonl",
        "joint_encoder": root_out / "joint_encoder" / "predictions_joint_encoder.jsonl",
        "joint_encoder_ce": root_out / "joint_encoder_ce" / "predictions_joint_encoder.jsonl",
    }

    metrics_list = []
    for m_key in args.models:
        res_file = result_files_map.get(m_key)
        metrics = extract_standard_metrics(m_key, res_file)
        metrics_list.append(metrics)

    # Compute McNemar tests between key pairs
    significance_tests = {}
    key_pairs = [
        ("Joint-Encoder (Focal) vs Dual-Encoder", "joint_encoder", "dual_encoder"),
        ("Joint-Encoder (Focal) vs Joint-Encoder (CE)", "joint_encoder", "joint_encoder_ce"),
        ("Joint-Encoder (Focal) vs TF-IDF Combined", "joint_encoder", "tfidf_combined"),
        ("Dual-Encoder vs TF-IDF Combined", "dual_encoder", "tfidf_combined"),
    ]

    for label, m_a, m_b in key_pairs:
        if m_a in args.models and m_b in args.models:
            pa = prediction_files_map.get(m_a)
            pb = prediction_files_map.get(m_b)
            if pa and pb and pa.is_file() and pb.is_file():
                test_res = compute_mcnemar_test(pa, pb)
                if test_res:
                    significance_tests[label] = test_res

    # Count test samples
    test_path = PROJECT_ROOT / args.test_file
    test_count = 0
    if test_path.is_file():
        with test_path.open("r", encoding="utf-8") as f:
            test_count = sum(1 for line in f if line.strip())

    benchmarks_dir = root_out / "benchmarks"
    generate_benchmark_reports(metrics_list, significance_tests, benchmarks_dir, test_count)

    print("\n" + "=" * 80)
    print("ALL EXPANSION V1 EXPERIMENTS COMPLETE!")
    print(f"Benchmark summary saved to : {benchmarks_dir / 'benchmark_summary.md'}")
    print(f"LaTeX table saved to       : {benchmarks_dir / 'benchmark_table.tex'}")
    print(f"JSON record saved to       : {benchmarks_dir / 'benchmark_summary.json'}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
