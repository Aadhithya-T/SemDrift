#!/usr/bin/env python3
"""
scripts/run_model_a.py — Run Model A (Baseline: Dual-Encoder + Cosine Similarity).

Features:
  - GPU (CUDA) acceleration enabled by default (with CPU fallback).
  - L2-normalization and mean-centering for anisotropic CodeBERT vectors.
  - Balanced Accuracy threshold sweep (Sensitivity + Specificity) / 2.
  - Docstring summary cleaning.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from semdrift.embedder.embed import get_embeddings, compute_divergence, load_config

DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def extract_docstring_summary(docstring: str) -> str:
    """Extract clean natural language summary from docstrings by stripping
    REPL code blocks (>>>), parameter tables, and return type specs."""
    if not docstring:
        return ""
    lines = docstring.strip().split("\n")
    summary_lines = []
    for line in lines:
        l = line.strip()
        if not l or l.startswith(">>>") or l.startswith("...") or l.startswith("Parameters") or l.startswith("Returns") or l.startswith("Examples") or l.startswith("See Also"):
            break
        summary_lines.append(l)

    cleaned = " ".join(summary_lines).strip()
    if len(cleaned) >= 10:
        return cleaned
    return lines[0].strip()


def load_jsonl(filepath: str, clean_docs: bool = False) -> list[dict]:
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                if clean_docs:
                    rec["docstring"] = extract_docstring_summary(rec.get("docstring", ""))
                records.append(rec)
    return records


def compute_dataset_divergences(
    records: list[dict],
    batch_size: int = 64,
    device: str = DEFAULT_DEVICE,
    pooling: str = "mean",
    model_name: str = "microsoft/codebert-base",
    normalize: bool = True,
) -> list[float]:
    """Extract code & docstrings from records, embed them, and calculate divergence scores."""
    codes = [r["code"] for r in records]
    docstrings = [r["docstring"] for r in records]

    print(f"Generating embeddings for {len(records)} pairs (batch_size={batch_size}, device={device}, pooling={pooling}, normalize={normalize})...", flush=True)
    start_time = time.time()

    print(" -> Embedding code snippets...", flush=True)
    code_embs = get_embeddings(
        codes, batch_size=batch_size, device=device, pooling=pooling, model_name=model_name, show_progress=True
    )
    print(" -> Embedding docstrings...", flush=True)
    doc_embs = get_embeddings(
        docstrings, batch_size=batch_size, device=device, pooling=pooling, model_name=model_name, show_progress=True
    )

    divergences = compute_divergence(code_embs, doc_embs, normalize=normalize)
    if isinstance(divergences, torch.Tensor):
        divergences = divergences.tolist()
    elif isinstance(divergences, float):
        divergences = [divergences]

    elapsed = time.time() - start_time
    print(f"Done in {elapsed:.2f}s ({len(records)/elapsed:.1f} pairs/sec)", flush=True)

    return divergences


def sweep_threshold(
    y_true: list[str],
    divergences: list[float],
    metric: str = "balanced_accuracy",
) -> tuple[float, dict[float, float]]:
    """Sweep threshold tau in [0.0, 1.0] to find tau* maximizing target metric."""
    y_binary = np.array([1 if label == "drifted" else 0 for label in y_true])
    div_array = np.array(divergences)

    thresholds = np.linspace(0.0, 1.0, 401)
    best_tau = 0.5
    best_score = -1.0
    history = {}

    for tau in thresholds:
        preds = (div_array >= tau).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_binary, preds, average="binary", zero_division=0
        )
        acc = accuracy_score(y_binary, preds)

        if metric == "accuracy":
            score = acc
        elif metric == "balanced_accuracy":
            sensitivity = recall
            specificity = accuracy_score(y_binary[y_binary == 0], preds[y_binary == 0]) if np.sum(y_binary == 0) > 0 else 0
            score = (sensitivity + specificity) / 2.0
        elif metric == "macro_f1":
            _, _, macro_f1, _ = precision_recall_fscore_support(
                y_binary, preds, average="macro", zero_division=0
            )
            score = macro_f1
        else:
            score = f1

        history[round(float(tau), 4)] = round(float(score), 4)

        if score > best_score:
            best_score = score
            best_tau = float(tau)

    return best_tau, history


def calculate_metrics(y_true: list[str], y_pred: list[str]) -> dict:
    y_b_true = [1 if label == "drifted" else 0 for label in y_true]
    y_b_pred = [1 if label == "drifted" else 0 for label in y_pred]

    acc = float(accuracy_score(y_b_true, y_b_pred))
    p, r, f1, _ = precision_recall_fscore_support(y_b_true, y_b_pred, average="binary", zero_division=0)
    _, _, macro_f1, _ = precision_recall_fscore_support(y_b_true, y_b_pred, average="macro", zero_division=0)

    return {
        "accuracy": round(acc, 4),
        "precision": round(float(p), 4),
        "recall": round(float(r), 4),
        "f1": round(float(f1), 4),
        "macro_f1": round(float(macro_f1), 4),
        "count": len(y_true),
    }


def evaluate_breakdowns(records: list[dict], predictions: list[str]) -> dict:
    by_drift_type = defaultdict(list)
    by_severity = defaultdict(list)
    by_repo = defaultdict(list)

    for record, pred in zip(records, predictions):
        true_label = record["label"]
        pair = (true_label, pred)

        drift_type = record.get("drift_type") or "aligned"
        by_drift_type[drift_type].append(pair)

        severity = record.get("severity") or "aligned"
        by_severity[severity].append(pair)

        repo = record.get("repo") or "unknown"
        by_repo[repo].append(pair)

    def calc_group_metrics(group_dict):
        result = {}
        for key, pairs in group_dict.items():
            yt, yp = zip(*pairs)
            result[key] = calculate_metrics(list(yt), list(yp))
        return result

    return {
        "by_drift_type": calc_group_metrics(by_drift_type),
        "by_severity": calc_group_metrics(by_severity),
        "by_repo": calc_group_metrics(by_repo),
    }


def main():
    parser = argparse.ArgumentParser(description="Run Model A (Baseline) Evaluation")
    parser.add_argument("--val", default="data/labeled/val.jsonl", help="Validation dataset path")
    parser.add_argument("--test", default="data/labeled/test.jsonl", help="Test dataset path")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for embedding")
    parser.add_argument("--pooling", choices=["mean", "cls"], default="mean", help="Pooling strategy")
    parser.add_argument("--metric", choices=["accuracy", "balanced_accuracy", "macro_f1", "f1"], default="balanced_accuracy", help="Sweep target metric")
    parser.add_argument("--clean_docstrings", action="store_true", default=True, help="Clean REPL examples from docstrings")
    parser.add_argument("--normalize", action="store_true", default=True, help="Apply L2 normalization & mean centering")
    parser.add_argument("--device", default=DEFAULT_DEVICE, help="Device (cuda/cpu)")
    parser.add_argument("--output_dir", default="data/labeled", help="Directory to write predictions and results")
    args = parser.parse_args()

    config = load_config()
    model_name = config.get("embedder", {}).get("model_name", "microsoft/codebert-base")

    print("======================================================================", flush=True)
    print("Model A Baseline — Dual-Encoder Cosine Similarity Evaluation", flush=True)
    print("======================================================================", flush=True)
    print(f"Model Name       : {model_name}", flush=True)
    print(f"Device           : {args.device}", flush=True)
    print(f"Pooling          : {args.pooling}", flush=True)
    print(f"Batch Size       : {args.batch_size}", flush=True)
    print(f"Clean Docstrings : {args.clean_docstrings}", flush=True)
    print(f"L2 Normalization : {args.normalize}", flush=True)
    print(f"Sweep Metric     : {args.metric}", flush=True)
    print("----------------------------------------------------------------------", flush=True)

    # 1. Load Datasets
    print(f"Loading validation set from: {args.val}", flush=True)
    val_records = load_jsonl(args.val, clean_docs=args.clean_docstrings)
    print(f"Loaded {len(val_records)} validation records.", flush=True)

    print(f"Loading test set from: {args.test}", flush=True)
    test_records = load_jsonl(args.test, clean_docs=args.clean_docstrings)
    print(f"Loaded {len(test_records)} test records.", flush=True)

    # 2. Compute Divergences on Validation Set
    print("\n[Step 1/3] Processing Validation Set...", flush=True)
    val_divs = compute_dataset_divergences(
        val_records, batch_size=args.batch_size, device=args.device, pooling=args.pooling, model_name=model_name, normalize=args.normalize
    )

    val_y_true = [r["label"] for r in val_records]

    # 3. Threshold Sweep on Validation Set
    print(f"\n[Step 2/3] Performing Threshold Sweep (target metric: {args.metric}) on Validation Set...", flush=True)
    best_tau, _ = sweep_threshold(val_y_true, val_divs, metric=args.metric)
    val_preds = ["drifted" if d >= best_tau else "aligned" for d in val_divs]
    val_metrics = calculate_metrics(val_y_true, val_preds)

    print(f"Optimal Threshold (tau*): {best_tau:.4f}", flush=True)
    print(f"Validation Performance  : Acc={val_metrics['accuracy']:.4f} | F1={val_metrics['f1']:.4f} | MacroF1={val_metrics['macro_f1']:.4f} | Prec={val_metrics['precision']:.4f} | Rec={val_metrics['recall']:.4f}", flush=True)

    # 4. Evaluate on Test Set using optimal threshold
    print("\n[Step 3/3] Processing Test Set with Optimal Threshold (tau*)...", flush=True)
    test_divs = compute_dataset_divergences(
        test_records, batch_size=args.batch_size, device=args.device, pooling=args.pooling, model_name=model_name, normalize=args.normalize
    )

    test_y_true = [r["label"] for r in test_records]
    test_preds = ["drifted" if d >= best_tau else "aligned" for d in test_divs]

    test_overall = calculate_metrics(test_y_true, test_preds)
    breakdowns = evaluate_breakdowns(test_records, test_preds)

    print("\n======================================================================", flush=True)
    print("FINAL MODEL A OPTIMIZED TEST RESULTS", flush=True)
    print("======================================================================", flush=True)
    print(f"Optimal Threshold (tau*) : {best_tau:.4f}", flush=True)
    print(f"Accuracy                 : {test_overall['accuracy']:.4f}", flush=True)
    print(f"Precision                : {test_overall['precision']:.4f}", flush=True)
    print(f"Recall                   : {test_overall['recall']:.4f}", flush=True)
    print(f"F1 Score (Binary)        : {test_overall['f1']:.4f}", flush=True)
    print(f"Macro F1 Score           : {test_overall['macro_f1']:.4f}", flush=True)

    print("\n--- Breakdown by Drift Type ---", flush=True)
    for dt, m in sorted(breakdowns["by_drift_type"].items()):
        print(f"  {dt:<22}: Acc={m['accuracy']:.4f} | F1={m['f1']:.4f} | Prec={m['precision']:.4f} | Rec={m['recall']:.4f} (N={m['count']})", flush=True)

    print("\n--- Breakdown by Severity ---", flush=True)
    for sev, m in sorted(breakdowns["by_severity"].items()):
        print(f"  {sev:<22}: Acc={m['accuracy']:.4f} | F1={m['f1']:.4f} | Prec={m['precision']:.4f} | Rec={m['recall']:.4f} (N={m['count']})", flush=True)

    print("\n--- Breakdown by Repository ---", flush=True)
    for repo, m in sorted(breakdowns["by_repo"].items()):
        print(f"  {repo:<22}: Acc={m['accuracy']:.4f} | F1={m['f1']:.4f} | Prec={m['precision']:.4f} | Rec={m['recall']:.4f} (N={m['count']})", flush=True)

    # 5. Save Outputs
    os.makedirs(args.output_dir, exist_ok=True)
    pred_path = os.path.join(args.output_dir, "predictions_model_a.jsonl")
    results_path = os.path.join(args.output_dir, "results_model_a.json")

    with open(pred_path, "w", encoding="utf-8") as f:
        for rec, div, pred in zip(test_records, test_divs, test_preds):
            output_rec = dict(rec)
            output_rec["divergence_score"] = round(div, 6)
            output_rec["predicted_label"] = pred
            output_rec["optimal_threshold"] = best_tau
            f.write(json.dumps(output_rec) + "\n")

    full_results = {
        "model": "Model A (CodeBERT Dual-Encoder Baseline)",
        "model_name": model_name,
        "pooling": args.pooling,
        "clean_docstrings": args.clean_docstrings,
        "normalize": args.normalize,
        "sweep_metric": args.metric,
        "optimal_threshold": best_tau,
        "val_metrics": val_metrics,
        "test_overall": test_overall,
        "breakdowns": breakdowns,
    }

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(full_results, f, indent=2)

    print("\n----------------------------------------------------------------------", flush=True)
    print(f"Saved predictions to : {pred_path}", flush=True)
    print(f"Saved test results to: {results_path}", flush=True)
    print("Model A Baseline execution complete!", flush=True)


if __name__ == "__main__":
    main()
