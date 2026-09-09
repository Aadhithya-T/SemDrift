#!/usr/bin/env python3
"""
scripts/training/run_tfidf_baseline.py — TF-IDF + Linear Classifier Baseline for SemDrift V2.

Provides a clean lexical baseline to evaluate whether CodeBERT-based models
capture predictive code–documentation relationships beyond token frequency,
lexical overlap, and identifier correspondence.

Experimental Ladder:
  1. TF-IDF Combined (Pure Lexical Baseline):
     doc + code -> single vocabulary -> Logistic Regression
  2. TF-IDF Relational (Enhanced Lexical Baseline):
     doc TF-IDF + code TF-IDF + explicit overlap features (Cosine, Jaccard, AST identifiers) -> Logistic Regression
  3. Zero-shot CodeBERT (Pretrained Semantic Baseline)
  4. Dual-Encoder CodeBERT (Fine-tuned Representation Baseline)
  5. Joint-Encoder CodeBERT (Fine-tuned Cross-Input Baseline)

Key Methodological Guarantees:
  - Canonical Model: LogisticRegression(random_state=42, class_weight='balanced')
  - Stopwords: stop_words=None by default (strictly preserving negation & condition operators)
  - Threshold Tuning: Strictly on validation Macro-F1 (locked test set is never exposed to tuning)
  - Provenance Diagnostics: Tracks prediction balance and score distributions by provenance
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.svm import LinearSVC

# Project root setup
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from semdrift.data.labels import extract_label_tuple, validate_dataset_split


# ---------------------------------------------------------------------------
# Preprocessing & Lexical Feature Extraction
# ---------------------------------------------------------------------------

def clean_docstring(doc: str) -> str:
    """Normalize docstring text while preserving all natural language and
    semantic operators (e.g. not, without, raises, returns)."""
    if not doc:
        return ""
    # Strip structured tags while keeping their contents
    cleaned = re.sub(r"\[(SUMMARY|PARAMETERS|RETURNS|RAISES)\]", " ", doc)
    # Collapse multiple whitespaces
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def extract_identifiers_from_code(code: str) -> set[str]:
    """Extract code identifier tokens using Python AST with regex fallback."""
    if not code:
        return set()
    identifiers: set[str] = set()
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                identifiers.add(node.id.lower())
            elif isinstance(node, ast.FunctionDef):
                identifiers.add(node.name.lower())
                for arg in node.args.args:
                    identifiers.add(arg.arg.lower())
                for arg in node.args.kwonlyargs:
                    identifiers.add(arg.arg.lower())
                if node.args.vararg:
                    identifiers.add(node.args.vararg.arg.lower())
                if node.args.kwarg:
                    identifiers.add(node.args.kwarg.arg.lower())
            elif isinstance(node, ast.Attribute):
                identifiers.add(node.attr.lower())
    except Exception:
        # Fallback to identifier regex
        tokens = re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", code)
        identifiers = {t.lower() for t in tokens}
    return identifiers


def extract_words_from_text(text: str) -> set[str]:
    """Extract lowercase word tokens from text."""
    if not text:
        return set()
    return {w.lower() for w in re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", text)}


def compute_relational_features(docs: List[str], codes: List[str]) -> np.ndarray:
    """Compute explicit lexical overlap and alignment features between doc and code.
    
    Features computed per sample:
      1. Jaccard token overlap between doc words and code identifiers
      2. Docstring identifier coverage (fraction of doc words in code)
      3. Code identifier coverage (fraction of code tokens in doc)
      4. Token intersection count
      5. Docstring word length
      6. Code token length
    """
    n = len(docs)
    features = np.zeros((n, 6), dtype=np.float32)

    for i, (doc, code) in enumerate(zip(docs, codes)):
        doc_tokens = extract_words_from_text(doc)
        code_tokens = extract_identifiers_from_code(code)

        doc_len = len(doc_tokens)
        code_len = len(code_tokens)
        intersection = doc_tokens.intersection(code_tokens)
        union = doc_tokens.union(code_tokens)

        jaccard = len(intersection) / len(union) if union else 0.0
        doc_cov = len(intersection) / (doc_len + 1)
        code_cov = len(intersection) / (code_len + 1)

        features[i, 0] = jaccard
        features[i, 1] = doc_cov
        features[i, 2] = code_cov
        features[i, 3] = float(len(intersection))
        features[i, 4] = float(doc_len)
        features[i, 5] = float(code_len)

    return features


# ---------------------------------------------------------------------------
# Metrics Computation
# ---------------------------------------------------------------------------

def compute_metrics(
    y_true: List[int],
    y_pred: List[int],
    y_prob: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """Compute comprehensive classification metrics matching SemDrift IEEE standard."""
    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    _, _, macro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    roc_auc: Optional[float] = None
    pr_auc: Optional[float] = None
    if y_prob is not None and len(set(y_true)) > 1:
        try:
            roc_auc = float(roc_auc_score(y_true, y_prob))
            pr_auc = float(average_precision_score(y_true, y_prob))
        except Exception:
            pass

    return {
        "accuracy": round(acc, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "precision": round(float(p), 4),
        "recall": round(float(r), 4),
        "f1": round(float(f1), 4),
        "macro_f1": round(float(macro_f1), 4),
        "roc_auc": round(roc_auc, 4) if roc_auc is not None else None,
        "pr_auc": round(pr_auc, 4) if pr_auc is not None else None,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "confusion_matrix": f"TN={tn}, FP={fp}, FN={fn}, TP={tp}",
        "pred_aligned": int(sum(1 for p in y_pred if p == 0)),
        "pred_drifted": int(sum(1 for p in y_pred if p == 1)),
        "count": len(y_true),
    }


def compute_score_distributions(probs: List[float], groups: List[str]) -> Dict[str, Dict[str, float]]:
    """Compute summary statistics (mean, median, std, min, max) of drift scores by group."""
    by_grp = defaultdict(list)
    for p, g in zip(probs, groups):
        by_grp[g].append(p)

    stats: Dict[str, Dict[str, float]] = {}
    for grp, vals in sorted(by_grp.items()):
        arr = np.array(vals)
        stats[grp] = {
            "count": len(vals),
            "mean": round(float(np.mean(arr)), 4),
            "median": round(float(np.median(arr)), 4),
            "std": round(float(np.std(arr)), 4),
            "min": round(float(np.min(arr)), 4),
            "max": round(float(np.max(arr)), 4),
        }
    return stats


# ---------------------------------------------------------------------------
# Threshold Tuning
# ---------------------------------------------------------------------------

def sweep_validation_threshold(
    y_true: List[int],
    y_prob: np.ndarray,
    metric_name: str = "macro_f1",
) -> Tuple[float, float, Dict[float, float]]:
    """Sweep threshold tau in [0.01, 0.99] to find optimal decision boundary.
    
    Canonical selection criterion: Macro-F1 strictly on validation data.
    """
    best_tau = 0.50
    best_score = -1.0
    history: Dict[float, float] = {}

    for tau_int in range(1, 100):
        tau = round(tau_int / 100.0, 2)
        y_pred = (y_prob >= tau).astype(int)
        _, _, macro_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0
        )
        history[tau] = round(float(macro_f1), 4)

        if macro_f1 > best_score:
            best_score = float(macro_f1)
            best_tau = tau

    return best_tau, round(best_score, 4), history


# ---------------------------------------------------------------------------
# Dataset Loading & Validation
# ---------------------------------------------------------------------------

def load_jsonl_dataset(path: Path) -> Tuple[List[Dict[str, Any]], List[int], List[str]]:
    """Load JSONL dataset enforcing canonical SemDrift label extraction."""
    assert path.is_file(), f"Dataset file not found: {path}"
    records = []
    labels_int = []
    labels_str = []

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            try:
                l_int, l_str = extract_label_tuple(rec)
            except Exception as e:
                raise ValueError(f"Label extraction failed at {path}:{line_no}: {e}")
            records.append(rec)
            labels_int.append(l_int)
            labels_str.append(l_str)

    return records, labels_int, labels_str


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="SemDrift TF-IDF + Linear Classifier Baseline")
    parser.add_argument(
        "--train_file",
        default="experiments/2026-09-07_clean_v2/dataset/train.jsonl",
        help="Path to training set JSONL",
    )
    parser.add_argument(
        "--val_file",
        default="experiments/2026-09-07_clean_v2/dataset/val.jsonl",
        help="Path to validation set JSONL",
    )
    parser.add_argument(
        "--test_file",
        default="experiments/2026-09-07_clean_v2/dataset/verified_test.jsonl",
        help="Path to test set JSONL",
    )
    parser.add_argument(
        "--feature_mode",
        choices=["combined", "dual_overlap"],
        default="combined",
        help="Feature strategy: 'combined' (pure lexical) or 'dual_overlap' (enhanced relational)",
    )
    parser.add_argument(
        "--model_type",
        choices=["logistic", "ridge", "linear_svc", "sgd"],
        default="logistic",
        help="Canonical model: 'logistic' (Logistic Regression)",
    )
    parser.add_argument("--ngram_max", type=int, default=2, help="Max n-gram range (default: 2)")
    parser.add_argument("--min_df", type=int, default=2, help="Minimum document frequency (default: 2)")
    parser.add_argument("--max_features", type=int, default=10000, help="Vocabulary size (default: 10000)")
    parser.add_argument("--C", type=float, default=1.0, help="Inverse regularization strength (default: 1.0)")
    parser.add_argument(
        "--class_weight",
        choices=["balanced", "none"],
        default="balanced",
        help="Class weighting strategy (default: balanced)",
    )
    parser.add_argument("--max_iter", type=int, default=1000, help="Maximum iterations for solver convergence (default: 1000)")
    parser.add_argument(
        "--tune_threshold",
        action="store_true",
        help="Optimize decision threshold on validation Macro-F1",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Optional directory to save results JSON and predictions JSONL",
    )
    parser.add_argument(
        "--output_results",
        default=None,
        help="Path to export results JSON",
    )
    parser.add_argument(
        "--output_preds",
        default=None,
        help="Path to export predictions JSONL",
    )
    parser.add_argument("--save_model", default=None, help="Optional path to save serialized model pipeline")
    return parser.parse_args()


def main():
    args = parse_args()
    start_time = time.time()

    if args.output_dir:
        out_d = Path(args.output_dir)
        if not args.output_results:
            args.output_results = str(out_d / f"results_tfidf_{args.feature_mode}.json")
        if not args.output_preds:
            args.output_preds = str(out_d / f"tfidf_{args.feature_mode}_predictions.jsonl")

    if not args.output_results:
        args.output_results = "experiments/2026-09-07_clean_v2/evaluation/results_tfidf_baseline.json"
    if not args.output_preds:
        args.output_preds = "experiments/2026-09-07_clean_v2/predictions/tfidf_baseline_predictions.jsonl"

    train_path = PROJECT_ROOT / args.train_file
    val_path = PROJECT_ROOT / args.val_file
    test_path = PROJECT_ROOT / args.test_file

    print("=" * 80)
    print("SEMDRIFT TF-IDF + LINEAR CLASSIFIER BASELINE")
    print(f"Feature Mode   : {args.feature_mode.upper()} "
          f"({'Pure Lexical' if args.feature_mode == 'combined' else 'Enhanced Lexical/Relational'})")
    print(f"Canonical Model: {args.model_type.upper()}")
    print(f"Hyperparams    : ngrams=(1, {args.ngram_max}), min_df={args.min_df}, "
          f"max_features={args.max_features}, C={args.C}, class_weight={args.class_weight}")
    print(f"Stopwords      : None (Preserving negation & semantic operators)")
    print(f"Train File     : {train_path}")
    print(f"Val File       : {val_path}")
    print(f"Test File      : {test_path}")
    print("=" * 80)

    # 1. Preflight Invariant Validation Gate
    print("\n[Step 1/5] Running Preflight Dataset Invariant Gates...", flush=True)
    for split_path, split_name in [(train_path, "train"), (val_path, "val"), (test_path, "test")]:
        report = validate_dataset_split(str(split_path), split_name=split_name)
        print(f"  -> [{split_name.upper()}] Verified {report['total']} samples "
              f"(Clean: {report['class_0_count']}, Drift: {report['class_1_count']}). Gate PASSED.")

    # 2. Load Datasets
    print("\n[Step 2/5] Loading datasets...", flush=True)
    train_records, y_train, _ = load_jsonl_dataset(train_path)
    val_records, y_val, _ = load_jsonl_dataset(val_path)
    test_records, y_test, _ = load_jsonl_dataset(test_path)

    def extract_doc_text(r):
        return clean_docstring(r.get("docstring") or r.get("docstring_after") or r.get("docstring_before") or r.get("raw_docstring") or "")

    def extract_code_text(r):
        return r.get("code") or r.get("code_after") or r.get("code_before") or ""

    train_docs = [extract_doc_text(r) for r in train_records]
    train_codes = [extract_code_text(r) for r in train_records]

    val_docs = [extract_doc_text(r) for r in val_records]
    val_codes = [extract_code_text(r) for r in val_records]

    test_docs = [extract_doc_text(r) for r in test_records]
    test_codes = [extract_code_text(r) for r in test_records]

    # 3. Vectorization & Feature Engineering
    print(f"\n[Step 3/5] Vectorizing text in '{args.feature_mode}' mode...", flush=True)

    if args.feature_mode == "combined":
        # Pure Lexical: doc + code -> single TF-IDF representation
        train_combined = [f"{d} {c}" for d, c in zip(train_docs, train_codes)]
        val_combined = [f"{d} {c}" for d, c in zip(val_docs, val_codes)]
        test_combined = [f"{d} {c}" for d, c in zip(test_docs, test_codes)]

        vec = TfidfVectorizer(
            ngram_range=(1, args.ngram_max),
            min_df=args.min_df,
            max_features=args.max_features,
            sublinear_tf=True,
            stop_words=None,  # Strictly preserve negation and conditions!
        )
        X_train = vec.fit_transform(train_combined)
        X_val = vec.transform(val_combined)
        X_test = vec.transform(test_combined)

        print(f"  -> Vocabulary fitted: {len(vec.vocabulary_)} features.")

    else:
        # Enhanced Lexical: Dual TF-IDF + Relational Alignment & Overlap features
        all_texts = train_docs + train_codes
        vec_shared = TfidfVectorizer(
            ngram_range=(1, args.ngram_max),
            min_df=args.min_df,
            max_features=args.max_features,
            sublinear_tf=True,
            stop_words=None,
        )
        vec_shared.fit(all_texts)

        def transform_dual_features(docs: List[str], codes: List[str]):
            X_d = vec_shared.transform(docs)
            X_c = vec_shared.transform(codes)

            # Cosine similarity row-wise
            cos_sim = np.array((X_d.multiply(X_c)).sum(axis=1), dtype=np.float32)

            # Structural & identifier overlap features
            rel_feats = compute_relational_features(docs, codes)

            dense_feats = np.hstack([cos_sim, rel_feats])
            return sp.hstack([X_d, X_c, sp.csr_matrix(dense_feats)], format="csr")

        X_train = transform_dual_features(train_docs, train_codes)
        X_val = transform_dual_features(val_docs, val_codes)
        X_test = transform_dual_features(test_docs, test_codes)

        print(f"  -> Dual vocabulary + overlap features shape: {X_train.shape[1]} features.")

    # 4. Model Training
    print(f"\n[Step 4/5] Training {args.model_type.upper()} classifier...", flush=True)
    cw = "balanced" if args.class_weight == "balanced" else None

    if args.model_type == "logistic":
        clf = LogisticRegression(
            C=args.C,
            max_iter=args.max_iter,
            random_state=42,
            class_weight=cw,
            solver="lbfgs",
        )
    elif args.model_type == "ridge":
        clf = RidgeClassifier(
            alpha=1.0 / max(args.C, 1e-4),
            random_state=42,
            class_weight=cw,
        )
    elif args.model_type == "linear_svc":
        clf = LinearSVC(
            C=args.C,
            random_state=42,
            class_weight=cw,
            max_iter=args.max_iter,
        )
    elif args.model_type == "sgd":
        clf = SGDClassifier(
            loss="log_loss",
            alpha=1.0 / (args.C * len(y_train)),
            random_state=42,
            class_weight=cw,
            max_iter=args.max_iter,
        )

    clf.fit(X_train, y_train)
    n_iters = int(clf.n_iter_[0]) if hasattr(clf, "n_iter_") else None
    converged = (n_iters < args.max_iter) if n_iters is not None else True
    print(f"  -> Model training complete in {n_iters} iterations (Converged: {converged}).", flush=True)

    # Helper function to get probability or score
    def get_probabilities(model, X) -> np.ndarray:
        if hasattr(model, "predict_proba"):
            return model.predict_proba(X)[:, 1]
        elif hasattr(model, "decision_function"):
            df = model.decision_function(X)
            # Sigmoid transform to map decision function to [0, 1]
            return 1.0 / (1.0 + np.exp(-df))
        else:
            return model.predict(X).astype(float)

    val_probs = get_probabilities(clf, X_val)
    test_probs = get_probabilities(clf, X_test)

    # 5. Validation Threshold Tuning
    selected_threshold = 0.50
    val_macro_f1_selected = None
    threshold_history = None

    if args.tune_threshold:
        print("\n[Step 5a] Sweeping decision threshold on validation set (Macro-F1 criterion)...", flush=True)
        best_tau, best_f1, threshold_history = sweep_validation_threshold(
            y_val, val_probs, metric_name="macro_f1"
        )
        selected_threshold = best_tau
        val_macro_f1_selected = best_f1
        print(f"  -> Default Threshold (0.50) Val Macro-F1: "
              f"{precision_recall_fscore_support(y_val, (val_probs >= 0.50).astype(int), average='macro', zero_division=0)[2]:.4f}")
        print(f"  -> Optimal Threshold ({selected_threshold:.2f}) Val Macro-F1: {val_macro_f1_selected:.4f}")
    else:
        y_val_pred_def = (val_probs >= 0.50).astype(int)
        val_macro_f1_selected = float(precision_recall_fscore_support(y_val, y_val_pred_def, average="macro", zero_division=0)[2])

    # 6. Test Set Evaluation
    print(f"\n[Step 5b] Evaluating on locked test set (N={len(y_test)}) at tau={selected_threshold:.2f}...", flush=True)
    test_preds = (test_probs >= selected_threshold).astype(int)
    test_metrics = compute_metrics(y_test, test_preds, test_probs.tolist())

    # Also compute at default threshold 0.50 for scientific comparison if tuned
    test_metrics_default_tau = None
    if selected_threshold != 0.50:
        test_preds_default = (test_probs >= 0.50).astype(int)
        test_metrics_default_tau = compute_metrics(y_test, test_preds_default, test_probs.tolist())

    # Multi-dimensional breakdowns
    provenances = [r.get("provenance") or r.get("source") or "unknown" for r in test_records]
    drift_types = [r.get("mutation_type") or r.get("drift_type") or ("aligned" if yt == 0 else "unknown")
                   for r, yt in zip(test_records, y_test)]
    severities = [r.get("severity") or ("none" if yt == 0 else "unknown") for r, yt in zip(test_records, y_test)]
    repos = [r.get("repo") or r.get("repo_or_origin") or "unknown" for r in test_records]

    def build_breakdown(group_keys: List[str]):
        grouped_true = defaultdict(list)
        grouped_pred = defaultdict(list)
        grouped_prob = defaultdict(list)
        for g, yt, yp, pr in zip(group_keys, y_test, test_preds, test_probs):
            grouped_true[g].append(yt)
            grouped_pred[g].append(yp)
            grouped_prob[g].append(pr)

        res = {}
        for g in sorted(grouped_true.keys()):
            res[g] = compute_metrics(grouped_true[g], grouped_pred[g], grouped_prob[g])
        return res

    prov_breakdown = build_breakdown(provenances)
    drift_type_breakdown = build_breakdown(drift_types)
    severity_breakdown = build_breakdown(severities)
    repo_breakdown = build_breakdown(repos)

    # Score distribution by provenance
    prov_score_stats = compute_score_distributions(test_probs.tolist(), provenances)

    # ---------------------------------------------------------------------------
    # Console Reporting
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("SEMDRIFT TF-IDF BASELINE TEST SET BENCHMARK RESULTS")
    print("=" * 80)
    print(f"Threshold Applied : {selected_threshold:.2f} "
          f"({'Tuned on Val Macro-F1' if args.tune_threshold else 'Default'})")
    print(f"Accuracy          : {test_metrics['accuracy']*100:.2f}%")
    print(f"Balanced Accuracy : {test_metrics['balanced_accuracy']*100:.2f}%")
    print(f"Precision (Drift) : {test_metrics['precision']*100:.2f}%")
    print(f"Recall (Drift)    : {test_metrics['recall']*100:.2f}%")
    print(f"Binary F1         : {test_metrics['f1']*100:.2f}%")
    print(f"Macro F1          : {test_metrics['macro_f1']*100:.2f}%")
    if test_metrics["roc_auc"] is not None:
        print(f"ROC-AUC           : {test_metrics['roc_auc']:.4f}")
        print(f"PR-AUC            : {test_metrics['pr_auc']:.4f}")
    print(f"Confusion Matrix  : {test_metrics['confusion_matrix']}")
    print(f"Prediction Balance: Aligned={test_metrics['pred_aligned']} | Drifted={test_metrics['pred_drifted']}")

    if test_metrics_default_tau:
        print("\n--- Contrast: Performance at Default Threshold (0.50) ---")
        print(f"BalAcc: {test_metrics_default_tau['balanced_accuracy']*100:.2f}% | "
              f"Macro-F1: {test_metrics_default_tau['macro_f1']*100:.2f}% | "
              f"Recall: {test_metrics_default_tau['recall']*100:.2f}% | "
              f"Prec: {test_metrics_default_tau['precision']*100:.2f}%")

    print("\n" + "-" * 80)
    print(f"{'PROVENANCE BREAKDOWN':<32} | {'N':<5} | {'Recall':<8} | {'Precision':<10} | {'F1':<8} | {'BalAcc':<8}")
    print("-" * 80)
    for prov, m in sorted(prov_breakdown.items()):
        print(f"{prov:<32} | {m['count']:<5} | {m['recall']:<8.4f} | {m['precision']:<10.4f} | {m['f1']:<8.4f} | {m['balanced_accuracy']:<8.4f}")
    print("-" * 80)

    print("\n" + "-" * 80)
    print(f"{'PROVENANCE DRIFT SCORE DISTRIBUTION (P(drift))':<32} | {'N':<5} | {'Mean':<8} | {'Median':<8} | {'Std':<8} | {'Range':<14}")
    print("-" * 80)
    for prov, s in sorted(prov_score_stats.items()):
        rng_str = f"[{s['min']:.2f}, {s['max']:.2f}]"
        print(f"{prov:<32} | {s['count']:<5} | {s['mean']:<8.4f} | {s['median']:<8.4f} | {s['std']:<8.4f} | {rng_str:<14}")
    print("-" * 80)

    print("\n--- Breakdown by Contract Drift Type ---")
    for dt, m in sorted(drift_type_breakdown.items()):
        print(f"  {dt:<28}: Acc={m['accuracy']:.4f} | F1={m['f1']:.4f} | Prec={m['precision']:.4f} | Rec={m['recall']:.4f} (N={m['count']})")

    # ---------------------------------------------------------------------------
    # Export Predictions & Results Artifacts
    # ---------------------------------------------------------------------------
    out_preds_path = PROJECT_ROOT / args.output_preds
    out_preds_path.parent.mkdir(parents=True, exist_ok=True)

    with out_preds_path.open("w", encoding="utf-8") as f:
        for rec, prob, pred in zip(test_records, test_probs, test_preds):
            out = dict(rec)
            pred_str = "drifted" if pred == 1 else "aligned"
            out["predicted_label"] = pred_str
            out["confidence"] = round(float(prob), 6)
            out["drift_score"] = round(float(prob), 6)
            out["threshold_used"] = selected_threshold
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
    print(f"\n[OK] Exported {len(test_records)} instance predictions to {out_preds_path}")

    # Results JSON export
    out_results_path = PROJECT_ROOT / args.output_results
    out_results_path.parent.mkdir(parents=True, exist_ok=True)

    import sklearn
    results_record = {
        "benchmark_name": f"SemDrift V2 TF-IDF Baseline ({args.feature_mode})",
        "feature_mode": args.feature_mode,
        "model_type": args.model_type,
        "config": {
            "ngram_max": args.ngram_max,
            "min_df": args.min_df,
            "max_features": args.max_features,
            "C": args.C,
            "class_weight": args.class_weight,
            "stop_words": None,
            "sublinear_tf": True,
            "max_iter": args.max_iter,
            "n_iter": n_iters,
            "converged": converged,
            "sklearn_version": sklearn.__version__,
        },
        "threshold_tuning": {
            "tuned": args.tune_threshold,
            "threshold_metric": "macro_f1",
            "selected_threshold": selected_threshold,
            "val_macro_f1_at_selected_threshold": val_macro_f1_selected,
        },
        "test_dataset": str(test_path),
        "test_samples": len(test_records),
        "overall_metrics": test_metrics,
        "overall_metrics_default_tau_0_50": test_metrics_default_tau,
        "provenance_breakdown": prov_breakdown,
        "provenance_score_distributions": prov_score_stats,
        "mutation_type_breakdown": drift_type_breakdown,
        "severity_breakdown": severity_breakdown,
        "repo_breakdown": repo_breakdown,
        "elapsed_seconds": round(time.time() - start_time, 2),
    }

    with out_results_path.open("w", encoding="utf-8") as f:
        json.dump(results_record, f, indent=2)
    print(f"[OK] Exported full evaluation record to {out_results_path}")

    # Optional model saving
    if args.save_model:
        import joblib
        save_model_path = PROJECT_ROOT / args.save_model
        save_model_path.parent.mkdir(parents=True, exist_ok=True)
        pipeline_obj = {
            "feature_mode": args.feature_mode,
            "classifier": clf,
            "threshold": selected_threshold,
            "vectorizer": vec if args.feature_mode == "combined" else vec_shared,
        }
        joblib.dump(pipeline_obj, save_model_path)
        print(f"[OK] Serialized model pipeline to {save_model_path}")


if __name__ == "__main__":
    main()
