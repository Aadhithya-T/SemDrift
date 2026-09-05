"""Multi-Signal Weak Supervision pipeline for generating Silver-Standard drift datasets.

Combines deterministic AST contract checks, lexical in-docstring parameter tracking,
structural priority scoring, and refactoring guards to compute a probabilistic consensus
pseudo-label P(drift | x) for all 5k mined candidates.

Calibrates and validates rules against the 76 human-verified gold benchmark samples.
"""

import argparse
import ast
import json
import logging
import math
import re
import sys
import textwrap
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CANDIDATE_ROOT = PROJECT_ROOT / "data" / "real_world" / "mined_candidates"
VERIFIED_ROOT = PROJECT_ROOT / "data" / "real_world" / "verified"
REJECTED_ROOT = PROJECT_ROOT / "data" / "real_world" / "rejected"
SILVER_ROOT = PROJECT_ROOT / "data" / "real_world" / "silver"

WORD_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

# Re-use contract checking functions
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
try:
    from contract_check_candidates import (
        parse_ast_function,
        extract_parameter_names,
        extract_documented_parameters,
        extract_parameter_defaults,
        extract_return_annotation,
        extract_raised_exceptions,
        extract_documented_exceptions,
        check_candidate_contracts,
    )
except ImportError:
    pass


def load_jsonl(path: Path) -> List[dict]:
    """Load JSON objects from a JSONL file."""
    records = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            try:
                records.append(json.loads(s))
            except json.JSONDecodeError:
                continue
    return records


def extract_identifiers(source: str) -> Set[str]:
    """Extract all valid Python identifier tokens from source code."""
    if not source:
        return set()
    return set(WORD_PATTERN.findall(source)) - {
        "def", "class", "return", "import", "from", "for", "in", "if", "else", "elif",
        "while", "try", "except", "finally", "with", "as", "pass", "None", "True", "False",
        "self", "cls", "and", "or", "not", "is", "lambda", "yield", "raise", "async", "await"
    }


def compute_docstring_signature_drift_signal(candidate: dict) -> float:
    """Check whether parameters added or removed from function signature are explicitly referenced in docstring."""
    code_before = candidate.get("code_before", "") or ""
    code_after = candidate.get("code_after", "") or ""
    docstring = candidate.get("docstring_after") or candidate.get("docstring_before") or ""
    func_name = candidate.get("function_name", "")

    if not docstring or not code_before or not code_after:
        return 0.0

    func_b = parse_ast_function(code_before, func_name)
    func_a = parse_ast_function(code_after, func_name)
    if func_b is None or func_a is None:
        return 0.0

    params_b = extract_parameter_names(func_b)
    params_a = extract_parameter_names(func_a)
    doc_params = extract_documented_parameters(docstring)

    # If parameters changed in signature and are documented
    changed_params = (params_b - params_a) | (params_a - params_b)
    if changed_params & doc_params:
        return 1.0

    # Check exception discrepancy
    exc_b = extract_raised_exceptions(func_b)
    exc_a = extract_raised_exceptions(func_a)
    doc_exc = extract_documented_exceptions(docstring)
    if (exc_b - exc_a) & doc_exc:
        return 1.0

    return 0.0


def compute_pure_refactor_negative_signal(candidate: dict) -> float:
    """Flag changes that are purely non-breaking (type annotations only, docstring untouched)."""
    code_before = candidate.get("code_before", "") or ""
    code_after = candidate.get("code_after", "") or ""
    func_name = candidate.get("function_name", "")

    func_b = parse_ast_function(code_before, func_name)
    func_a = parse_ast_function(code_after, func_name)

    if func_b is None or func_a is None:
        return 0.0

    params_b = extract_parameter_names(func_b)
    params_a = extract_parameter_names(func_a)

    # Identical parameter names and identical raised exceptions
    if params_b == params_a:
        exc_b = extract_raised_exceptions(func_b)
        exc_a = extract_raised_exceptions(func_a)
        if exc_b == exc_a:
            doc_b = candidate.get("docstring_before", "") or ""
            doc_a = candidate.get("docstring_after", "") or ""
            if doc_b.strip() == doc_a.strip():
                return 1.0  # Strong signal for no-drift

    return 0.0


def compute_silver_probability(candidate: dict) -> Tuple[float, float, Dict[str, float]]:
    """Compute consensus pseudo-probability P(drift | x) using multiple weak labeling functions."""
    # Ensure contract checks are computed
    if "contract_violation_count" not in candidate:
        candidate = check_candidate_contracts(candidate)

    # LF 1: AST Contract Violations (Hard deterministic rule)
    contract_count = candidate.get("contract_violation_count", 0)
    lf_contract = 1.0 if contract_count >= 1 else 0.0

    # LF 2: Signature vs Docstring mismatch
    lf_sig_doc = compute_docstring_signature_drift_signal(candidate)

    # LF 3: Pure Refactor / Negative Guard
    lf_refactor = compute_pure_refactor_negative_signal(candidate)

    # Combine signals
    if lf_contract > 0 or lf_sig_doc > 0:
        p_drift = 0.90 if (lf_contract > 0 and lf_sig_doc > 0) else 0.75
    elif lf_refactor > 0:
        p_drift = 0.05
    else:
        # Subtle candidate with neither explicit contract violation nor pure refactor
        p_drift = 0.20

    confidence = abs(p_drift - 0.5) * 2.0  # 0.0 at decision boundary, 1.0 at extremes

    signals = {
        "lf_contract": round(lf_contract, 4),
        "lf_sig_doc": round(lf_sig_doc, 4),
        "lf_refactor": round(lf_refactor, 4),
    }

    return round(p_drift, 4), round(confidence, 4), signals


def load_gold_dataset() -> List[dict]:
    """Load the 76 human-verified gold instances across Click, FastAPI, and Django."""
    gold_records = []
    for repo in ["click", "fastapi", "django"]:
        v_path = VERIFIED_ROOT / f"{repo}_verified.jsonl"
        r_path = REJECTED_ROOT / f"{repo}_rejected.jsonl"
        for item in load_jsonl(v_path):
            item["gold_label"] = "drift"
            gold_records.append(item)
        for item in load_jsonl(r_path):
            item["gold_label"] = "no_drift"
            gold_records.append(item)
    return gold_records


def calibrate_on_gold_benchmark(gold_records: List[dict], threshold_high: float = 0.60, threshold_low: float = 0.25) -> dict:
    """Measure Silver labeling performance against the 76 human-verified gold ground truth."""
    tp, fp, tn, fn, skipped = 0, 0, 0, 0, 0

    for item in gold_records:
        p_drift, conf, _ = compute_silver_probability(item)
        true_label = item.get("gold_label")

        if p_drift >= threshold_high:
            pred = "drift"
        elif p_drift <= threshold_low:
            pred = "no_drift"
        else:
            skipped += 1
            continue

        if pred == "drift":
            if true_label == "drift":
                tp += 1
            else:
                fp += 1
        elif pred == "no_drift":
            if true_label == "no_drift":
                tn += 1
            else:
                fn += 1

    decided = tp + fp + tn + fn
    accuracy = (tp + tn) / decided if decided > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    neg_precision = tn / (tn + fn) if (tn + fn) > 0 else 0.0

    return {
        "gold_total": len(gold_records),
        "evaluated_samples": decided,
        "ambiguous_skipped": skipped,
        "accuracy": round(accuracy, 4),
        "drift_precision": round(precision, 4),
        "drift_recall": round(recall, 4),
        "drift_f1": round(f1, 4),
        "no_drift_precision": round(neg_precision, 4),
        "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate Silver-Standard dataset using multi-signal weak supervision.")
    parser.add_argument("--threshold-high", type=float, default=0.60, help="P(drift) cutoff for Silver Positives")
    parser.add_argument("--threshold-low", type=float, default=0.25, help="P(drift) cutoff for Silver Negatives")
    args = parser.parse_args()

    SILVER_ROOT.mkdir(parents=True, exist_ok=True)

    # 1. Calibrate on Gold Benchmark
    gold_records = load_gold_dataset()
    print(f"Loaded {len(gold_records)} gold ground-truth verification instances.")
    metrics = calibrate_on_gold_benchmark(gold_records, args.threshold_high, args.threshold_low)

    print("=" * 64)
    print("GOLD BENCHMARK CALIBRATION RESULTS (N = 76)")
    print("=" * 64)
    print(f"Evaluated Decisions:      {metrics['evaluated_samples']} / {metrics['gold_total']}")
    print(f"Accuracy on Gold:         {metrics['accuracy'] * 100:.2f}%")
    print(f"Drift Precision (Pos):    {metrics['drift_precision'] * 100:.2f}%")
    print(f"Drift Recall (Pos):       {metrics['drift_recall'] * 100:.2f}%")
    print(f"Drift F1 Score:           {metrics['drift_f1'] * 100:.2f}%")
    print(f"No-Drift Precision (Neg): {metrics['no_drift_precision'] * 100:.2f}%")
    print(f"Confusion Matrix:         TP={metrics['confusion_matrix']['tp']}, FP={metrics['confusion_matrix']['fp']}, TN={metrics['confusion_matrix']['tn']}, FN={metrics['confusion_matrix']['fn']}")
    print("=" * 64)

    # 2. Process All 5k Candidates
    repos = ["click", "fastapi", "django"]
    all_silver_positives = []
    all_silver_negatives = []
    all_ambiguous = []

    repo_stats = {}

    for repo in repos:
        in_path = CANDIDATE_ROOT / f"{repo}_contract_checked.jsonl"
        if not in_path.is_file():
            in_path = CANDIDATE_ROOT / f"{repo}_scored.jsonl"
        if not in_path.is_file():
            continue

        records = load_jsonl(in_path)
        pos_cnt, neg_cnt, amb_cnt = 0, 0, 0

        for candidate in records:
            p_drift, conf, signals = compute_silver_probability(candidate)
            candidate["silver_drift_probability"] = p_drift
            candidate["silver_confidence"] = conf
            candidate["silver_signals"] = signals

            if p_drift >= args.threshold_high:
                candidate["silver_label"] = "drift"
                all_silver_positives.append(candidate)
                pos_cnt += 1
            elif p_drift <= args.threshold_low:
                candidate["silver_label"] = "no_drift"
                all_silver_negatives.append(candidate)
                neg_cnt += 1
            else:
                candidate["silver_label"] = "ambiguous"
                all_ambiguous.append(candidate)
                amb_cnt += 1

        repo_stats[repo] = {
            "total": len(records),
            "silver_positives": pos_cnt,
            "silver_negatives": neg_cnt,
            "ambiguous": amb_cnt,
        }

    # 3. Write Partitioned Silver Datasets
    pos_file = SILVER_ROOT / "silver_drift_positives.jsonl"
    neg_file = SILVER_ROOT / "silver_clean_negatives.jsonl"
    amb_file = SILVER_ROOT / "silver_ambiguous.jsonl"
    meta_file = SILVER_ROOT / "silver_dataset_summary.json"

    # Sort positives descending by probability and confidence
    all_silver_positives.sort(key=lambda c: (c["silver_drift_probability"], c["silver_confidence"]), reverse=True)
    all_silver_negatives.sort(key=lambda c: (1.0 - c["silver_drift_probability"], c["silver_confidence"]), reverse=True)

    with pos_file.open("w", encoding="utf-8") as f:
        for c in all_silver_positives:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    with neg_file.open("w", encoding="utf-8") as f:
        for c in all_silver_negatives:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    with amb_file.open("w", encoding="utf-8") as f:
        for c in all_ambiguous:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    summary = {
        "dataset_name": "SemDrift Silver-Standard Real-World Dataset",
        "total_candidates_processed": len(all_silver_positives) + len(all_silver_negatives) + len(all_ambiguous),
        "silver_positives_count": len(all_silver_positives),
        "silver_negatives_count": len(all_silver_negatives),
        "ambiguous_filtered_count": len(all_ambiguous),
        "silver_drift_rate": round(len(all_silver_positives) / (len(all_silver_positives) + len(all_silver_negatives)), 4),
        "repo_breakdown": repo_stats,
        "gold_calibration": metrics,
    }

    with meta_file.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 64)
    print("SILVER DATASET GENERATION SUMMARY")
    print("=" * 64)
    print(f"Total Candidates Processed:     {summary['total_candidates_processed']}")
    print(f"Silver Drift Positives (+):      {summary['silver_positives_count']} ({summary['silver_positives_count'] / summary['total_candidates_processed'] * 100:.1f}%)")
    print(f"Silver Clean Negatives (-):      {summary['silver_negatives_count']} ({summary['silver_negatives_count'] / summary['total_candidates_processed'] * 100:.1f}%)")
    print(f"Ambiguous / Discarded:          {summary['ambiguous_filtered_count']} ({summary['ambiguous_filtered_count'] / summary['total_candidates_processed'] * 100:.1f}%)")
    print("-" * 64)
    for repo, s in repo_stats.items():
        print(f"  - {repo.upper()}: Total={s['total']} | Positives={s['silver_positives']} | Negatives={s['silver_negatives']} | Ambiguous={s['ambiguous']}")
    print("=" * 64)
    print(f"Artifacts exported to: {SILVER_ROOT.relative_to(PROJECT_ROOT)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
