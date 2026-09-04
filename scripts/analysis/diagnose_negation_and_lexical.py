#!/usr/bin/env python3
"""
scripts/analysis/diagnose_negation_and_lexical.py — Deep Diagnostic of doc_negation Failure & TF-IDF Baseline.

Phase 1: Deep Diagnostic of doc_negation
  - Probability distribution analysis of False Negatives (FNs)
  - Docstring summary truncation & mutation preservation analysis
  - Linguistic pattern analysis of mutations
  - Concrete case inspection (code, docstring, P(drift))

Phase 2: Lexical Baseline (TF-IDF + Logistic Regression)
  - Train on train.jsonl
  - Evaluate on test.jsonl
  - Benchmark overall and per-drift-type performance
"""

import json
import os
import re
import sys
from collections import Counter, defaultdict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

NEGATION_SWAPS = {
    r"\breturns\b": "does not return",
    r"\bwill\b": "will not",
    r"\bcan\b": "cannot",
    r"\bshould\b": "should not",
    r"\bmust\b": "must not",
    r"\balways\b": "never",
    r"\bautomatically\b": "manually",
    r"\bdefault\b": "non-default",
    r"\braises\b": "suppresses",
    r"\bvalid\b": "invalid",
    r"\benabled\b": "disabled",
    r"\boptional\b": "required",
    r"\bsupported\b": "unsupported",
    r"\ballow\b": "disallow",
    r"\ballows\b": "disallows",
    r"\btrue\b": "false",
    r"\bincludes\b": "excludes",
    r"\bignores\b": "enforces",
    r"\bwith\b": "without",
}

SWAPPED_PHRASES = list(NEGATION_SWAPS.values())


def extract_docstring_summary(docstring: str) -> str:
    """Same function used in dataset preprocessing."""
    if not docstring:
        return ""
    lines = docstring.strip().split("\n")
    summary_lines = []
    for line in lines:
        stripped = line.strip()
        if (
            not stripped
            or stripped.startswith(">>>")
            or stripped.startswith("...")
            or stripped.startswith("Parameters")
            or stripped.startswith("Returns")
            or stripped.startswith("Examples")
            or stripped.startswith("See Also")
            or stripped.startswith("Notes")
            or stripped.startswith("Raises")
            or stripped.startswith("Warnings")
            or stripped.startswith("References")
        ):
            break
        summary_lines.append(stripped)

    cleaned = " ".join(summary_lines).strip()
    if len(cleaned) >= 10:
        return cleaned
    return lines[0].strip()


def run_diagnostic():
    pred_path = os.path.join(
        PROJECT_ROOT,
        "data",
        "experiments",
        "v2",
        "joint_focal_controlled",
        "predictions_joint_encoder.jsonl",
    )
    with open(pred_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f]

    neg_records = [r for r in records if r.get("drift_type") == "doc_negation"]
    print("=" * 80)
    print(f"STEP 1: DIAGNOSTIC OF doc_negation (Total N = {len(neg_records)})")
    print("=" * 80)

    tps = [r for r in neg_records if r["predicted_label"] == "drifted"]
    fns = [r for r in neg_records if r["predicted_label"] == "aligned"]

    print(f"True Positives  (Detected Drift) : {len(tps)} ({len(tps)/len(neg_records)*100:.2f}%)")
    print(f"False Negatives (Missed Drift)   : {len(fns)} ({len(fns)/len(neg_records)*100:.2f}%)")

    # Step 2: Look at the probabilities
    fn_probs = [r["confidence"] for r in fns]  # P(drift)
    tp_probs = [r["confidence"] for r in tps]

    print("\n" + "-" * 80)
    print("STEP 2: PROBABILITY DISTRIBUTION ANALYSIS FOR FALSE NEGATIVES")
    print("-" * 80)
    print(f"FN P(drift) — Mean: {np.mean(fn_probs):.4f}, Median: {np.median(fn_probs):.4f}, Std: {np.std(fn_probs):.4f}")
    print(f"FN P(drift) — Min:  {np.min(fn_probs):.4f}, Max:    {np.max(fn_probs):.4f}")
    print(f"TP P(drift) — Mean: {np.mean(tp_probs):.4f}, Median: {np.median(tp_probs):.4f}")

    bins = [
        (0.40, 0.50),
        (0.35, 0.40),
        (0.30, 0.35),
        (0.20, 0.30),
        (0.10, 0.20),
        (0.00, 0.10),
    ]
    print("\nFN Probability Histogram P(drift):")
    for low, high in bins:
        count = sum(1 for p in fn_probs if low <= p < high)
        bar = "#" * int(count * 40 / len(fns))
        print(f"  [{low:.2f} - {high:.2f}): {count:>2} ({count/len(fns)*100:>5.1f}%) | {bar}")

    # Check summary truncation effect
    print("\n" + "-" * 80)
    print("STEP 3: SUMMARY TRUNCATION & MUTATION SURVIVAL CHECK")
    print("-" * 80)

    stripped_count = 0
    preserved_count = 0
    mutation_locations = []

    for r in neg_records:
        raw_doc = r["docstring"]
        cleaned_doc = extract_docstring_summary(raw_doc)

        # Check which negation phrase is present
        found_phrase = None
        for p in SWAPPED_PHRASES:
            if re.search(r"\b" + re.escape(p) + r"\b", raw_doc, re.IGNORECASE):
                found_phrase = p
                break

        if found_phrase:
            # Check if it survived in cleaned docstring
            survived = bool(re.search(r"\b" + re.escape(found_phrase) + r"\b", cleaned_doc, re.IGNORECASE))
            if survived:
                preserved_count += 1
            else:
                stripped_count += 1
        else:
            # Phrase might be composite or original
            pass

    print(f"Total doc_negation records with identified mutation phrase : {preserved_count + stripped_count}")
    print(f"Mutation preserved in summary (Seen by model)              : {preserved_count}")
    print(f"Mutation STRIPPED by extract_docstring_summary (Invisible!): {stripped_count} ({stripped_count/(preserved_count+stripped_count)*100:.1f}%)")

    # Let's see survival broken down by TP vs FN
    tp_stripped = 0
    fn_stripped = 0
    for r in tps:
        raw_doc = r["docstring"]
        cleaned_doc = extract_docstring_summary(raw_doc)
        found = any(re.search(r"\b" + re.escape(p) + r"\b", raw_doc, re.IGNORECASE) for p in SWAPPED_PHRASES)
        survived = any(re.search(r"\b" + re.escape(p) + r"\b", cleaned_doc, re.IGNORECASE) for p in SWAPPED_PHRASES)
        if found and not survived:
            tp_stripped += 1

    for r in fns:
        raw_doc = r["docstring"]
        cleaned_doc = extract_docstring_summary(raw_doc)
        found = any(re.search(r"\b" + re.escape(p) + r"\b", raw_doc, re.IGNORECASE) for p in SWAPPED_PHRASES)
        survived = any(re.search(r"\b" + re.escape(p) + r"\b", cleaned_doc, re.IGNORECASE) for p in SWAPPED_PHRASES)
        if found and not survived:
            fn_stripped += 1

    print(f"  -> TPs where mutation was stripped: {tp_stripped}/{len(tps)}")
    print(f"  -> FNs where mutation was stripped: {fn_stripped}/{len(fns)} ({fn_stripped/len(fns)*100:.1f}%)")

    # Inspect concrete examples
    print("\n" + "-" * 80)
    print("STEP 4: INSPECTING 10 CONCRETE FALSE NEGATIVES")
    print("-" * 80)
    for i, r in enumerate(fns[:10], 1):
        raw_doc = r["docstring"]
        cleaned_doc = extract_docstring_summary(raw_doc)
        code_snip = r["code"][:250].replace("\n", " ")

        found_phrase = "unknown"
        for p in SWAPPED_PHRASES:
            if re.search(r"\b" + re.escape(p) + r"\b", raw_doc, re.IGNORECASE):
                found_phrase = p
                break

        survived = re.search(r"\b" + re.escape(found_phrase) + r"\b", cleaned_doc, re.IGNORECASE) is not None

        print(f"\n[Case #{i}] Function: {r['function_name']} ({r['repo']}) | Severity: {r['severity']}")
        print(f"  P(drift)          : {r['confidence']:.4f}")
        print(f"  Mutation Phrase   : '{found_phrase}'")
        print(f"  Survived in Summary: {'YES' if survived else 'NO (STRIPPED BY PREPROCESSING!)'}")
        print(f"  Cleaned Docstring : {cleaned_doc[:120]}...")
        if not survived:
            print(f"  Full Raw Docstring: {raw_doc[:160]}...")
        print(f"  Code Snippet      : {code_snip[:120]}...")


def run_tfidf_baseline():
    print("\n" + "=" * 80)
    print("STEP 5: LEXICAL BASELINE (TF-IDF + LOGISTIC REGRESSION)")
    print("=" * 80)

    train_path = os.path.join(PROJECT_ROOT, "data", "experiments", "v2", "train.jsonl")
    test_path = os.path.join(PROJECT_ROOT, "data", "experiments", "v2", "test.jsonl")

    print("Loading train & test sets...")
    with open(train_path, "r", encoding="utf-8") as f:
        train_data = [json.loads(line) for line in f]
    with open(test_path, "r", encoding="utf-8") as f:
        test_data = [json.loads(line) for line in f]

    def prep_text(rec):
        doc = extract_docstring_summary(rec.get("docstring", ""))
        code = rec.get("code", "")
        return doc + " " + code

    train_texts = [prep_text(r) for r in train_data]
    train_labels = [1 if r.get("label") in (1, "drifted", "1") else 0 for r in train_data]

    test_texts = [prep_text(r) for r in test_data]
    test_labels = [1 if r.get("label") in (1, "drifted", "1") else 0 for r in test_data]

    print(f"Vectorizing texts with TF-IDF (max_features=10,000, ngram_range=(1,2))...")
    vec = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), stop_words="english")
    X_train = vec.fit_transform(train_texts)
    X_test = vec.transform(test_texts)

    print("Training Logistic Regression classifier (C=1.0, max_iter=1000)...")
    clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    clf.fit(X_train, train_labels)

    test_preds = clf.predict(X_test)
    test_probs = clf.predict_proba(X_test)[:, 1]

    acc = accuracy_score(test_labels, test_preds)
    p, r, f1, _ = precision_recall_fscore_support(test_labels, test_preds, average="binary", zero_division=0)
    _, _, macro_f1, _ = precision_recall_fscore_support(test_labels, test_preds, average="macro", zero_division=0)

    print("\n--- Overall TF-IDF Baseline Performance ---")
    print(f"Accuracy  : {acc*100:.2f}%")
    print(f"Precision : {p*100:.2f}%")
    print(f"Recall    : {r*100:.2f}%")
    print(f"Binary F1 : {f1*100:.2f}%")
    print(f"Macro F1  : {macro_f1*100:.2f}%")

    # Breakdown by drift type
    by_dt = defaultdict(lambda: {"true": [], "pred": []})
    for r, yt, yp in zip(test_data, test_labels, test_preds):
        dt = r.get("drift_type") or ("aligned" if yt == 0 else "unknown")
        by_dt[dt]["true"].append(yt)
        by_dt[dt]["pred"].append(yp)

    print("\n--- TF-IDF Performance by Drift Type ---")
    print(f"{'Drift Type':<24} | {'Count':<8} | {'Accuracy':<10} | {'Recall':<10} | {'F1-Score':<10}")
    print("-" * 70)
    for dt, data in sorted(by_dt.items()):
        yt = data["true"]
        yp = data["pred"]
        dt_acc = accuracy_score(yt, yp) * 100
        _, dt_r, dt_f1, _ = precision_recall_fscore_support(yt, yp, average="binary", zero_division=0)
        print(f"{dt:<24} | {len(yt):<8} | {dt_acc:>8.2f}% | {dt_r*100:>8.2f}% | {dt_f1*100:>8.2f}%")


if __name__ == "__main__":
    run_diagnostic()
    run_tfidf_baseline()
