#!/usr/bin/env python3
"""
scripts/training/evaluate_phase2.py — SemDrift Sliding-Window Joint-Encoder Phase 2 Locked Evaluation.

Evaluates 4 locked configurations on:
1. Expansion V1 Test Set (N = 1,328)
2. Authentic Diagnostic Test Set (N = 104: 52 authentic historical drift, 52 clean grounded)

Configurations:
- Config A: Original single-window baseline (Window 0 only, tau = 0.50)
- Config B: Validation-tuned single-window baseline (Window 0 only, tau = 0.52)
- Config C: Sliding-window mean (Mean probability, tau = 0.52)
- Config D: Locked sliding-window winner (Max probability, tau = 0.58)

Computes:
- Expansion V1 metrics, stratified latencies, window distributions, and McNemar's test.
- Disaggregated diagnostic metrics (authentic drift recall vs clean false-positive rate) with Wilson score 95% CIs.
- Predeclared success criteria PASS/FAIL evaluation.
- Full prediction exports and JSON metadata.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import scipy
from scipy import stats
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
)
from transformers import AutoTokenizer, __version__ as transformers_version

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


def compute_sha256(filepath: Path) -> str:
    """Compute SHA-256 cryptographic hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def wilson_score_interval(successes: int, total: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Calculate Wilson score interval for binomial proportions."""
    if total == 0:
        return 0.0, 0.0
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    p = successes / total
    denominator = 1 + (z**2) / total
    centre = (p + (z**2) / (2 * total)) / denominator
    spread = (z / denominator) * math.sqrt((p * (1 - p) / total) + (z**2) / (4 * (total**2)))
    lower = max(0.0, centre - spread)
    upper = min(1.0, centre + spread)
    return round(float(lower) * 100, 2), round(float(upper) * 100, 2)


def calculate_full_metrics(y_true: list[int], y_pred: list[int], y_probs: list[float] = None) -> dict:
    """Calculate accuracy, macro-F1, weighted-F1, balanced accuracy, precision, recall, and confusion matrix."""
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    _, _, macro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
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
        "accuracy": round(float(acc) * 100, 2),
        "balanced_accuracy": round(float(bal_acc) * 100, 2),
        "macro_f1": round(float(macro_f1) * 100, 2),
        "weighted_f1": round(float(weighted_f1) * 100, 2),
        "precision": round(float(p) * 100, 2),
        "recall": round(float(r) * 100, 2),
        "f1": round(float(f1) * 100, 2),
        "roc_auc": round(float(roc_auc), 4) if roc_auc is not None else None,
        "pr_auc": round(float(pr_auc), 4) if pr_auc is not None else None,
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
        "confusion_matrix": f"TN={cm[0, 0]}, FP={cm[0, 1]}, FN={cm[1, 0]}, TP={cm[1, 1]}",
        "count": len(y_true),
    }


def mcnemar_test(y_true: list[int], y_pred_a: list[int], y_pred_d: list[int]) -> dict:
    """Perform McNemar's test on paired correctness between Config A and Config D."""
    n01 = 0  # Config D incorrect, Config A correct (Config A unique win)
    n10 = 0  # Config D correct, Config A incorrect (Config D unique win)
    n11 = 0  # Both correct
    n00 = 0  # Both incorrect

    for yt, pa, pd in zip(y_true, y_pred_a, y_pred_d):
        corr_a = (pa == yt)
        corr_d = (pd == yt)
        if corr_a and corr_d:
            n11 += 1
        elif corr_a and not corr_d:
            n01 += 1
        elif not corr_a and corr_d:
            n10 += 1
        else:
            n00 += 1

    discordant = n01 + n10
    if discordant == 0:
        chi2_stat = 0.0
        p_val = 1.0
    else:
        # Edwards continuity correction
        chi2_stat = ((abs(n10 - n01) - 1.0) ** 2) / discordant
        p_val = float(stats.chi2.sf(chi2_stat, df=1))

    return {
        "contingency_table": {
            "both_correct_n11": n11,
            "config_d_unique_wins_n10": n10,
            "config_a_unique_wins_n01": n01,
            "both_incorrect_n00": n00,
        },
        "discordant_pairs": discordant,
        "chi2_statistic": round(float(chi2_stat), 4),
        "p_value": float(f"{p_val:.6e}"),
        "statistically_significant_005": bool(p_val < 0.05),
    }


def warmup_gpu(model: torch.nn.Module, device: str, num_batches: int = 10, batch_size: int = 16):
    """Run warm-up passes to initialize CUDA kernels."""
    if not device.startswith("cuda"):
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
    dataset_name: str = "Test",
) -> Tuple[list[dict], dict]:
    """Execute sliding-window inference and record per-function latency."""
    model.eval()
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 1

    results = []
    strata_latencies = defaultdict(list)
    total_samples = len(records)
    print(f"Running inference on {dataset_name} ({total_samples} samples)...")

    for i, rec in enumerate(records):
        doc = rec.get("docstring", "")
        code = rec.get("code", "")

        if device.startswith("cuda"):
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        # Tokenization & window slicing
        doc_ids = tokenizer.encode(doc, add_special_tokens=False, verbose=False)
        code_ids = tokenizer.encode(code, add_special_tokens=False, verbose=False)

        windows = slice_docstring_windows(doc_ids, window_size=window_size, stride=stride)
        k = len(windows)

        # Assemble sequences
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

        # Batch K windows
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

        with torch.no_grad():
            logits = model(tensor_inputs)
            func_probs = F.softmax(logits, dim=1)[:, 1].cpu().tolist()

        p_single = float(func_probs[0])
        p_mean = float(np.mean(func_probs))
        p_max = float(np.max(func_probs))

        if device.startswith("cuda"):
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        func_duration_ms = (t1 - t0) * 1000.0

        res = {
            "sample_index": i,
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

        stratum = f"K={k}" if k <= 3 else "K>=4"
        strata_latencies[stratum].append(func_duration_ms)
        strata_latencies["Overall"].append(func_duration_ms)

        if (i + 1) % 200 == 0 or (i + 1 == total_samples):
            print(f"  [{dataset_name}] Processed {i + 1:4d} / {total_samples} samples...")

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


def evaluate_dataset_configurations(results: list[dict], configs: dict) -> dict:
    """Evaluate all 4 locked configurations on a dataset's results."""
    y_true = [r["label"] for r in results]
    eval_output = {}

    for cfg_id, cfg_info in configs.items():
        prob_key = cfg_info["prob_key"]
        tau = cfg_info["threshold"]
        probs = [r[prob_key] for r in results]
        preds = [1 if p >= tau else 0 for p in probs]

        metrics = calculate_full_metrics(y_true, preds, probs)
        eval_output[cfg_id] = {
            "config_id": cfg_id,
            "name": cfg_info["name"],
            "aggregation": cfg_info["aggregation"],
            "threshold": tau,
            "metrics": metrics,
            "predictions": preds,
            "probabilities": probs,
        }
    return eval_output


def main():
    start_time = time.time()
    timestamp_str = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    output_dir = PROJECT_ROOT / "experiments/expansion_v1/joint_sliding_window/phase2"
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = PROJECT_ROOT / "experiments/expansion_v1/joint_doc192/joint_encoder_checkpoint.pt"
    exp_test_path = PROJECT_ROOT / "data/real_dataset_expansion_v1/splits/test.jsonl"
    diag_test_path = PROJECT_ROOT / "experiments/2026-09-07_clean_v2/dataset/verified_test.jsonl"
    selected_config_path = PROJECT_ROOT / "experiments/expansion_v1/joint_sliding_window/selected_config.json"

    print("=" * 80)
    print("PHASE 2: SEMDRIFT SLIDING-WINDOW JOINT-ENCODER LOCKED EVALUATION")
    print("=" * 80)

    # ------------------------------------------------------------------------
    # 1. Preflight Verification
    # ------------------------------------------------------------------------
    print("Running Preflight Verification...")
    assert ckpt_path.is_file(), f"Checkpoint missing: {ckpt_path}"
    assert exp_test_path.is_file(), f"Expansion V1 test missing: {exp_test_path}"
    assert diag_test_path.is_file(), f"Diagnostic test missing: {diag_test_path}"
    assert selected_config_path.is_file(), f"Locked config missing: {selected_config_path}"

    ckpt_sha256 = compute_sha256(ckpt_path)
    exp_test_sha256 = compute_sha256(exp_test_path)
    diag_test_sha256 = compute_sha256(diag_test_path)

    with open(selected_config_path, "r", encoding="utf-8") as f:
        locked_cfg = json.load(f)

    expected_sha = "6930634acab6448bc6753fbd1a987bcb39a6ebe50f05502992c00dfb5a9132f0"
    assert ckpt_sha256 == expected_sha, f"SHA mismatch! Expected {expected_sha}, got {ckpt_sha256}"
    assert locked_cfg["selected_strategy"] == "sliding_max", f"Unexpected strategy: {locked_cfg['selected_strategy']}"
    assert locked_cfg["locked_threshold"] == 0.58, f"Unexpected threshold: {locked_cfg['locked_threshold']}"

    print(f"  [OK] Checkpoint SHA-256   : {ckpt_sha256}")
    print(f"  [OK] Expansion V1 SHA-256 : {exp_test_sha256}")
    print(f"  [OK] Diagnostic SHA-256   : {diag_test_sha256}")
    print(f"  [OK] Locked Strategy      : {locked_cfg['selected_strategy']} @ threshold {locked_cfg['locked_threshold']}")

    # ------------------------------------------------------------------------
    # 2. Dataset Loading & Diagnostics Inspection
    # ------------------------------------------------------------------------
    print("\nLoading Datasets...")
    exp_dataset = SemDriftDataset(str(exp_test_path), clean_docs=False)
    diag_dataset = SemDriftDataset(str(diag_test_path), clean_docs=False)

    assert len(exp_dataset) == 1328, f"Expansion V1 test count is {len(exp_dataset)}, expected 1328"
    assert len(diag_dataset) == 104, f"Diagnostic test count is {len(diag_dataset)}, expected 104"

    # Confirm diagnostic composition
    diag_labels = [r["label_int"] for r in diag_dataset.records]
    diag_provs = [r.get("provenance", "") for r in diag_dataset.records]
    auth_drift_count = sum(1 for l, p in zip(diag_labels, diag_provs) if l == 1 and p == "authentic_historical_mined")
    clean_count = sum(1 for l, p in zip(diag_labels, diag_provs) if l == 0 and p == "clean_grounded")

    assert auth_drift_count == 52, f"Expected 52 authentic historical drift, found {auth_drift_count}"
    assert clean_count == 52, f"Expected 52 clean grounded, found {clean_count}"
    print(f"  [OK] Expansion V1 Test    : {len(exp_dataset)} samples")
    print(f"  [OK] Diagnostic Test      : {len(diag_dataset)} samples (52 authentic historical drift, 52 clean grounded)")

    # ------------------------------------------------------------------------
    # 3. Model Loading & GPU Warmup
    # ------------------------------------------------------------------------
    print(f"\nInstantiating JointEncoderModel on {DEFAULT_DEVICE}...")
    model = JointEncoderModel(
        model_name="microsoft/codebert-base",
        pooling="cls",
        num_labels=2,
        dropout=0.1,
    )
    state_dict = torch.load(ckpt_path, map_location=DEFAULT_DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEFAULT_DEVICE)
    model.eval()
    print("[OK] Model weights successfully restored to evaluation mode.")

    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    warmup_gpu(model, DEFAULT_DEVICE, num_batches=10, batch_size=16)

    # ------------------------------------------------------------------------
    # 4. Locked Configurations Definition
    # ------------------------------------------------------------------------
    CONFIGS = {
        "A": {
            "name": "Original single-window baseline",
            "aggregation": "First window only",
            "prob_key": "p_single",
            "threshold": 0.50,
        },
        "B": {
            "name": "Validation-tuned single-window baseline",
            "aggregation": "First window only",
            "prob_key": "p_single",
            "threshold": 0.52,
        },
        "C": {
            "name": "Sliding-window mean",
            "aggregation": "Mean of window probabilities",
            "prob_key": "p_mean",
            "threshold": 0.52,
        },
        "D": {
            "name": "Locked sliding-window winner",
            "aggregation": "Maximum of window probabilities",
            "prob_key": "p_max",
            "threshold": 0.58,
        },
    }

    # ------------------------------------------------------------------------
    # 5. Step 1: Expansion V1 Test Evaluation (N = 1,328)
    # ------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STEP 1: EXPANSION V1 TEST EVALUATION (N = 1,328)")
    print("=" * 80)

    exp_results, exp_latency = run_sliding_window_inference(
        model=model,
        tokenizer=tokenizer,
        records=exp_dataset.records,
        device=DEFAULT_DEVICE,
        max_length=512,
        window_size=192,
        stride=128,
        code_truncation="head_tail",
        dataset_name="Expansion V1 Test",
    )

    exp_eval = evaluate_dataset_configurations(exp_results, CONFIGS)

    # McNemar's Test: Config D vs Config A
    y_true_exp = [r["label"] for r in exp_results]
    mcnemar_exp = mcnemar_test(
        y_true=y_true_exp,
        y_pred_a=exp_eval["A"]["predictions"],
        y_pred_d=exp_eval["D"]["predictions"],
    )

    # Window count distribution for Expansion V1
    exp_win_dist = defaultdict(int)
    for r in exp_results:
        exp_win_dist[r["window_count"]] += 1

    print("\nEXPANSION V1 TEST RESULTS SUMMARY:")
    print(f"{'Config':<35} {'Threshold':<11} {'Macro-F1':<11} {'Bal Acc':<11} {'Recall':<10} {'Precision':<11} {'Confusion Matrix'}")
    print("-" * 80)
    for cid in ["A", "B", "C", "D"]:
        m = exp_eval[cid]["metrics"]
        print(f"{exp_eval[cid]['name']:<35} {exp_eval[cid]['threshold']:<11.2f} {m['macro_f1']:<11.2f}% {m['balanced_accuracy']:<11.2f}% {m['recall']:<10.2f}% {m['precision']:<11.2f}% {m['confusion_matrix']}")
    print("-" * 80)
    print(f"McNemar's Test (Config D vs Config A):")
    print(f"  Discordant pairs: {mcnemar_exp['discordant_pairs']} (D unique wins: {mcnemar_exp['contingency_table']['config_d_unique_wins_n10']}, A unique wins: {mcnemar_exp['contingency_table']['config_a_unique_wins_n01']})")
    print(f"  Chi-Square: {mcnemar_exp['chi2_statistic']:.4f}, p-value: {mcnemar_exp['p_value']:.4e}, Significant: {mcnemar_exp['statistically_significant_005']}")

    # ------------------------------------------------------------------------
    # 6. Step 2: Authentic Diagnostic Test Evaluation (N = 104)
    # ------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STEP 2: AUTHENTIC DIAGNOSTIC TEST EVALUATION (N = 104)")
    print("=" * 80)

    diag_results, diag_latency = run_sliding_window_inference(
        model=model,
        tokenizer=tokenizer,
        records=diag_dataset.records,
        device=DEFAULT_DEVICE,
        max_length=512,
        window_size=192,
        stride=128,
        code_truncation="head_tail",
        dataset_name="Authentic Diagnostic Test",
    )

    diag_eval = evaluate_dataset_configurations(diag_results, CONFIGS)

    # Disaggregate Diagnostic Subsets
    diag_disaggregated = {}
    for cid in ["A", "B", "C", "D"]:
        preds = diag_eval[cid]["predictions"]
        
        # Authentic historical drift (label == 1, provenance == 'authentic_historical_mined')
        tp_auth = sum(1 for r, p in zip(diag_results, preds) if r["label"] == 1 and p == 1)
        fn_auth = sum(1 for r, p in zip(diag_results, preds) if r["label"] == 1 and p == 0)
        auth_recall_pct = round((tp_auth / 52.0) * 100, 2)
        ci_recall_low, ci_recall_high = wilson_score_interval(tp_auth, 52)

        # Clean grounded (label == 0, provenance == 'clean_grounded')
        fp_clean = sum(1 for r, p in zip(diag_results, preds) if r["label"] == 0 and p == 1)
        tn_clean = sum(1 for r, p in zip(diag_results, preds) if r["label"] == 0 and p == 0)
        clean_fpr_pct = round((fp_clean / 52.0) * 100, 2)
        clean_spec_pct = round((tn_clean / 52.0) * 100, 2)
        ci_fpr_low, ci_fpr_high = wilson_score_interval(fp_clean, 52)

        diag_disaggregated[cid] = {
            "authentic_drift_subset": {
                "total": 52,
                "true_positives": tp_auth,
                "false_negatives": fn_auth,
                "recall_pct": auth_recall_pct,
                "recall_ratio": f"{tp_auth}/52",
                "recall_wilson_ci_95": [ci_recall_low, ci_recall_high],
            },
            "clean_grounded_subset": {
                "total": 52,
                "false_positives": fp_clean,
                "true_negatives": tn_clean,
                "false_positive_rate_pct": clean_fpr_pct,
                "fpr_ratio": f"{fp_clean}/52",
                "specificity_pct": clean_spec_pct,
                "fpr_wilson_ci_95": [ci_fpr_low, ci_fpr_high],
            },
            "overall_metrics": diag_eval[cid]["metrics"],
        }

    print("\nAUTHENTIC DIAGNOSTIC TEST DISAGGREGATED SUMMARY:")
    print(f"{'Config':<35} {'Threshold':<10} {'Authentic Recall':<20} {'Clean False Alarms':<20} {'Macro-F1':<10} {'Confusion Matrix'}")
    print("-" * 80)
    for cid in ["A", "B", "C", "D"]:
        sub = diag_disaggregated[cid]
        rec_str = f"{sub['authentic_drift_subset']['true_positives']}/52 ({sub['authentic_drift_subset']['recall_pct']:.2f}%)"
        fp_str = f"{sub['clean_grounded_subset']['false_positives']}/52 ({sub['clean_grounded_subset']['false_positive_rate_pct']:.2f}%)"
        print(f"{diag_eval[cid]['name']:<35} {diag_eval[cid]['threshold']:<10.2f} {rec_str:<20} {fp_str:<20} {sub['overall_metrics']['macro_f1']:<10.2f}% {sub['overall_metrics']['confusion_matrix']}")
    print("-" * 80)

    # ------------------------------------------------------------------------
    # 7. Success Criteria Evaluation
    # ------------------------------------------------------------------------
    target_macro_f1 = 85.0
    target_recall_exp = 85.0
    target_auth_drift_tp = 10
    target_clean_fp = 5

    measured_exp_macro_f1 = exp_eval["D"]["metrics"]["macro_f1"]
    measured_exp_recall = exp_eval["D"]["metrics"]["recall"]
    measured_auth_drift_tp = diag_disaggregated["D"]["authentic_drift_subset"]["true_positives"]
    measured_clean_fp = diag_disaggregated["D"]["clean_grounded_subset"]["false_positives"]

    pass_criteria_1 = bool(measured_exp_macro_f1 >= target_macro_f1)
    pass_criteria_2 = bool(measured_exp_recall >= target_recall_exp)
    pass_criteria_3 = bool(measured_auth_drift_tp >= target_auth_drift_tp)
    pass_criteria_4 = bool(measured_clean_fp <= target_clean_fp)

    criteria_table = [
        {
            "criterion": "Expansion V1 Test Macro-F1",
            "target": f">={target_macro_f1:.1f}%",
            "baseline_val": f"{exp_eval['A']['metrics']['macro_f1']:.2f}%",
            "measured_val": f"{measured_exp_macro_f1:.2f}%",
            "delta": f"{measured_exp_macro_f1 - exp_eval['A']['metrics']['macro_f1']:+.2f}%",
            "status": "PASS" if pass_criteria_1 else "FAIL",
        },
        {
            "criterion": "Expansion V1 Test Drift Recall",
            "target": f">={target_recall_exp:.1f}%",
            "baseline_val": f"{exp_eval['A']['metrics']['recall']:.2f}%",
            "measured_val": f"{measured_exp_recall:.2f}%",
            "delta": f"{measured_exp_recall - exp_eval['A']['metrics']['recall']:+.2f}%",
            "status": "PASS" if pass_criteria_2 else "FAIL",
        },
        {
            "criterion": "Authentic Historical Drift Caught",
            "target": f">={target_auth_drift_tp}/52",
            "baseline_val": f"{diag_disaggregated['A']['authentic_drift_subset']['true_positives']}/52",
            "measured_val": f"{measured_auth_drift_tp}/52",
            "delta": f"{measured_auth_drift_tp - diag_disaggregated['A']['authentic_drift_subset']['true_positives']:+d}",
            "status": "PASS" if pass_criteria_3 else "FAIL",
        },
        {
            "criterion": "Clean-Grounded False Positives",
            "target": f"<={target_clean_fp}/52",
            "baseline_val": f"{diag_disaggregated['A']['clean_grounded_subset']['false_positives']}/52",
            "measured_val": f"{measured_clean_fp}/52",
            "delta": f"{measured_clean_fp - diag_disaggregated['A']['clean_grounded_subset']['false_positives']:+d}",
            "status": "PASS" if pass_criteria_4 else "FAIL",
        },
    ]

    print("\nSUCCESS CRITERIA VERIFICATION:")
    print(f"{'Criterion':<35} {'Target':<12} {'Baseline (A)':<14} {'Measured (D)':<14} {'Delta':<10} {'Status'}")
    print("-" * 80)
    for c in criteria_table:
        print(f"{c['criterion']:<35} {c['target']:<12} {c['baseline_val']:<14} {c['measured_val']:<14} {c['delta']:<10} {c['status']}")
    print("-" * 80)

    # ------------------------------------------------------------------------
    # 8. Save Artifacts
    # ------------------------------------------------------------------------
    print("\nWriting Phase 2 Artifacts...")

    # A. expansion_v1_test_results.json
    exp_output_file = output_dir / "expansion_v1_test_results.json"
    exp_output_data = {
        "dataset": str(exp_test_path),
        "sample_count": len(exp_dataset),
        "configurations": {
            cid: {
                "name": exp_eval[cid]["name"],
                "aggregation": exp_eval[cid]["aggregation"],
                "threshold": exp_eval[cid]["threshold"],
                "metrics": exp_eval[cid]["metrics"],
            }
            for cid in CONFIGS
        },
        "mcnemar_test_d_vs_a": mcnemar_exp,
        "latency_summary": exp_latency,
        "window_distribution": dict(exp_win_dist),
    }
    with exp_output_file.open("w", encoding="utf-8") as f:
        json.dump(exp_output_data, f, indent=2)
    print(f"  [OK] Saved {exp_output_file}")

    # B. authentic_diagnostic_results.json
    diag_output_file = output_dir / "authentic_diagnostic_results.json"
    diag_output_data = {
        "dataset": str(diag_test_path),
        "sample_count": len(diag_dataset),
        "configurations": diag_disaggregated,
        "latency_summary": diag_latency,
    }
    with diag_output_file.open("w", encoding="utf-8") as f:
        json.dump(diag_output_data, f, indent=2)
    print(f"  [OK] Saved {diag_output_file}")

    # C. expansion_v1_test_predictions.jsonl
    exp_preds_file = output_dir / "expansion_v1_test_predictions.jsonl"
    with exp_preds_file.open("w", encoding="utf-8") as f:
        for i, r in enumerate(exp_results):
            rec = dict(r["record"])
            rec["sample_index"] = i
            rec["ground_truth"] = r["label"]
            rec["window_count"] = r["window_count"]
            rec["window_probs"] = r["window_probs"]
            rec["p_single"] = r["p_single"]
            rec["p_mean"] = r["p_mean"]
            rec["p_max"] = r["p_max"]
            rec["latency_ms"] = r["latency_ms"]
            rec["predictions"] = {
                "config_a": exp_eval["A"]["predictions"][i],
                "config_b": exp_eval["B"]["predictions"][i],
                "config_c": exp_eval["C"]["predictions"][i],
                "config_d": exp_eval["D"]["predictions"][i],
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"  [OK] Saved {exp_preds_file}")

    # D. authentic_diagnostic_predictions.jsonl
    diag_preds_file = output_dir / "authentic_diagnostic_predictions.jsonl"
    with diag_preds_file.open("w", encoding="utf-8") as f:
        for i, r in enumerate(diag_results):
            rec = dict(r["record"])
            rec["sample_index"] = i
            rec["ground_truth"] = r["label"]
            rec["provenance"] = rec.get("provenance", "unknown")
            rec["window_count"] = r["window_count"]
            rec["window_probs"] = r["window_probs"]
            rec["p_single"] = r["p_single"]
            rec["p_mean"] = r["p_mean"]
            rec["p_max"] = r["p_max"]
            rec["latency_ms"] = r["latency_ms"]
            rec["predictions"] = {
                "config_a": diag_eval["A"]["predictions"][i],
                "config_b": diag_eval["B"]["predictions"][i],
                "config_c": diag_eval["C"]["predictions"][i],
                "config_d": diag_eval["D"]["predictions"][i],
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"  [OK] Saved {diag_preds_file}")

    # E. phase2_run_metadata.json
    metadata_file = output_dir / "phase2_run_metadata.json"
    metadata = {
        "evaluation_name": "SemDrift Sliding-Window Joint-Encoder Phase 2 Locked Evaluation",
        "timestamp_utc": timestamp_str,
        "elapsed_seconds": round(time.time() - start_time, 2),
        "environment": {
            "python_version": sys.version,
            "torch_version": torch.__version__,
            "transformers_version": transformers_version,
            "scipy_version": scipy.__version__,
            "platform": platform.platform(),
            "device": DEFAULT_DEVICE,
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None",
        },
        "model": {
            "checkpoint_path": str(ckpt_path),
            "checkpoint_sha256": ckpt_sha256,
            "base_model": "microsoft/codebert-base",
            "pooling": "cls",
            "max_length": 512,
            "window_size": 192,
            "stride": 128,
            "code_truncation": "head_tail",
            "clean_docstrings": False,
        },
        "datasets": {
            "expansion_v1_test": {
                "path": str(exp_test_path),
                "sha256": exp_test_sha256,
                "sample_count": len(exp_dataset),
            },
            "authentic_diagnostic_test": {
                "path": str(diag_test_path),
                "sha256": diag_test_sha256,
                "sample_count": len(diag_dataset),
            },
        },
        "locked_configurations": CONFIGS,
        "success_criteria_results": criteria_table,
    }
    with metadata_file.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"  [OK] Saved {metadata_file}")

    # F. phase2_comparison_report.md
    report_file = output_dir / "phase2_comparison_report.md"
    report_md = f"""# SemDrift Phase 2 Comparison Report: Sliding-Window Joint Encoder

**Evaluation Date**: {timestamp_str}  
**Checkpoint**: `{ckpt_path.name}` (`{ckpt_sha256[:16]}...`)  
**Hardware**: {metadata['environment']['gpu_name']} ({DEFAULT_DEVICE})  

---

## 1. Executive Summary & Success Criteria Assessment

| Criterion | Target | Baseline (Config A) | Locked Winner (Config D) | Absolute Delta | Status |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Expansion V1 Macro-F1** | $\\ge 85.0\\%$ | **{exp_eval['A']['metrics']['macro_f1']:.2f}%** | **{measured_exp_macro_f1:.2f}%** | **{measured_exp_macro_f1 - exp_eval['A']['metrics']['macro_f1']:+.2f}%** | **{'PASS' if pass_criteria_1 else 'FAIL'}** |
| **Expansion V1 Drift Recall** | $\\ge 85.0\\%$ | **{exp_eval['A']['metrics']['recall']:.2f}%** | **{measured_exp_recall:.2f}%** | **{measured_exp_recall - exp_eval['A']['metrics']['recall']:+.2f}%** | **{'PASS' if pass_criteria_2 else 'FAIL'}** |
| **Authentic Drift Detected ($N=52$)** | $\\ge 10/52$ | **{diag_disaggregated['A']['authentic_drift_subset']['true_positives']}/52** | **{measured_auth_drift_tp}/52** | **{measured_auth_drift_tp - diag_disaggregated['A']['authentic_drift_subset']['true_positives']:+d}** | **{'PASS' if pass_criteria_3 else 'FAIL'}** |
| **Clean False Positives ($N=52$)** | $\\le 5/52$ | **{diag_disaggregated['A']['clean_grounded_subset']['false_positives']}/52** | **{measured_clean_fp}/52** | **{measured_clean_fp - diag_disaggregated['A']['clean_grounded_subset']['false_positives']:+d}** | **{'PASS' if pass_criteria_4 else 'FAIL'}** |

---

## 2. Expansion V1 Test Set Performance ($N = 1,328$)

| ID | Configuration | Aggregation | $\\tau$ | Accuracy | Macro-F1 | Bal Acc | Precision | Recall | F1 | Confusion Matrix |
|:---:|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **A** | Original Baseline | First window only | 0.50 | {exp_eval['A']['metrics']['accuracy']:.2f}% | {exp_eval['A']['metrics']['macro_f1']:.2f}% | {exp_eval['A']['metrics']['balanced_accuracy']:.2f}% | {exp_eval['A']['metrics']['precision']:.2f}% | {exp_eval['A']['metrics']['recall']:.2f}% | {exp_eval['A']['metrics']['f1']:.2f}% | `{exp_eval['A']['metrics']['confusion_matrix']}` |
| **B** | Tuned Baseline | First window only | 0.52 | {exp_eval['B']['metrics']['accuracy']:.2f}% | {exp_eval['B']['metrics']['macro_f1']:.2f}% | {exp_eval['B']['metrics']['balanced_accuracy']:.2f}% | {exp_eval['B']['metrics']['precision']:.2f}% | {exp_eval['B']['metrics']['recall']:.2f}% | {exp_eval['B']['metrics']['f1']:.2f}% | `{exp_eval['B']['metrics']['confusion_matrix']}` |
| **C** | Sliding Mean | Window Mean | 0.52 | {exp_eval['C']['metrics']['accuracy']:.2f}% | {exp_eval['C']['metrics']['macro_f1']:.2f}% | {exp_eval['C']['metrics']['balanced_accuracy']:.2f}% | {exp_eval['C']['metrics']['precision']:.2f}% | {exp_eval['C']['metrics']['recall']:.2f}% | {exp_eval['C']['metrics']['f1']:.2f}% | `{exp_eval['C']['metrics']['confusion_matrix']}` |
| **D** | **Locked Winner** | **Window Max** | **0.58** | **{exp_eval['D']['metrics']['accuracy']:.2f}%** | **{exp_eval['D']['metrics']['macro_f1']:.2f}%** | **{exp_eval['D']['metrics']['balanced_accuracy']:.2f}%** | **{exp_eval['D']['metrics']['precision']:.2f}%** | **{exp_eval['D']['metrics']['recall']:.2f}%** | **{exp_eval['D']['metrics']['f1']:.2f}%** | `{exp_eval['D']['metrics']['confusion_matrix']}` |

### McNemar's Paired Significance Test (Config D vs Config A)
- **Both Correct ($n_{{11}}$)**: {mcnemar_exp['contingency_table']['both_correct_n11']}
- **Config D Unique Wins ($n_{{10}}$)**: {mcnemar_exp['contingency_table']['config_d_unique_wins_n10']}
- **Config A Unique Wins ($n_{{01}}$)**: {mcnemar_exp['contingency_table']['config_a_unique_wins_n01']}
- **Both Incorrect ($n_{{00}}$)**: {mcnemar_exp['contingency_table']['both_incorrect_n00']}
- **Discordant Pairs**: {mcnemar_exp['discordant_pairs']}
- **Chi-Square (with continuity correction)**: **{mcnemar_exp['chi2_statistic']:.4f}**
- **p-value**: **{mcnemar_exp['p_value']:.4e}** (Statistically Significant at $\\alpha=0.05$: **{'Yes' if mcnemar_exp['statistically_significant_005'] else 'No'}**)

---

## 3. Authentic Diagnostic Test Set Breakdown ($N = 104$)

### A. Authentic Historical Human Drift ($N = 52$)
| ID | Configuration | $\\tau$ | Caught (TP) | Missed (FN) | Authentic Drift Recall (95% Wilson CI) |
|:---:|:---|:---:|:---:|:---:|:---|
| **A** | Original Baseline | 0.50 | {diag_disaggregated['A']['authentic_drift_subset']['true_positives']} | {diag_disaggregated['A']['authentic_drift_subset']['false_negatives']} | **{diag_disaggregated['A']['authentic_drift_subset']['recall_ratio']} ({diag_disaggregated['A']['authentic_drift_subset']['recall_pct']:.2f}%)** [{diag_disaggregated['A']['authentic_drift_subset']['recall_wilson_ci_95'][0]}% - {diag_disaggregated['A']['authentic_drift_subset']['recall_wilson_ci_95'][1]}%] |
| **B** | Tuned Baseline | 0.52 | {diag_disaggregated['B']['authentic_drift_subset']['true_positives']} | {diag_disaggregated['B']['authentic_drift_subset']['false_negatives']} | **{diag_disaggregated['B']['authentic_drift_subset']['recall_ratio']} ({diag_disaggregated['B']['authentic_drift_subset']['recall_pct']:.2f}%)** [{diag_disaggregated['B']['authentic_drift_subset']['recall_wilson_ci_95'][0]}% - {diag_disaggregated['B']['authentic_drift_subset']['recall_wilson_ci_95'][1]}%] |
| **C** | Sliding Mean | 0.52 | {diag_disaggregated['C']['authentic_drift_subset']['true_positives']} | {diag_disaggregated['C']['authentic_drift_subset']['false_negatives']} | **{diag_disaggregated['C']['authentic_drift_subset']['recall_ratio']} ({diag_disaggregated['C']['authentic_drift_subset']['recall_pct']:.2f}%)** [{diag_disaggregated['C']['authentic_drift_subset']['recall_wilson_ci_95'][0]}% - {diag_disaggregated['C']['authentic_drift_subset']['recall_wilson_ci_95'][1]}%] |
| **D** | **Locked Winner** | **0.58** | **{diag_disaggregated['D']['authentic_drift_subset']['true_positives']}** | **{diag_disaggregated['D']['authentic_drift_subset']['false_negatives']}** | **{diag_disaggregated['D']['authentic_drift_subset']['recall_ratio']} ({diag_disaggregated['D']['authentic_drift_subset']['recall_pct']:.2f}%)** [{diag_disaggregated['D']['authentic_drift_subset']['recall_wilson_ci_95'][0]}% - {diag_disaggregated['D']['authentic_drift_subset']['recall_wilson_ci_95'][1]}%] |

### B. Clean-Grounded Production Functions ($N = 52$)
| ID | Configuration | $\\tau$ | False Alarms (FP) | Clean Correct (TN) | Clean FPR (95% Wilson CI) | Specificity |
|:---:|:---|:---:|:---:|:---:|:---|:---:|
| **A** | Original Baseline | 0.50 | {diag_disaggregated['A']['clean_grounded_subset']['false_positives']} | {diag_disaggregated['A']['clean_grounded_subset']['true_negatives']} | **{diag_disaggregated['A']['clean_grounded_subset']['fpr_ratio']} ({diag_disaggregated['A']['clean_grounded_subset']['false_positive_rate_pct']:.2f}%)** [{diag_disaggregated['A']['clean_grounded_subset']['fpr_wilson_ci_95'][0]}% - {diag_disaggregated['A']['clean_grounded_subset']['fpr_wilson_ci_95'][1]}%] | {diag_disaggregated['A']['clean_grounded_subset']['specificity_pct']:.2f}% |
| **B** | Tuned Baseline | 0.52 | {diag_disaggregated['B']['clean_grounded_subset']['false_positives']} | {diag_disaggregated['B']['clean_grounded_subset']['true_negatives']} | **{diag_disaggregated['B']['clean_grounded_subset']['fpr_ratio']} ({diag_disaggregated['B']['clean_grounded_subset']['false_positive_rate_pct']:.2f}%)** [{diag_disaggregated['B']['clean_grounded_subset']['fpr_wilson_ci_95'][0]}% - {diag_disaggregated['B']['clean_grounded_subset']['fpr_wilson_ci_95'][1]}%] | {diag_disaggregated['B']['clean_grounded_subset']['specificity_pct']:.2f}% |
| **C** | Sliding Mean | 0.52 | {diag_disaggregated['C']['clean_grounded_subset']['false_positives']} | {diag_disaggregated['C']['clean_grounded_subset']['true_negatives']} | **{diag_disaggregated['C']['clean_grounded_subset']['fpr_ratio']} ({diag_disaggregated['C']['clean_grounded_subset']['false_positive_rate_pct']:.2f}%)** [{diag_disaggregated['C']['clean_grounded_subset']['fpr_wilson_ci_95'][0]}% - {diag_disaggregated['C']['clean_grounded_subset']['fpr_wilson_ci_95'][1]}%] | {diag_disaggregated['C']['clean_grounded_subset']['specificity_pct']:.2f}% |
| **D** | **Locked Winner** | **0.58** | **{diag_disaggregated['D']['clean_grounded_subset']['false_positives']}** | **{diag_disaggregated['D']['clean_grounded_subset']['true_negatives']}** | **{diag_disaggregated['D']['clean_grounded_subset']['fpr_ratio']} ({diag_disaggregated['D']['clean_grounded_subset']['false_positive_rate_pct']:.2f}%)** [{diag_disaggregated['D']['clean_grounded_subset']['fpr_wilson_ci_95'][0]}% - {diag_disaggregated['D']['clean_grounded_subset']['fpr_wilson_ci_95'][1]}%] | **{diag_disaggregated['D']['clean_grounded_subset']['specificity_pct']:.2f}%** |

---

## 4. Stratified Latency Benchmarking (Expansion V1 Test, $N=1,328$)

| Stratum | Definition | Sample Count | Share | Mean Latency | Median Latency | P95 Latency | Min – Max |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **$K = 1$** | $N_{{doc}} \\le 192$ | {exp_latency['K=1']['count']} | {exp_latency['K=1']['count']/1328*100:.1f}% | {exp_latency['K=1']['mean_ms']:.2f} ms | {exp_latency['K=1']['median_ms']:.2f} ms | {exp_latency['K=1']['p95_ms']:.2f} ms | {exp_latency['K=1']['min_ms']:.2f} – {exp_latency['K=1']['max_ms']:.2f} ms |
| **$K = 2$** | $193 \\le N_{{doc}} \\le 320$ | {exp_latency['K=2']['count']} | {exp_latency['K=2']['count']/1328*100:.1f}% | {exp_latency['K=2']['mean_ms']:.2f} ms | {exp_latency['K=2']['median_ms']:.2f} ms | {exp_latency['K=2']['p95_ms']:.2f} ms | {exp_latency['K=2']['min_ms']:.2f} – {exp_latency['K=2']['max_ms']:.2f} ms |
| **$K = 3$** | $321 \\le N_{{doc}} \\le 448$ | {exp_latency['K=3']['count']} | {exp_latency['K=3']['count']/1328*100:.1f}% | {exp_latency['K=3']['mean_ms']:.2f} ms | {exp_latency['K=3']['median_ms']:.2f} ms | {exp_latency['K=3']['p95_ms']:.2f} ms | {exp_latency['K=3']['min_ms']:.2f} – {exp_latency['K=3']['max_ms']:.2f} ms |
| **$K \\ge 4$** | $N_{{doc}} > 448$ | {exp_latency['K>=4']['count']} | {exp_latency['K>=4']['count']/1328*100:.1f}% | {exp_latency['K>=4']['mean_ms']:.2f} ms | {exp_latency['K>=4']['median_ms']:.2f} ms | {exp_latency['K>=4']['p95_ms']:.2f} ms | {exp_latency['K>=4']['min_ms']:.2f} – {exp_latency['K>=4']['max_ms']:.2f} ms |
| **Overall** | All test instances | 1328 | 100.0% | **{exp_latency['Overall']['mean_ms']:.2f} ms** | **{exp_latency['Overall']['median_ms']:.2f} ms** | **{exp_latency['Overall']['p95_ms']:.2f} ms** | {exp_latency['Overall']['min_ms']:.2f} – {exp_latency['Overall']['max_ms']:.2f} ms |
"""
    with report_file.open("w", encoding="utf-8") as f:
        f.write(report_md)
    print(f"  [OK] Saved {report_file}")
    print("\nPhase 2 Locked Evaluation Complete.")


if __name__ == "__main__":
    main()
