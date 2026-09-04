#!/usr/bin/env python3
"""
scripts/training/train_joint_encoder.py — Fine-tune Fine-Tuned Joint-Encoder (Primary Contribution).

Architecture:
  CodeBERT receives docstring and code as a SINGLE concatenated input:
      [CLS] docstring_tokens [SEP] code_tokens [SEP]
  Self-attention sees ALL tokens from both sides in one forward pass.
  The [CLS] hidden state is fed to a classification head → aligned / drifted.

This version (v2) includes options for head-tail token truncation, customizable pooling,
robust metric-based checkpoint selection (macro-F1 or balanced accuracy), and detailed
evaluation logging per epoch.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, balanced_accuracy_score
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from semdrift.models.joint_encoder import (
    JointEncoderModel,
    SemDriftDataset,
    FocalLoss,
    make_collate_fn,
    extract_docstring_summary,
)

DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_epoch(model, dataloader, optimizer, scheduler, loss_fn, device, category_weighting: bool = False):
    """Run one training epoch. Returns average loss."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch_idx, (inputs, labels, metas) in enumerate(dataloader):
        inputs = {k: v.to(device) for k, v in inputs.items()}
        labels = labels.to(device)

        sample_weights = None
        if category_weighting:
            weights = []
            for meta in metas:
                m_type = meta.get("drift_type") or meta.get("mutation_type")
                if m_type == "doc_negation":
                    weights.append(1.5)
                elif m_type == "doc_sentence_delete":
                    weights.append(1.2)
                else:
                    weights.append(1.0)
            sample_weights = torch.tensor(weights, dtype=torch.float, device=device)

        optimizer.zero_grad()
        logits = model(inputs)

        if isinstance(loss_fn, FocalLoss):
            loss = loss_fn(logits, labels, sample_weights=sample_weights)
        else:
            if sample_weights is not None:
                unweighted = F.cross_entropy(logits, labels, reduction="none")
                loss = (unweighted * sample_weights).mean()
            else:
                loss = loss_fn(logits, labels)

        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        num_batches += 1

        # Progress logging every 50 batches
        if (batch_idx + 1) % 50 == 0:
            print(f"    Batch {batch_idx + 1}/{len(dataloader)} | Loss: {loss.item():.4f}", flush=True)

    return total_loss / max(num_batches, 1)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(model, dataloader, device):
    """Run evaluation. Returns labels, predictions, probabilities, and metadata."""
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []
    all_metas = []

    with torch.no_grad():
        for inputs, labels, metas in dataloader:
            inputs = {k: v.to(device) for k, v in inputs.items()}
            logits = model(inputs)
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)

            all_labels.extend(labels.tolist())
            all_preds.extend(preds.cpu().tolist())
            all_probs.extend(probs[:, 1].cpu().tolist())  # P(drifted)
            all_metas.extend(metas)

    label_strings = ["drifted" if l == 1 else "aligned" for l in all_labels]
    pred_strings = ["drifted" if p == 1 else "aligned" for p in all_preds]

    return label_strings, pred_strings, all_probs, all_metas


def calculate_metrics(y_true: list[str], y_pred: list[str]) -> dict:
    """Compute overall metrics including balanced accuracy, confusion matrix, and prediction balance."""
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

    # Confusion matrix extraction
    cm = confusion_matrix(y_b_true, y_b_pred)
    tn, fp, fn, tp = 0, 0, 0, 0
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        # Handle dry runs or subsets where only one label is present
        if len(set(y_b_true)) == 1:
            val = list(set(y_b_true))[0]
            if val == 0:
                tn = len(y_b_true)
            else:
                tp = len(y_b_true)

    pred_aligned = sum(1 for p in y_b_pred if p == 0)
    pred_drifted = sum(1 for p in y_b_pred if p == 1)

    return {
        "accuracy": round(acc, 4),
        "precision": round(float(p), 4),
        "recall": round(float(r), 4),
        "f1": round(float(f1), 4),
        "macro_f1": round(float(macro_f1), 4),
        "balanced_accuracy": round(balanced_acc, 4),
        "confusion_matrix": f"TN={tn}, FP={fp}, FN={fn}, TP={tp}",
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "pred_aligned": pred_aligned,
        "pred_drifted": pred_drifted,
        "count": len(y_true),
    }


def evaluate_breakdowns(y_true: list[str], y_pred: list[str], metas: list[dict]) -> dict:
    """Compute metrics broken down by drift_type, severity, and repo."""
    by_drift_type: dict[str, list] = defaultdict(list)
    by_severity: dict[str, list] = defaultdict(list)
    by_repo: dict[str, list] = defaultdict(list)

    for yt, yp, m in zip(y_true, y_pred, metas):
        pair = (yt, yp)
        by_drift_type[m["drift_type"]].append(pair)
        by_severity[m["severity"]].append(pair)
        by_repo[m["repo"]].append(pair)

    def calc_group(group_dict):
        result = {}
        for key, pairs in group_dict.items():
            yt_g, yp_g = zip(*pairs)
            result[key] = calculate_metrics(list(yt_g), list(yp_g))
        return result

    return {
        "by_drift_type": calc_group(by_drift_type),
        "by_severity": calc_group(by_severity),
        "by_repo": calc_group(by_repo),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune Model B (Joint Encoder — Primary Contribution) v2"
    )

    # Data
    parser.add_argument("--train", default="data/experiments/v2/train.jsonl", help="Train dataset")
    parser.add_argument("--val", default="data/experiments/v2/val.jsonl", help="Validation dataset")
    parser.add_argument("--test", default="data/experiments/v2/test.jsonl", help="Test dataset")

    # Architecture & Tokenization configs
    parser.add_argument("--model_name", default="microsoft/codebert-base",
                        help="HuggingFace model checkpoint")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout before classifier head")
    parser.add_argument("--max_length", type=int, default=512,
                        help="Max token length for joint (docstring + code) input (Cap: 512)")
    parser.add_argument("--doc_max_tokens", type=int, default=96,
                        help="Max token budget for docstrings")
    parser.add_argument("--code_truncation", choices=["head", "tail", "head_tail"], default="head_tail",
                        help="Code truncation strategy if budget is exceeded")
    parser.add_argument("--pooling", choices=["cls", "mean"], default="cls",
                        help="Pooling strategy for sentence representations")

    # Training
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="L2 weight decay")
    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                        help="Fraction of total steps for LR warmup")
    parser.add_argument("--checkpoint_metric", choices=["macro_f1", "balanced_accuracy", "f1", "accuracy"], default="macro_f1",
                        help="Validation metric to pick the best checkpoint")

    # Loss & Class Weighting
    parser.add_argument("--use_focal_loss", dest="use_focal_loss", action="store_true", default=True,
                        help="Use Focal Loss to focus on hard samples (e.g. doc_negation)")
    parser.add_argument("--no_focal_loss", dest="use_focal_loss", action="store_false",
                        help="Disable Focal Loss (use standard CrossEntropyLoss)")
    parser.add_argument("--focal_gamma", type=float, default=2.0, help="Gamma parameter for Focal Loss")
    parser.add_argument("--focal_alpha", type=float, default=0.5, help="Alpha parameter for Focal Loss")
    parser.add_argument("--category_weighting", dest="category_weighting", action="store_true", default=True,
                        help="Enable sample loss weighting for underperforming categories (e.g., doc_negation)")
    parser.add_argument("--no_category_weighting", dest="category_weighting", action="store_false",
                        help="Disable sample loss weighting across categories")

    # Flags
    parser.add_argument("--no_clean_docstrings", dest="clean_docstrings",
                        action="store_false", default=True,
                        help="Disable docstring summary extraction (use full docstrings)")
    parser.add_argument("--device", default=DEFAULT_DEVICE, help="Device (cuda / cpu)")
    parser.add_argument("--output_dir", default="data/experiments/v2/joint_encoder_results",
                        help="Directory to write predictions and results")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--dry_run", action="store_true", default=False,
                        help="Quick run on 16-sample subset to verify shapes/gradients")

    args = parser.parse_args()
    set_seed(args.seed)

    # Force clip max_length to 512 to avoid index out of bound for CodeBERT positional embeddings
    if args.max_length > 512:
        print(f"Warning: --max_length {args.max_length} exceeds CodeBERT limit. Clipping to 512.", flush=True)
        args.max_length = 512

    print("=" * 70, flush=True)
    print("Fine-Tuned Joint-Encoder (Primary Contribution) — V2 Pipeline", flush=True)
    print("=" * 70, flush=True)
    print(f"Base Model        : {args.model_name}", flush=True)
    print(f"Pooling Strategy  : {args.pooling.upper()}", flush=True)
    print(f"Code Truncation   : {args.code_truncation}", flush=True)
    print(f"Doc Max Tokens    : {args.doc_max_tokens}", flush=True)
    print(f"Max Seq Length    : {args.max_length}", flush=True)
    print(f"Checkpoint Metric : {args.checkpoint_metric}", flush=True)
    print(f"Use Focal Loss    : {args.use_focal_loss} (gamma={args.focal_gamma}, alpha={args.focal_alpha})", flush=True)
    print(f"Category Weighting: {args.category_weighting}", flush=True)
    print(f"Device            : {args.device}", flush=True)
    print(f"Epochs            : {args.epochs}", flush=True)
    print(f"Batch Size        : {args.batch_size}", flush=True)
    print(f"Learning Rate     : {args.lr}", flush=True)
    print(f"Clean Docstrings  : {args.clean_docstrings}", flush=True)
    print(f"Dry Run Mode      : {args.dry_run}", flush=True)
    print("-" * 70, flush=True)

    # ------------------------------------------------------------------
    # 1. Load Datasets
    # ------------------------------------------------------------------
    print("Loading datasets...", flush=True)
    train_dataset = SemDriftDataset(args.train, clean_docs=args.clean_docstrings)
    val_dataset = SemDriftDataset(args.val, clean_docs=args.clean_docstrings)
    test_dataset = SemDriftDataset(args.test, clean_docs=args.clean_docstrings)

    if args.dry_run:
        print(f"[Dry Run] Subsetting datasets to 16 samples each.", flush=True)
        train_dataset.records = train_dataset.records[:16]
        val_dataset.records = val_dataset.records[:16]
        test_dataset.records = test_dataset.records[:16]

    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | "
          f"Test: {len(test_dataset)} samples.", flush=True)

    # ------------------------------------------------------------------
    # 2. Tokenizer & DataLoaders
    # ------------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    collate_fn = make_collate_fn(
        tokenizer,
        max_length=args.max_length,
        doc_max_tokens=args.doc_max_tokens,
        truncation_strategy=args.code_truncation
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn
    )

    # ------------------------------------------------------------------
    # 3. Model
    # ------------------------------------------------------------------
    model = JointEncoderModel(
        model_name=args.model_name, pooling=args.pooling, num_labels=2, dropout=args.dropout
    )
    model.to(args.device)
    print("Model loaded successfully. Full fine-tuning enabled.", flush=True)

    # ------------------------------------------------------------------
    # 4. Optimization Setup
    # ------------------------------------------------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    if args.use_focal_loss:
        loss_fn = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma)
    else:
        loss_fn = nn.CrossEntropyLoss()

    # ------------------------------------------------------------------
    # 5. Training Loop
    # ------------------------------------------------------------------
    print(f"\nStarting training loop ({args.epochs} epochs)...", flush=True)

    best_val_metric_val = -1.0
    best_epoch = -1
    checkpoint_path = os.path.join(args.output_dir, "joint_encoder_checkpoint.pt")
    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(args.epochs):
        epoch_start = time.time()
        avg_loss = train_epoch(
            model, train_loader, optimizer, scheduler, loss_fn, args.device,
            category_weighting=args.category_weighting
        )

        # Validation evaluation
        val_y_true, val_y_pred, _, _ = evaluate(model, val_loader, args.device)
        val_metrics = calculate_metrics(val_y_true, val_y_pred)

        elapsed = time.time() - epoch_start
        print(f"Epoch {epoch + 1}/{args.epochs} | Loss: {avg_loss:.4f} | ({elapsed:.1f}s)", flush=True)
        print(f"  Accuracy: {val_metrics['accuracy']:.4f} | F1: {val_metrics['f1']:.4f} | Macro-F1: {val_metrics['macro_f1']:.4f} | Balanced Acc: {val_metrics['balanced_accuracy']:.4f}", flush=True)
        print(f"  Confusion Matrix: {val_metrics['confusion_matrix']}", flush=True)
        print(f"  Prediction Balance: Aligned={val_metrics['pred_aligned']} | Drifted={val_metrics['pred_drifted']}", flush=True)

        # Checkpoint selection with Bias Collapse Guard
        current_metric_val = val_metrics[args.checkpoint_metric]
        drifted_ratio = val_metrics['pred_drifted'] / max(val_metrics['count'], 1)
        aligned_ratio = val_metrics['pred_aligned'] / max(val_metrics['count'], 1)

        is_collapsed = (drifted_ratio > 0.88) or (aligned_ratio > 0.88)

        if is_collapsed:
            print(f"  -> [WARNING] Bias collapse detected! (Drifted ratio: {drifted_ratio:.1%}, Aligned ratio: {aligned_ratio:.1%}). Skipping checkpoint save.", flush=True)
        elif current_metric_val > best_val_metric_val:
            best_val_metric_val = current_metric_val
            best_epoch = epoch + 1
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  -> Saved best checkpoint to {checkpoint_path} (Val {args.checkpoint_metric}={best_val_metric_val:.4f})", flush=True)
        print("-" * 50, flush=True)

    # ------------------------------------------------------------------
    # 6. Load Best Checkpoint & Run Test Evaluation
    # ------------------------------------------------------------------
    print(f"\nLoading best checkpoint from Epoch {best_epoch} for final test evaluation...", flush=True)
    model.load_state_dict(torch.load(checkpoint_path, map_location=args.device))

    # Evaluate validation metrics with best checkpoint
    val_y_true, val_y_pred, _, _ = evaluate(model, val_loader, args.device)
    val_metrics = calculate_metrics(val_y_true, val_y_pred)

    # Evaluate test metrics with best checkpoint
    test_y_true, test_y_pred, test_probs, test_metas = evaluate(
        model, test_loader, args.device
    )
    test_overall = calculate_metrics(test_y_true, test_y_pred)
    breakdowns = evaluate_breakdowns(test_y_true, test_y_pred, test_metas)

    # Confusion matrix test set
    cm = confusion_matrix(
        [1 if l == "drifted" else 0 for l in test_y_true],
        [1 if p == "drifted" else 0 for p in test_y_pred],
    )

    print("\n" + "=" * 70, flush=True)
    print("FINAL TEST RESULTS — Fine-Tuned Joint-Encoder V2", flush=True)
    print("=" * 70, flush=True)
    print(f"Best Validation Epoch   : {best_epoch}", flush=True)
    print(f"Accuracy                : {test_overall['accuracy']:.4f}", flush=True)
    print(f"Precision               : {test_overall['precision']:.4f}", flush=True)
    print(f"Recall                  : {test_overall['recall']:.4f}", flush=True)
    print(f"F1 Score (Binary)       : {test_overall['f1']:.4f}", flush=True)
    print(f"Macro F1 Score          : {test_overall['macro_f1']:.4f}", flush=True)
    print(f"Balanced Accuracy       : {test_overall['balanced_accuracy']:.4f}", flush=True)
    print("Confusion Matrix:", flush=True)
    print(f"  TN: {cm[0, 0]}  |  FP: {cm[0, 1]}", flush=True)
    print(f"  FN: {cm[1, 0]}  |  TP: {cm[1, 1]}", flush=True)

    print("\n--- Breakdown by Drift Type ---", flush=True)
    for dt, m in sorted(breakdowns["by_drift_type"].items()):
        print(f"  {dt:<22}: Acc={m['accuracy']:.4f} | "
              f"F1={m['f1']:.4f} | Prec={m['precision']:.4f} | "
              f"Rec={m['recall']:.4f} (N={m['count']})", flush=True)

    print("\n--- Breakdown by Severity ---", flush=True)
    for sev, m in sorted(breakdowns["by_severity"].items()):
        print(f"  {sev:<22}: Acc={m['accuracy']:.4f} | "
              f"F1={m['f1']:.4f} | Prec={m['precision']:.4f} | "
              f"Rec={m['recall']:.4f} (N={m['count']})", flush=True)

    print("\n--- Breakdown by Repository ---", flush=True)
    for repo, m in sorted(breakdowns["by_repo"].items()):
        print(f"  {repo:<22}: Acc={m['accuracy']:.4f} | "
              f"F1={m['f1']:.4f} | Prec={m['precision']:.4f} | "
              f"Rec={m['recall']:.4f} (N={m['count']})", flush=True)

    # ------------------------------------------------------------------
    # 7. Save outputs
    # ------------------------------------------------------------------
    pred_path = os.path.join(args.output_dir, "predictions_joint_encoder.jsonl")
    results_path = os.path.join(args.output_dir, "results_joint_encoder.json")

    with open(pred_path, "w", encoding="utf-8") as f:
        for rec, prob, pred in zip(test_dataset.records, test_probs, test_y_pred):
            out = dict(rec)
            out["predicted_label"] = pred
            out["confidence"] = round(prob, 6)
            f.write(json.dumps(out) + "\n")

    full_results = {
        "model": "Fine-Tuned Joint Encoder (Joint Code–Documentation Self-Attention) — V2",
        "model_name": args.model_name,
        "architecture": "joint_encoder",
        "pooling": args.pooling,
        "code_truncation": args.code_truncation,
        "doc_max_tokens": args.doc_max_tokens,
        "max_length": args.max_length,
        "dropout": args.dropout,
        "use_focal_loss": args.use_focal_loss,
        "category_weighting": args.category_weighting,
        "checkpoint_metric": args.checkpoint_metric,
        "best_epoch": best_epoch,
        "val_metrics": val_metrics,
        "test_overall": test_overall,
        "confusion_matrix": {
            "tn": int(cm[0, 0]),
            "fp": int(cm[0, 1]),
            "fn": int(cm[1, 0]),
            "tp": int(cm[1, 1]),
        },
        "breakdowns": breakdowns,
    }

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(full_results, f, indent=2)

    print("\n" + "-" * 70, flush=True)
    print(f"Predictions : {pred_path}", flush=True)
    print(f"Results     : {results_path}", flush=True)
    print("Fine-Tuned Joint-Encoder v2 execution complete!", flush=True)


if __name__ == "__main__":
    main()
