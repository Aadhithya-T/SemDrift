#!/usr/bin/env python3
"""
scripts/train_model_b.py — Fine-tune Model B (Dual-Encoder + Similarity / Classifier Head).

Supports:
  - Variant 2 (Default): Shared encoder + classifier head on [u; v; |u-v|]. Trained using CrossEntropyLoss.
  - Variant 1: Shared encoder + Cosine Embedding Loss. Classification done via post-training threshold sweep.
  - Complete validation checkpointing and evaluation breakdowns by drift type, severity, and repo.
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
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def extract_docstring_summary(docstring: str) -> str:
    """Extract clean natural language summary from docstrings."""
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


class SemDriftDataset(Dataset):
    def __init__(self, filepath: str, clean_docs: bool = True):
        self.records = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    if clean_docs:
                        rec["docstring"] = extract_docstring_summary(rec.get("docstring", ""))
                    self.records.append(rec)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        code = rec.get("code", "")
        docstring = rec.get("docstring", "")
        label_str = rec.get("label", "aligned")
        label = 1 if label_str == "drifted" else 0
        
        # Meta dictionary for evaluating breakdowns later
        meta = {
            "repo": rec.get("repo", "unknown"),
            "drift_type": rec.get("drift_type") or "aligned",
            "severity": rec.get("severity") or "aligned",
            "label_str": label_str
        }
        return code, docstring, label, meta


def make_collate_fn(tokenizer: AutoTokenizer, max_length: int):
    def collate_fn(batch):
        codes = [item[0] for item in batch]
        docstrings = [item[1] for item in batch]
        labels = torch.tensor([item[2] for item in batch], dtype=torch.long)
        metas = [item[3] for item in batch]

        # Tokenize code and docstring separately to avoid cross-attention
        code_inputs = tokenizer(
            codes,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        doc_inputs = tokenizer(
            docstrings,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        return code_inputs, doc_inputs, labels, metas
    return collate_fn


class DualEncoderModel(nn.Module):
    def __init__(self, model_name: str, variant: str = "variant_2", freeze_base: bool = False):
        super().__init__()
        self.variant = variant
        self.encoder = AutoModel.from_pretrained(model_name)
        
        if freeze_base:
            print("Freezing base model layers. Fine-tuning only the classifier head.")
            for param in self.encoder.parameters():
                param.requires_grad = False
        else:
            print("Fine-tuning base model + classification/projection layers end-to-end.")

        if self.variant == "variant_2":
            # Classifier head on top of [u; v; |u-v|]
            # u and v are mean-pooled representations (dimension: 768)
            hidden_size = self.encoder.config.hidden_size
            self.classifier = nn.Linear(3 * hidden_size, 2)

    def mean_pooling(self, last_hidden_state, attention_mask):
        token_embeddings = last_hidden_state
        mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        summed = torch.sum(token_embeddings * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        return summed / counts

    def forward(self, code_inputs, doc_inputs):
        # Step 1: Independently encode code and docstrings
        code_outputs = self.encoder(**code_inputs)
        doc_outputs = self.encoder(**doc_inputs)

        # Step 2: Mean-pool over all token representations (excluding padding tokens)
        code_emb = self.mean_pooling(code_outputs.last_hidden_state, code_inputs["attention_mask"])
        doc_emb = self.mean_pooling(doc_outputs.last_hidden_state, doc_inputs["attention_mask"])

        if self.variant == "variant_2":
            # Concatenate u, v, and |u - v|
            feat = torch.cat([code_emb, doc_emb, torch.abs(code_emb - doc_emb)], dim=1)
            logits = self.classifier(feat)
            return logits, code_emb, doc_emb
        else:
            # Variant 1: return representations directly for distance/similarity training
            return code_emb, doc_emb


def train_epoch(model, dataloader, optimizer, scheduler, loss_fn_v1, loss_fn_v2, device, variant):
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        code_inputs, doc_inputs, labels, _ = batch
        
        code_inputs = {k: v.to(device) for k, v in code_inputs.items()}
        doc_inputs = {k: v.to(device) for k, v in doc_inputs.items()}
        labels = labels.to(device)

        optimizer.zero_grad()

        if variant == "variant_2":
            logits, _, _ = model(code_inputs, doc_inputs)
            loss = loss_fn_v2(logits, labels)
        else:
            code_emb, doc_emb = model(code_inputs, doc_inputs)
            # CosineEmbeddingLoss target: 1 for aligned (label=0), -1 for drifted (label=1)
            targets = torch.where(labels == 0, 1.0, -1.0)
            loss = loss_fn_v1(code_emb, doc_emb, targets)

        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
    return total_loss / len(dataloader)


def evaluate(model, dataloader, device, variant, sweep_threshold_metric="balanced_accuracy", train_divs_val_tau=None):
    model.eval()
    all_labels = []
    all_preds = []
    all_divs = []
    all_metas = []

    with torch.no_grad():
        for batch in dataloader:
            code_inputs, doc_inputs, labels, metas = batch
            code_inputs = {k: v.to(device) for k, v in code_inputs.items()}
            doc_inputs = {k: v.to(device) for k, v in doc_inputs.items()}

            if variant == "variant_2":
                logits, code_emb, doc_emb = model(code_inputs, doc_inputs)
                preds = torch.argmax(logits, dim=1).cpu().tolist()
                # Compute cosine divergence as score output
                sim = F.cosine_similarity(code_emb, doc_emb, dim=1)
                divs = (1.0 - sim).cpu().tolist()
                
                all_preds.extend(preds)
                all_divs.extend(divs)
            else:
                code_emb, doc_emb = model(code_inputs, doc_inputs)
                sim = F.cosine_similarity(code_emb, doc_emb, dim=1)
                divs = (1.0 - sim).cpu().tolist()
                all_divs.extend(divs)

            all_labels.extend(labels.tolist())
            all_metas.extend(metas)

    # For Variant 1, classification depends on threshold sweep
    if variant == "variant_1":
        if train_divs_val_tau is not None:
            tau = train_divs_val_tau
        else:
            # Sweeping validation/train to find best similarity-based divergence threshold
            tau = sweep_threshold(all_labels, all_divs, metric=sweep_threshold_metric)
        all_preds = [1 if d >= tau else 0 for d in all_divs]
    else:
        tau = 0.5  # Standard decision boundary for classifier head probabilities

    # Convert binary 0/1 prediction to label string "aligned"/"drifted"
    pred_strings = ["drifted" if p == 1 else "aligned" for p in all_preds]
    label_strings = ["drifted" if l == 1 else "aligned" for l in all_labels]

    return label_strings, pred_strings, all_divs, tau, all_metas


def sweep_threshold(y_true: list[int], divergences: list[float], metric: str = "balanced_accuracy") -> float:
    y_binary = np.array(y_true)
    div_array = np.array(divergences)
    thresholds = np.linspace(0.0, 1.0, 401)
    best_tau = 0.5
    best_score = -1.0

    for tau in thresholds:
        preds = (div_array >= tau).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(y_binary, preds, average="binary", zero_division=0)
        acc = accuracy_score(y_binary, preds)

        if metric == "accuracy":
            score = acc
        elif metric == "balanced_accuracy":
            sensitivity = recall
            specificity = accuracy_score(y_binary[y_binary == 0], preds[y_binary == 0]) if np.sum(y_binary == 0) > 0 else 0
            score = (sensitivity + specificity) / 2.0
        elif metric == "macro_f1":
            _, _, macro_f1, _ = precision_recall_fscore_support(y_binary, preds, average="macro", zero_division=0)
            score = macro_f1
        else:
            score = f1

        if score > best_score:
            best_score = score
            best_tau = float(tau)
    return best_tau


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


def evaluate_breakdowns(y_true: list[str], y_pred: list[str], metas: list[dict]) -> dict:
    by_drift_type = defaultdict(list)
    by_severity = defaultdict(list)
    by_repo = defaultdict(list)

    for yt, yp, m in zip(y_true, y_pred, metas):
        pair = (yt, yp)
        by_drift_type[m["drift_type"]].append(pair)
        by_severity[m["severity"]].append(pair)
        by_repo[m["repo"]].append(pair)

    def calc_group_metrics(group_dict):
        result = {}
        for key, pairs in group_dict.items():
            yt_g, yp_g = zip(*pairs)
            result[key] = calculate_metrics(list(yt_g), list(yp_g))
        return result

    return {
        "by_drift_type": calc_group_metrics(by_drift_type),
        "by_severity": calc_group_metrics(by_severity),
        "by_repo": calc_group_metrics(by_repo),
    }


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Model B (Dual-Encoder ablation model)")
    parser.add_argument("--train", default="data/labeled/train.jsonl", help="Train dataset path")
    parser.add_argument("--val", default="data/labeled/val.jsonl", help="Validation dataset path")
    parser.add_argument("--test", default="data/labeled/test.jsonl", help="Test dataset path")
    parser.add_argument("--variant", choices=["variant_1", "variant_2"], default="variant_2", help="Model B Architecture variant")
    parser.add_argument("--model_name", default="microsoft/codebert-base", help="Hugging Face model checkpoint")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs to train")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="L2 weight decay")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio for linear LR schedule")
    parser.add_argument("--max_length", type=int, default=512, help="Max token sequence length")
    parser.add_argument("--no_clean_docstrings", dest="clean_docstrings", action="store_false", default=True, help="Disable extracting summary from docstrings (train/eval on full docstrings)")
    parser.add_argument("--freeze_base", action="store_true", default=False, help="Freeze encoder parameters")
    parser.add_argument("--device", default=DEFAULT_DEVICE, help="Execution device (cuda/cpu)")
    parser.add_argument("--output_dir", default="data/labeled", help="Output results directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--dry_run", action="store_true", default=False, help="Dry run on subset to verify shapes/gradients")
    args = parser.parse_args()

    set_seed(args.seed)

    print("======================================================================", flush=True)
    print("Fine-tuning Model B (Dual-Encoder Ablation Model)", flush=True)
    print("======================================================================", flush=True)
    print(f"Base Model Name  : {args.model_name}", flush=True)
    print(f"Model B Variant  : {args.variant.upper()}", flush=True)
    print(f"Device           : {args.device}", flush=True)
    print(f"Epochs           : {args.epochs}", flush=True)
    print(f"Batch Size       : {args.batch_size}", flush=True)
    print(f"Learning Rate    : {args.lr}", flush=True)
    print(f"Clean Docstrings : {args.clean_docstrings}", flush=True)
    print(f"Freeze Base      : {args.freeze_base}", flush=True)
    print(f"Dry Run Mode     : {args.dry_run}", flush=True)
    print("----------------------------------------------------------------------", flush=True)

    # 1. Load Datasets
    print("Loading datasets...", flush=True)
    train_dataset = SemDriftDataset(args.train, clean_docs=args.clean_docstrings)
    val_dataset = SemDriftDataset(args.val, clean_docs=args.clean_docstrings)
    test_dataset = SemDriftDataset(args.test, clean_docs=args.clean_docstrings)

    if args.dry_run:
        print(f"[Dry Run] Subsetting datasets from (Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}) to 16 samples each.", flush=True)
        train_dataset.records = train_dataset.records[:16]
        val_dataset.records = val_dataset.records[:16]
        test_dataset.records = test_dataset.records[:16]

    print(f"Loaded Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)} samples.", flush=True)

    # 2. Tokenizer & DataLoaders
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    collate_fn = make_collate_fn(tokenizer, args.max_length)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # 3. Initialize Model
    model = DualEncoderModel(args.model_name, variant=args.variant, freeze_base=args.freeze_base)
    model.to(args.device)

    # 4. Setup Optimization
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    # Loss Functions
    loss_fn_v1 = nn.CosineEmbeddingLoss(margin=0.0)
    loss_fn_v2 = nn.CrossEntropyLoss()

    # 5. Training Loop
    print("\nStarting training loop...", flush=True)
    best_val_score = -1.0
    best_checkpoint_path = os.path.join(args.output_dir, "model_b_checkpoint.pt")
    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(args.epochs):
        epoch_start = time.time()
        avg_loss = train_epoch(model, train_loader, optimizer, scheduler, loss_fn_v1, loss_fn_v2, args.device, args.variant)
        
        # Evaluate on validation set
        val_y_true, val_y_pred, val_divs, val_tau, _ = evaluate(model, val_loader, args.device, args.variant)
        val_metrics = calculate_metrics(val_y_true, val_y_pred)
        
        elapsed = time.time() - epoch_start
        print(f"Epoch {epoch+1}/{args.epochs} | Avg Loss: {avg_loss:.4f} | Val Acc: {val_metrics['accuracy']:.4f} | Val F1: {val_metrics['f1']:.4f} | Threshold (tau*): {val_tau:.4f} ({elapsed:.1f}s)", flush=True)

        # Track best model using Validation F1 Score
        val_score = val_metrics["f1"]
        if val_score > best_val_score:
            best_val_score = val_score
            torch.save(model.state_dict(), best_checkpoint_path)
            print(f" -> Saved new best model checkpoint to {best_checkpoint_path}", flush=True)

    # 6. Load Best Checkpoint and Run Final Test Set Evaluation
    print("\nLoading best model checkpoint for final test evaluation...", flush=True)
    model.load_state_dict(torch.load(best_checkpoint_path, map_location=args.device))
    
    # Validation step to compute the optimal threshold tau* using best weights
    val_y_true, val_y_pred, val_divs, best_tau, _ = evaluate(model, val_loader, args.device, args.variant)
    val_metrics = calculate_metrics(val_y_true, val_y_pred)
    print(f"Loaded validation metrics using best weights: F1={val_metrics['f1']:.4f} | Optimal Threshold (tau*): {best_tau:.4f}")

    # Evaluate on the Test set
    test_y_true, test_y_pred, test_divs, _, test_metas = evaluate(
        model, test_loader, args.device, args.variant, train_divs_val_tau=best_tau
    )
    test_overall = calculate_metrics(test_y_true, test_y_pred)
    breakdowns = evaluate_breakdowns(test_y_true, test_y_pred, test_metas)

    # Compute confusion matrix
    cm = confusion_matrix(
        [1 if l == "drifted" else 0 for l in test_y_true],
        [1 if p == "drifted" else 0 for p in test_y_pred]
    )

    print("\n======================================================================", flush=True)
    print("FINAL TEST RESULTS (MODEL B)", flush=True)
    print("======================================================================", flush=True)
    print(f"Test Accuracy            : {test_overall['accuracy']:.4f}", flush=True)
    print(f"Test Precision           : {test_overall['precision']:.4f}", flush=True)
    print(f"Test Recall              : {test_overall['recall']:.4f}", flush=True)
    print(f"Test F1 Score (Binary)   : {test_overall['f1']:.4f}", flush=True)
    print(f"Test Macro F1 Score      : {test_overall['macro_f1']:.4f}", flush=True)
    print("Confusion Matrix:", flush=True)
    print(f"  True Negatives (TN): {cm[0, 0]} | False Positives (FP): {cm[0, 1]}", flush=True)
    print(f"  False Negatives (FN): {cm[1, 0]} | True Positives (TP): {cm[1, 1]}", flush=True)

    print("\n--- Breakdown by Drift Type ---", flush=True)
    for dt, m in sorted(breakdowns["by_drift_type"].items()):
        print(f"  {dt:<22}: Acc={m['accuracy']:.4f} | F1={m['f1']:.4f} | Prec={m['precision']:.4f} | Rec={m['recall']:.4f} (N={m['count']})", flush=True)

    print("\n--- Breakdown by Severity ---", flush=True)
    for sev, m in sorted(breakdowns["by_severity"].items()):
        print(f"  {sev:<22}: Acc={m['accuracy']:.4f} | F1={m['f1']:.4f} | Prec={m['precision']:.4f} | Rec={m['recall']:.4f} (N={m['count']})", flush=True)

    print("\n--- Breakdown by Repository ---", flush=True)
    for repo, m in sorted(breakdowns["by_repo"].items()):
        print(f"  {repo:<22}: Acc={m['accuracy']:.4f} | F1={m['f1']:.4f} | Prec={m['precision']:.4f} | Rec={m['recall']:.4f} (N={m['count']})", flush=True)

    # 7. Write predictions and results
    pred_path = os.path.join(args.output_dir, "predictions_model_b.jsonl")
    results_path = os.path.join(args.output_dir, "results_model_b.json")

    with open(pred_path, "w", encoding="utf-8") as f:
        for rec, div, pred in zip(test_dataset.records, test_divs, test_y_pred):
            output_rec = dict(rec)
            output_rec["divergence_score"] = round(div, 6)
            output_rec["predicted_label"] = pred
            output_rec["optimal_threshold"] = best_tau
            f.write(json.dumps(output_rec) + "\n")

    full_results = {
        "model": "Model B (CodeBERT Dual-Encoder Ablation)",
        "model_name": args.model_name,
        "variant": args.variant,
        "clean_docstrings": args.clean_docstrings,
        "freeze_base": args.freeze_base,
        "optimal_threshold": best_tau,
        "val_metrics": val_metrics,
        "test_overall": test_overall,
        "confusion_matrix": {
            "tn": int(cm[0, 0]),
            "fp": int(cm[0, 1]),
            "fn": int(cm[1, 0]),
            "tp": int(cm[1, 1])
        },
        "breakdowns": breakdowns,
    }

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(full_results, f, indent=2)

    print("\n----------------------------------------------------------------------", flush=True)
    print(f"Saved predictions to : {pred_path}", flush=True)
    print(f"Saved test results to: {results_path}", flush=True)
    print("Model B Ablation execution complete!", flush=True)


if __name__ == "__main__":
    main()
