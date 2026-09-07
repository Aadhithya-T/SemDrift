#!/usr/bin/env python3
"""
independent_evaluate.py

Independent Evaluation Script for SemDrift Clean V2 Experiment:
- Enforces cryptographic dataset integrity gate (Layer A + Layer B) before execution
- Starts in a fresh Python process
- Instantiates fresh model architecture from base CodeBERT
- Loads checkpoint from disk: experiments/2026-09-07_clean_v2/checkpoints/joint_encoder_checkpoint.pt
- Evaluates on the locked 104-instance balanced diagnostic verified test set
- Computes comprehensive test metrics, confusion matrix, and dedicated PROVENANCE BREAKDOWN:
    authentic_historical_mined vs contract_grounded_generated vs clean_grounded
- Saves results to:
    experiments/2026-09-07_clean_v2/evaluation/eval_results.json
    experiments/2026-09-07_clean_v2/predictions/independent_predictions.jsonl
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from semdrift.models.joint_encoder import JointEncoderModel, make_collate_fn
from semdrift.data.integrity import verify_dataset_integrity
from scripts.training.train_joint_encoder import SemDriftDataset, calculate_metrics, evaluate_breakdowns


def parse_args():
    parser = argparse.ArgumentParser(description="Independent Clean-Slate V2 Evaluation")
    parser.add_argument("--checkpoint", default="experiments/2026-09-07_clean_v2/checkpoints/joint_encoder_checkpoint.pt")
    parser.add_argument("--test_file", default="experiments/2026-09-07_clean_v2/dataset/verified_test.jsonl")
    parser.add_argument("--output_results", default="experiments/2026-09-07_clean_v2/evaluation/eval_results.json")
    parser.add_argument("--output_preds", default="experiments/2026-09-07_clean_v2/predictions/independent_predictions.jsonl")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--pooling", default="mean")
    parser.add_argument("--code_truncation", default="head_tail")
    parser.add_argument("--doc_max_tokens", type=int, default=96)
    parser.add_argument("--max_length", type=int, default=512)
    return parser.parse_args()


def main():
    args = parse_args()
    
    ckpt_path = PROJECT_ROOT / args.checkpoint
    test_path = PROJECT_ROOT / args.test_file
    
    assert ckpt_path.is_file(), f"Checkpoint not found at: {ckpt_path}"
    assert test_path.is_file(), f"Test set not found at: {test_path}"
    
    print("=" * 70)
    print("SEMDRIFT INDEPENDENT EVALUATION (CLEAN-SLATE V2)")
    print(f"Checkpoint : {ckpt_path}")
    print(f"Test Set   : {test_path}")
    print(f"Device     : {args.device}")
    print("=" * 70)
    
    # 1. Authoritative Dataset Integrity Verification Gate
    print(f"Verifying cryptographic integrity gate for {test_path.parent}...", flush=True)
    verify_dataset_integrity(test_path.parent, required_files=[test_path.name])
    print("  -> Cryptographic integrity verified (Layer A + Layer B).", flush=True)
    
    # 2. Load Dataset
    test_dataset = SemDriftDataset(str(test_path), clean_docs=False)
    assert len(test_dataset) >= 100, f"Expected at least 100 test samples, got {len(test_dataset)}"
    print(f"[OK] Loaded {len(test_dataset)} human-verified test samples.")
    
    # 3. Tokenizer & DataLoader
    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    collate_fn = make_collate_fn(
        tokenizer,
        max_length=args.max_length,
        doc_max_tokens=args.doc_max_tokens,
        truncation_strategy=args.code_truncation
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn
    )
    
    # 4. Model Architecture & Checkpoint Loading
    print("Instantiating fresh JointEncoderModel...")
    model = JointEncoderModel(
        model_name="microsoft/codebert-base",
        pooling=args.pooling,
        num_labels=2,
        dropout=0.1
    )
    print(f"Loading checkpoint weights from {ckpt_path}...")
    state_dict = torch.load(ckpt_path, map_location=args.device)
    model.load_state_dict(state_dict)
    model.to(args.device)
    model.eval()
    print("[OK] Model restored to eval mode.")
    
    # 5. Evaluation Loop
    all_labels = []
    all_preds = []
    all_probs = []
    all_metas = []
    
    with torch.no_grad():
        for inputs, labels, metas in test_loader:
            inputs = {k: v.to(args.device) for k, v in inputs.items()}
            logits = model(inputs)
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            
            all_labels.extend(labels.tolist())
            all_preds.extend(preds.cpu().tolist())
            all_probs.extend(probs[:, 1].cpu().tolist())  # P(drifted)
            all_metas.extend(metas)
            
    label_strings = ["drifted" if l == 1 else "aligned" for l in all_labels]
    pred_strings = ["drifted" if p == 1 else "aligned" for p in all_preds]
    
    overall_metrics = calculate_metrics(label_strings, pred_strings)
    breakdowns = evaluate_breakdowns(label_strings, pred_strings, all_metas)
    
    print("\n" + "=" * 70)
    print("FINAL INDEPENDENT EVALUATION RESULTS — BALANCED DIAGNOSTIC TEST SET")
    print("=" * 70)
    print(f"Accuracy               : {overall_metrics['accuracy']:.4f}")
    print(f"Precision              : {overall_metrics['precision']:.4f}")
    print(f"Recall                 : {overall_metrics['recall']:.4f}")
    print(f"F1 Score (Binary)      : {overall_metrics['f1']:.4f}")
    print(f"Macro F1 Score         : {overall_metrics['macro_f1']:.4f}")
    print(f"Balanced Accuracy      : {overall_metrics['balanced_accuracy']:.4f}")
    print("Confusion Matrix:")
    print(f"  TN: {overall_metrics['tn']:<4} | FP: {overall_metrics['fp']}")
    print(f"  FN: {overall_metrics['fn']:<4} | TP: {overall_metrics['tp']}")
    
    # Dedicated Provenance Breakdown
    print("\n" + "=" * 70)
    print("DEDICATED PROVENANCE BREAKDOWN (HISTORICAL VS GENERATED)")
    print("=" * 70)
    prov_breakdown = breakdowns.get("by_provenance", {})
    print(f"{'Provenance':<30} | {'N':<5} | {'Recall':<8} | {'Precision':<10} | {'F1':<8} | {'BalAcc':<8}")
    print("-" * 75)
    for prov, m in sorted(prov_breakdown.items()):
        print(f"{prov:<30} | {m['count']:<5} | {m['recall']:<8.4f} | {m['precision']:<10.4f} | {m['f1']:<8.4f} | {m['balanced_accuracy']:<8.4f}")
    print("-" * 75)
    print(f"{'Overall':<30} | {overall_metrics['count']:<5} | {overall_metrics['recall']:<8.4f} | {overall_metrics['precision']:<10.4f} | {overall_metrics['f1']:<8.4f} | {overall_metrics['balanced_accuracy']:<8.4f}")
    
    # Breakdown by drift type
    print("\n--- Breakdown by Contract Drift Type ---")
    for dt, m in sorted(breakdowns.get("by_drift_type", {}).items()):
        print(f"  {dt:<28}: Acc={m['accuracy']:.4f} | F1={m['f1']:.4f} | Prec={m['precision']:.4f} | Rec={m['recall']:.4f} (N={m['count']})")
        
    # Breakdown by repo
    print("\n--- Breakdown by Repository ---")
    for repo, m in sorted(breakdowns.get("by_repo", {}).items()):
        print(f"  {repo:<20}: Acc={m['accuracy']:.4f} | F1={m['f1']:.4f} | Prec={m['precision']:.4f} | Rec={m['recall']:.4f} (N={m['count']})")
        
    # 6. Save independent predictions
    out_preds_path = PROJECT_ROOT / args.output_preds
    out_preds_path.parent.mkdir(parents=True, exist_ok=True)
    with out_preds_path.open("w", encoding="utf-8") as f:
        for rec, prob, pred in zip(test_dataset.records, all_probs, pred_strings):
            out = dict(rec)
            out["predicted_label"] = pred
            out["confidence"] = round(prob, 6)
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
    print(f"\n[OK] Saved independent predictions to {out_preds_path}")
    
    # 7. Save eval results JSON
    eval_record = {
        "evaluation_name": "Independent Clean-Slate V2 Balanced Diagnostic Evaluation",
        "checkpoint_evaluated": str(ckpt_path),
        "test_dataset": str(test_path),
        "test_samples": len(test_dataset),
        "device": args.device,
        "overall_metrics": overall_metrics,
        "confusion_matrix": {
            "tn": overall_metrics["tn"],
            "fp": overall_metrics["fp"],
            "fn": overall_metrics["fn"],
            "tp": overall_metrics["tp"],
        },
        "provenance_breakdown": prov_breakdown,
        "breakdowns": breakdowns,
    }
    
    out_results_path = PROJECT_ROOT / args.output_results
    out_results_path.parent.mkdir(parents=True, exist_ok=True)
    with out_results_path.open("w", encoding="utf-8") as f:
        json.dump(eval_record, f, indent=2)
    print(f"[OK] Saved independent evaluation summary to {out_results_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
