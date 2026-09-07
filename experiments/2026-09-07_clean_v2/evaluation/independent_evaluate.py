#!/usr/bin/env python3
"""
independent_evaluate.py

Independent Evaluation Script for SemDrift Clean V2 Experiment:
- Starts in a fresh Python process
- Instantiates fresh model architecture from base CodeBERT
- Loads checkpoint from disk: experiments/2026-09-07_clean_v2/checkpoints/joint_encoder_checkpoint.pt
- Evaluates on canonical verified test anchor: experiments/2026-09-07_clean_v2/dataset/verified_test.jsonl (N=101)
- Computes comprehensive test metrics, confusion matrix, and category breakdowns
- Cross-references against training run's logged results (if present) to verify reproducibility
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
from scripts.training.train_joint_encoder import SemDriftDataset, calculate_metrics, evaluate_breakdowns


def parse_args():
    parser = argparse.ArgumentParser(description="Independent Clean-Slate V2 Evaluation")
    parser.add_argument("--checkpoint", default="experiments/2026-09-07_clean_v2/checkpoints/joint_encoder_checkpoint.pt")
    parser.add_argument("--test_file", default="experiments/2026-09-07_clean_v2/dataset/verified_test.jsonl")
    parser.add_argument("--training_results", default="experiments/2026-09-07_clean_v2/checkpoints/results_joint_encoder.json")
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
    
    # 1. Load Dataset
    test_dataset = SemDriftDataset(str(test_path), clean_docs=False)
    assert len(test_dataset) == 101, f"Expected 101 test samples, got {len(test_dataset)}"
    print(f"[OK] Loaded {len(test_dataset)} human-verified test samples.")
    
    # 2. Tokenizer & DataLoader
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
    
    # 3. Model Architecture & Checkpoint Loading
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
    
    # 4. Evaluation Loop
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
    print("INDEPENDENT EVALUATION RESULTS")
    print("=" * 70)
    print(f"Accuracy          : {overall_metrics['accuracy']:.4f}")
    print(f"Precision         : {overall_metrics['precision']:.4f}")
    print(f"Recall            : {overall_metrics['recall']:.4f}")
    print(f"F1 Score (Binary) : {overall_metrics['f1']:.4f}")
    print(f"Macro F1 Score    : {overall_metrics['macro_f1']:.4f}")
    print(f"Balanced Accuracy : {overall_metrics['balanced_accuracy']:.4f}")
    print("Confusion Matrix:")
    print(f"  TN: {overall_metrics['tn']:<4} | FP: {overall_metrics['fp']}")
    print(f"  FN: {overall_metrics['fn']:<4} | TP: {overall_metrics['tp']}")
    
    # 5. Cross-reference against training run's logged metrics
    training_res_path = PROJECT_ROOT / args.training_results
    cross_verified = False
    if training_res_path.is_file():
        print(f"\nVerifying against training test results: {training_res_path}")
        with training_res_path.open("r", encoding="utf-8") as rf:
            train_results = json.load(rf)
        t_overall = train_results.get("test_overall", {})
        
        # Exact discrete match
        discrete_match = (
            overall_metrics["tn"] == t_overall.get("tn") and
            overall_metrics["fp"] == t_overall.get("fp") and
            overall_metrics["fn"] == t_overall.get("fn") and
            overall_metrics["tp"] == t_overall.get("tp") and
            overall_metrics["pred_aligned"] == t_overall.get("pred_aligned") and
            overall_metrics["pred_drifted"] == t_overall.get("pred_drifted")
        )
        assert discrete_match, f"Discrete mismatch with training results!\nEval: {overall_metrics}\nTrain: {t_overall}"
        
        # Numerical tolerance check on continuous scores (1e-5)
        for key in ["accuracy", "macro_f1", "balanced_accuracy", "precision", "recall"]:
            diff = abs(overall_metrics[key] - t_overall.get(key, 0.0))
            assert diff < 1e-4, f"Metric {key} differs by {diff} (tolerance 1e-4)"
            
        print("[OK] Exact match confirmed between independent evaluation and training log!")
        cross_verified = True
    else:
        print(f"[NOTE] Training results not yet found at {training_res_path}, skipping cross-verification.")
        
    # 6. Save independent predictions
    out_preds_path = PROJECT_ROOT / args.output_preds
    out_preds_path.parent.mkdir(parents=True, exist_ok=True)
    with out_preds_path.open("w", encoding="utf-8") as f:
        for rec, prob, pred in zip(test_dataset.records, all_probs, pred_strings):
            out = dict(rec)
            out["predicted_label"] = pred
            out["confidence"] = round(prob, 6)
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
    print(f"[OK] Saved independent predictions to {out_preds_path}")
    
    # 7. Save eval results JSON
    eval_record = {
        "evaluation_name": "Independent Clean-Slate V2 Verified Evaluation",
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
        "breakdowns": breakdowns,
        "cross_verified_with_training": cross_verified
    }
    
    out_results_path = PROJECT_ROOT / args.output_results
    out_results_path.parent.mkdir(parents=True, exist_ok=True)
    with out_results_path.open("w", encoding="utf-8") as f:
        json.dump(eval_record, f, indent=2)
    print(f"[OK] Saved independent evaluation summary to {out_results_path}")


if __name__ == "__main__":
    main()
