#!/usr/bin/env bash
# Shell script to train, tune, and test CodeBERT Joint-Encoder (CE Ablation) on Expansion V1 dataset (10K)
set -e

DEVICE="${1:-cuda}"
EPOCHS="${2:-3}"
BATCH_SIZE="${3:-8}"
LR="${4:-2e-5}"
METRIC="${5:-macro_f1}"

echo "======================================================================"
echo "Fine-Tuning Joint-Encoder (Standard CrossEntropy) on Expansion V1 (10K samples)"
echo "Device: $DEVICE | Epochs: $EPOCHS | BatchSize: $BATCH_SIZE | LR: $LR"
echo "Objective: Standard CrossEntropyLoss (Focal=OFF, CategoryWeighting=OFF)"
echo "======================================================================"

python scripts/expansion_v1_experiments/train_eval_joint_encoder_ce.py \
    --device "$DEVICE" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --lr "$LR" \
    --checkpoint_metric "$METRIC" \
    "${@:6}"

echo "[SUCCESS] Joint-Encoder (CE) training & evaluation complete!"
