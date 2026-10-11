#!/usr/bin/env bash
# Shell script to train, tune, and test CodeBERT Joint-Encoder (Focal) on Expansion V1 dataset (10K)
set -e

DEVICE="${1:-cuda}"
EPOCHS="${2:-3}"
BATCH_SIZE="${3:-8}"
LR="${4:-2e-5}"
FOCAL_GAMMA="${5:-2.0}"

echo "======================================================================"
echo "Fine-Tuning Joint-Encoder (Focal) on Expansion V1 (10K samples)"
echo "Device: $DEVICE | Epochs: $EPOCHS | BatchSize: $BATCH_SIZE | LR: $LR | Gamma: $FOCAL_GAMMA"
echo "======================================================================"

python scripts/expansion_v1_experiments/train_eval_joint_encoder.py \
    --device "$DEVICE" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --lr "$LR" \
    --focal_gamma "$FOCAL_GAMMA" \
    "${@:6}"

echo "[SUCCESS] Joint-Encoder (Focal) training & evaluation complete!"
