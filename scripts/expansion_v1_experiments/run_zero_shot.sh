#!/usr/bin/env bash
# Shell script to run Zero-Shot CodeBERT baseline on Expansion V1 dataset (10K)
set -e

DEVICE="${1:-cuda}"
BATCH_SIZE="${2:-32}"
METRIC="${3:-macro_f1}"

echo "======================================================================"
echo "Running Zero-Shot CodeBERT on Expansion V1 Dataset (10K samples)"
echo "Device: $DEVICE | BatchSize: $BATCH_SIZE | Metric: $METRIC"
echo "======================================================================"

python scripts/expansion_v1_experiments/run_zero_shot.py \
    --device "$DEVICE" \
    --batch_size "$BATCH_SIZE" \
    --metric "$METRIC"

echo "[SUCCESS] Zero-Shot CodeBERT evaluation completed successfully!"
