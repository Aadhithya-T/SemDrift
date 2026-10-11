#!/usr/bin/env bash
# Master Bash runner to train, evaluate, and test ALL 6 models on the Expansion V1 dataset (10K).
set -e

DEVICE="${1:-cuda}"
EPOCHS="${2:-3}"
BATCH_SIZE="${3:-8}"
LR="${4:-2e-5}"
OUTPUT_DIR="${5:-experiments/expansion_v1}"

echo "======================================================================"
echo "SemDrift Master Expansion V1 Benchmark Suite (10,000 samples)"
echo "Device: $DEVICE | Epochs: $EPOCHS | BatchSize: $BATCH_SIZE | LR: $LR"
echo "Output: $OUTPUT_DIR"
echo "======================================================================"

python scripts/expansion_v1_experiments/run_all_models.py \
    --device "$DEVICE" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --lr "$LR" \
    --output_dir "$OUTPUT_DIR" \
    "${@:6}"

echo "[SUCCESS] Expansion V1 benchmark suite completed successfully!"
