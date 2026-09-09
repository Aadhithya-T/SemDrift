#!/usr/bin/env bash
# scripts/runners/evaluate_all_models_on_eval_set.sh
# Benchmark all 5 SemDrift models on V2 eval_set.jsonl

set -e

TEST_FILE="${1:-data/v2_real_world/evaluation/Evaluation_set2.jsonl}"
VAL_FILE="${2:-experiments/2026-09-07_clean_v2/dataset/val.jsonl}"
TRAIN_FILE="${3:-experiments/2026-09-07_clean_v2/dataset/train.jsonl}"
OUTPUT_DIR="${4:-experiments/2026-09-07_clean_v2/evaluation/evaluation_set2_benchmarks}"
DEVICE="${5:-cuda}"

echo "======================================================================"
echo "SemDrift V2 Eval Set Benchmark Orchestration"
echo "Test File  : $TEST_FILE"
echo "Device     : $DEVICE"
echo "Output Dir : $OUTPUT_DIR"
echo "======================================================================"

python scripts/runners/evaluate_all_models_on_eval_set.py \
    --test_file "$TEST_FILE" \
    --val_file "$VAL_FILE" \
    --train_file "$TRAIN_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --device "$DEVICE"

echo "Benchmark evaluation on eval_set.jsonl complete!"
