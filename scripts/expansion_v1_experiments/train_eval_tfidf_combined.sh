#!/usr/bin/env bash
# Shell script to train, tune, and test TF-IDF Combined baseline on Expansion V1 dataset (10K)
set -e

MAX_ITER="${1:-2000}"
C_VAL="${2:-1.0}"

echo "======================================================================"
echo "Training TF-IDF Combined (Pure Lexical) on Expansion V1 (10K samples)"
echo "MaxIter: $MAX_ITER | C: $C_VAL"
echo "======================================================================"

python scripts/expansion_v1_experiments/train_eval_tfidf_combined.py \
    --max_iter "$MAX_ITER" \
    --C "$C_VAL"

echo "[SUCCESS] TF-IDF Combined training & testing complete!"
