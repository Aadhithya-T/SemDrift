@echo off
REM ==============================================================================
REM SemDrift — Individual Retraining & Benchmark Script (Batch)
REM ==============================================================================

echo [1/4] Running Zero-Shot Baseline (Dual Encoder + Cosine Similarity)...
python scripts/training/run_zero_shot_baseline.py ^
    --val data/v2_real_world/training/val.jsonl ^
    --test data/v2_real_world/evaluation/verified_test.jsonl ^
    --output_dir data/v2_real_world/baseline_results ^
    --device cuda

echo.
echo [2/4] Retraining Fine-Tuned Dual-Encoder (Ablation Model)...
python scripts/training/train_dual_encoder.py ^
    --train data/v2_real_world/training/train.jsonl ^
    --val data/v2_real_world/training/val.jsonl ^
    --test data/v2_real_world/evaluation/verified_test.jsonl ^
    --device cuda ^
    --epochs 3 ^
    --batch_size 8 ^
    --output_dir data/v2_real_world/dual_encoder_results/

echo.
echo [3/4] Retraining Fine-Tuned Joint-Encoder (Primary Contribution with Focal Loss & Category Weighting)...
python scripts/training/train_joint_encoder.py ^
    --train data/v2_real_world/training/train.jsonl ^
    --val data/v2_real_world/training/val.jsonl ^
    --test data/v2_real_world/evaluation/verified_test.jsonl ^
    --device cuda ^
    --epochs 3 ^
    --batch_size 8 ^
    --code_truncation head_tail ^
    --pooling cls ^
    --checkpoint_metric macro_f1 ^
    --use_focal_loss ^
    --category_weighting ^
    --output_dir data/v2_real_world/joint_encoder_results/

echo.
echo [4/4] Generating Updated IEEE Paper Tables & McNemar Significance Tests...
python scripts/analysis/generate_ieee_results.py ^
    --v2_dir data/v2_real_world ^
    --output_dir data/v2_real_world

echo.
echo All models successfully retrained and evaluated!
