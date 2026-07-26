@echo off
REM ==============================================================================
REM SemDrift — Individual Retraining & Benchmark Script (Batch)
REM ==============================================================================

echo [1/4] Running Model A (Zero-Shot Dual Encoder Baseline)...
python scripts/training/run_model_a.py ^
    --val data/experiments/v2/val.jsonl ^
    --test data/experiments/v2/test.jsonl ^
    --output_dir data/experiments/v2/model_a_results ^
    --device cuda

echo.
echo [2/4] Retraining Model B (Dual-Encoder Ablation)...
python scripts/training/train_dual_encoder.py ^
    --train data/experiments/v2/train.jsonl ^
    --val data/experiments/v2/val.jsonl ^
    --test data/experiments/v2/test.jsonl ^
    --device cuda ^
    --epochs 3 ^
    --batch_size 8 ^
    --output_dir data/experiments/v2/dual_encoder_results/

echo.
echo [3/4] Retraining Model B (Joint-Encoder Primary with Focal Loss & Category Weighting)...
python scripts/training/train_model_b.py ^
    --train data/experiments/v2/train.jsonl ^
    --val data/experiments/v2/val.jsonl ^
    --test data/experiments/v2/test.jsonl ^
    --device cuda ^
    --epochs 3 ^
    --batch_size 8 ^
    --code_truncation head_tail ^
    --pooling cls ^
    --checkpoint_metric macro_f1 ^
    --use_focal_loss ^
    --category_weighting ^
    --output_dir data/experiments/v2/joint_encoder_results/

echo.
echo [4/4] Generating Updated IEEE Paper Tables & McNemar Significance Tests...
python scripts/analysis/generate_ieee_results.py ^
    --v2_dir data/experiments/v2 ^
    --output_dir data/experiments/v2

echo.
echo All models successfully retrained and evaluated!
