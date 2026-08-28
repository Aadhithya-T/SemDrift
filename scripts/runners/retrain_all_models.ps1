# ==============================================================================
# SemDrift — Individual Retraining & Benchmark Script (PowerShell)
# ==============================================================================

Write-Host "[1/4] Running Zero-Shot Baseline (Dual Encoder + Cosine Similarity)..." -ForegroundColor Cyan
python scripts/training/run_zero_shot_baseline.py `
    --val data/experiments/v2/val.jsonl `
    --test data/experiments/v2/test.jsonl `
    --output_dir data/experiments/v2/baseline_results `
    --device cuda

Write-Host "`n[2/4] Retraining Fine-Tuned Dual-Encoder (Ablation Model)..." -ForegroundColor Cyan
python scripts/training/train_dual_encoder.py `
    --train data/experiments/v2/train.jsonl `
    --val data/experiments/v2/val.jsonl `
    --test data/experiments/v2/test.jsonl `
    --device cuda `
    --epochs 3 `
    --batch_size 8 `
    --output_dir data/experiments/v2/dual_encoder_results/

Write-Host "`n[3/4] Retraining Fine-Tuned Joint-Encoder (Primary Contribution with Focal Loss & Category Weighting)..." -ForegroundColor Cyan
python scripts/training/train_joint_encoder.py `
    --train data/experiments/v2/train.jsonl `
    --val data/experiments/v2/val.jsonl `
    --test data/experiments/v2/test.jsonl `
    --device cuda `
    --epochs 3 `
    --batch_size 8 `
    --code_truncation head_tail `
    --pooling cls `
    --checkpoint_metric macro_f1 `
    --use_focal_loss `
    --category_weighting `
    --output_dir data/experiments/v2/joint_encoder_results/

Write-Host "`n[4/4] Generating Updated IEEE Paper Tables & McNemar Significance Tests..." -ForegroundColor Cyan
python scripts/analysis/generate_ieee_results.py `
    --v2_dir data/experiments/v2 `
    --output_dir data/experiments/v2

Write-Host "`nAll models successfully retrained and evaluated!" -ForegroundColor Green
