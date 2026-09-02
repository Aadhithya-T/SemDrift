# ==============================================================================
# SemDrift — Controlled Loss Ablation: Joint + Focal (No Category Weighting)
#
# Isolates Focal Loss vs CrossEntropy under the exact same Joint-Encoder setup:
#   - Backbone         : microsoft/codebert-base
#   - Random Seed      : 42
#   - Learning Rate    : 2e-5
#   - Epochs / Batch   : 3 / 8
#   - Dropout          : 0.1
#   - Code Truncation  : head_tail
#   - Pooling          : cls
#   - Checkpoint Metric: macro_f1
#   - Loss Objective   : FocalLoss (gamma=2.0, alpha=0.5)
#   - Category Weights : OFF (--no_category_weighting)
# ==============================================================================

param(
    [string]$Device = "cuda"
)

$ErrorActionPreference = "Stop"

$BaseDir = "data/experiments/v2"
$TrainPath = "$BaseDir/train.jsonl"
$ValPath = "$BaseDir/val.jsonl"
$TestPath = "$BaseDir/test.jsonl"
$OutputDir = "$BaseDir/joint_focal_controlled"

Write-Host "==============================================================================" -ForegroundColor Cyan
Write-Host "  Controlled Loss Ablation: Joint-Encoder + Focal Loss (Unweighted)          " -ForegroundColor Cyan
Write-Host "==============================================================================" -ForegroundColor Cyan
Write-Host "Architecture       : Joint-Encoder (CodeBERT Joint Self-Attention)"
Write-Host "Loss Function      : FocalLoss (alpha=0.5, gamma=2.0)"
Write-Host "Category Weighting : OFF"
Write-Host "Device             : $Device"
Write-Host "Random Seed        : 42"
Write-Host "Learning Rate      : 2e-5"
Write-Host "Epochs             : 3"
Write-Host "Batch Size         : 8"
Write-Host "Dropout            : 0.1"
Write-Host "Checkpoint Metric  : macro_f1"
Write-Host "Output Directory   : $OutputDir"
Write-Host "------------------------------------------------------------------------------"

# [1/2] Train Joint-Encoder with Focal Loss (Unweighted)
Write-Host "`n[1/2] Training Joint-Encoder + FocalLoss (CategoryWeighting=OFF)..." -ForegroundColor Yellow
python scripts/training/train_joint_encoder.py `
    --train $TrainPath `
    --val $ValPath `
    --test $TestPath `
    --device $Device `
    --epochs 3 `
    --batch_size 8 `
    --lr 2e-5 `
    --dropout 0.1 `
    --code_truncation head_tail `
    --pooling cls `
    --checkpoint_metric macro_f1 `
    --use_focal_loss `
    --no_category_weighting `
    --seed 42 `
    --output_dir $OutputDir

# [2/2] Compare Joint-CE vs. Joint-Focal
Write-Host "`n[2/2] Running Comparative Loss Ablation Analysis..." -ForegroundColor Yellow
python scripts/analysis/analyze_loss_ablation.py `
    --ce_preds "$BaseDir/joint_ce_controlled/predictions_joint_encoder.jsonl" `
    --focal_preds "$OutputDir/predictions_joint_encoder.jsonl" `
    --output_dir "$BaseDir/loss_ablation"

Write-Host "`n[SUCCESS] Loss ablation complete!" -ForegroundColor Green
