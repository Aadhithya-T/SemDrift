# ==============================================================================
# SemDrift — Controlled Joint vs. Dual Architectural Experiment (PowerShell)
#
# Isolates architecture by holding all training hyperparameters, random seeds,
# splits, and the training objective strictly constant:
#
#   Dual-Encoder  + CrossEntropy (Standard unweighted loss)
#         VS
#   Joint-Encoder + CrossEntropy (Standard unweighted loss: focal=OFF, category_weighting=OFF)
#
# ==============================================================================

param(
    [string]$Device = "cuda",
    [int]$Epochs = 3,
    [int]$BatchSize = 8,
    [int]$Seed = 42,
    [double]$Lr = 2e-5,
    [double]$Dropout = 0.1,
    [string]$CheckpointMetric = "macro_f1"
)

$ErrorActionPreference = "Stop"

$BaseDir = "data/experiments/v2"
$TrainPath = "$BaseDir/train.jsonl"
$ValPath = "$BaseDir/val.jsonl"
$TestPath = "$BaseDir/test.jsonl"
$AblationDir = "$BaseDir/controlled_ablation"
$DualOut = "$AblationDir/dual_ce"
$JointOut = "$AblationDir/joint_ce"

Write-Host "==============================================================================" -ForegroundColor Cyan
Write-Host "  Controlled Architecture Experiment: Dual-Encoder vs Joint-Encoder (CE)      " -ForegroundColor Cyan
Write-Host "==============================================================================" -ForegroundColor Cyan
Write-Host "Backbone          : microsoft/codebert-base"
Write-Host "Device            : $Device"
Write-Host "Random Seed       : $Seed"
Write-Host "Epochs            : $Epochs"
Write-Host "Batch Size        : $BatchSize"
Write-Host "Learning Rate     : $Lr"
Write-Host "Dropout           : $Dropout"
Write-Host "Checkpoint Metric : $CheckpointMetric"
Write-Host "Dual Objective    : Standard CrossEntropyLoss"
Write-Host "Joint Objective   : Standard CrossEntropyLoss (Focal=OFF, CategoryWeighting=OFF)"
Write-Host "------------------------------------------------------------------------------"

# [1/3] Train Dual-Encoder with CrossEntropy
Write-Host "`n[1/3] Training Dual-Encoder + CrossEntropy..." -ForegroundColor Yellow
python scripts/training/train_dual_encoder.py `
    --train $TrainPath `
    --val $ValPath `
    --test $TestPath `
    --device $Device `
    --epochs $Epochs `
    --batch_size $BatchSize `
    --lr $Lr `
    --dropout $Dropout `
    --checkpoint_metric $CheckpointMetric `
    --seed $Seed `
    --output_dir $DualOut

# [2/3] Train Joint-Encoder with CrossEntropy (focal loss OFF, category weighting OFF)
Write-Host "`n[2/3] Training Joint-Encoder + CrossEntropy (Focal=OFF, CategoryWeighting=OFF)..." -ForegroundColor Yellow
python scripts/training/train_joint_encoder.py `
    --train $TrainPath `
    --val $ValPath `
    --test $TestPath `
    --device $Device `
    --epochs $Epochs `
    --batch_size $BatchSize `
    --lr $Lr `
    --dropout $Dropout `
    --code_truncation head_tail `
    --pooling cls `
    --checkpoint_metric $CheckpointMetric `
    --no_focal_loss `
    --no_category_weighting `
    --seed $Seed `
    --output_dir $JointOut

# [3/3] Analyze results and generate comparative report & LaTeX table
Write-Host "`n[3/3] Comparing Dual-CE vs. Joint-CE & Computing McNemar's Test..." -ForegroundColor Yellow
python scripts/analysis/analyze_controlled_experiment.py `
    --dual_preds "$DualOut/predictions_dual_encoder.jsonl" `
    --joint_preds "$JointOut/predictions_joint_encoder.jsonl" `
    --output_dir $AblationDir

Write-Host "`n[SUCCESS] Controlled architectural experiment complete!" -ForegroundColor Green
Write-Host "Summary report saved to : $AblationDir/controlled_experiment_summary.md" -ForegroundColor Green
Write-Host "LaTeX table saved to    : $AblationDir/controlled_experiment_table.tex" -ForegroundColor Green
Write-Host "JSON results saved to   : $AblationDir/controlled_experiment_results.json" -ForegroundColor Green
