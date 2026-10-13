# PowerShell script to train, tune, and test CodeBERT Joint-Encoder (CE Ablation) on Expansion V1 dataset (10K)
param(
    [string]$Device = "cuda",
    [int]$Epochs = 3,
    [int]$BatchSize = 8,
    [double]$Lr = 2e-5,
    [string]$CheckpointMetric = "macro_f1",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "Fine-Tuning Joint-Encoder (Standard CrossEntropy) on Expansion V1 (10K samples)" -ForegroundColor Cyan
Write-Host "Device: $Device | Epochs: $Epochs | BatchSize: $BatchSize | LR: $Lr"
Write-Host "Objective: Standard CrossEntropyLoss (Focal=OFF, CategoryWeighting=OFF)"
Write-Host "======================================================================" -ForegroundColor Cyan

$extraArgs = @()
if ($DryRun) {
    $extraArgs += "--dry_run"
}

python scripts/expansion_v1_experiments/train_eval_joint_encoder_ce.py `
    --device $Device `
    --epochs $Epochs `
    --batch_size $BatchSize `
    --lr $Lr `
    --checkpoint_metric $CheckpointMetric `
    @extraArgs

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n[SUCCESS] Joint-Encoder (CE) training & evaluation complete!" -ForegroundColor Green
} else {
    Write-Host "`n[FAILURE] Execution failed with code $LASTEXITCODE" -ForegroundColor Red
    exit $LASTEXITCODE
}
