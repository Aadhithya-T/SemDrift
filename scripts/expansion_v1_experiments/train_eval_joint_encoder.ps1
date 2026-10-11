# PowerShell script to train, tune, and test CodeBERT Joint-Encoder (Focal) on Expansion V1 dataset (10K)
param(
    [string]$Device = "cuda",
    [int]$Epochs = 3,
    [int]$BatchSize = 8,
    [double]$Lr = 2e-5,
    [double]$FocalGamma = 2.0,
    [double]$FocalAlpha = 0.5,
    [string]$CheckpointMetric = "macro_f1",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "Fine-Tuning Joint-Encoder (Focal) on Expansion V1 (10K samples)" -ForegroundColor Cyan
Write-Host "Device: $Device | Epochs: $Epochs | BatchSize: $BatchSize | LR: $Lr | Gamma: $FocalGamma"
Write-Host "======================================================================" -ForegroundColor Cyan

$extraArgs = @()
if ($DryRun) {
    $extraArgs += "--dry_run"
}

python scripts/expansion_v1_experiments/train_eval_joint_encoder.py `
    --device $Device `
    --epochs $Epochs `
    --batch_size $BatchSize `
    --lr $Lr `
    --focal_gamma $FocalGamma `
    --focal_alpha $FocalAlpha `
    --checkpoint_metric $CheckpointMetric `
    @extraArgs

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n[SUCCESS] Joint-Encoder (Focal) training & evaluation complete!" -ForegroundColor Green
} else {
    Write-Host "`n[FAILURE] Execution failed with code $LASTEXITCODE" -ForegroundColor Red
    exit $LASTEXITCODE
}
