# PowerShell script to train, tune, and test CodeBERT Dual-Encoder on Expansion V1 dataset (10K)
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
Write-Host "Fine-Tuning Dual-Encoder on Expansion V1 (10K samples)" -ForegroundColor Cyan
Write-Host "Device: $Device | Epochs: $Epochs | BatchSize: $BatchSize | LR: $Lr | Metric: $CheckpointMetric"
Write-Host "======================================================================" -ForegroundColor Cyan

$extraArgs = @()
if ($DryRun) {
    $extraArgs += "--dry_run"
}

python scripts/expansion_v1_experiments/train_eval_dual_encoder.py `
    --device $Device `
    --epochs $Epochs `
    --batch_size $BatchSize `
    --lr $Lr `
    --checkpoint_metric $CheckpointMetric `
    @extraArgs

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n[SUCCESS] Dual-Encoder training & evaluation complete!" -ForegroundColor Green
} else {
    Write-Host "`n[FAILURE] Execution failed with code $LASTEXITCODE" -ForegroundColor Red
    exit $LASTEXITCODE
}
