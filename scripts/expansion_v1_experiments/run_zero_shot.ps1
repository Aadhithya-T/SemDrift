# PowerShell script to run Zero-Shot CodeBERT baseline on Expansion V1 dataset (10K)
param(
    [string]$Device = "cuda",
    [int]$BatchSize = 32,
    [string]$Metric = "macro_f1"
)

$ErrorActionPreference = "Stop"

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "Running Zero-Shot CodeBERT on Expansion V1 Dataset (10K samples)" -ForegroundColor Cyan
Write-Host "Device: $Device | BatchSize: $BatchSize | Metric: $Metric"
Write-Host "======================================================================" -ForegroundColor Cyan

python scripts/expansion_v1_experiments/run_zero_shot.py `
    --device $Device `
    --batch_size $BatchSize `
    --metric $Metric

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n[SUCCESS] Zero-Shot CodeBERT evaluation completed successfully!" -ForegroundColor Green
} else {
    Write-Host "`n[FAILURE] Execution failed with code $LASTEXITCODE" -ForegroundColor Red
    exit $LASTEXITCODE
}
