# PowerShell script to train, tune, and test TF-IDF Combined baseline on Expansion V1 dataset (10K)
param(
    [int]$MaxIter = 2000,
    [double]$C = 1.0
)

$ErrorActionPreference = "Stop"

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "Training TF-IDF Combined (Pure Lexical) on Expansion V1 (10K samples)" -ForegroundColor Cyan
Write-Host "MaxIter: $MaxIter | C: $C"
Write-Host "======================================================================" -ForegroundColor Cyan

python scripts/expansion_v1_experiments/train_eval_tfidf_combined.py `
    --max_iter $MaxIter `
    --C $C

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n[SUCCESS] TF-IDF Combined training & testing complete!" -ForegroundColor Green
} else {
    Write-Host "`n[FAILURE] Execution failed with code $LASTEXITCODE" -ForegroundColor Red
    exit $LASTEXITCODE
}
