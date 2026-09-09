<#
.SYNOPSIS
    Evaluates all 5 SemDrift models on the V2 eval_set.jsonl.

.DESCRIPTION
    Runs Zero-Shot CodeBERT, TF-IDF Combined, TF-IDF Relational, Dual-Encoder,
    and Joint-Encoder against the 2,200-sample V2 evaluation dataset.

.EXAMPLE
    .\scripts\runners\evaluate_all_models_on_eval_set.ps1
    .\scripts\runners\evaluate_all_models_on_eval_set.ps1 -Device cpu -BatchSize 8
    .\scripts\runners\evaluate_all_models_on_eval_set.ps1 -Models tfidf_combined,tfidf_relational
#>

[CmdletBinding()]
param (
    [string]$TestFile = "data/v2_real_world/evaluation/Evaluation_set2.jsonl",
    [string]$ValFile = "experiments/2026-09-07_clean_v2/dataset/val.jsonl",
    [string]$TrainFile = "experiments/2026-09-07_clean_v2/dataset/train.jsonl",
    [string]$OutputDir = "experiments/2026-09-07_clean_v2/evaluation/evaluation_set2_benchmarks",
    [string[]]$Models = @("zero_shot", "tfidf_combined", "tfidf_relational", "dual_encoder", "joint_encoder"),
    [string]$Device = "cuda",
    [int]$BatchSize = 16,
    [int]$MaxIter = 5000
)

$ErrorActionPreference = "Stop"

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "SemDrift V2 Eval Set Benchmark Orchestration" -ForegroundColor Cyan
Write-Host "Test File  : $TestFile"
Write-Host "Models     : $($Models -join ', ')"
Write-Host "Device     : $Device"
Write-Host "Output Dir : $OutputDir"
Write-Host "======================================================================" -ForegroundColor Cyan

$modelArgs = $Models -join " "

python scripts/runners/evaluate_all_models_on_eval_set.py `
    --test_file $TestFile `
    --val_file $ValFile `
    --train_file $TrainFile `
    --output_dir $OutputDir `
    --models $Models `
    --device $Device `
    --batch_size $BatchSize `
    --max_iter $MaxIter

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n[SUCCESS] Benchmark evaluation on eval_set.jsonl complete!" -ForegroundColor Green
    Write-Host "Results saved in: $OutputDir" -ForegroundColor Green
} else {
    Write-Host "`n[FAILURE] Benchmark evaluation failed with exit code $LASTEXITCODE." -ForegroundColor Red
    exit $LASTEXITCODE
}
