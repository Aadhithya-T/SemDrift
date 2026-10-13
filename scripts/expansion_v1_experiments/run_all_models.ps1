<#
.SYNOPSIS
    Master PowerShell runner to train, evaluate, and test ALL 6 models on the Expansion V1 dataset (10K).

.DESCRIPTION
    Runs:
      1. Zero-Shot CodeBERT (Cosine Divergence baseline)
      2. TF-IDF Combined (Pure Lexical baseline)
      3. TF-IDF Relational (Enhanced Lexical / Dual Overlap baseline)
      4. Dual-Encoder (Fine-tuned CodeBERT Bi-Encoder)
      5. Joint-Encoder (Fine-tuned Cross-Encoder + Focal Loss + Category Weighting)
      6. Joint-Encoder CE (Controlled Cross-Encoder + Standard CrossEntropy Ablation)

    Aggregates test metrics into benchmark_summary.md, benchmark_table.tex, and benchmark_summary.json.

.EXAMPLE
    .\scripts\expansion_v1_experiments\run_all_models.ps1 -Device cuda -Epochs 3 -BatchSize 8
    .\scripts\expansion_v1_experiments\run_all_models.ps1 -DryRun
    .\scripts\expansion_v1_experiments\run_all_models.ps1 -Models tfidf_combined,tfidf_relational
#>

param(
    [string[]]$Models = @("zero_shot", "tfidf_combined", "tfidf_relational", "dual_encoder", "joint_encoder", "joint_encoder_ce"),
    [string]$Device = "cuda",
    [int]$Epochs = 3,
    [int]$BatchSize = 8,
    [double]$Lr = 2e-5,
    [string]$OutputDir = "experiments/expansion_v1",
    [switch]$DryRun,
    [switch]$SkipExisting
)

$ErrorActionPreference = "Stop"

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "SemDrift Master Expansion V1 Benchmark Suite (10,000 samples)" -ForegroundColor Cyan
Write-Host "Models     : $($Models -join ', ')"
Write-Host "Device     : $Device"
Write-Host "Epochs     : $Epochs"
Write-Host "Batch Size : $BatchSize"
Write-Host "LR         : $Lr"
Write-Host "Output Dir : $OutputDir"
Write-Host "Dry Run    : $DryRun"
Write-Host "======================================================================" -ForegroundColor Cyan

$extraArgs = @()
if ($DryRun) {
    $extraArgs += "--dry_run"
}
if ($SkipExisting) {
    $extraArgs += "--skip_existing"
}

python scripts/expansion_v1_experiments/run_all_models.py `
    --models $Models `
    --device $Device `
    --epochs $Epochs `
    --batch_size $BatchSize `
    --lr $Lr `
    --output_dir $OutputDir `
    @extraArgs

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n[SUCCESS] Expansion V1 benchmark suite completed successfully!" -ForegroundColor Green
    Write-Host "Results available at: $OutputDir/benchmarks/benchmark_summary.md" -ForegroundColor Green
} else {
    Write-Host "`n[FAILURE] Benchmark execution failed with exit code $LASTEXITCODE" -ForegroundColor Red
    exit $LASTEXITCODE
}
