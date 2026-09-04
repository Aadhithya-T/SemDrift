@echo off
REM ==============================================================================
REM SemDrift — Controlled Loss Ablation: Joint + Focal (No Category Weighting)
REM ==============================================================================

set BASE_DIR=data\experiments\v2
set OUT_DIR=%BASE_DIR%\joint_focal_controlled

echo ==============================================================================
echo   Controlled Loss Ablation: Joint-Encoder + Focal Loss (Unweighted)
echo ==============================================================================
echo Architecture       : Joint-Encoder (CodeBERT Joint Self-Attention)
echo Loss Function      : FocalLoss (alpha=0.5, gamma=2.0)
echo Category Weighting : OFF
echo Random Seed        : 42
echo Learning Rate      : 2e-5
echo Epochs             : 3
echo Batch Size         : 8
echo Dropout            : 0.1
echo Checkpoint Metric  : macro_f1
echo Output Directory   : %OUT_DIR%
echo ------------------------------------------------------------------------------

echo.
echo [1/2] Training Joint-Encoder + FocalLoss (CategoryWeighting=OFF)...
python scripts\training\train_joint_encoder.py ^
    --train %BASE_DIR%\train.jsonl ^
    --val %BASE_DIR%\val.jsonl ^
    --test %BASE_DIR%\test.jsonl ^
    --device cuda ^
    --epochs 3 ^
    --batch_size 8 ^
    --lr 2e-5 ^
    --dropout 0.1 ^
    --code_truncation head_tail ^
    --pooling cls ^
    --checkpoint_metric macro_f1 ^
    --use_focal_loss ^
    --no_category_weighting ^
    --seed 42 ^
    --output_dir %OUT_DIR%

if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Training failed.
    exit /b %ERRORLEVEL%
)

echo.
echo [2/2] Running Comparative Loss Ablation Analysis...
python scripts\analysis\analyze_loss_ablation.py ^
    --ce_preds %BASE_DIR%\joint_ce_controlled\predictions_joint_encoder.jsonl ^
    --focal_preds %OUT_DIR%\predictions_joint_encoder.jsonl ^
    --output_dir %BASE_DIR%\loss_ablation

echo.
echo [SUCCESS] Loss ablation complete!
