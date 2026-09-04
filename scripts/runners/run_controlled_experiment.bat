@echo off
REM ==============================================================================
REM SemDrift — Controlled Joint vs. Dual Architectural Experiment (Batch)
REM ==============================================================================

set BASE_DIR=data\experiments\v2
set ABLATION_DIR=%BASE_DIR%\controlled_ablation
set DUAL_OUT=%ABLATION_DIR%\dual_ce
set JOINT_OUT=%ABLATION_DIR%\joint_ce

echo ==============================================================================
echo   Controlled Architecture Experiment: Dual-Encoder vs Joint-Encoder (CE)
echo ==============================================================================
echo Backbone          : microsoft/codebert-base
echo Random Seed       : 42
echo Epochs            : 3
echo Batch Size        : 8
echo Learning Rate     : 2e-5
echo Dropout           : 0.1
echo Checkpoint Metric : macro_f1
echo Dual Objective    : Standard CrossEntropyLoss
echo Joint Objective   : Standard CrossEntropyLoss (Focal=OFF, CategoryWeighting=OFF)
echo ------------------------------------------------------------------------------

echo.
echo [1/3] Training Dual-Encoder + CrossEntropy...
python scripts\training\train_dual_encoder.py ^
    --train %BASE_DIR%\train.jsonl ^
    --val %BASE_DIR%\val.jsonl ^
    --test %BASE_DIR%\test.jsonl ^
    --device cuda ^
    --epochs 3 ^
    --batch_size 8 ^
    --lr 2e-5 ^
    --dropout 0.1 ^
    --checkpoint_metric macro_f1 ^
    --seed 42 ^
    --output_dir %DUAL_OUT%

if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Dual-Encoder training failed.
    exit /b %ERRORLEVEL%
)

echo.
echo [2/3] Training Joint-Encoder + CrossEntropy (Focal=OFF, CategoryWeighting=OFF)...
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
    --no_focal_loss ^
    --no_category_weighting ^
    --seed 42 ^
    --output_dir %JOINT_OUT%

if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Joint-Encoder training failed.
    exit /b %ERRORLEVEL%
)

echo.
echo [3/3] Comparing Dual-CE vs. Joint-CE ^& Computing McNemar's Test...
python scripts\analysis\analyze_controlled_experiment.py ^
    --dual_preds %DUAL_OUT%\predictions_dual_encoder.jsonl ^
    --joint_preds %JOINT_OUT%\predictions_joint_encoder.jsonl ^
    --output_dir %ABLATION_DIR%

echo.
echo [SUCCESS] Controlled architectural experiment complete!
