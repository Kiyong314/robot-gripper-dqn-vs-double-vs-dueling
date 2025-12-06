@echo off
REM =====================================================================
REM Off-Policy Learning 전체 실행 스크립트
REM 
REM 1. 4가지 DQN 모델 오프라인 학습
REM 2. 학습된 모델 평가 (시뮬레이션)
REM 3. 결과 비교 리포트 생성
REM =====================================================================

echo =====================================================================
echo OFF-POLICY LEARNING - FULL PIPELINE
echo =====================================================================
echo.

REM 현재 디렉토리 저장
set SCRIPT_DIR=%~dp0
cd /d "%SCRIPT_DIR%"

REM Python 경로 확인
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Python not found in PATH
    pause
    exit /b 1
)

echo [INFO] Script directory: %SCRIPT_DIR%
echo [INFO] Python: 
python --version
echo.

REM =====================================================================
REM STEP 1: 오프라인 학습
REM =====================================================================
echo =====================================================================
echo STEP 1: OFFLINE TRAINING (4 models)
echo =====================================================================
echo.
echo This will train all 4 DQN variants:
echo   - Standard DQN
echo   - Double DQN
echo   - Dueling DQN
echo   - Double + Dueling DQN
echo.
echo Estimated time: 1-2 hours
echo.

python train_offline.py --model all

if %errorlevel% neq 0 (
    echo [ERROR] Training failed!
    pause
    exit /b 1
)

echo.
echo [SUCCESS] Training completed!
echo.

REM =====================================================================
REM STEP 2: 모델 평가 (시뮬레이션 필요)
REM =====================================================================
echo =====================================================================
echo STEP 2: MODEL EVALUATION
echo =====================================================================
echo.
echo This will evaluate all checkpoints in CoppeliaSim.
echo Make sure CoppeliaSim is running with the IRB360 scene.
echo.
echo Trials per checkpoint: 10
echo Objects per trial: 10
echo.
echo Estimated time: 2-3 hours
echo.

set /p RUN_EVAL="Run evaluation? (y/n): "
if /i "%RUN_EVAL%"=="y" (
    python evaluate_models.py --trials 10 --objects 10
    
    if %errorlevel% neq 0 (
        echo [WARNING] Evaluation failed or was interrupted.
    ) else (
        echo.
        echo [SUCCESS] Evaluation completed!
    )
) else (
    echo [SKIP] Evaluation skipped.
    echo You can run it later with: python evaluate_models.py --trials 10 --objects 10
)

echo.

REM =====================================================================
REM STEP 3: 결과 비교
REM =====================================================================
echo =====================================================================
echo STEP 3: COMPARISON REPORT
echo =====================================================================
echo.

python compare_results.py

if %errorlevel% neq 0 (
    echo [WARNING] Comparison failed!
) else (
    echo.
    echo [SUCCESS] Comparison completed!
)

echo.
echo =====================================================================
echo PIPELINE COMPLETE
echo =====================================================================
echo.
echo Results are saved in:
echo   - results/      : Training logs and models
echo   - evaluation/   : Evaluation JSONs
echo   - evaluation/reports/ : Comparison graphs
echo.

pause


