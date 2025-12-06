@echo off
REM =============================================================================
REM 모든 DQN 변형 오프라인 학습 스크립트
REM 
REM 사용법: train_all_variants.bat [데이터폴더경로]
REM 예시: train_all_variants.bat logs\2025-12-02.22.21.35
REM =============================================================================

SET DATA_DIR=%1
IF "%DATA_DIR%"=="" SET DATA_DIR=logs\2025-12-02.22.21.35

echo.
echo ====================================================
echo    DQN Variants Offline Training
echo    Data: %DATA_DIR%
echo ====================================================
echo.

REM 1. Standard DQN
echo [1/4] Training Standard DQN...
python train_offline.py --data_dir %DATA_DIR% --epochs 30
echo.

REM 2. Double DQN
echo [2/4] Training Double DQN...
python train_offline.py --data_dir %DATA_DIR% --double_dqn --epochs 30
echo.

REM 3. Dueling DQN
echo [3/4] Training Dueling DQN...
python train_offline.py --data_dir %DATA_DIR% --dueling_dqn --epochs 30
echo.

REM 4. Double + Dueling DQN
echo [4/4] Training Double + Dueling DQN...
python train_offline.py --data_dir %DATA_DIR% --double_dqn --dueling_dqn --epochs 30
echo.

echo ====================================================
echo    All training completed!
echo    Check logs\ folder for results.
echo ====================================================

pause


