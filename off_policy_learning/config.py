#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Off-Policy Learning 설정 파일

학습 및 평가에 사용되는 모든 설정값을 관리합니다.
"""

import os

# =============================================================================
# 경로 설정
# =============================================================================

# 프로젝트 루트 경로
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Off-Policy Learning 폴더 경로
OFF_POLICY_ROOT = os.path.dirname(os.path.abspath(__file__))

# 학습 데이터 소스 폴더 (2개 폴더 결합)
DATA_SOURCES = [
    os.path.join(PROJECT_ROOT, 'logs', '2025-12-04.00.03.30_double_dueling'),  # 1,500 샘플, 83.7% 성공
    os.path.join(PROJECT_ROOT, 'logs', '2025-12-05.16.25.50'),                  # ~1,260 샘플, 41.6% 성공
]

# 결과 저장 경로
RESULTS_DIR = os.path.join(OFF_POLICY_ROOT, 'results')
EVALUATION_DIR = os.path.join(OFF_POLICY_ROOT, 'evaluation')

# =============================================================================
# 모델 설정
# =============================================================================

# 4가지 DQN 변형
MODEL_CONFIGS = {
    'standard_dqn': {
        'name': 'Standard DQN',
        'double_dqn': False,
        'dueling_dqn': False,
    },
    'double_dqn': {
        'name': 'Double DQN',
        'double_dqn': True,
        'dueling_dqn': False,
    },
    'dueling_dqn': {
        'name': 'Dueling DQN',
        'double_dqn': False,
        'dueling_dqn': True,
    },
    'double_dueling_dqn': {
        'name': 'Double + Dueling DQN',
        'double_dqn': True,
        'dueling_dqn': True,
    },
}

# =============================================================================
# 학습 설정
# =============================================================================

# 학습 하이퍼파라미터
TRAINING_CONFIG = {
    'epochs': 100,                    # 총 에폭 수
    'batch_size': 16,                 # 배치 크기
    'learning_rate': 1e-4,            # 학습률
    'save_interval': 50,              # 가중치 저장 간격 (에폭)
    'target_update_freq': 100,        # Target network 업데이트 주기 (Double DQN)
    'future_reward_discount': 0.2,    # 미래 보상 할인율 (γ)
}

# 샘플링 설정
SAMPLING_CONFIG = {
    'strategy': 'balanced',           # 'balanced' (1:1) 또는 'random'
    'success_ratio': 0.5,             # 균형 샘플링 시 성공 샘플 비율
    'filter_floor_selection': False,  # 바닥선택(-1) 필터링 여부 (False = 전체 포함)
}

# =============================================================================
# 평가 설정
# =============================================================================

EVALUATION_CONFIG = {
    'trials_per_checkpoint': 10,      # 체크포인트당 테스트 횟수
    'objects_per_trial': 10,          # 시뮬레이션당 물체 개수
    'evaluate_all_checkpoints': True, # 모든 체크포인트 평가 여부
}

# 시뮬레이션 설정 (IRB360)
SIMULATION_CONFIG = {
    'workspace_limits': [[-0.7, -0.3], [-0.2, 0.2], [0.001, 0.20]],
    'workspace_limit_place': [[-0.5, 0.5], [-0.6, -0.4], [0.001, 0.20]],
    'heightmap_resolution': 0.00125,
    'obj_mesh_dir': os.path.join(PROJECT_ROOT, 'objects', 'blocks'),
}

# =============================================================================
# 시각화 설정
# =============================================================================

VISUALIZATION_CONFIG = {
    'figsize': (12, 8),               # 그래프 크기
    'dpi': 150,                       # 해상도
    'colors': {
        'standard_dqn': '#1f77b4',    # 파란색
        'double_dqn': '#ff7f0e',      # 주황색
        'dueling_dqn': '#2ca02c',     # 초록색
        'double_dueling_dqn': '#d62728',  # 빨간색
    },
}


def get_model_result_dir(model_name):
    """모델별 결과 저장 경로 반환"""
    return os.path.join(RESULTS_DIR, model_name)


def get_model_eval_dir(model_name):
    """모델별 평가 결과 저장 경로 반환"""
    return os.path.join(EVALUATION_DIR, model_name)


def get_reports_dir():
    """리포트 저장 경로 반환"""
    return os.path.join(EVALUATION_DIR, 'reports')


if __name__ == '__main__':
    # 설정 확인
    print('=== Off-Policy Learning Configuration ===')
    print(f'Project Root: {PROJECT_ROOT}')
    print(f'Data Sources:')
    for src in DATA_SOURCES:
        exists = os.path.exists(src)
        print(f'  - {src} (exists: {exists})')
    print(f'Results Dir: {RESULTS_DIR}')
    print(f'Evaluation Dir: {EVALUATION_DIR}')
    print(f'\nModels: {list(MODEL_CONFIGS.keys())}')
    print(f'\nTraining Config: {TRAINING_CONFIG}')
    print(f'Sampling Config: {SAMPLING_CONFIG}')
    print(f'Evaluation Config: {EVALUATION_CONFIG}')


