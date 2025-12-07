# 모델 평가 및 비교 시스템

저장된 체크포인트 모델들을 자동으로 평가하고 비교하는 시스템입니다.

## 📁 파일 구조

```
├── evaluate_checkpoints.py      # 체크포인트 평가 스크립트
├── plot_evaluation_results.py   # 결과 시각화 스크립트
├── logs/
│   ├── 2025-12-07_dqn/          # DQN 학습 로그
│   ├── 2025-12-07_double_dqn/   # Double DQN 학습 로그
│   └── 2025-12-07_dueling_dqn/  # Dueling DQN 학습 로그
│       ├── models/               # 체크포인트 모델들
│       │   ├── snapshot-000000.reinforcement.pth
│       │   ├── snapshot-000050.reinforcement.pth
│       │   └── ...
│       └── transitions/          # 학습 로그 파일들
└── evaluation/                   # 평가 로그 (자동 생성)
    ├── 2025-12-07_dqn/          # DQN 평가 결과
    │   ├── iter_000000/
    │   │   ├── transitions/
    │   │   │   ├── grasp-success.log.txt
    │   │   │   ├── predicted-value.log.txt
    │   │   │   ├── object-count.log.txt
    │   │   │   └── ...
    │   │   ├── visualizations/   # 시각화 이미지
    │   │   │   ├── 000000.grasp.png
    │   │   │   └── ...
    │   │   └── evaluation_summary.log
    │   ├── iter_000050/
    │   ├── ...
    │   └── evaluation_results.csv  # 전체 결과 CSV
    ├── 2025-12-07_double_dqn/   # Double DQN 평가 결과
    └── 2025-12-07_dueling_dqn/  # Dueling DQN 평가 결과
```

## 🚀 사용 방법

### 1단계: 단일 모델 평가

```bash
# 기본 사용 (30회 시도)
python evaluate_checkpoints.py --log_dir logs/2025-12-07_dqn

# 모델 이름 직접 지정
python evaluate_checkpoints.py \
  --log_dir logs/2025-12-07.17.45.26 \
  --model_name "dqn_model"

# 옵션 지정
python evaluate_checkpoints.py \
  --log_dir logs/2025-12-07_dqn \
  --model_name "standard_dqn" \
  --trials 50 \
  --num_obj 10 \
  --start_iter 0 \
  --end_iter 2000
```

**옵션 설명:**
- `--log_dir`: 평가할 모델이 있는 로그 디렉토리
- `--model_name`: 평가 폴더 이름 (기본: log_dir의 폴더명 자동 추출)
- `--trials`: 체크포인트당 평가 횟수 (기본: 30)
- `--num_obj`: 초기 물체 개수 (기본: 10)
- `--output`: 결과 CSV 파일명 (기본: evaluation_results.csv, evaluation/{model_name}/ 폴더에 저장)
- `--start_iter`: 시작 iteration (기본: 0)
- `--end_iter`: 종료 iteration (기본: 999999)

### 평가 결과 저장 위치

모든 평가 결과는 `evaluation/{model_name}/` 폴더에 저장됩니다:
- `evaluation/{model_name}/iter_000000/`: Iteration 0 평가 로그 및 시각화
- `evaluation/{model_name}/iter_000050/`: Iteration 50 평가 로그 및 시각화
- `evaluation/{model_name}/evaluation_results.csv`: 전체 결과 CSV

### 2단계: 결과 시각화

#### 단일 모델 시각화
```bash
python plot_evaluation_results.py \
  --csv evaluation/dqn/evaluation_results.csv \
  --labels "Standard DQN" \
  --output_dir plots_dqn
```

**생성되는 그래프:**
- `success_rate.png`: 성공률 변화
- `qvalue_stats.png`: Q-value 통계
- `outcome_distribution.png`: 결과 분포 (성공/실패/바닥선택)
- `failure_analysis.png`: 물체 개수별 실패 분석

#### 여러 모델 비교
```bash
python plot_evaluation_results.py \
  --csv evaluation/dqn/evaluation_results.csv \
        evaluation/double_dqn/evaluation_results.csv \
        evaluation/dueling_dqn/evaluation_results.csv \
  --labels "Standard DQN" "Double DQN" "Dueling DQN" \
  --output_dir comparison_plots
```

**생성되는 비교 그래프:**
- `success_rate_comparison.png`: 성공률 비교
- `qvalue_comparison.png`: Q-value 비교
- `final_performance.png`: 최종 성능 막대 그래프
- `learning_curve.png`: 초기 학습 곡선
- `statistics_table.png`: 통계 요약 테이블
- `comparison_statistics.csv`: 상세 통계 CSV

## 📊 출력 데이터

### evaluation_results.csv 컬럼

| 컬럼명 | 설명 |
|--------|------|
| `iteration` | 체크포인트 iteration |
| `success_rate` | 성공률 (%) |
| `total_grasps` | 총 시도 횟수 |
| `successful_grasps` | 성공 횟수 |
| `failed_grasps` | 실패 횟수 (일반) |
| `floor_selections` | 바닥 선택 횟수 |
| `avg_q_value` | 평균 Q-value |
| `max_q_value` | 최대 Q-value |
| `min_q_value` | 최소 Q-value |
| `std_q_value` | Q-value 표준편차 |
| `avg_objects_per_scene` | 씬당 평균 물체 개수 |
| `failure_with_many_objects` | 물체 많을 때 실패 (≥5개) |
| `failure_with_few_objects` | 물체 적을 때 실패 (≤2개) |
| `scene_restarts` | 씬 재시작 횟수 |

### evaluation_summary.log 예시

```
============================================================
EVALUATION SUMMARY
============================================================

Iteration: 1000
Total Grasps: 30
Success Rate: 73.33%
  - Successful: 22
  - Failed: 5
  - Floor Selection: 3

Q-value Statistics:
  - Average: 12.3456
  - Maximum: 45.6789
  - Minimum: -2.1234
  - Std Dev: 8.9012

Object Count Analysis:
  - Average objects per scene: 6.5
  - Failures with many objects (>=5): 2
  - Failures with few objects (<=2): 1

Scene Restarts: 3
```

## 🔬 분석 지표

평가 시스템은 다음 지표들을 자동으로 계산합니다:

### 1. 성능 지표
- **Success Rate**: 전체 성공률
- **Convergence Speed**: 80% 성공률 도달 iteration
- **Final Performance**: 마지막 체크포인트 성능
- **Stability**: 후반부 성공률의 표준편차

### 2. Q-value 분석
- **Average Q-value**: 평균 Q-value 추이
- **Q-value Range**: 최대/최소 범위
- **Q-value Stability**: 표준편차 변화

### 3. Failure Case 분석
- **물체 개수별 실패 패턴**: 적은 물체 vs 많은 물체
- **바닥 선택 빈도**: 잘못된 위치 선택 비율
- **씬 재시작 빈도**: 무한 루프 발생 횟수

## 🎯 PPT용 데이터 활용

생성된 데이터를 논문/PPT에 활용하는 방법:

### 1. 핵심 성능 비교
```python
import pandas as pd

# CSV 로드
dqn = pd.read_csv('evaluation_results_dqn.csv')
double = pd.read_csv('evaluation_results_double.csv')
dueling = pd.read_csv('evaluation_results_dueling.csv')

# 최종 성능
print(f"Standard DQN: {dqn.iloc[-1]['success_rate']:.2f}%")
print(f"Double DQN: {double.iloc[-1]['success_rate']:.2f}%")
print(f"Dueling DQN: {dueling.iloc[-1]['success_rate']:.2f}%")
```

### 2. 수렴 속도 분석
```python
def iterations_to_80_percent(df):
    idx = df[df['success_rate'] >= 80].index
    return df.iloc[idx[0]]['iteration'] if len(idx) > 0 else -1

print(f"DQN converged at: {iterations_to_80_percent(dqn)}")
print(f"Double DQN converged at: {iterations_to_80_percent(double)}")
print(f"Dueling DQN converged at: {iterations_to_80_percent(dueling)}")
```

### 3. 그래프 직접 생성
생성된 PNG 파일들을 PPT에 직접 삽입하거나, CSV 데이터로 커스텀 그래프를 만들 수 있습니다.

## 🐛 문제 해결

### 평가 중 에러 발생
- CoppeliaSim이 실행 중인지 확인
- IRB360 씬이 로드되어 있는지 확인
- ZMQ Remote API 서버가 활성화되어 있는지 확인

### 메모리 부족
- `--trials` 값을 줄이기 (예: 30 → 10)
- 체크포인트를 나눠서 평가 (`--start_iter`, `--end_iter` 사용)

### 시간이 너무 오래 걸림
- 체크포인트 간격을 늘리기 (예: 50 iteration마다 → 100 iteration마다)
- 병렬 실행 스크립트 작성 (추후 구현 가능)

## 📝 주의사항

1. **테스트 모드 실패 감지**: 같은 위치에서 2회 연속 실패 시 자동으로 씬 재시작
2. **물체 개수 로깅**: `is_testing=True`일 때만 물체 개수가 로그에 기록됨
3. **평가 로그 저장**: 각 평가마다 `evaluation_logs/` 폴더에 별도 저장
4. **Summary 파일**: 각 평가 폴더에 `evaluation_summary.log` 자동 생성

## 🔗 관련 파일

- `main_irb360.py`: 물체 개수 로깅 및 반복 실패 감지 추가
- `DQNTrainer.py`: `object_count_log` 로그 버퍼 추가
- `logger.py`: 기존 로그 시스템 (수정 불필요)

