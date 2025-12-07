# Epsilon-greedy 로그 기능 구현 완료

## ✅ 구현 내용

### 1. 로그 버퍼 추가 (`DQNTrainer.py`)
```python
self.epsilon_log = []  # [epsilon, is_random]
```

### 2. 로그 복원 기능 (`DQNTrainer.py`)
```python
'epsilon.log.txt': ('epsilon_log', 2)  # [epsilon, is_random]
```

### 3. Epsilon 기록 (`main_irb360.py`)
```python
trainer.epsilon_log.append([current_epsilon, 1 if is_random_action else 0])
logger.write_to_log('epsilon', trainer.epsilon_log)
```

## 📊 로그 형식

### `epsilon.log.txt`
```
컬럼 1: epsilon 값 (0.0~1.0)
컬럼 2: is_random (1=Exploration, 0=Exploitation)

예시:
1.0000 1
0.9950 1
0.9900 0
0.9850 1
```

## 🔍 분석 방법

### 1. 간단한 확인
```python
import numpy as np

# 로그 로드
epsilon_log = np.loadtxt('logs/2025-12-06.22.22.16/transitions/epsilon.log.txt')
epsilon = epsilon_log[:, 0]
is_random = epsilon_log[:, 1]

# 통계
print(f"Epsilon 범위: [{epsilon.min():.4f}, {epsilon.max():.4f}]")
print(f"Random 비율: {is_random.mean()*100:.1f}%")
```

### 2. 고급 분석 스크립트
```bash
# 가장 최근 로그 자동 분석
python analyze_epsilon.py

# 특정 로그 분석
python analyze_epsilon.py logs/2025-12-06.22.22.16
```

**분석 결과**:
- Epsilon decay 그래프
- Exploration vs Exploitation 비율
- Success rate 추이
- Epsilon과 성공률 상관관계
- Random vs Greedy 성공률 비교

## 📈 활용 예시

### 성공률 추이 분석
```python
import numpy as np
import matplotlib.pyplot as plt

# 로그 로드
epsilon_log = np.loadtxt('logs/.../epsilon.log.txt')
success_log = np.loadtxt('logs/.../grasp-success.log.txt')

epsilon = epsilon_log[:, 0]
is_random = epsilon_log[:, 1]
success = (success_log > 0).astype(float)

# 이동 평균 (100 step window)
window = 100
epsilon_ma = np.convolve(epsilon, np.ones(window)/window, mode='valid')
success_ma = np.convolve(success, np.ones(window)/window, mode='valid')

# 그래프
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(epsilon_ma, label='Epsilon')
plt.plot(success_ma, label='Success Rate')
plt.xlabel('Iteration')
plt.ylabel('Value')
plt.legend()
plt.title('Epsilon vs Success Rate')

plt.subplot(1, 2, 2)
random_mask = is_random == 1
greedy_mask = is_random == 0
plt.bar(['Random', 'Greedy'], 
        [success[random_mask].mean(), success[greedy_mask].mean()])
plt.ylabel('Success Rate')
plt.title('Random vs Greedy Success Rate')

plt.tight_layout()
plt.show()
```

### Random vs Greedy 성공률 비교
```python
random_mask = is_random == 1
greedy_mask = is_random == 0

random_success = success[random_mask].mean()
greedy_success = success[greedy_mask].mean()

print(f"Random action success: {random_success*100:.2f}%")
print(f"Greedy action success: {greedy_success*100:.2f}%")
print(f"Difference: {(greedy_success - random_success)*100:.2f}%p")
```

## 🎯 기대 효과

1. **학습 진행 모니터링**
   - Epsilon이 올바르게 감소하는지 확인
   - Exploration/Exploitation 균형 확인

2. **성능 분석**
   - 어느 시점부터 Greedy가 Random보다 좋은지
   - Epsilon decay가 너무 빠르거나 느린지 판단

3. **하이퍼파라미터 튜닝**
   - `epsilon_start`, `epsilon_end`, `epsilon_decay_steps` 조정
   - Curriculum Learning 전략 개선

4. **논문 작성 자료**
   - Exploration-Exploitation 전략 시각화
   - 학습 안정성 증명

## 📝 주의사항

1. **기존 로그와 호환성**
   - 새 로그 파일이므로 기존 학습에는 없음
   - 이 업데이트 이후부터 기록됨

2. **Continue logging**
   - `--continue_logging` 사용 시 epsilon.log.txt도 복원됨
   - 이전 로그가 없으면 경고만 출력하고 계속 진행

3. **Testing mode**
   - `--is_testing` 모드에서는 epsilon=0 (항상 greedy)
   - is_random은 항상 0

## 🚀 다음 단계

학습을 시작하면 자동으로 `epsilon.log.txt`가 생성됩니다:

```bash
python main_irb360.py --is_sim --obj_mesh_dir objects/blocks --num_obj 10 --save_visualizations
```

학습 중 또는 학습 후에 분석:

```bash
python analyze_epsilon.py logs/2025-12-06.22.22.16
```

---

**구현 완료!** ✅

