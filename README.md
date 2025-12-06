# Robot Gripper DQN vs Double DQN vs Dueling DQN

IRB360 델타 로봇과 진공 컵을 사용한 Pick-and-Place 작업을 위한 Deep Reinforcement Learning 구현.  
DQN, Double DQN, Dueling DQN 세 가지 알고리즘을 비교 분석합니다.

![Grasp Visualization](visualization.grasp.png)

## 📋 프로젝트 개요

이 프로젝트는 **CoppeliaSim 시뮬레이션** 환경에서 IRB360 델타 로봇이 물체를 집어 올리는 **Grasp** 작업을 강화학습으로 학습합니다.

### 주요 특징

- **3가지 DQN 알고리즘 비교**
  - **DQN**: 기본 Deep Q-Network
  - **Double DQN**: Q값 과대추정 방지
  - **Dueling DQN**: Value + Advantage 분리 구조

- **IRB360 델타 로봇 최적화**
  - 진공 컵 그리퍼 (회전 불필요 → 학습 속도 향상)
  - ZMQ Remote API로 CoppeliaSim 연동

- **Curriculum Learning 기반 탐색 전략**
  - 초기(0~500): 물체 위치에서만 탐색
  - 중반(500~1000): 80% 물체 + 20% 전체 영역
  - 후반(1000+): 전체 영역 탐색 (바닥 회피 학습)

## 🛠️ 설치 방법

### 1. 필수 요구사항

- Python 3.8+
- CoppeliaSim 4.4+ (시뮬레이션 환경)
- CUDA (선택, GPU 가속용)

### 2. 패키지 설치

```bash
# 저장소 클론
git clone https://github.com/Kiyong314/robot-gripper-dqn-vs-double-vs-dueling.git
cd robot-gripper-dqn-vs-double-vs-dueling

# 의존성 설치
pip install -r requirements.txt
```

### 3. PyTorch 설치 (CUDA 버전별)

```bash
# CPU only
pip install torch torchvision

# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## 🚀 실행 방법

### 1. CoppeliaSim 시뮬레이션 시작

CoppeliaSim을 실행하고 `simulation.ttt` 씬 파일을 로드합니다.

### 2. 학습 실행

```bash
# 기본 DQN 학습
python main_irb360.py --is_sim --save_visualizations

# Double DQN 학습
python main_irb360.py --is_sim --double_dqn --save_visualizations

# Dueling DQN 학습
python main_irb360.py --is_sim --dueling_dqn --save_visualizations

# Double + Dueling DQN 학습
python main_irb360.py --is_sim --double_dqn --dueling_dqn --save_visualizations
```

### 3. 주요 인자 설명

| 인자 | 기본값 | 설명 |
|------|--------|------|
| `--is_sim` | False | 시뮬레이션 모드 활성화 |
| `--double_dqn` | False | Double DQN 사용 |
| `--dueling_dqn` | False | Dueling DQN 사용 |
| `--num_obj` | 10 | 시뮬레이션 물체 개수 |
| `--save_visualizations` | False | 예측 시각화 저장 |
| `--experience_replay` | True | Experience Replay 사용 |
| `--target_update_freq` | 100 | Target network 업데이트 주기 |
| `--gripper_diameter` | 0.015 | 그리퍼 지름 (m) |

## 📁 프로젝트 구조

```
robot-gripper-dqn-vs-double-vs-dueling/
├── main_irb360.py          # 메인 학습 스크립트
├── DQNModels.py            # DQN/Dueling DQN 모델 정의
├── DQNTrainer.py           # 학습 로직 (Double DQN 포함)
├── network.py              # FeatureTrunk (DenseNet 기반)
├── utils.py                # 유틸리티 함수
├── logger.py               # 학습 로그 관리
├── requirements.txt        # 의존성 패키지
├── simulation.ttt          # CoppeliaSim 씬 파일
├── objects/                # 3D 물체 모델
│   └── blocks/             # 블록 객체 (.obj)
├── test/                   # 테스트 및 캘리브레이션
│   ├── robot_zmq_irb360.py # 로봇 ZMQ 통신 클래스
│   ├── test_camera.py      # 카메라 테스트
│   └── test_calibration.py # 캘리브레이션 테스트
└── logs/                   # 학습 로그 (자동 생성)
    └── YYYY-MM-DD.HH.MM.SS/
        ├── data/           # 이미지 데이터
        ├── models/         # 모델 스냅샷
        ├── transitions/    # 상태 전이 데이터
        └── visualizations/ # 예측 시각화
```

## 🔬 알고리즘 비교

### DQN (Deep Q-Network)
기본 Q-learning에 신경망을 적용한 방법.

### Double DQN
- **문제**: DQN은 Q값을 과대추정하는 경향
- **해결**: 행동 선택과 Q값 평가에 다른 네트워크 사용
- **수식**: `Q_target = r + γ * Q_target(s', argmax_a Q_main(s', a))`

### Dueling DQN
- **아이디어**: Q값을 Value(상태 가치)와 Advantage(행동 이점)로 분리
- **수식**: `Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))`
- **장점**: 상태 자체의 가치를 더 잘 학습

## 📊 학습 결과

학습 중 생성되는 시각화:
- `visualization.grasp.png`: 현재 예측 히트맵
- `visualization.best_grasp.png`: 선택된 최적 그리핑 위치

로그 폴더에서 학습 곡선 확인:
- `predicted-value.log`: 예측 Q값
- `label-value.log`: 실제 레이블 값
- `reward-value.log`: 보상 값
- `grasp-success.log`: 그리핑 성공 여부

## 🔧 캘리브레이션

카메라-로봇 캘리브레이션:
```bash
cd test
python compute_calibration.py
```

생성되는 파일:
- `camera_calibration.npy`: Homography 행렬
- `camera_calibration_inv.npy`: 역변환 행렬

## 📚 참고 문헌

### 논문
- [DQN Paper](https://www.nature.com/articles/nature14236) - Mnih et al., "Human-level control through deep reinforcement learning", Nature, 2015
- [Double DQN Paper](https://arxiv.org/abs/1509.06461) - Van Hasselt et al., "Deep Reinforcement Learning with Double Q-learning", AAAI, 2016
- [Dueling DQN Paper](https://arxiv.org/abs/1511.06581) - Wang et al., "Dueling Network Architectures for Deep Reinforcement Learning", ICML, 2016

### 참고 코드
- [Visual Pushing Grasping](https://github.com/andyzeng/visual-pushing-grasping) - Andy Zeng et al.
- [Learning Pick-to-Place Objects](https://github.com/Marwanon/Learning-Pick-to-Place-Objects-in-a-cluttered-scene-using-deep-reinforcement-learning) - Marwan Qaid Mohammed

---

## 🙏 Acknowledgments & Code Attribution

이 프로젝트는 다음 오픈소스 프로젝트들을 기반으로 개발되었습니다.

### 1. 기본 DQN 구현
- **Repository**: [Marwanon/Learning-Pick-to-Place-Objects](https://github.com/Marwanon/Learning-Pick-to-Place-Objects-in-a-cluttered-scene-using-deep-reinforcement-learning)
- **Author**: Marwan Qaid Mohammed
- **Citation**: 
  ```
  Marwan Qaid Mohammed, Lee Chung Kwek and Shing Chyi Chua, 
  "Learning Pick to Place Objects using Self-supervised Learning with Minimal Training Resources"
  International Journal of Advanced Computer Science and Applications (IJACSA), 12(10), 2021.
  ```
- **사용된 파일**: `network.py`, `logger.py`, `DQNModels.py` (일부), `DQNTrainer.py` (일부), `utils.py` (일부)

### 2. Visual Pushing Grasping
- **Repository**: [andyzeng/visual-pushing-grasping](https://github.com/andyzeng/visual-pushing-grasping)
- **Author**: Andy Zeng et al.

---

## ✨ 본 프로젝트의 독자적 기여 (My Original Contributions)

| 기능 | 파일 | 설명 |
|------|------|------|
| **Double DQN** | `DQNTrainer.py` | Target Network 분리, Q값 과대추정 방지 |
| **Dueling DQN** | `DQNModels.py` | Value + Advantage 스트림 분리 아키텍처 |
| **IRB360 델타 로봇 통합** | `robot_zmq_irb360.py` | CoppeliaSim ZMQ API 기반 로봇 제어 (완전 새로 작성) |
| **진공 컵 제어** | `robot_zmq_irb360.py` | 흡착/해제 시그널, 센서 기반 하강 |
| **Curriculum Learning** | `main_irb360.py` | 단계적 탐색 영역 확장 전략 |
| **바닥 감지** | `main_irb360.py` | 그리퍼 영역 내 바닥 선택 시 즉시 실패 및 음수 보상 |
| **카메라 캘리브레이션** | `utils.py`, `test/` | Homography 기반 좌표 변환 |
| **Heightmap 검증** | `utils.py` | 물체 위치와 heightmap 정확도 검증 |
| **Orthographic 카메라** | `utils.py` | Ortho 카메라용 heightmap 생성 |
| **동일 이미지 감지** | `main_irb360.py` | 물체가 로봇에 붙어있는 경우 감지 |

---

## 📝 License

This project is open source and available under the MIT License.

## 👤 Author

- GitHub: [@Kiyong314](https://github.com/Kiyong314)

