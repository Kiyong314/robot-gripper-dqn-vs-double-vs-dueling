# IRB360 델타 로봇 Pick & Place 프로젝트

이 폴더는 IRB360 델타 로봇 + BaxterVacuumCup 진공 컵을 Python으로 제어하여 Pick & Place 작업을 수행하는 프로젝트입니다.

## 파일 구조

```
test/
├── robot_zmq_irb360.py         # IRB360 로봇 제어 클래스 (핵심)
├── test_pick_place.py          # Pick & Place 통합 테스트
├── test_irb360.py              # 기본 기능 테스트 (연결/이동/진공/Grasp)
├── test_descent_and_grasp.py  # 하강 및 흡착 상세 테스트
├── irb360_python_wrapper.lua   # IRB360 IK 제어 Lua 스크립트 (필수)
├── utils.py                    # 유틸리티 (heightmap 생성 등)
├── DQNModels.py                # DQN 모델 (향후 학습용)
├── DQNTrainer.py               # DQN 트레이너 (향후 학습용)
├── network.py                  # 신경망 (DQN용)
├── logger.py                   # 로깅 (DQN용)
├── main_irb360.py              # DQN 학습 메인 스크립트
└── README.md                   # 이 문서
```

## 요구사항

### 소프트웨어
- **CoppeliaSim 4.10 이상** (IRB360 씬 포함)
- **Python 3.7 이상**
- **PyTorch** (DQN 학습용, 선택)

### Python 패키지
```bash
pip install coppeliasim-zmqremoteapi-client numpy opencv-python torch torchvision
```

## CoppeliaSim 씬 설정

### 씬 구성
```
IRB360 베이스: Z = 1.100m
Plane (Pick 작업대): (-0.500, 0.000, 0.001m)
Plane1 (Place 작업대): (0.075, -0.500, 0.001m)
Vision_sensor_ortho: (-0.500, 0.000, 0.180m)
```

### 시작 전 체크리스트
1. ✅ CoppeliaSim 실행
2. ✅ IRB360 씬 로드 (IRB360 + BaxterVacuumCup 포함)
3. ✅ ZMQ Remote API 서버 활성화: `Add-ons → ZMQ remote API server`
4. ✅ `irb360_python_wrapper.lua` 스크립트가 IRB360 오브젝트에 적용되어 있는지 확인

## 사용법

### 1. 기본 기능 테스트 (test_irb360.py)

로봇의 기본 기능을 개별적으로 테스트합니다.

```bash
cd test

# 모든 테스트 실행
python test_irb360.py

# 특정 테스트만 실행
python test_irb360.py --test connection  # 연결 테스트
python test_irb360.py --test move        # 이동 테스트
python test_irb360.py --test vacuum      # 진공 테스트
python test_irb360.py --test grasp       # Grasp/Place 테스트
```

**출력 예시:**
```
======================================================================
테스트 1: CoppeliaSim 연결 테스트
======================================================================
[TEST] Connecting to CoppeliaSim...
[TEST] Connected successfully!
  [OK] irb360: 123
  [OK] ikTarget: 456
  [OK] BaxterVacuumCup: 789
```

### 2. 하강 및 흡착 테스트 (test_descent_and_grasp.py)

센서 기반 하강과 진공 흡착을 상세하게 테스트합니다.

```bash
python test_descent_and_grasp.py
```

**기능:**
- 물체 1개 생성
- 물체 위치로 이동
- 센서 기반 정밀 하강
- 진공 흡착 테스트
- 물체 들어올리기

### 3. Pick & Place 통합 테스트 (test_pick_place.py)

Heightmap 기반 물체 감지 및 Pick & Place를 수행합니다.

```bash
python test_pick_place.py
```

**테스트 시나리오:**
1. **Heightmap 생성 테스트**: 카메라 데이터로 heightmap 생성
2. **물체 등록 테스트**: workspace 내 물체 배치
3. **Pick & Place 테스트**: 물체 감지 → Pick → Place

**출력 예시:**
```
======================================================================
테스트 3: Pick & Place 테스트
======================================================================
[ATTEMPT 1/3]
[STEP 1] 카메라 데이터 가져오기...
[STEP 2] Heightmap 생성...
[STEP 3] 물체 위치 감지...
[INFO] Pick 영역 내 물체 발견: (-0.450, 0.050, 0.035)
[STEP 4] Pick 시도...
[GRASP] Executing at (-0.450, 0.050, 0.055)
[GRASP] Phase 1: Fast descent
[GRASP] Sensor ON! Z=0.0650m, Distance=12.3mm
[GRASP] Phase 2: Precision descent
[GRASP] Object contact! Stopping at Z=0.0580m
[SUCCESS] Pick & Place 성공! (1/1)
```

## Python 코드에서 사용

### 기본 사용법

```python
from robot_zmq_irb360 import RobotIRB360ZMQ

# 로봇 초기화 (물체 없이)
robot = RobotIRB360ZMQ(
    is_sim=True,
    obj_mesh_dir=None,
    num_obj=0
)

# 홈 위치로 이동
robot.move_to(robot.home_position, 0)

# 특정 위치로 이동 (월드 좌표)
robot.move_to([0.1, 0.2, 0.3], rotation_angle=0)

# 진공 제어
robot.activate_vacuum()           # 흡착 활성화
attached = robot.check_vacuum_attached()  # 흡착 확인
robot.deactivate_vacuum()         # 흡착 해제

# 현재 VacuumCup 위치 확인
pos = robot.get_current_position()
print(f"현재 위치: {pos}")
```

### 물체와 함께 사용

```python
import os

# 물체 메쉬 디렉토리 경로
OBJECTS_DIR = os.path.join('..', 'objects', 'blocks')

# 로봇 초기화 (물체 3개)
robot = RobotIRB360ZMQ(
    is_sim=True,
    obj_mesh_dir=OBJECTS_DIR,
    num_obj=3,
    place=True
)

# 물체 추가 (workspace_limits 내에 랜덤 배치)
robot.add_objects()

# 물체 위치 확인
obj_positions = robot.get_obj_positions()
for i, pos in enumerate(obj_positions):
    print(f"물체 {i}: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")

# Grasp 실행 (센서 기반 하강 + 진공 흡착)
pick_position = [obj_positions[0][0], obj_positions[0][1], obj_positions[0][2] + 0.02]
result = robot.grasp(
    pick_position,
    rotation_angle=0,
    workspace_limits=robot.workspace_limits,
    trainer=None,
    workspace_limit_place=robot.workspace_limit_place
)

# 결과 확인
if result == 1:
    print("성공!")
elif result == 0.5:
    print("물체가 흔들려서 떨어짐")
else:
    print("실패 (센서 감지 실패 또는 흡착 실패)")  0 or -1
```

### 카메라 및 Heightmap 사용

```python
import utils
import cv2
import numpy as np

# 카메라 데이터 가져오기
color_img, depth_img = robot.get_camera_data()

# Heightmap 생성
heightmap_resolution = 0.002  # 2mm per pixel
color_heightmap, depth_heightmap = utils.get_heightmap(
    color_img,
    depth_img,
    robot.cam_intrinsics,
    robot.cam_pose,
    robot.workspace_limits,
    heightmap_resolution
)

# Heightmap 저장
cv2.imwrite('heightmap_color.png', cv2.cvtColor(color_heightmap, cv2.COLOR_RGB2BGR))

# 가장 높은 점 찾기 (물체 위치)
valid_depth = depth_heightmap.copy()
valid_depth[np.isnan(valid_depth)] = 0
max_idx = np.unravel_index(np.argmax(valid_depth), valid_depth.shape)
max_height = valid_depth[max_idx]
print(f"최대 높이: {max_height:.3f}m")
```

## 주요 좌표 및 설정

### 씬 좌표 (월드 좌표)

| 위치 | 월드 좌표 (x, y, z) | 설명 |
|------|---------------------|------|
| **IRB360 베이스** | (0.0, 0.0, 1.100) | 로봇 베이스 |
| **Home** | (0.0, 0.0, 0.600) | 홈 포지션 (대기) |
| **Plane (Pick)** | (-0.500, 0.0, 0.001) | Pick 작업대 |
| **Plane1 (Place)** | (0.075, -0.500, 0.001) | Place 작업대 |
| **Place Position** | (0.075, 0.0, 0.200) | 물체 배치 위치 |
| **Camera** | (-0.500, 0.0, 0.180) | 비전 센서 |

### Workspace 설정

#### Pick Workspace (Plane 영역)
```python
workspace_limits = [
    [-0.7, -0.3],   # x: -0.7m ~ -0.3m
    [-0.2, 0.2],    # y: -0.2m ~ 0.2m
    [0.001, 0.20]   # z: 0.001m ~ 0.2m
]
```

#### Place Workspace (Plane1 영역)
```python
workspace_limit_place = [
    [-0.5, 0.5],    # x: -0.5m ~ 0.5m
    [-0.6, -0.4],   # y: -0.6m ~ -0.4m
    [0.001, 0.20]   # z: 0.001m ~ 0.2m
]
```

## 좌표 시스템

### 월드 좌표 vs 로봇 좌표

- **월드 좌표**: CoppeliaSim 전체 씬 기준 절대 좌표
- **로봇 좌표**: IRB360 베이스 기준 상대 좌표

```python
# 좌표 변환 (내부적으로 처리됨)
robot_pos = robot.world_to_robot_coord(world_pos)
world_pos = robot.robot_to_world_coord(robot_pos)
```

### ikTarget vs VacuumCup

```
ikTarget Z = VacuumCup Z + 0.066m
```

- `move_to()` 함수는 **ikTarget** 위치를 제어
- 실제 흡착은 **VacuumCup** 위치에서 발생
- 6.6cm 오프셋이 자동으로 고려됨

## 진공 컵 동작 원리

### BaxterVacuumCup 제어

진공 컵은 **신호 기반**으로 동작합니다:

```python
# 활성화 (Python)
robot.activate_vacuum()
# 내부적으로: sim.setIntProperty(sim.handle_scene, 'signal.BaxterVacuumCup__XXX__active', 1)

# 비활성화 (Python)
robot.deactivate_vacuum()
# 내부적으로: sim.setIntProperty(sim.handle_scene, 'signal.BaxterVacuumCup__XXX__active', 0)

# 흡착 확인
attached = robot.check_vacuum_attached()
# loopClosureDummy1의 부모가 VacuumCup이 아니면 물체가 흡착됨
```

### Lua 스크립트 동작 (BaxterVacuumCup)

1. **신호 감지**: `signal.BaxterVacuumCup__XXX__active` 확인
2. **물체 감지**: Proximity sensor로 respondable shape 탐지
3. **Loop Closure**: `loopClosureDummy1`을 감지된 shape에 연결
4. **흡착 유지**: 신호가 1인 동안 연결 유지
5. **해제**: 신호가 0이면 연결 해제

## 센서 기반 하강 (Grasp 동작)

### 2단계 하강 전략

```python
# grasp() 메서드 내부 동작:
# Phase 1: Fast descent (15mm step, 센서 범위 밖)
while current_z > target_z and not in_sensor_range:
    current_z -= 0.015  # 15mm
    check_sensor()

# Phase 2: Precision descent (10mm step, 센서 활성)
while current_z > target_z and not sensor_detected:
    current_z -= 0.010  # 10mm
    if sensor_detects_object:
        if distance < 5mm:
            break  # 접촉 감지, 정지
```

### 안전 장치

- **최소 하강 제한**: `min_descent_z = 0.031m` (바닥 + 3cm)
- **센서 실패 시**: 최소 높이까지만 하강
- **최대 반복 횟수**: 300회 (무한 루프 방지)

### Grasp 반환 값

```python
result = robot.grasp(...)

# result == 1: 성공 (Pick → Place 완료)
# result == 0.5: 물체 흔들림 (들어올릴 때 떨어짐)
# result == -1: 실패 (센서 감지 실패 또는 흡착 실패)
```

## 문제 해결

### 연결 실패
```
Failed to connect to CoppeliaSim
```
✅ **해결:**
1. CoppeliaSim이 실행 중인지 확인
2. `Add-ons → ZMQ remote API server` 활성화 확인
3. 포트 충돌 확인 (기본 포트: 23000)

### 핸들을 찾을 수 없음
```
Object does not exist: /irb360
```
✅ **해결:**
1. 씬에 IRB360 로봇이 있는지 확인
2. 오브젝트 이름이 정확히 `irb360`인지 확인
3. BaxterVacuumCup이 씬에 있는지 확인

### 로봇이 움직이지 않음
✅ **해결:**
1. 시뮬레이션이 시작되었는지 확인
2. `irb360_python_wrapper.lua` 스크립트가 적용되어 있는지 확인
3. 이동 목표가 로봇 도달 범위 내인지 확인
4. 터미널 로그에서 `[MOVE]` 메시지 확인

### 진공 컵이 흡착하지 않음
✅ **해결:**
1. 물체가 `respondable` 속성을 가지고 있는지 확인
2. VacuumCup이 물체에 충분히 가까이 있는지 확인 (5mm 이내)
3. `activate_vacuum()` 후 충분한 대기 시간 (0.5초)
4. Lua 스크립트 로그 확인 (CoppeliaSim 콘솔)

### 센서가 물체를 감지하지 못함
```
[GRASP] Reached minimum Z=0.0310m (no sensor detection)
```
✅ **해결:**
1. 물체가 Pick workspace 내에 있는지 확인
2. 물체 위치가 정확한지 확인 (`get_obj_positions()`)
3. Proximity sensor 범위 확인 (Lua 스크립트)
4. 물체가 너무 작거나 센서 각도 밖에 있지 않은지 확인

### 물체가 Plane 밖으로 떨어짐
✅ **해결:**
1. `add_objects()` 시 `drop_z` 높이 확인 (0.001 + 0.15m)
2. Workspace limits 마진 확인 (0.05m)
3. 물체 스폰 후 충분한 안착 시간 (2초)

### Heightmap이 이상함 (NaN 또는 빈 값)
✅ **해결:**
1. 카메라가 workspace를 향하고 있는지 확인
2. 카메라 높이 확인 (0.18m)
3. `cam_intrinsics` 설정 확인
4. Depth 센서 near/far plane 설정 확인

## 고급 사용

### DQN 학습 (향후)

```bash
# DQN 학습 시작
python main_irb360.py --num_obj 5 --episode_num 1000
```

### 커스텀 Workspace

```python
# 커스텀 workspace 정의
custom_workspace = [
    [-0.6, -0.4],  # x 범위
    [-0.1, 0.1],   # y 범위
    [0.001, 0.15]  # z 범위
]

robot = RobotIRB360ZMQ(
    is_sim=True,
    workspace_limits=custom_workspace,
    obj_mesh_dir=OBJECTS_DIR,
    num_obj=5
)
```

### 물체 재배치

```python
# 현재 물체를 새 위치로 재배치
robot.reposition_objects()

# 또는 특정 workspace로 재배치
robot.reposition_objects(workspace_limits=custom_workspace)
```

## 참고 자료

- CoppeliaSim ZMQ Remote API: https://www.coppeliarobotics.com/helpFiles/en/zmqRemoteApiOverview.htm
- IRB360 델타 로봇: https://new.abb.com/products/robotics/industrial-robots/irb-360



