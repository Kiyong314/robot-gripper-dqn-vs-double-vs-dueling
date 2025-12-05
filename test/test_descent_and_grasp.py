"""
로봇 하강 및 흡착 테스트 스크립트

1. 물체 생성
2. 물체 위치 확인
3. 로봇을 물체 위로 이동
4. 하강
5. 센서 점검
6. 흡착 테스트
"""

from coppeliasim_zmqremoteapi_client import RemoteAPIClient
from robot_zmq_irb360 import RobotIRB360ZMQ
import os
import time
import numpy as np

# 물체 디렉토리
OBJECTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    '23.Learning-Pick-to-Place-Objects-in-a-cluttered-scene-using-deep-reinforcement-learning-main',
    'objects',
    'blocks'
)

print("=" * 70)
print("IRB360 하강 및 흡착 테스트")
print("=" * 70)

# 로봇 초기화 및 물체 생성
print("\n[STEP 1] 로봇 초기화 및 물체 생성...")
robot = RobotIRB360ZMQ(
    is_sim=True,
    obj_mesh_dir=OBJECTS_DIR,
    num_obj=1  # 테스트를 위해 1개만
)

robot.add_objects()
time.sleep(2.0)  # 물체 낙하 대기

# 물체 위치 확인
print("\n[STEP 2] 물체 위치 확인...")
obj_positions = robot.get_obj_positions()
if len(obj_positions) == 0:
    print("[ERROR] 물체가 없습니다!")
    exit(1)

target_obj_pos = obj_positions[0]
print(f"물체 위치: ({target_obj_pos[0]:.3f}, {target_obj_pos[1]:.3f}, {target_obj_pos[2]:.3f})")

# 물체 높이 추정
object_center_height = target_obj_pos[2] + 0.001  # Plane 표면 기준
estimated_height = object_center_height * 2
object_top_z = target_obj_pos[2] + estimated_height / 2

print(f"물체 중심 높이: {object_center_height:.4f}m")
print(f"추정 물체 높이: {estimated_height:.4f}m")
print(f"물체 상단 Z: {object_top_z:.4f}m")

# VacuumCup 오프셋
vacuum_cup_offset = 0.066  # ikTarget과 VacuumCup 차이

# 목표 위치 계산
ikTarget_z = object_top_z + vacuum_cup_offset
expected_vacuum_z = object_top_z

print(f"목표 ikTarget Z: {ikTarget_z:.4f}m")
print(f"예상 VacuumCup Z: {expected_vacuum_z:.4f}m")

# 3. 물체 위로 이동 (상공)
print(f"\n[STEP 3] 물체 위 상공으로 이동...")
above_position = [target_obj_pos[0], target_obj_pos[1], ikTarget_z + 0.15]
print(f"상공 위치: ({above_position[0]:.3f}, {above_position[1]:.3f}, {above_position[2]:.3f})")

robot.move_to(above_position, 0)
time.sleep(2.0)

# 현재 VacuumCup 위치 확인
vacuum_pos = robot.sim.getObjectPosition(robot.vacuumCup_body_handle, -1)
print(f"현재 VacuumCup 위치: ({vacuum_pos[0]:.3f}, {vacuum_pos[1]:.3f}, {vacuum_pos[2]:.3f})")

# 진공 활성화
print(f"진공 활성화...")
robot.activate_vacuum()
time.sleep(2.0)  # 충분히 대기


# 4. 하강
print(f"\n[STEP 4] 물체 위로 하강 중...")
target_position = [target_obj_pos[0], target_obj_pos[1], ikTarget_z]
print(f"하강 목표: ({target_position[0]:.3f}, {target_position[1]:.3f}, {target_position[2]:.3f})")

robot.move_to(target_position, 0)
time.sleep(2.0)  # 충분히 대기

# 하강 후 위치 확인
vacuum_pos_after = robot.sim.getObjectPosition(robot.vacuumCup_body_handle, -1)
print(f"하강 후 VacuumCup 위치: ({vacuum_pos_after[0]:.3f}, {vacuum_pos_after[1]:.3f}, {vacuum_pos_after[2]:.3f})")
print(f"예상 VacuumCup Z: {expected_vacuum_z:.4f}m")
print(f"실제 VacuumCup Z: {vacuum_pos_after[2]:.4f}m")
print(f"오차: {abs(vacuum_pos_after[2] - expected_vacuum_z)*1000:.1f}mm")

# 물체와의 거리
distance_to_object = np.linalg.norm(np.array(vacuum_pos_after) - np.array(target_obj_pos))
print(f"VacuumCup-물체 거리: {distance_to_object*1000:.1f}mm")

# 5. 센서 점검
print(f"\n[STEP 5] 센서 점검...")

# 모든 shape 확인
client = RemoteAPIClient()
sim = client.require('sim')

sensor_handle = sim.getObject('/BaxterVacuumCup/sensor')
print(f"Sensor handle: {sensor_handle}")

# Proximity sensor로 물체 감지 테스트
print(f"\n물체 감지 테스트:")
index = 0
detected_shapes = []
while True:
    try:
        shape = sim.getObjects(index, sim.sceneobject_shape)
        if shape == -1:
            break
        
        shape_alias = sim.getObjectAlias(shape, 0)
        if 'shape_' in shape_alias:
            # checkProximitySensor 테스트
            detection_result = sim.checkProximitySensor(sensor_handle, shape)
            is_respondable = sim.getBoolProperty(shape, 'respondable')
            shape_pos = sim.getObjectPosition(shape, -1)
            
            print(f"  {shape_alias}:")
            print(f"    Respondable: {is_respondable}")
            print(f"    Detection: {detection_result}")
            print(f"    Position: ({shape_pos[0]:.3f}, {shape_pos[1]:.3f}, {shape_pos[2]:.3f})")
            
            if detection_result == 1:
                detected_shapes.append(shape_alias)
        
        index += 1
    except:
        break

if detected_shapes:
    print(f"\n[OK] 센서가 {len(detected_shapes)}개 물체 감지!")
    for name in detected_shapes:
        print(f"  - {name}")
else:
    print(f"\n[FAIL] 센서가 물체를 감지하지 못했습니다!")
    print(f"[INFO] VacuumCup이 물체에 충분히 가까이 있지 않을 수 있습니다.")

# 3. 물체 위로 이동 (상공)
print(f"\n[STEP 3] 물체 위 상공으로 이동...")
above_position = [target_obj_pos[0], target_obj_pos[1], ikTarget_z + 0.15]
print(f"상공 위치: ({above_position[0]:.3f}, {above_position[1]:.3f}, {above_position[2]:.3f})")


# 6. 흡착 테스트
print(f"\n[STEP 6] 흡착 테스트...")

# 로봇의 check_vacuum_attached 메서드 사용
print(f"흡착 상태 확인 (robot.check_vacuum_attached)...")
is_attached_before = robot.check_vacuum_attached()
print(f"진공 활성화 전 흡착 상태: {is_attached_before}")

# 진공 비활성화 확인
print(f"\n진공 비활성화...")
robot.deactivate_vacuum()
time.sleep(0.5)

# 진공 활성화
print(f"진공 활성화...")
robot.activate_vacuum()
time.sleep(2.0)  # 충분히 대기

# 흡착 상태 확인
is_attached_after = robot.check_vacuum_attached()
print(f"\n진공 활성화 후 흡착 상태: {is_attached_after}")

if is_attached_after:
    print(f"[SUCCESS] 물체 흡착 성공!")
    
    # 들어올리기 테스트
    print(f"\n물체 들어올리기...")
    lift_position = [target_obj_pos[0], target_obj_pos[1], ikTarget_z + 0.2]
    robot.move_to(lift_position, 0)
    time.sleep(2.0)
    
    # 물체 위치 확인 (올라갔는지)
    obj_handle = robot.object_handles[0] if robot.object_handles else None
    if obj_handle:
        obj_pos_after_lift = sim.getObjectPosition(obj_handle, -1)
        print(f"들어올린 후 물체 위치: ({obj_pos_after_lift[0]:.3f}, {obj_pos_after_lift[1]:.3f}, {obj_pos_after_lift[2]:.3f})")
        
        if obj_pos_after_lift[2] > target_obj_pos[2] + 0.05:
            print(f"[SUCCESS] 물체가 성공적으로 들어올려졌습니다!")
        else:
            print(f"[FAIL] 물체가 들어올려지지 않았습니다.")
    
    # 진공 해제
    print(f"\n진공 해제...")
    robot.deactivate_vacuum()
    time.sleep(1.0)
    
else:
    print(f"[FAIL] 물체 흡착 실패!")
    print(f"\n가능한 원인:")
    if not detected_shapes:
        print(f"  1. Proximity sensor가 물체를 감지하지 못함")
        print(f"     -> VacuumCup을 더 낮추거나 센서 범위 조정 필요")
    else:
        print(f"  1. Sensor는 감지했지만 Lua 스크립트가 작동하지 않음")
        print(f"     -> Lua 스크립트 로그 확인 필요")

# 홈으로 복귀
print(f"\n홈으로 복귀...")
robot.move_to(robot.home_position, 0)
time.sleep(1.0)

print("\n" + "=" * 70)
print("테스트 완료")
print("=" * 70)

