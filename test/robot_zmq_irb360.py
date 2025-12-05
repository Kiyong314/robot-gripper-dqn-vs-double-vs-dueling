"""
CoppeliaSim 4.10 ZMQ Remote API를 사용하는 IRB360 델타 로봇 클래스

IRB360 델타 로봇 + BaxterVacuumCup 진공 컵 제어

주요 특징:
- IK 모드로 직접 XYZ 좌표 제어 (로봇 상대 좌표)
- 진공 컵 신호로 물체 흡착/해제
- Lua moveToConfig 함수 호출 또는 직접 조인트 제어
- 월드 좌표 ↔ 로봇 상대 좌표 변환

좌표 시스템:
- Place 위치: x=0.2, y=-0.5, z=0.3 (월드 좌표, 컨베이어 위치)
- Pick 높이: z=0.3 (월드 좌표)
"""

import time
import os
import sys
import math
import numpy as np

from coppeliasim_zmqremoteapi_client import RemoteAPIClient


def euler2rotm(euler_angles):
    """
    오일러 각도를 회전 행렬로 변환
    
    Args:
        euler_angles: [rx, ry, rz] (라디안)
        
    Returns:
        np.array: 3x3 회전 행렬
    """
    rx, ry, rz = euler_angles
    
    # Rotation matrices
    Rx = np.array([
        [1, 0, 0],
        [0, math.cos(rx), -math.sin(rx)],
        [0, math.sin(rx), math.cos(rx)]
    ])
    
    Ry = np.array([
        [math.cos(ry), 0, math.sin(ry)],
        [0, 1, 0],
        [-math.sin(ry), 0, math.cos(ry)]
    ])
    
    Rz = np.array([
        [math.cos(rz), -math.sin(rz), 0],
        [math.sin(rz), math.cos(rz), 0],
        [0, 0, 1]
    ])
    
    return np.dot(Rz, np.dot(Ry, Rx))


class RobotIRB360ZMQ(object):
    """
    IRB360 델타 로봇 + BaxterVacuumCup 제어 클래스
    
    CoppeliaSim 4.10 ZMQ API를 사용하여 IRB360 로봇을 제어합니다.
    
    주요 특징:
    1. Lua moveToConfig 함수 호출로 IK 모드 이동
    2. 진공 컵 신호로 물체 흡착/해제
    3. 월드 좌표 → 로봇 상대 좌표 변환
    """
    
    def __init__(self, is_sim=True, obj_mesh_dir=None, num_obj=0, 
                 workspace_limits=None, workspace_limit_place=None,
                 is_testing=False, test_preset_cases=False, 
                 test_preset_file=None, place=False):
        """
        IRB360 로봇 초기화
        
        Args:
            is_sim: 시뮬레이션 모드 여부 (기본값: True)
            obj_mesh_dir: 물체 메쉬 파일 디렉토리 경로
            num_obj: 씬에 추가할 물체 개수
            workspace_limits: Pick 작업 공간 제한 [[x_min, x_max], [y_min, y_max], [z_min, z_max]]
            workspace_limit_place: Place 작업 공간 제한
            is_testing: 테스트 모드 여부
            test_preset_cases: 사전 설정된 테스트 케이스 사용 여부
            test_preset_file: 테스트 케이스 파일 경로
            place: Place 작업 수행 여부
        """
        self.is_sim = is_sim
        self.place_task = place
        
        # IRB360용 기본 workspace 설정 (월드 좌표)
        # 현재 씬 정보:
        # - IRB360 베이스: Z=1.100m
        # - Plane (Pick): 위치=(-0.500, 0.000, 0.001m)
        # - Plane1 (Place): 위치=(0.075, -0.500, 0.001m)
        # - VacuumCup 초기: Z=0.289m
        # - ikTarget 초기: Z=0.355m (VacuumCup과의 차이: 0.066m)
        # - 카메라: 위치=(-0.500, 0.000, 0.180m)
        
        # Pick 작업 영역: Plane 위치 기준
        if workspace_limits is None:
            self.workspace_limits = np.asarray([
                [-0.7, -0.3],   # x: Plane 중심(-0.5) ± 0.2
                [-0.2, 0.2],    # y: Plane 중심(0) ± 0.2
                [0.001, 0.20]   # z: Plane 표면(0.001) ~ 물체 최대 높이(0.20)
            ])
        else:
            self.workspace_limits = np.asarray(workspace_limits)
        
        # Place 작업 영역: Plane1 위치 기준
        if workspace_limit_place is None:
            self.workspace_limit_place = np.asarray([
                [-0.5, 0.5],    # x: 로봇의 전체 작업 범위
                [-0.6, -0.4],   # y: Plane1 중심(-0.5) ± 0.1
                [0.001, 0.20]   # z: Plane1 표면(0.001) ~ 0.2m
            ])
        else:
            self.workspace_limit_place = np.asarray(workspace_limit_place)
        
        # IRB360 홈 포지션 (월드 좌표) - 작업 대기 위치
        # VacuumCup 초기 위치가 (-0.250, 0.000, 0.289)이므로 이와 유사하게 설정
        # 카메라 시야를 방해하지 않으면서 Pick 영역에서 충분히 떨어진 위치
        # 카메라 위치: (-0.5, 0, 0.18), Workspace: x[-0.7, -0.3]
        # Home은 Workspace 밖이면서 카메라 시야 밖이어야 함
        # X: 0.3 (Workspace 오른쪽 밖, 카메라에서 멀리)
        # Y: 0.4 (Workspace 위쪽 밖)
        # Z: 0.60 (충분히 높은 위치)
        self.home_position = [0.0, 0.0, 0.50]
        
        # Place 위치 (월드 좌표) - 로봇 도달 가능한 위치
        # IRB360의 작업 범위를 고려하여 설정
        # Pick: X≈-0.5 (왼쪽), Place: X=0 (중앙, 더 안전한 위치)
        self.place_position = [-0, -0.0, 0.2]
        
        # Pick 높이 (월드 좌표) - Plane 표면 높이
        self.pick_height = 0.001
        
        # 물체 관련 초기화
        self.object_handles = []
        self.obj_mesh_dir = obj_mesh_dir
        self.num_obj = num_obj
        
        if self.is_sim:
            # 물체 색상 팔레트 (Tableau palette)
            self.color_space = np.asarray([
                [78.0, 121.0, 167.0],    # blue
                [89.0, 161.0, 79.0],     # green
                [156, 117, 95],          # brown
                [242, 142, 43],          # orange
                [237.0, 201.0, 72.0],    # yellow
                [186, 176, 172],         # gray
                [255.0, 87.0, 89.0],     # red
                [176, 122, 161],         # purple
                [118, 183, 178],         # cyan
                [255, 157, 167]          # pink
            ]) / 255.0
            
            # 물체 메쉬 관련 초기화
            if obj_mesh_dir and os.path.exists(obj_mesh_dir):
                self.mesh_list = os.listdir(obj_mesh_dir)
                self.obj_mesh_ind = np.random.randint(0, len(self.mesh_list), size=max(num_obj, 1))
                self.obj_mesh_color = self.color_space[np.asarray(range(max(num_obj, 1))) % 10, :]
            else:
                self.mesh_list = []
                self.obj_mesh_ind = []
                self.obj_mesh_color = []
            
            # CoppeliaSim ZMQ Remote API 연결
            print('Connecting to CoppeliaSim via ZMQ Remote API (IRB360)...')
            try:
                self.client = RemoteAPIClient()
                self.sim = self.client.require('sim')
                print('Connected to CoppeliaSim successfully.')
                
                # Stepping 모드 비활성화 (비동기 모드)
                self.sim.setStepping(False)
                print('Stepping mode disabled (asynchronous mode).')
                
                # 기존 시뮬레이션이 실행 중이면 정지
                state = self.sim.getSimulationState()
                if state != self.sim.simulation_stopped:
                    print(f'Stopping existing simulation (state: {state})...')
                    self.sim.stopSimulation(True)
                    time.sleep(0.5)
                    print('Existing simulation stopped successfully.')
                    
            except Exception as e:
                print(f'Failed to connect to CoppeliaSim: {e}')
                print('Make sure:')
                print('  1. CoppeliaSim is running')
                print('  2. ZMQ Remote API server is started (Add-ons → ZMQ remote API server)')
                raise
            
            # IRB360 관련 핸들 설정
            self._setup_irb360_handles()
            
            # 테스트 설정
            self.is_testing = is_testing
            self.test_preset_cases = test_preset_cases
            self.test_preset_file = test_preset_file
            
            # 카메라 설정
            self.setup_sim_camera()
            
            # 시뮬레이션 시작
            self.restart_sim()
    
    def _setup_irb360_handles(self):
        """
        IRB360 로봇 관련 오브젝트 핸들 설정
        
        설정하는 핸들:
        - irb360_handle: IRB360 로봇 베이스
        - ikTarget_handle: IK 타겟 (이동 목표점)
        - irb360_script_handle: IRB360 Lua 스크립트 핸들
        - vacuumCup_handle: BaxterVacuumCup 핸들
        - loopClosureDummy1_handle: 흡착 상태 확인용 더미
        """
        try:
            # IRB360 로봇 베이스 핸들
            self.irb360_handle = self.sim.getObject('/irb360')
            print(f'[HANDLE] irb360: {self.irb360_handle}')
            
            # ikTarget 핸들 (cartesianX/Y/Z 하위에 있음)
            self.ikTarget_handle = self.sim.getObject('/irb360/cartesianX/cartesianY/cartesianZ/ikTarget')
            print(f'[HANDLE] ikTarget: {self.ikTarget_handle}')
            
            # IRB360 스크립트 핸들 (moveToConfig 호출용)
            try:
                self.irb360_script_handle = self.sim.getScript(self.sim.scripttype_childscript, self.irb360_handle)
                print(f'[HANDLE] irb360_script: {self.irb360_script_handle}')
            except:
                self.irb360_script_handle = None
                print('[HANDLE] irb360_script: Not found (will use direct joint control)')
            
            # BaxterVacuumCup 핸들 (alias로 접근)
            self.vacuumCup_handle = self.sim.getObject('/BaxterVacuumCup')
            print(f'[HANDLE] vacuumCup: {self.vacuumCup_handle}')
            
            # loopClosureDummy1 핸들 (흡착 상태 확인용)
            self.loopClosureDummy1_handle = self.sim.getObject('/BaxterVacuumCup/loopClosureDummy1')
            print(f'[HANDLE] loopClosureDummy1: {self.loopClosureDummy1_handle}')
            
            # 진공 컵 본체 핸들 (부모 비교용)
            self.vacuumCup_body_handle = self.vacuumCup_handle
            
            # IRB360 베이스 위치 저장 (좌표 변환용)
            self.irb360_base_position = self.sim.getObjectPosition(self.irb360_handle, -1)
            print(f'[INFO] IRB360 base position: {self.irb360_base_position}')
            
            # 진공 컵 위치 확인 (디버깅용)
            vacuum_pos = self.sim.getObjectPosition(self.vacuumCup_handle, -1)
            print(f'[INFO] VacuumCup position: {vacuum_pos}')
            
            # Cartesian 조인트 핸들 (직접 제어용)
            self.cartesianX_handle = self.sim.getObject('/irb360/cartesianX')
            self.cartesianY_handle = self.sim.getObject('/irb360/cartesianX/cartesianY')
            self.cartesianZ_handle = self.sim.getObject('/irb360/cartesianX/cartesianY/cartesianZ')
            self.motor_handle = self.sim.getObject('/irb360/motor')
            print('[HANDLE] Cartesian joints acquired')
            
        except Exception as e:
            print(f'Failed to setup IRB360 handles: {e}')
            print('Make sure the IRB360 scene is loaded with correct object names.')
            raise
    
    # =========================================================================
    # 진공 컵 제어 메서드
    # =========================================================================
    
    def activate_vacuum(self):
        """
        진공 컵 활성화 (물체 흡착)
        
        BaxterVacuumCup 스크립트의 'signal.BaxterVacuumCup_active' 신호를 1로 설정
        스크립트에서 이 신호가 1이면 proximity sensor로 감지된 물체를 흡착합니다.
        """
        if self.is_sim:
            try:
                # VacuumCup의 실제 Alias를 가져와서 신호 이름 구성
                # 파라미터 4: Lua 스크립트와 동일하게 (handle 포함)
                alias = self.sim.getObjectAlias(self.vacuumCup_handle, 4)
                signal_name = f'signal.{alias}_active'
                
                self.sim.setIntProperty(
                    self.sim.handle_scene, 
                    signal_name, 
                    1
                )
                print(f'[VACUUM] Activated ({signal_name}=1) - waiting for attachment...')
                time.sleep(0.3)
                
                attached = self.check_vacuum_attached()
                print(f'[VACUUM] Attachment status: {attached}')
                
            except Exception as e:
                print(f'[VACUUM] Failed to activate: {e}')

    def deactivate_vacuum(self):
        """
        진공 컵 비활성화 (물체 해제)
        
        BaxterVacuumCup 스크립트의 'signal.BaxterVacuumCup_active' 신호를 0으로 설정
        """
        if self.is_sim:
            try:
                # 파라미터 4: Lua 스크립트와 동일
                alias = self.sim.getObjectAlias(self.vacuumCup_handle, 4)
                signal_name = f'signal.{alias}_active'
                
                self.sim.setIntProperty(
                    self.sim.handle_scene, 
                    signal_name, 
                    0
                )
                print(f'[VACUUM] Deactivated ({signal_name}=0) - object released')
                time.sleep(0.2)
                
            except Exception as e:
                print(f'[VACUUM] Failed to deactivate: {e}')
    
    def check_vacuum_attached(self):
        """
        물체가 흡착되었는지 확인
        
        BaxterVacuumCup 스크립트에서:
        - loopClosureDummy1의 부모가 VacuumCup 본체(b)면 물체 없음
        - loopClosureDummy1의 부모가 다른 shape면 물체가 흡착됨
        
        Returns:
            bool: 물체가 흡착되었으면 True, 아니면 False
        """
        if self.is_sim:
            try:
                parent_handle = self.sim.getObjectParent(self.loopClosureDummy1_handle)
                attached = (parent_handle != self.vacuumCup_body_handle)
                
                if attached:
                    try:
                        parent_alias = self.sim.getObjectAlias(parent_handle, 0)
                        print(f'[VACUUM] Attached to object: {parent_alias} (handle: {parent_handle})')
                    except:
                        pass
                
                return attached
                
            except Exception as e:
                print(f'[VACUUM] Failed to check attachment: {e}')
                return False
        
        return False
    
    # =========================================================================
    # 이동 메서드
    # =========================================================================
    
    def world_to_robot_coord(self, world_pos):
        """
        월드 좌표를 로봇 상대 좌표로 변환
        
        IRB360의 좌표계 특성 (축 반전 등) 고려
        Args:
            world_pos: 월드 좌표 [x, y, z]
            
        Returns:
            list: 로봇 상대 좌표 [x, y, z]
        """
        # 로봇 베이스 위치
        base_pos = self.irb360_base_position
        
        # X, Y 변환 (IRB360은 축이 반대임: test_vacuum.py에서 검증됨)
        # robot_x = -(world_x - base_x)
        robot_x = -(world_pos[0] - base_pos[0])
        robot_y = -(world_pos[1] - base_pos[1])
        
        # Z 변환
        # robot_z = 0일 때 VacuumCup Z 높이 (약 0.689m)
        # robot_z = world_z - vacuum_origin_z
        # 예: 목표 0.3m -> 0.3 - 0.689 = -0.389
        vacuum_origin_z = 0.689 
        robot_z = world_pos[2] - vacuum_origin_z
        
        return [robot_x, robot_y, robot_z]
    
    def robot_to_world_coord(self, robot_pos):
        """
        로봇 상대 좌표를 월드 좌표로 변환
        
        Args:
            robot_pos: 로봇 상대 좌표 [x, y, z]
            
        Returns:
            list: 월드 좌표 [x, y, z]
        """
        irb360_base_pos = self.sim.getObjectPosition(self.irb360_handle, -1)
        
        world_pos = [
            robot_pos[0] + irb360_base_pos[0],
            robot_pos[1] + irb360_base_pos[1],
            robot_pos[2] + irb360_base_pos[2]
        ]
        
        return world_pos
    
    def move_to(self, world_position, rotation_angle=0, wait_arrival=True, timeout=5.0):
        """
        로봇을 지정된 월드 좌표로 이동 (ikTarget 직접 제어 방식)
        
        Lua 스크립트의 sysCall_actuation에서 IK Solver가 
        ikTip을 ikTarget 위치로 지속적으로 맞추고 있으므로,
        ikTarget의 위치만 변경하면 로봇이 따라 움직입니다.
        
        Args:
            world_position: 목표 위치 (월드 좌표) [x, y, z]
            rotation_angle: 회전 각도 (라디안), 기본값 0
            wait_arrival: True면 실제 도착 확인, False면 고정 시간 대기
            timeout: 도착 대기 최대 시간 (초)
        """
        if self.is_sim:
            try:
                #print(f'[MOVE] Moving ikTarget to World: {world_position}, Rot: {rotation_angle:.2f}')
                
                # 1. ikTarget 위치 설정 (월드 좌표 기준)
                # 주의: world_position은 [x, y, z] 리스트여야 함
                self.sim.setObjectPosition(self.ikTarget_handle, -1, world_position)
                
                # 2. 회전 각도 (Motor Joint) 설정
                # Delta 로봇의 특성상 4번째 축(회전)은 별도로 제어될 수 있음
                self.sim.setJointTargetPosition(self.motor_handle, rotation_angle)
                
                # 3. 이동 완료 대기
                if wait_arrival:
                    # 실제 도착 확인
                    arrival_threshold = 0.01  # 1cm 이내
                    start_time = time.time()
                    
                    while time.time() - start_time < timeout:
                        current_pos = self.sim.getObjectPosition(self.ikTarget_handle, -1)
                        distance = np.linalg.norm(np.array(current_pos) - np.array(world_position))
                        
                        if distance < arrival_threshold:
                            #print(f'[MOVE] Arrived at target (distance: {distance*1000:.1f}mm)')
                            time.sleep(0.2)  # 안정화
                            break
                        time.sleep(0.1)
                    else:
                        print(f'[MOVE] Timeout after {timeout}s, distance: {distance*1000:.1f}mm')
                else:
                    # 기존 방식: 고정 시간 대기
                    time.sleep(0.5)
                
            except Exception as e:
                print(f'[MOVE] Failed: {e}')
                # Fallback: 직접 조인트 제어 시도 (좌표 변환 필요)
                try:
                    robot_pos = self.world_to_robot_coord(world_position)
                    self._move_to_direct(world_position, rotation_angle)
                except:
                    pass
    
    def _move_to_direct(self, world_position, rotation_angle=0):
        """
        직접 cartesianX/Y/Z 조인트 위치를 설정하여 이동
        (test_irb360.py 방식 참고 - setJointTargetPosition 사용)
        
        Args:
            world_position: 목표 위치 (월드 좌표) [x, y, z]
            rotation_angle: 회전 각도 (라디안)
        """
        if self.is_sim:
            try:
                robot_pos = self.world_to_robot_coord(world_position)
                target_x, target_y, target_z = robot_pos
                
                # IRB360 cartesian 조인트 범위 제한 (테스트로 확인된 범위)
                # X: -0.4 ~ 0.5, Y: -0.4 ~ 0.4, Z: -0.4 ~ 0.2
                # 조인트 범위를 초과하면 클램핑
                target_x = np.clip(target_x, -0.4, 0.5)
                target_y = np.clip(target_y, -0.4, 0.4)
                target_z = np.clip(target_z, -0.4, 0.2)
                
                print(f'[MOVE-DIRECT] Target: X={target_x:.3f}, Y={target_y:.3f}, Z={target_z:.3f}')
                
                # setJointTargetPosition 사용 (모터가 실제로 이동)
                self.sim.setJointTargetPosition(self.cartesianX_handle, target_x)
                self.sim.setJointTargetPosition(self.cartesianY_handle, target_y)
                self.sim.setJointTargetPosition(self.cartesianZ_handle, target_z)
                self.sim.setJointTargetPosition(self.motor_handle, rotation_angle)
                
                # 이동 완료 대기 (0.8초)
                time.sleep(0.8)
                
                # 현재 위치 확인
                current_x = self.sim.getJointPosition(self.cartesianX_handle)
                current_y = self.sim.getJointPosition(self.cartesianY_handle)
                current_z = self.sim.getJointPosition(self.cartesianZ_handle)
                print(f'[MOVE-DIRECT] Actual: X={current_x:.3f}, Y={current_y:.3f}, Z={current_z:.3f}')
                
            except Exception as e:
                print(f'[MOVE-DIRECT] Failed: {e}')
    
    def get_current_position(self):
        """
        현재 엔드이펙터(진공 컵) 위치 가져오기 (월드 좌표)
        
        Returns:
            list: 현재 위치 [x, y, z] (월드 좌표)
        """
        if self.is_sim:
            try:
                position = self.sim.getObjectPosition(self.vacuumCup_handle, -1)
                return list(position)
            except Exception as e:
                print(f'[POS] Failed to get current position: {e}')
                return [0, 0, 0]
        return [0, 0, 0]
    
    # =========================================================================
    # Grasp/Place 메서드
    # =========================================================================
    
    def grasp(self, position, heightmap_rotation_angle, workspace_limits, trainer, workspace_limit_place):
        """
        Grasp primitive - 물체를 집는 동작 (진공 흡착 방식)
        
        시퀀스:
        1. Pick 위치 위로 이동 (z + margin)
        2. 진공 비활성화 확인
        3. Pick 위치로 하강
        4. 진공 활성화
        5. 위로 상승
        6. 흡착 성공 여부 확인
        7. 성공 시 Place 수행
        
        Args:
            position: 집을 위치 [x, y, z] (월드 좌표, z는 물체 상단 Z)
            heightmap_rotation_angle: 회전 각도 (라디안)
            workspace_limits: 작업 공간 제한
            trainer: DQN 트레이너 (place_success_log 기록용)
            workspace_limit_place: Place 공간 제한
        
        Returns:
            grasp_success: 물체를 성공적으로 집었는지 여부
        """
        if workspace_limits is None:
            workspace_limits = self.workspace_limits
        
        print(f'[GRASP] Executing at ({position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f})')
        
        if self.is_sim:
            tool_rotation_angle = (heightmap_rotation_angle % np.pi) - np.pi / 2
            
            # position[2]는 물체 상단 Z 좌표 (test에서 계산)
            position = np.asarray(position).copy()
            object_top_z = position[2]
            
            # VacuumCup 오프셋 계산
            vacuum_cup_length_offset = 0.066  # ikTarget ~ VacuumCup = 6.6cm
            position_correction = -0.03  # 위치 보정 (-3cm, 확실한 접촉)
            ikTarget_z = object_top_z + vacuum_cup_length_offset + position_correction
            
            print(f'[GRASP] Object top Z: {object_top_z:.4f}m')
            print(f'[GRASP] ikTarget Z (corrected): {ikTarget_z:.4f}m')
            
            position[2] = ikTarget_z
            
            grasp_location_margin = 0.2
            location_above_grasp_target = [
                position[0], 
                position[1], 
                0.3
            ]
            
            # 1. Pick 위치 위로 이동
            print(f'[GRASP] Step 1: Moving above target')
            self.move_to(location_above_grasp_target, tool_rotation_angle)
            #time.sleep(0.5)
            
            # 2. 진공 비활성화 확인
            print('[GRASP] Step 2: Ensuring vacuum is OFF')
            self.deactivate_vacuum()
            #time.sleep(0.2)
            
            # 3. 진공 활성화 (하강 전에 미리 켜둠)
            print('[GRASP] Step 3: Activating vacuum (before descent)')
            self.activate_vacuum()
            #time.sleep(0.3)
            
            # 4. 센서 기반 하강 (바닥 3cm까지만)
            print(f'[GRASP] Step 4: Descending with sensor feedback')
            
            # 센서 핸들 가져오기
            try:
                sensor_handle = self.sim.getObject('/BaxterVacuumCup/sensor')
            except:
                print(f'[GRASP] WARNING: Sensor not found - cannot proceed')
                self.deactivate_vacuum()
                self.move_to(self.home_position, 0)
                return False
            
            # 센서 기반 하강 파라미터
            # VacuumCup이 바닥에서 3cm 떨어지려면 ikTarget은 더 높아야 함
            # VacuumCup = ikTarget - 0.066m 이므로
            # ikTarget_min = 0.03 (VacuumCup 바닥 제한) + 0.066 (오프셋) = 0.096m
            vacuum_cup_offset = 0.066  # ikTarget ~ VacuumCup 거리
            min_vacuum_z = 0  # VacuumCup이 바닥에서 최소 3cm
            min_ikTarget_z = min_vacuum_z + vacuum_cup_offset  # ikTarget 최소 높이
            
            descent_step = 0.015  # 1cm씩 하강
            max_iterations = 50
            
            current_z = location_above_grasp_target[2]
            object_found = False
            grasped_object_handle = None  # 집힌 물체의 핸들 저장
            
            print(f'[GRASP] Starting descent from Z={current_z:.4f}m')
            print(f'[GRASP] Min ikTarget Z={min_ikTarget_z:.4f}m (VacuumCup at {min_vacuum_z:.4f}m)')
            
            for iteration in range(max_iterations):
                # 센서로 물체 감지 체크 (이동 전에 먼저 확인)
                detected = False
                detected_distance = None
                
                for obj_handle in self.object_handles:
                    try:
                        detection_result = self.sim.checkProximitySensor(sensor_handle, obj_handle)
                        if detection_result[0] == 1:  # 감지됨
                            distance = detection_result[1]
                            print(f'[GRASP] Sensor detected object at Z={current_z:.4f}m, Distance={distance*1000:.1f}mm (iteration {iteration+1})')
                            
                            # 센서가 물체를 감지하면 즉시 정지
                            # 거리 30mm 이내면 충분히 가까움
                            if distance < 0.030:  # 30mm 이내
                                print(f'[GRASP] Object within range ({distance*1000:.1f}mm) - stopping descent')
                                object_found = True
                                detected = True
                                detected_distance = distance
                                grasped_object_handle = obj_handle  # 감지된 물체 핸들 저장
                                break
                    except:
                        pass
                
                # 물체 감지되면 즉시 하강 중지
                if detected:
                    print(f'[GRASP] Descent stopped at Z={current_z:.4f}m')
                    break
                
                # 바닥 제한 체크 (ikTarget 기준으로 체크)
                next_z = current_z - descent_step
                if next_z < min_ikTarget_z:
                    print(f'[GRASP] Reached minimum ikTarget Z ({min_ikTarget_z:.4f}m)')
                    print(f'[GRASP] VacuumCup would be at Z={next_z - vacuum_cup_offset:.4f}m - stopping')
                    print(f'[GRASP] No object found after {iteration+1} iterations')
                    break
                
                # 이동 (물체 감지 안 되고 바닥 제한도 안 걸린 경우만)
                current_z = next_z
                temp_position = [position[0], position[1], current_z]
                self.move_to(temp_position, tool_rotation_angle)
                #time.sleep(0.05)
            
            # 물체를 찾지 못한 경우
            if not object_found:
                print(f'[GRASP] [X] No object found after {iteration+1} iterations')
 
                # Home 이동은 main에서 처리 (안정적인 촬영 보장)

                                # 실패 시 위로 복귀
                self.deactivate_vacuum()
                time.sleep(0.2)
                
                lift_position = [position[0], position[1], position[2]+0.3]
                self.move_to(lift_position, tool_rotation_angle)
                time.sleep(0.5)
                
                self.move_to(self.home_position, 0)
                time.sleep(0.5)

                return False
            
            # 실제 VacuumCup 위치 확인
            actual_vacuum_pos = self.sim.getObjectPosition(self.vacuumCup_body_handle, -1)
            print(f'[GRASP] Final VacuumCup position: Z={actual_vacuum_pos[2]:.4f}m')
            print(f'[GRASP] Target object top Z: {object_top_z:.4f}m')
            
            # 5. 흡착 확인 (진공은 이미 켜져 있음)
            print('[GRASP] Step 5: Checking vacuum attachment')
            time.sleep(0.3)
            
            grasp_success = self.check_vacuum_attached()
            
            if grasp_success:
                print('[GRASP] [OK] Object attached successfully!')
                
                # 6. 위로 상승
                print('[GRASP] Step 6: Lifting object')
                lift_position = [position[0], position[1], position[2] + 0.3]
                self.move_to(lift_position, tool_rotation_angle)
                time.sleep(0.5)
                
                # 7. Place로 이동
                print('[GRASP] Step 7: Moving to place location')
                place_position_lift = [
                    self.place_position[0],
                    self.place_position[1],
                    0.5
                ]
                self.move_to(place_position_lift, 0)
                time.sleep(0.5)
                
                # 8. Place 하강
                print('[GRASP] Step 8: Descending to place')
                self.move_to(self.place_position, 0)
                time.sleep(0.5)
                
                # 9. 진공 해제
                print('[GRASP] Step 9: Releasing object')
                self.deactivate_vacuum()
                time.sleep(0.5)
                
                # 9.5. 물체가 제대로 떨어졌는지 확인
                attached_object_name = self.check_vacuum_attached()
                if attached_object_name:
                    print(f'[GRASP] [ERROR] Object still attached after release: {attached_object_name}')
                    print(f'[GRASP] Abnormal situation - restarting scene')
                    self.restart_sim()
                    self.add_objects()
                    return -1  # 비정상 실패
                else:
                    print('[GRASP] [OK] Object released successfully')
                
                # 10. 위로 복귀
                print('[GRASP] Step 10: Returning to safe height')
                self.move_to(place_position_lift, 0)
                time.sleep(0.3)
                
                # 11. Home 위치로 복귀
                print('[GRASP] Step 11: Returning to home')
                self.move_to(self.home_position, 0)
                time.sleep(0.5)
                
                # 12. 성공한 물체를 object_handles에서 제거 (물체 개수 추적용)
                if grasped_object_handle is not None and grasped_object_handle in self.object_handles:
                    self.object_handles.remove(grasped_object_handle)
                    print(f'[GRASP] Removed object from tracking (remaining: {len(self.object_handles)})')
                
                print('[SUCCESS] Pick and Place complete!')
                return True
                
            else:
                print('[GRASP] [X] Failed to attach object')
                
                # 실패 시 위로 복귀
                self.deactivate_vacuum()
                time.sleep(0.2)
                
                lift_position = [position[0], position[1], position[2] + 0.3]
                self.move_to(lift_position, tool_rotation_angle)
                time.sleep(0.3)
                
                self.move_to(self.home_position, 0)
                time.sleep(0.3)
                
                return False
        
        else:
            print('[GRASP] Not in simulation mode')
            return False
            return grasp_success
        
        return False
    
    def place(self, place_position, heightmap_rotation_angle, trainer, workspace_limit_place=None):
        """
        Place primitive - 물체를 놓는 동작 (진공 해제 방식)
        
        시퀀스:
        1. 물체 흡착 확인
        2. Place 위치로 이동 (컨베이어: x=0.2, y=-0.5, z=0.3)
        3. 진공 비활성화
        4. 위로 상승
        5. 홈 위치로 복귀
        
        Args:
            place_position: 기준 위치 [x, y, z]
            heightmap_rotation_angle: 회전 각도 (라디안)
            trainer: DQN 트레이너 (place_success_log 기록용)
            workspace_limit_place: Place 공간 제한
        
        Returns:
            tuple: (place_success, place_not_success)
        """
        if workspace_limit_place is None:
            workspace_limit_place = self.workspace_limit_place
        
        print(f'[PLACE] Executing at conveyor: {self.place_position}')
        
        if self.is_sim:
            tool_rotation_angle = (heightmap_rotation_angle % np.pi) - np.pi / 2
            target_place_position = self.place_position.copy()
            
            place_location_margin = 0.1
            location_above_place_target = [
                target_place_position[0],
                target_place_position[1],
                target_place_position[2] + place_location_margin
            ]
            
            # 1. 물체 흡착 확인
            print('[PLACE] Step 1: Checking attachment')
            still_attached = self.check_vacuum_attached()
            
            if still_attached:
                place_success = True
                place_not_success = False
                
                # 2. Place 위치 위로 이동
                print(f'[PLACE] Step 2: Moving above place target')
                self.move_to(location_above_place_target, tool_rotation_angle)
                time.sleep(0.2)
                
                # 3. Place 위치로 하강
                print(f'[PLACE] Step 3: Descending')
                self.move_to(target_place_position, tool_rotation_angle)
                time.sleep(0.2)
                
                # 4. 진공 비활성화
                print('[PLACE] Step 4: Releasing object')
                self.deactivate_vacuum()
                time.sleep(0.3)
                
                # 5. 위로 상승
                print(f'[PLACE] Step 5: Rising')
                self.move_to(location_above_place_target, tool_rotation_angle)
                time.sleep(0.2)
                
                # 6. 홈으로 복귀
                print(f'[PLACE] Step 6: Returning to home')
                self.move_to(self.home_position, 0)
                time.sleep(0.2)
                
                print(f'[PLACE] Success: {place_success}')
                
            else:
                place_success = False
                place_not_success = True
                print('[PLACE] Object was not attached - failed')
                self.move_to(self.home_position, 0)
                time.sleep(0.2)
            
            if trainer is not None and hasattr(trainer, 'place_success_log'):
                trainer.place_success_log.append(place_success)
            
            return place_success, place_not_success
        
        return False, True
    
    # =========================================================================
    # 카메라 및 유틸리티 메서드
    # =========================================================================
    
    def setup_sim_camera(self):
        """카메라 설정 - Vision_sensor_ortho 사용"""
        try:
            self.cam_handle = self.sim.getObject('/Vision_sensor_ortho')
            print(f'[CAMERA] Using Vision_sensor_ortho (handle: {self.cam_handle})')
            
            cam_position = self.sim.getObjectPosition(self.cam_handle, -1)
            cam_orientation = self.sim.getObjectOrientation(self.cam_handle, -1)
            
            cam_trans = np.eye(4, 4)
            cam_trans[0:3, 3] = np.asarray(cam_position)
            cam_orientation_neg = [-cam_orientation[0], -cam_orientation[1], -cam_orientation[2]]
            cam_rotm = np.eye(4, 4)
            cam_rotm[0:3, 0:3] = np.linalg.inv(euler2rotm(cam_orientation_neg))
            self.cam_pose = np.dot(cam_trans, cam_rotm)
            
            self.cam_intrinsics = np.asarray([
                [618.62, 0, 320], 
                [0, 618.62, 240], 
                [0, 0, 1]
            ])
            self.cam_depth_scale = 1
            
            self.bg_color_img, self.bg_depth_img = self.get_camera_data()
            if self.bg_depth_img is not None:
                self.bg_depth_img = self.bg_depth_img * self.cam_depth_scale
            
            print('[CAMERA] Setup completed')
            
        except Exception as e:
            print(f'[CAMERA] Setup failed: {e}')
    
    def get_camera_data(self):
        """카메라 데이터 가져오기
        
        Note: 호출 전에 이미 로봇이 home 위치에 있어야 합니다.
        """
        if self.is_sim:
            try:
                # 이미 Home 위치에 있다고 가정 (main에서 이동)
                time.sleep(0.3)  # 진동 안정화만 대기
                
                img, resolution = self.sim.getVisionSensorImg(self.cam_handle)
                img = np.frombuffer(img, dtype=np.uint8)
                img = img.reshape((resolution[1], resolution[0], 3))
                color_img = np.flipud(img)
                
                depth_result = self.sim.getVisionSensorDepth(self.cam_handle, 0)
                
                if isinstance(depth_result, tuple) and len(depth_result) == 2:
                    depth_data, depth_res = depth_result
                    if isinstance(depth_data, bytes):
                        depth_img = np.frombuffer(depth_data, dtype=np.float32)
                        depth_img = depth_img.reshape((depth_res[1], depth_res[0]))
                    else:
                        depth_img = np.array(depth_data, dtype=np.float32)
                        depth_img = depth_img.reshape((depth_res[1], depth_res[0]))
                else:
                    if isinstance(depth_result, bytes):
                        depth_img = np.frombuffer(depth_result, dtype=np.float32)
                    else:
                        depth_img = np.array(depth_result, dtype=np.float32)
                    depth_img = depth_img.reshape((resolution[1], resolution[0]))
                
                depth_img = np.flipud(depth_img)
                
                zNear = 0.01
                zFar = 10
                depth_img = depth_img * (zFar - zNear) + zNear
                
                return color_img, depth_img
                
            except Exception as e:
                print(f'[CAMERA] Failed to get data: {e}')
                return None, None
        
        return None, None
    
    def restart_sim(self):
        """시뮬레이션 재시작
        
        주의: IRB360 스크립트의 초기화 moveToConfig가 완료될 때까지 대기해야 함
        """
        try:
            self.sim.stopSimulation(True)
            time.sleep(0.5)
            
            self.sim.setStepping(False)
            self.sim.startSimulation()
            
            # IRB360 스크립트 초기화 대기 (중요!)
            # sysCall_thread의 moveToConfig가 완료될 때까지 대기
            print('[RESTART] Waiting for IRB360 script initialization (3 seconds)...')
            time.sleep(3.0)
            
            # 이제 API 호출 가능
            self.irb360_base_position = self.sim.getObjectPosition(self.irb360_handle, -1)
            self.deactivate_vacuum()
            
            # 홈 위치로 이동
            print('[RESTART] Moving robot to home position...')
            self.move_to(self.home_position, 0)
            time.sleep(1.0)
            print(f'[RESTART] Robot at home: {self.home_position}')
            
            print('[RESTART] Simulation restarted successfully')
            
        except Exception as e:
            print(f'[RESTART] Failed: {e}')
            raise
    
    def add_objects(self):
        """
        물체 추가 - workspace_limits 내에 물체를 랜덤하게 배치
        
        물체 메쉬 파일을 로드하여 workspace_limits 범위 내에 랜덤 위치/방향으로 배치합니다.
        기존 물체를 먼저 삭제하고 새로 생성합니다.
        """
        if not self.obj_mesh_dir or not os.path.exists(self.obj_mesh_dir):
            print('[OBJECTS] No mesh directory specified')
            return
        
        # 1. 기존 물체 삭제 (핸들 리스트에서)
        if hasattr(self, 'object_handles') and self.object_handles:
            print(f'[OBJECTS] Removing {len(self.object_handles)} existing objects...')
            for handle in self.object_handles:
                try:
                    self.sim.removeObject(handle)
                except:
                    pass
        
        # 2. 씬에서 "shape_" 이름의 기존 물체들도 삭제 (4.10Test 참고)
        print('[OBJECTS] Cleaning up existing shape_ objects in scene...')
        removed_count = 0
        for i in range(100):  # shape_00 ~ shape_99
            try:
                # getObject는 찾지 못하면 예외 발생
                existing_handle = self.sim.getObject(f'/shape_{i:02d}')
                self.sim.removeObject(existing_handle)
                removed_count += 1
                print(f'[OBJECTS] Removed /shape_{i:02d}')
            except:
                # 해당 이름의 물체가 없으면 패스
                pass
        
        if removed_count > 0:
            print(f'[OBJECTS] Total removed: {removed_count} objects')
        
        self.object_handles = []
        
        # workspace_limits에서 범위 추출
        x_min, x_max = self.workspace_limits[0]
        y_min, y_max = self.workspace_limits[1]
        
        # 마진 적용 (가장자리 회피)
        margin = 0.05
        
        for object_idx in range(len(self.obj_mesh_ind)):
            curr_mesh_file = os.path.join(
                self.obj_mesh_dir, 
                self.mesh_list[self.obj_mesh_ind[object_idx]]
            )
            curr_shape_name = 'shape_%02d' % object_idx
            
            # 테스트 케이스 파일이 있으면 해당 위치 사용
            if self.is_testing and self.test_preset_cases and hasattr(self, 'test_obj_positions'):
                object_position = self.test_obj_positions[object_idx]
                object_orientation = self.test_obj_orientations[object_idx]
            else:
                # workspace_limits 범위 내에서 랜덤 위치 생성
                drop_x = (x_max - x_min - 2*margin) * np.random.random_sample() + x_min + margin
                drop_y = (y_max - y_min - 2*margin) * np.random.random_sample() + y_min + margin
                # Plane 표면(0.001m) 위에서 낙하 (15cm 높이에서 떨어뜨림)
                drop_z = 0.001 + 0.15
                object_position = [drop_x, drop_y, drop_z]
                object_orientation = [
                    2 * np.pi * np.random.random_sample(),
                    2 * np.pi * np.random.random_sample(),
                    2 * np.pi * np.random.random_sample()
                ]
            
            object_color = self.obj_mesh_color[object_idx]
            
            try:
                # ZMQ API로 shape import
                shape_handle = self.sim.importShape(0, curr_mesh_file, 0, 0, 1.0)
                
                # 물체 속성 설정
                self.sim.setObjectAlias(shape_handle, curr_shape_name)
                self.sim.setObjectPosition(shape_handle, -1, object_position)
                self.sim.setObjectOrientation(shape_handle, -1, object_orientation)
                self.sim.setShapeColor(
                    shape_handle, None, 
                    self.sim.colorcomponent_ambient_diffuse, 
                    object_color.tolist()
                )
                
                # 동적/반응 가능 설정
                self.sim.setObjectInt32Param(shape_handle, self.sim.shapeintparam_static, 0)
                self.sim.setObjectInt32Param(shape_handle, self.sim.shapeintparam_respondable, 1)
                self.sim.setShapeMass(shape_handle, 0.1)
                
                # Convex로 설정하여 시뮬레이션 속도 향상
                # Non-convex 물체는 시뮬레이션과 센서 체크를 느리게 만듦
                try:
                    # 방법 1: convex 속성 설정 시도
                    self.sim.setObjectInt32Param(shape_handle, self.sim.shapeintparam_convex, 1)
                    print(f'[OBJECTS] Set {curr_shape_name} as convex')
                except:
                    try:
                        # 방법 2: convexDecompose 사용 (더 정확)
                        # 복잡한 형상을 여러 개의 convex 부분으로 분해
                        self.sim.convexDecompose(shape_handle, 0, [0, 0, 0, 0, 0, 0], 0)
                        print(f'[OBJECTS] Applied convex decomposition to {curr_shape_name}')
                    except Exception as e:
                        # Convex 설정 실패 시 그냥 진행 (성능 저하는 있지만 작동은 함)
                        print(f'[OBJECTS] WARNING: Could not optimize {curr_shape_name} (will be slower)')
                
                self.object_handles.append(shape_handle)
                print(f'[OBJECTS] Added: {curr_shape_name} at ({object_position[0]:.2f}, {object_position[1]:.2f})')
                
            except Exception as e:
                print(f'[OBJECTS] Failed to add object {object_idx}: {e}')
                print(f'         Mesh file: {curr_mesh_file}')
        
        self.prev_obj_positions = []
        self.obj_positions = []
        print(f'[OBJECTS] Successfully added {len(self.object_handles)} objects.')
        
        # 물체 낙하 및 안정화 대기
        print('[OBJECTS] Running object drop simulation...')
        self.sim.setStepping(False)
        self.sim.startSimulation()
        time.sleep(2.0)  # 초기 낙하
        
        # 물체가 완전히 정지할 때까지 대기
        print('[OBJECTS] Waiting for objects to stabilize...')
        self.wait_for_objects_to_settle()
        
        # 로봇을 홈 포지션으로 이동 (카메라 촬영 시 로봇이 안 보이도록)
        self.move_to(self.home_position, 0)
        print('[OBJECTS] Robot moved to home position after object spawn.')
    
    def wait_for_objects_to_settle(self, max_wait_time=10.0, velocity_threshold=0.001):
        """
        물체들이 완전히 정지할 때까지 대기
        
        Args:
            max_wait_time: 최대 대기 시간 (초)
            velocity_threshold: 정지 판단 속도 임계값 (m/s)
        """
        start_time = time.time()
        check_interval = 0.2  # 0.2초마다 체크
        
        while (time.time() - start_time) < max_wait_time:
            all_settled = True
            
            for obj_handle in self.object_handles:
                try:
                    # 물체의 선속도 가져오기
                    lin_vel, ang_vel = self.sim.getObjectVelocity(obj_handle)
                    
                    # 속도 크기 계산
                    lin_speed = np.sqrt(lin_vel[0]**2 + lin_vel[1]**2 + lin_vel[2]**2)
                    ang_speed = np.sqrt(ang_vel[0]**2 + ang_vel[1]**2 + ang_vel[2]**2)
                    
                    # 선속도나 각속도가 임계값보다 크면 아직 움직이는 중
                    if lin_speed > velocity_threshold or ang_speed > velocity_threshold * 10:
                        all_settled = False
                        break
                        
                except Exception as e:
                    print(f'[OBJECTS] Warning: Could not check velocity for object {obj_handle}: {e}')
                    pass
            
            if all_settled:
                elapsed = time.time() - start_time
                print(f'[OBJECTS] All objects settled in {elapsed:.2f} seconds')
                return True
            
            time.sleep(check_interval)
        
        # 최대 시간 초과
        elapsed = time.time() - start_time
        print(f'[OBJECTS] Max wait time reached ({elapsed:.2f}s), proceeding anyway')
        return False
    
    def get_obj_positions(self):
        """모든 객체의 위치 가져오기"""
        obj_positions = []
        for object_handle in self.object_handles:
            try:
                object_position = self.sim.getObjectPosition(object_handle, -1)
                obj_positions.append(list(object_position))
            except:
                obj_positions.append([0, 0, 0])
        return obj_positions
    
    def get_obj_positions_and_orientations(self):
        """모든 객체의 위치와 방향 가져오기"""
        obj_positions = []
        obj_orientations = []
        for object_handle in self.object_handles:
            try:
                object_position = self.sim.getObjectPosition(object_handle, -1)
                object_orientation = self.sim.getObjectOrientation(object_handle, -1)
                obj_positions.append(list(object_position))
                obj_orientations.append(list(object_orientation))
            except:
                obj_positions.append([0, 0, 0])
                obj_orientations.append([0, 0, 0])
        return obj_positions, obj_orientations
    
    def check_sim(self):
        """
        시뮬레이션 안정성 확인
        
        엔드이펙터(진공 컵)가 workspace 범위 내에 있는지 확인합니다.
        범위를 벗어나면 시뮬레이션을 재시작합니다.
        
        단, Home 위치와 Place 위치에 있을 때는 체크하지 않습니다.
        """
        if not self.is_sim:
            return
        
        buffer_meters = 0.1
        
        try:
            # 진공 컵 위치 확인
            vacuum_position = self.sim.getObjectPosition(self.vacuumCup_handle, -1)
            
            # Home 위치에 있으면 체크 안 함 (±5cm 허용)
            home_threshold = 0.05
            is_at_home = (
                abs(vacuum_position[0] - self.home_position[0]) < home_threshold and
                abs(vacuum_position[1] - self.home_position[1]) < home_threshold and
                abs(vacuum_position[2] - self.home_position[2]) < home_threshold
            )
            
            # Place 위치 근처에 있으면 체크 안 함 (±15cm 허용, 이동 중 고려)
            place_threshold = 0.15
            is_at_place = (
                abs(vacuum_position[0] - self.place_position[0]) < place_threshold and
                abs(vacuum_position[1] - self.place_position[1]) < place_threshold and
                vacuum_position[2] < 0.5  # Z가 50cm 이하면 Place 작업 중
            )
            
            if is_at_home or is_at_place:
                # Home 또는 Place 위치에 있으면 안정적으로 간주
                return
            
            # workspace 범위 확인 (버퍼 포함)
            sim_ok = (
                vacuum_position[0] > self.workspace_limits[0][0] - buffer_meters and
                vacuum_position[0] < self.workspace_limits[0][1] + buffer_meters and
                vacuum_position[1] > self.workspace_limits[1][0] - buffer_meters and
                vacuum_position[1] < self.workspace_limits[1][1] + buffer_meters and
                vacuum_position[2] > self.workspace_limits[2][0] - buffer_meters and
                vacuum_position[2] < self.workspace_limits[2][1] + buffer_meters
            )
            
            if not sim_ok:
                print(f'[CHECK_SIM] Simulation unstable. VacuumCup at {vacuum_position}, outside workspace.')
                self.restart_sim()
                self.add_objects()
                
        except Exception as e:
            print(f'[CHECK_SIM] Error: {e}')
    
    def reposition_objects(self, workspace_limits=None):
        """
        물체들을 workspace_limits 내 랜덤 위치로 재배치
        
        Args:
            workspace_limits: 작업 공간 제한 (None이면 self.workspace_limits 사용)
        """
        if workspace_limits is None:
            workspace_limits = self.workspace_limits
        
        # 로봇을 안전한 위치로 이동
        self.move_to(self.home_position, 0)
        
        # workspace_limits에서 범위 추출
        x_min, x_max = workspace_limits[0]
        y_min, y_max = workspace_limits[1]
        margin = 0.05
        
        for object_handle in self.object_handles:
            try:
                # workspace 범위 내에서 랜덤 위치 생성
                drop_x = (x_max - x_min - 2*margin) * np.random.random_sample() + x_min + margin
                drop_y = (y_max - y_min - 2*margin) * np.random.random_sample() + y_min + margin
                # Plane 표면(0.001m) 위에서 낙하 (15cm 높이에서 떨어뜨림)
                drop_z = 0.001 + 0.15
                object_position = [drop_x, drop_y, drop_z]
                object_orientation = [
                    2 * np.pi * np.random.random_sample(),
                    2 * np.pi * np.random.random_sample(),
                    2 * np.pi * np.random.random_sample()
                ]
                
                self.sim.setObjectPosition(object_handle, -1, object_position)
                self.sim.setObjectOrientation(object_handle, -1, object_orientation)
                time.sleep(2)
                
            except Exception as e:
                print(f'[REPOSITION] Failed: {e}')
    
    def start_simulation(self):
        """시뮬레이션 시작"""
        if not self.is_sim:
            return False
        try:
            self.sim.setStepping(False)
            self.sim.startSimulation()
            time.sleep(1.0)
            return True
        except Exception as e:
            print(f"[SIM] Failed to start: {e}")
            return False
    
    def stop_simulation(self):
        """시뮬레이션 정지"""
        if not self.is_sim:
            return False
        try:
            self.sim.stopSimulation(True)
            time.sleep(0.5)
            return True
        except Exception as e:
            print(f"[SIM] Failed to stop: {e}")
            return False
    
    def get_object_info(self):
        """
        현재 시뮬레이션에 있는 모든 물체의 정보를 반환
        
        Returns:
            objects_info: [{'handle': int, 'name': str, 'position': [x,y,z], 'size': [x,y,z]}, ...]
        """
        objects_info = []
        
        for i, obj_handle in enumerate(self.object_handles):
            try:
                # 물체 이름
                obj_name = f'shape_{i:02d}'
                
                # 물체 위치 (월드 좌표)
                obj_position = self.sim.getObjectPosition(obj_handle, -1)
                
                # 물체 크기 (bounding box)
                # CoppeliaSim에서 shape의 크기를 얻는 방법: getObjectFloatParam
                # 또는 메시 정보에서 추정
                try:
                    # 물체의 바운딩 박스 계산
                    min_vals = self.sim.getObjectFloatParam(obj_handle, self.sim.objfloatparam_objbbox_min_x)
                    max_vals = self.sim.getObjectFloatParam(obj_handle, self.sim.objfloatparam_objbbox_max_x)
                    # 간단히 0.02~0.05m 범위로 가정 (블록 크기)
                    obj_size = [0.04, 0.04, 0.03]  # 기본값
                except:
                    obj_size = [0.04, 0.04, 0.03]  # fallback
                
                objects_info.append({
                    'handle': obj_handle,
                    'name': obj_name,
                    'position': obj_position,
                    'size': obj_size
                })
            except Exception as e:
                print(f'[OBJECT_INFO] Error getting info for object {i}: {e}')
        
        return objects_info

