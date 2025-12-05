"""
IRB360 델타 로봇 + BaxterVacuumCup 테스트 스크립트

CoppeliaSim 4.10 ZMQ API를 사용하여 IRB360 로봇을 테스트합니다.

씬 구성:
- IRB360 베이스: Z=1.200m
- Plane (Pick 작업대): 위치=(-0.5, 0, 0.001m)
- Plane1 (Place 작업대): 위치=(0.075, -0.5, 0.001m)
- VacuumCup 초기: Z=0.389m
- ikTarget 초기: Z=0.455m

테스트 시나리오:
1. 연결 테스트: CoppeliaSim 연결 및 핸들 확인
2. 이동 테스트: move_to 함수로 로봇 이동
3. 진공 테스트: 진공 컵 활성화/비활성화
4. Grasp/Place 테스트: 물체 집기 및 놓기

사용법:
    python test_irb360.py [--test TEST_NAME]
    
    TEST_NAME:
        connection  - 연결 테스트만
        move        - 이동 테스트만
        vacuum      - 진공 테스트만
        grasp       - Grasp/Place 테스트만
        all         - 모든 테스트 (기본값)
"""

import time
import sys
import os
import argparse
import numpy as np


class DummyTrainer:
    """테스트용 더미 Trainer 클래스"""
    def __init__(self):
        self.place_success_log = []


def test_connection():
    """테스트 1: CoppeliaSim 연결 및 핸들 확인"""
    print("\n" + "=" * 60)
    print("테스트 1: CoppeliaSim 연결 테스트")
    print("=" * 60)
    
    try:
        from coppeliasim_zmqremoteapi_client import RemoteAPIClient
        
        print("[TEST] Connecting to CoppeliaSim...")
        client = RemoteAPIClient()
        sim = client.require('sim')
        print("[TEST] Connected successfully!")
        
        # 시뮬레이션 상태 확인
        sim_state = sim.getSimulationState()
        state_names = {
            sim.simulation_stopped: "STOPPED",
            sim.simulation_paused: "PAUSED", 
            sim.simulation_advancing_running: "RUNNING",
        }
        state_name = state_names.get(sim_state, f"UNKNOWN({sim_state})")
        print(f"[TEST] Simulation state: {state_name}")
        
        print("\n[TEST] Checking IRB360 handles...")
        
        handles = {
            'irb360': '/irb360',
            'ikTarget': '/irb360/cartesianX/cartesianY/cartesianZ/ikTarget',
            'BaxterVacuumCup': '/BaxterVacuumCup',
            'loopClosureDummy1': '/BaxterVacuumCup/loopClosureDummy1',
            'Vision_sensor_ortho': '/Vision_sensor_ortho',
        }
        
        all_ok = True
        for name, path in handles.items():
            try:
                handle = sim.getObject(path)
                print(f"  [OK] {name}: {handle}")
            except Exception as e:
                print(f"  [FAIL] {name}: {e}")
                all_ok = False
        
        irb360_handle = sim.getObject('/irb360')
        base_pos = sim.getObjectPosition(irb360_handle, -1)
        print(f"\n[TEST] IRB360 base position: [{base_pos[0]:.3f}, {base_pos[1]:.3f}, {base_pos[2]:.3f}]")
        
        vacuum_handle = sim.getObject('/BaxterVacuumCup')
        vacuum_pos = sim.getObjectPosition(vacuum_handle, -1)
        print(f"[TEST] VacuumCup position: [{vacuum_pos[0]:.3f}, {vacuum_pos[1]:.3f}, {vacuum_pos[2]:.3f}]")
        
        if all_ok:
            print("\n[RESULT] 연결 테스트 성공!")
        else:
            print("\n[RESULT] 일부 핸들을 찾을 수 없습니다.")
        
        return all_ok
        
    except Exception as e:
        print(f"[ERROR] Connection test failed: {e}")
        return False


def test_move():
    """테스트 2: 이동 테스트"""
    print("\n" + "=" * 60)
    print("테스트 2: 로봇 이동 테스트")
    print("=" * 60)
    
    try:
        from coppeliasim_zmqremoteapi_client import RemoteAPIClient
        
        client = RemoteAPIClient()
        sim = client.require('sim')
        
        # 핸들을 시뮬레이션 시작 전에 먼저 가져오기 (중요!)
        # 시뮬레이션 시작 후에는 IRB360 스크립트의 moveToConfig가 블로킹되어 응답 불가
        print("[TEST] 핸들 가져오는 중...")
        cartesianX = sim.getObject('/irb360/cartesianX')
        cartesianY = sim.getObject('/irb360/cartesianX/cartesianY')
        cartesianZ = sim.getObject('/irb360/cartesianX/cartesianY/cartesianZ')
        vacuum_handle = sim.getObject('/BaxterVacuumCup')
        print("[TEST] 핸들 획득 완료")
        
        # 시뮬레이션 시작
        sim_state = sim.getSimulationState()
        if sim_state == sim.simulation_stopped:
            print("[TEST] 시뮬레이션 시작 중...")
            sim.setStepping(False)
            sim.startSimulation()
            time.sleep(2.0)  # IRB360 스크립트 초기화 대기
            print("[TEST] 시뮬레이션 시작됨")
        
        current_x = sim.getJointPosition(cartesianX)
        current_y = sim.getJointPosition(cartesianY)
        current_z = sim.getJointPosition(cartesianZ)
        print(f"[TEST] Current: X={current_x:.3f}, Y={current_y:.3f}, Z={current_z:.3f}")
        
        vacuum_pos_before = sim.getObjectPosition(vacuum_handle, -1)
        print(f"[TEST] VacuumCup 초기: {vacuum_pos_before}")
        
        test_positions = [
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [0.1, 0.1, 0.0],
            [0.0, 0.0, -0.05],
            [0.0, 0.0, 0.0],
        ]
        
        print("\n[TEST] 이동 테스트 시작...")
        for i, pos in enumerate(test_positions):
            print(f"\n[TEST] 이동 {i+1}: {pos}")
            
            # 조인트 위치 설정 (targetPosition 사용)
            sim.setJointTargetPosition(cartesianX, pos[0])
            sim.setJointTargetPosition(cartesianY, pos[1])
            sim.setJointTargetPosition(cartesianZ, pos[2])
            
            time.sleep(0.5)  # 이동 완료 대기 (0.5초로 단축)
            
            new_x = sim.getJointPosition(cartesianX)
            new_y = sim.getJointPosition(cartesianY)
            new_z = sim.getJointPosition(cartesianZ)
            print(f"[TEST] 조인트: X={new_x:.3f}, Y={new_y:.3f}, Z={new_z:.3f}")
            
            vacuum_pos = sim.getObjectPosition(vacuum_handle, -1)
            print(f"[TEST] VacuumCup: [{vacuum_pos[0]:.3f}, {vacuum_pos[1]:.3f}, {vacuum_pos[2]:.3f}]")
        
        print("\n[RESULT] 이동 테스트 완료!")
        return True
        
    except Exception as e:
        print(f"[ERROR] Move test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_vacuum():
    """테스트 3: 진공 컵 테스트"""
    print("\n" + "=" * 60)
    print("테스트 3: 진공 컵 테스트")
    print("=" * 60)
    
    try:
        from coppeliasim_zmqremoteapi_client import RemoteAPIClient
        
        client = RemoteAPIClient()
        sim = client.require('sim')
        
        # 핸들을 시뮬레이션 시작 전에 먼저 가져오기 (중요!)
        print("[TEST] 핸들 가져오는 중...")
        vacuum_handle = sim.getObject('/BaxterVacuumCup')
        loopDummy1 = sim.getObject('/BaxterVacuumCup/loopClosureDummy1')
        print("[TEST] 핸들 획득 완료")
        
        # 시뮬레이션 시작 (진공 스크립트가 동작하려면 시뮬레이션이 실행 중이어야 함)
        sim_state = sim.getSimulationState()
        if sim_state == sim.simulation_stopped:
            print("[TEST] 시뮬레이션 시작 중...")
            sim.setStepping(False)
            sim.startSimulation()
            time.sleep(2.0)  # IRB360 스크립트 초기화 대기
            print("[TEST] 시뮬레이션 시작됨")
        
        print("[TEST] 초기 상태 확인...")
        parent = sim.getObjectParent(loopDummy1)
        print(f"[TEST] loopClosureDummy1 parent: {parent}")
        print(f"[TEST] VacuumCup handle: {vacuum_handle}")
        attached = (parent != vacuum_handle)
        print(f"[TEST] 물체 흡착 상태: {attached}")
        
        print("\n[TEST] 진공 활성화...")
        sim.setIntProperty(sim.handle_scene, 'signal.BaxterVacuumCup_active', 1)
        time.sleep(0.5)  # 0.5초로 단축
        
        parent = sim.getObjectParent(loopDummy1)
        attached = (parent != vacuum_handle)
        print(f"[TEST] 진공 ON 후: {attached}")
        if attached:
            try:
                parent_name = sim.getObjectAlias(parent, 0)
                print(f"[TEST] 흡착된 물체: {parent_name}")
            except:
                pass
        
        print("\n[TEST] 진공 비활성화...")
        sim.setIntProperty(sim.handle_scene, 'signal.BaxterVacuumCup_active', 0)
        time.sleep(0.5)  # 0.5초로 단축
        
        parent = sim.getObjectParent(loopDummy1)
        attached = (parent != vacuum_handle)
        print(f"[TEST] 진공 OFF 후: {attached}")
        
        print("\n[NOTE] 물체가 없으면 흡착 상태는 항상 False입니다.")
        print("[NOTE] 실제 물체 흡착 테스트는 test_grasp를 사용하세요.")
        
        print("\n[RESULT] 진공 컵 테스트 완료!")
        return True
        
    except Exception as e:
        print(f"[ERROR] Vacuum test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_grasp():
    """테스트 4: Grasp/Place 테스트 (RobotIRB360ZMQ 클래스 사용)"""
    print("\n" + "=" * 60)
    print("테스트 4: Grasp/Place 테스트")
    print("=" * 60)
    
    try:
        from robot_zmq_irb360 import RobotIRB360ZMQ
        
        print("\n[TEST] RobotIRB360ZMQ 인스턴스 생성...")
        
        robot = RobotIRB360ZMQ(
            is_sim=True,
            obj_mesh_dir=None,
            num_obj=0,
            workspace_limits=None,
            workspace_limit_place=None,
            place=True
        )
        
        print("[TEST] 인스턴스 생성 완료!")
        
        # 현재 위치 확인
        current_pos = robot.get_current_position()
        print(f"[TEST] 현재 VacuumCup 위치: [{current_pos[0]:.3f}, {current_pos[1]:.3f}, {current_pos[2]:.3f}]")
        
        trainer = DummyTrainer()
        
        # 이동 테스트 (짧은 거리로)
        print("\n[TEST] 이동 테스트...")
        
        # 홈 위치로 이동
        print("[TEST] 1. 홈 위치로 이동")
        robot.move_to(robot.home_position, 0)
        pos = robot.get_current_position()
        print(f"[TEST]    위치: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
        
        # Place 위치로 이동
        print("[TEST] 2. Place 위치로 이동")
        robot.move_to(robot.place_position, 0)
        pos = robot.get_current_position()
        print(f"[TEST]    위치: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
        
        # 다시 홈으로
        print("[TEST] 3. 홈으로 복귀")
        robot.move_to(robot.home_position, 0)
        pos = robot.get_current_position()
        print(f"[TEST]    위치: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
        
        # 진공 테스트
        print("\n[TEST] 진공 테스트...")
        robot.activate_vacuum()
        attached = robot.check_vacuum_attached()
        print(f"[TEST] 흡착 상태: {attached}")
        robot.deactivate_vacuum()
        
        print("\n[NOTE] 물체가 없으면 흡착 상태는 항상 False입니다.")
        print("[NOTE] 실제 Pick&Place는 물체가 있는 씬에서 테스트하세요.")
        
        print("\n[RESULT] Grasp/Place 테스트 완료!")
        return True
        
    except Exception as e:
        print(f"[ERROR] Grasp test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='IRB360 로봇 테스트')
    parser.add_argument('--test', type=str, default='all',
                       choices=['connection', 'move', 'vacuum', 'grasp', 'all'],
                       help='실행할 테스트 (기본값: all)')
    args = parser.parse_args()
    
    print("=" * 60)
    print("IRB360 델타 로봇 + BaxterVacuumCup 테스트")
    print("=" * 60)
    print("\n주의:")
    print("  1. CoppeliaSim이 실행 중이어야 합니다.")
    print("  2. IRB360 씬이 로드되어 있어야 합니다.")
    print("  3. ZMQ Remote API 서버가 활성화되어 있어야 합니다.")
    print("     (Add-ons → ZMQ remote API server)\n")
    
    results = {}
    
    if args.test in ['connection', 'all']:
        results['connection'] = test_connection()
    
    if args.test in ['move', 'all']:
        results['move'] = test_move()
    
    if args.test in ['vacuum', 'all']:
        results['vacuum'] = test_vacuum()
    
    if args.test in ['grasp', 'all']:
        results['grasp'] = test_grasp()
    
    print("\n" + "=" * 60)
    print("테스트 결과 요약")
    print("=" * 60)
    for test_name, success in results.items():
        status = "성공" if success else "실패"
        print(f"  {test_name}: {status}")
    
    all_passed = all(results.values())
    print("\n" + ("모든 테스트 통과!" if all_passed else "일부 테스트 실패"))
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())

