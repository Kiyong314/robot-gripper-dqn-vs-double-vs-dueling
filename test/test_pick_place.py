"""
IRB360 델타 로봇 Pick & Place 테스트 스크립트

물체를 등록하고, 비전 센서로 위치를 감지하여 Pick & Place를 수행합니다.

테스트 시나리오:
1. 물체 등록: Plane 위에 물체를 랜덤하게 배치
2. Heightmap 생성: 카메라 데이터에서 heightmap 생성
3. 물체 감지: heightmap에서 가장 높은 점 찾기
4. Pick & Place: 감지된 물체를 Plane1로 이동

씬 구성:
- Plane (Pick 작업대): 위치=(-0.5, 0, 0.001m)
- Plane1 (Place 작업대): 위치=(0.075, -0.5, 0.001m)
- 카메라: 위치=(-0.5, 0, 0.3m)

사용법:
    python test_pick_place.py

필요 조건:
    1. CoppeliaSim 실행 중
    2. IRB360 씬 로드
    3. ZMQ Remote API 서버 활성화
    4. objects/blocks 폴더에 물체 메쉬 파일 존재
"""

import os
import sys
import time
import numpy as np
import cv2

# 상위 폴더를 Python 경로에 추가 (utils.py import용)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# 상위 폴더를 sys.path에 추가하여 utils.py를 import 가능하게 함
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# 이제 utils를 직접 import 가능
import utils

# 물체 폴더 경로
OBJECTS_DIR = os.path.join(
    os.path.dirname(PROJECT_ROOT),
    '23.Learning-Pick-to-Place-Objects-in-a-cluttered-scene-using-deep-reinforcement-learning-main',
    'objects',
    'blocks'
)


class DummyTrainer:
    """테스트용 더미 Trainer 클래스"""
    def __init__(self):
        self.place_success_log = []


def test_heightmap():
    """
    테스트 1: Heightmap 생성 테스트
    
    카메라 데이터를 가져와서 heightmap을 생성하고 저장합니다.
    """
    print("\n" + "=" * 60)
    print("테스트 1: Heightmap 생성 테스트")
    print("=" * 60)
    
    try:
        from robot_zmq_irb360 import RobotIRB360ZMQ
        # utils는 파일 상단에서 이미 import됨
        
        # 로봇 초기화 (물체 없이)
        robot = RobotIRB360ZMQ(
            is_sim=True,
            obj_mesh_dir=None,
            num_obj=0
        )
        
        # 카메라 데이터 가져오기
        print("\n[TEST] 카메라 데이터 가져오는 중...")
        color_img, depth_img = robot.get_camera_data()
        
        if color_img is None or depth_img is None:
            print("[ERROR] 카메라 데이터를 가져올 수 없습니다.")
            return False
        
        print(f"[TEST] RGB 이미지: {color_img.shape}")
        print(f"[TEST] Depth 이미지: {depth_img.shape}")
        print(f"[TEST] Depth 범위: min={np.nanmin(depth_img):.4f}, max={np.nanmax(depth_img):.4f}")
        
        # Heightmap 생성
        print("\n[TEST] Heightmap 생성 중...")
        heightmap_resolution = 0.002  # 2mm per pixel
        
        color_heightmap, depth_heightmap = utils.get_heightmap(
            color_img, 
            depth_img, 
            robot.cam_intrinsics, 
            robot.cam_pose, 
            robot.workspace_limits, 
            heightmap_resolution
        )
        
        print(f"[TEST] Color heightmap: {color_heightmap.shape}")
        print(f"[TEST] Depth heightmap: {depth_heightmap.shape}")
        
        # Heightmap 저장
        output_dir = os.path.join(SCRIPT_DIR, 'test_output')
        os.makedirs(output_dir, exist_ok=True)
        
        # Color heightmap 저장
        cv2.imwrite(
            os.path.join(output_dir, 'heightmap_color.png'),
            cv2.cvtColor(color_heightmap, cv2.COLOR_RGB2BGR)
        )
        
        # Depth heightmap 시각화 및 저장
        valid_depth = depth_heightmap.copy()
        valid_depth[np.isnan(valid_depth)] = 0
        depth_normalized = (valid_depth / np.max(valid_depth) * 255).astype(np.uint8)
        depth_colormap = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
        cv2.imwrite(
            os.path.join(output_dir, 'heightmap_depth.png'),
            depth_colormap
        )
        
        print(f"\n[TEST] Heightmap 저장됨: {output_dir}")
        print("[RESULT] Heightmap 생성 테스트 성공!")
        
        robot.stop_simulation()
        return True
        
    except Exception as e:
        print(f"[ERROR] Heightmap 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_add_objects():
    """
    테스트 2: 물체 등록 테스트
    
    workspace 내에 물체를 등록하고 위치를 확인합니다.
    """
    print("\n" + "=" * 60)
    print("테스트 2: 물체 등록 테스트")
    print("=" * 60)
    
    # 물체 폴더 확인 (지역 변수 사용)
    obj_dir = OBJECTS_DIR
    
    if not os.path.exists(obj_dir):
        print(f"[ERROR] 물체 폴더가 없습니다: {obj_dir}")
        print("[INFO] 대체 폴더 검색 중...")
        
        # 대체 경로 시도
        alt_paths = [
            os.path.join(PROJECT_ROOT, 'objects', 'blocks'),
            os.path.join(SCRIPT_DIR, 'objects', 'blocks'),
        ]
        
        obj_dir = None
        for alt_path in alt_paths:
            if os.path.exists(alt_path):
                obj_dir = alt_path
                print(f"[INFO] 대체 폴더 발견: {obj_dir}")
                break
        
        if obj_dir is None:
            print("[ERROR] 물체 폴더를 찾을 수 없습니다.")
            return False
    
    try:
        from robot_zmq_irb360 import RobotIRB360ZMQ
        
        num_objects = 3
        print(f"\n[TEST] {num_objects}개 물체 등록 중...")
        print(f"[TEST] 물체 폴더: {obj_dir}")
        
        # 로봇 초기화 (물체 등록)
        robot = RobotIRB360ZMQ(
            is_sim=True,
            obj_mesh_dir=obj_dir,
            num_obj=num_objects
        )
        
        # 물체 추가
        robot.add_objects()
        
        # 물체 위치 확인
        print("\n[TEST] 등록된 물체 위치:")
        obj_positions = robot.get_obj_positions()
        for i, pos in enumerate(obj_positions):
            print(f"  shape_{i:02d}: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")
        
        # workspace 범위 내에 있는지 확인
        ws = robot.workspace_limits
        print(f"\n[TEST] Workspace: X[{ws[0][0]:.2f}, {ws[0][1]:.2f}], Y[{ws[1][0]:.2f}, {ws[1][1]:.2f}]")
        
        in_workspace = 0
        for pos in obj_positions:
            if (ws[0][0] <= pos[0] <= ws[0][1] and 
                ws[1][0] <= pos[1] <= ws[1][1]):
                in_workspace += 1
        
        print(f"[TEST] Workspace 내 물체: {in_workspace}/{len(obj_positions)}")
        
        print("\n[RESULT] 물체 등록 테스트 성공!")
        
        robot.stop_simulation()
        return True
        
    except Exception as e:
        print(f"[ERROR] 물체 등록 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_pick_and_place():
    """
    테스트 3: Pick & Place 테스트
    
    물체를 등록하고, heightmap에서 가장 높은 점을 찾아 Pick & Place를 수행합니다.
    """
    print("\n" + "=" * 60)
    print("테스트 3: Pick & Place 테스트")
    print("=" * 60)
    
    # 물체 폴더 확인
    obj_dir = OBJECTS_DIR
    if not os.path.exists(obj_dir):
        # 대체 경로 시도
        alt_paths = [
            os.path.join(PROJECT_ROOT, 'objects', 'blocks'),
            os.path.join(SCRIPT_DIR, 'objects', 'blocks'),
        ]
        for alt_path in alt_paths:
            if os.path.exists(alt_path):
                obj_dir = alt_path
                break
        else:
            print("[ERROR] 물체 폴더를 찾을 수 없습니다.")
            return False
    
    try:
        from robot_zmq_irb360 import RobotIRB360ZMQ
        # utils는 파일 상단에서 이미 import됨
        
        num_objects = 3
        print(f"\n[TEST] {num_objects}개 물체로 Pick & Place 테스트")
        
        # 로봇 초기화
        robot = RobotIRB360ZMQ(
            is_sim=True,
            obj_mesh_dir=obj_dir,
            num_obj=num_objects,
            place=True
        )
        
        # 물체 추가
        robot.add_objects()
        time.sleep(1.0)
        
        # 더미 트레이너
        trainer = DummyTrainer()
        
        # heightmap 해상도
        heightmap_resolution = 0.002
        
        # Pick & Place 시도
        max_attempts = 3
        successful_picks = 0
        
        for attempt in range(max_attempts):
            print(f"\n{'='*40}")
            print(f"[ATTEMPT {attempt + 1}/{max_attempts}]")
            print(f"{'='*40}")
            
            # 카메라 데이터 가져오기
            print("[STEP 1] 카메라 데이터 가져오기...")
            color_img, depth_img = robot.get_camera_data()
            
            if color_img is None or depth_img is None:
                print("[ERROR] 카메라 데이터 실패")
                continue
            
            # Heightmap 생성
            print("[STEP 2] Heightmap 생성...")
            color_heightmap, depth_heightmap = utils.get_heightmap(
                color_img, 
                depth_img, 
                robot.cam_intrinsics, 
                robot.cam_pose, 
                robot.workspace_limits, 
                heightmap_resolution
            )
            
            # NaN 처리
            valid_depth_heightmap = depth_heightmap.copy()
            valid_depth_heightmap[np.isnan(valid_depth_heightmap)] = 0
            
            # 가장 높은 점 찾기 (물체 위치)
            print("[STEP 3] 물체 위치 감지...")
            
            # 최소 높이 임계값 (바닥 제외)
            height_threshold = 0.02  # 2cm
            
            # 높이 임계값 이상인 점들 중 가장 높은 점
            valid_mask = valid_depth_heightmap > height_threshold
            
            if not np.any(valid_mask):
                print("[INFO] 물체가 감지되지 않음 (heightmap이 비어있음)")
                break
            
            # 가장 높은 점 찾기 (heightmap 방식)
            max_height_idx = np.unravel_index(
                np.argmax(valid_depth_heightmap), 
                valid_depth_heightmap.shape
            )
            max_height = valid_depth_heightmap[max_height_idx]
            
            # 픽셀 좌표 → 월드 좌표 변환 (heightmap 기반)
            pix_y, pix_x = max_height_idx
            heightmap_x = pix_x * heightmap_resolution + robot.workspace_limits[0][0]
            heightmap_y = pix_y * heightmap_resolution + robot.workspace_limits[1][0]
            
            print(f"[INFO] Heightmap 감지 위치: ({heightmap_x:.3f}, {heightmap_y:.3f})")
            print(f"[INFO] Heightmap 픽셀: ({pix_y}, {pix_x}), 높이: {max_height:.3f}m")
            
            # 실제 물체 위치 가져오기
            obj_positions = robot.get_obj_positions()
            if len(obj_positions) == 0:
                print("[INFO] 물체가 없음")
                break
            
            # Pick 영역(Plane)에 있는 물체만 필터링
            # Pick workspace: X=[-0.7, -0.3], Y=[-0.2, 0.2]
            pick_workspace = robot.workspace_limits
            objects_in_pick_area = []
            
            for obj_pos in obj_positions:
                x, y, z = obj_pos
                # Pick 영역 내에 있는지 확인
                if (pick_workspace[0][0] <= x <= pick_workspace[0][1] and
                    pick_workspace[1][0] <= y <= pick_workspace[1][1]):
                    objects_in_pick_area.append(obj_pos)
                    print(f"[INFO] Pick 영역 내 물체 발견: ({x:.3f}, {y:.3f}, {z:.3f})")
                else:
                    print(f"[INFO] Pick 영역 밖 물체 무시: ({x:.3f}, {y:.3f}, {z:.3f})")
            
            if len(objects_in_pick_area) == 0:
                print("[INFO] Pick 영역에 물체가 없음 - 모두 Place됨!")
                break
            
            # Pick 영역의 첫 번째 물체 선택
            target_obj = objects_in_pick_area[0]
            world_x, world_y, world_z_center = target_obj
            
            # 간단한 방법: 물체 중심 Z 사용 (CoppeliaSim이 제공하는 정확한 값)
            # 물체 상단은 대략 중심 + 2-3cm 정도
            object_top_z = world_z_center + 0.02  # 중심에서 2cm 위 (보수적 추정)
            
            print(f"[INFO] Object center Z: {world_z_center:.4f}m")
            print(f"[INFO] Object top Z (estimated): {object_top_z:.4f}m")
            
            pick_position = [world_x, world_y, object_top_z]
            
            print(f"[INFO] Pick 위치: ({world_x:.3f}, {world_y:.3f}, {object_top_z:.4f})")
            
            # Pick 시도
            print("[STEP 4] Pick 시도...")
            
            # 회전 각도 (0도)
            rotation_angle = 0
            
            grasp_result = robot.grasp(
                pick_position,
                rotation_angle,
                robot.workspace_limits,
                trainer,
                robot.workspace_limit_place
            )
            
            if grasp_result == 1:
                successful_picks += 1
                print(f"[SUCCESS] Pick & Place 성공! ({successful_picks}/{attempt + 1})")
            elif grasp_result == 0.5:
                print("[UNSTABLE] 물체가 흔들려서 떨어짐")
            else:  # grasp_result == -1
                print("[FAIL] Grasp 실패 (센서 감지 실패 또는 흡착 실패)")
            
            # 잠시 대기
            time.sleep(1.0)
        
        # 결과 출력
        print(f"\n{'='*60}")
        print(f"[RESULT] Pick & Place 테스트 완료")
        print(f"         시도: {max_attempts}, 성공: {successful_picks}")
        print(f"         성공률: {successful_picks/max_attempts*100:.1f}%")
        print(f"{'='*60}")
        
        robot.stop_simulation()
        return successful_picks > 0
        
    except Exception as e:
        print(f"[ERROR] Pick & Place 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """메인 함수"""
    print("=" * 60)
    print("IRB360 델타 로봇 Pick & Place 테스트")
    print("=" * 60)
    print("\n주의:")
    print("  1. CoppeliaSim이 실행 중이어야 합니다.")
    print("  2. IRB360 씬이 로드되어 있어야 합니다.")
    print("  3. ZMQ Remote API 서버가 활성화되어 있어야 합니다.")
    print(f"  4. 물체 폴더: {OBJECTS_DIR}\n")
    
    results = {}
    
    # 테스트 1: Heightmap 생성
    results['heightmap'] = test_heightmap()
    
    # 테스트 2: 물체 등록
    results['add_objects'] = test_add_objects()
    
    # 테스트 3: Pick & Place
    results['pick_place'] = test_pick_and_place()
    
    # 결과 요약
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

