"""
Camera Test Script - Depth 이미지와 좌표계 검증

이 스크립트는 다음을 수행합니다:
1. CoppeliaSim에서 카메라 이미지 (Color, Depth) 캡처
2. Depth 이미지 시각화 (colormap 적용)
3. 물체 위치와 depth 값 매칭 검증
4. 픽셀 좌표 → 월드 좌표 변환 검증
"""

import sys
import os
import time
import numpy as np
import cv2
from datetime import datetime

# 프로젝트 루트를 path에 추가
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# 현재 디렉토리(test)를 path에 추가
test_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, test_dir)

from robot_zmq_irb360 import RobotIRB360ZMQ
import utils


def world_to_pixel(world_pos, cam_intrinsics, cam_pose, img_height, img_width, H=None):
    """
    월드 좌표를 카메라 픽셀 좌표로 변환
    
    캘리브레이션 행렬(H)이 있으면 사용, 없으면 orthographic projection 근사
    
    Args:
        world_pos: 월드 좌표 [x, y, z]
        cam_intrinsics: 카메라 내부 파라미터 (3x3)
        cam_pose: 카메라 포즈 (4x4 변환 행렬)
        img_height: 이미지 높이
        img_width: 이미지 너비
        H: Homography 행렬 (3x3, optional)
        
    Returns:
        (pixel_x, pixel_y, cam_z): 픽셀 좌표 및 카메라 Z
    """
    # 월드 좌표를 homogeneous 좌표로 변환
    world_pos_homo = np.array([world_pos[0], world_pos[1], world_pos[2], 1.0])
    
    # 카메라 좌표계로 변환 (Z 값 계산용)
    cam_pose_inv = np.linalg.inv(cam_pose)
    cam_pos = np.dot(cam_pose_inv, world_pos_homo)
    cam_z = cam_pos[2]
    
    # Homography 행렬이 있으면 사용
    if H is not None:
        world_xy = np.array([[world_pos[0], world_pos[1], 1.0]]).T
        pixel_h = H @ world_xy
        pixel_x = int(pixel_h[0, 0] / pixel_h[2, 0])
        pixel_y = int(pixel_h[1, 0] / pixel_h[2, 0])
    else:
        # Orthographic projection 근사 (이전 방식)
        cam_x, cam_y = cam_pos[0], cam_pos[1]
        ortho_scale = 1.0
        pixel_x = int(img_width / 2 - cam_x * cam_intrinsics[0, 0] * ortho_scale)
        pixel_y = int(img_height / 2 - cam_y * cam_intrinsics[1, 1] * ortho_scale)
    
    # 이미지 범위 체크
    pixel_x = np.clip(pixel_x, 0, img_width - 1)
    pixel_y = np.clip(pixel_y, 0, img_height - 1)
    
    return pixel_x, pixel_y, cam_z


def visualize_depth(depth_img, min_depth=None, max_depth=None):
    """
    Depth 이미지를 컬러맵으로 시각화
    
    Args:
        depth_img: Depth 이미지 (H x W)
        min_depth: 최소 depth 값 (None이면 자동 계산)
        max_depth: 최대 depth 값 (None이면 자동 계산)
        
    Returns:
        depth_colormap: 컬러맵 적용된 이미지 (H x W x 3, BGR)
    """
    # NaN 제거
    valid_depth = depth_img.copy()
    valid_mask = ~np.isnan(valid_depth)
    
    if not np.any(valid_mask):
        return np.zeros((depth_img.shape[0], depth_img.shape[1], 3), dtype=np.uint8)
    
    # 범위 계산
    if min_depth is None:
        min_depth = np.min(valid_depth[valid_mask])
    if max_depth is None:
        max_depth = np.max(valid_depth[valid_mask])
    
    # 정규화 (0~255)
    depth_normalized = np.zeros_like(depth_img)
    if max_depth > min_depth:
        depth_normalized[valid_mask] = (valid_depth[valid_mask] - min_depth) / (max_depth - min_depth) * 255
    depth_normalized = depth_normalized.astype(np.uint8)
    
    # 컬러맵 적용 (JET: 파란색=가까움, 빨간색=멀리)
    depth_colormap = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
    
    # NaN 영역은 검은색으로
    depth_colormap[~valid_mask] = [0, 0, 0]
    
    return depth_colormap


def test_camera_capture():
    """카메라 캡처 및 좌표계 검증 테스트"""
    
    print("="*60)
    print("Camera Test - Depth Image & Coordinate System Verification")
    print("="*60)
    
    # 캘리브레이션 행렬 로드
    calib_file = 'camera_calibration.npy'
    H = None
    if os.path.exists(calib_file):
        H = np.load(calib_file)
        print(f"\n[CALIBRATION] Loaded homography matrix from {calib_file}")
        print("Using calibrated transformation")
    else:
        print(f"\n[CALIBRATION] No calibration file found")
        print("Using orthographic projection approximation")
    
    # 출력 디렉토리 생성
    output_dir = os.path.join('test', 'test_output')
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 로봇 초기화 (물체 3개)
    print("\n[1] Initializing robot...")
    workspace_limits = np.asarray([
        [-0.7, -0.3],   # x
        [-0.2, 0.2],    # y
        [0.001, 0.20]   # z
    ])
    
    # 프로젝트 루트 디렉토리 기준 경로
    obj_mesh_dir = os.path.join(project_root, 'objects', 'blocks')
    print(f"  Object directory: {obj_mesh_dir}")
    
    robot = RobotIRB360ZMQ(
        is_sim=True,
        obj_mesh_dir=obj_mesh_dir,
        num_obj=3,
        workspace_limits=workspace_limits,
        workspace_limit_place=None,
        is_testing=False,
        test_preset_cases=False,
        test_preset_file=None,
        place=False
    )
    
    print("Robot initialized successfully!")
    time.sleep(1.0)
    
    # 물체 추가
    print("\n[2] Adding objects to scene...")
    robot.add_objects()
    time.sleep(5.0)
    
    # 물체 위치 가져오기
    print("\n[3] Getting object positions...")
    obj_positions = robot.get_obj_positions()
    print(f"Number of objects: {len(obj_positions)}")
    for i, pos in enumerate(obj_positions):
        print(f"  Object {i}: X={pos[0]:.3f}, Y={pos[1]:.3f}, Z={pos[2]:.3f}")
    
    # 카메라 데이터 가져오기
    print("\n[4] Capturing camera data...")
    color_img, depth_img = robot.get_camera_data()
    
    if color_img is None or depth_img is None:
        print("[ERROR] Failed to get camera data!")
        return
    
    print(f"Color image shape: {color_img.shape}")
    print(f"Depth image shape: {depth_img.shape}")
    print(f"Depth range: {np.nanmin(depth_img):.3f}m ~ {np.nanmax(depth_img):.3f}m")
    
    # 카메라 정보
    print("\n[5] Camera information:")
    print(f"Camera position: {robot.cam_pose[0:3, 3]}")
    print(f"Camera intrinsics:\n{robot.cam_intrinsics}")
    
    # Depth 시각화
    print("\n[6] Visualizing depth image...")
    depth_colormap = visualize_depth(depth_img, min_depth=0.0, max_depth=0.3)
    
    # 물체 위치와 depth 매칭 검증
    print("\n[7] Verifying object positions vs depth values...")
    img_height, img_width = depth_img.shape
    
    # Color 이미지 복사 (마킹용)
    color_marked = color_img.copy()
    depth_marked = depth_colormap.copy()
    
    verification_results = []
    
    for i, obj_pos in enumerate(obj_positions):
        # 월드 좌표 → 픽셀 좌표
        pixel_x, pixel_y, cam_z = world_to_pixel(
            obj_pos, 
            robot.cam_intrinsics, 
            robot.cam_pose,
            img_height,
            img_width,
            H  # 캘리브레이션 행렬
        )
        
        # Depth 값 가져오기
        depth_value = depth_img[pixel_y, pixel_x]
        
        # 결과 출력
        print(f"\n  Object {i}:")
        print(f"    World pos: ({obj_pos[0]:.3f}, {obj_pos[1]:.3f}, {obj_pos[2]:.3f})")
        print(f"    Pixel pos: ({pixel_x}, {pixel_y})")
        print(f"    Camera Z: {cam_z:.3f}m")
        print(f"    Depth value: {depth_value:.3f}m")
        print(f"    Difference: {abs(cam_z - depth_value):.3f}m")
        
        verification_results.append({
            'object_id': i,
            'world_pos': obj_pos,
            'pixel_pos': (pixel_x, pixel_y),
            'cam_z': cam_z,
            'depth_value': depth_value,
            'error': abs(cam_z - depth_value)
        })
        
        # 이미지에 마킹 (십자 표시)
        cross_size = 10
        color = [(0, 0, 255), (0, 255, 0), (255, 0, 0)][i % 3]  # BGR
        
        # Color 이미지에 표시
        cv2.line(color_marked, (pixel_x - cross_size, pixel_y), (pixel_x + cross_size, pixel_y), color, 2)
        cv2.line(color_marked, (pixel_x, pixel_y - cross_size), (pixel_x, pixel_y + cross_size), color, 2)
        cv2.circle(color_marked, (pixel_x, pixel_y), 3, color, -1)
        cv2.putText(color_marked, f"#{i}", (pixel_x + 15, pixel_y - 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Depth 이미지에 표시
        cv2.line(depth_marked, (pixel_x - cross_size, pixel_y), (pixel_x + cross_size, pixel_y), (255, 255, 255), 2)
        cv2.line(depth_marked, (pixel_x, pixel_y - cross_size), (pixel_x, pixel_y + cross_size), (255, 255, 255), 2)
        cv2.circle(depth_marked, (pixel_x, pixel_y), 3, (255, 255, 255), -1)
        cv2.putText(depth_marked, f"#{i}", (pixel_x + 15, pixel_y - 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    # 이미지 저장
    print("\n[8] Saving images...")
    
    # Color 이미지 (RGB → BGR 변환)
    color_bgr = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)
    color_marked_bgr = cv2.cvtColor(color_marked, cv2.COLOR_RGB2BGR)
    
    cv2.imwrite(os.path.join(output_dir, f'color_{timestamp}.png'), color_bgr)
    cv2.imwrite(os.path.join(output_dir, f'color_marked_{timestamp}.png'), color_marked_bgr)
    print(f"  Saved: color_{timestamp}.png")
    print(f"  Saved: color_marked_{timestamp}.png")
    
    # Depth 이미지
    cv2.imwrite(os.path.join(output_dir, f'depth_{timestamp}.png'), depth_colormap)
    cv2.imwrite(os.path.join(output_dir, f'depth_marked_{timestamp}.png'), depth_marked)
    print(f"  Saved: depth_{timestamp}.png")
    print(f"  Saved: depth_marked_{timestamp}.png")
    
    # Raw depth 저장
    np.save(os.path.join(output_dir, f'depth_raw_{timestamp}.npy'), depth_img)
    print(f"  Saved: depth_raw_{timestamp}.npy")
    
    # Combined 이미지 (Color + Depth)
    combined = np.hstack([color_marked_bgr, depth_marked])
    cv2.imwrite(os.path.join(output_dir, f'combined_{timestamp}.png'), combined)
    print(f"  Saved: combined_{timestamp}.png")
    
    # 통계 정보 저장
    print("\n[9] Saving depth statistics...")
    stats_file = os.path.join(output_dir, f'depth_info_{timestamp}.txt')
    with open(stats_file, 'w') as f:
        f.write("="*60 + "\n")
        f.write("Depth Image & Coordinate System Verification Report\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"Timestamp: {timestamp}\n\n")
        
        f.write("Camera Information:\n")
        f.write(f"  Position: {robot.cam_pose[0:3, 3]}\n")
        f.write(f"  Intrinsics:\n")
        for row in robot.cam_intrinsics:
            f.write(f"    {row}\n")
        f.write("\n")
        
        f.write("Depth Statistics:\n")
        valid_depth = depth_img[~np.isnan(depth_img)]
        f.write(f"  Min: {np.min(valid_depth):.4f}m\n")
        f.write(f"  Max: {np.max(valid_depth):.4f}m\n")
        f.write(f"  Mean: {np.mean(valid_depth):.4f}m\n")
        f.write(f"  Std: {np.std(valid_depth):.4f}m\n")
        f.write(f"  Valid pixels: {len(valid_depth)}/{depth_img.size}\n\n")
        
        f.write("Object Verification Results:\n")
        for result in verification_results:
            f.write(f"\n  Object {result['object_id']}:\n")
            f.write(f"    World Position: ({result['world_pos'][0]:.3f}, {result['world_pos'][1]:.3f}, {result['world_pos'][2]:.3f})\n")
            f.write(f"    Pixel Position: {result['pixel_pos']}\n")
            f.write(f"    Camera Z: {result['cam_z']:.3f}m\n")
            f.write(f"    Depth Value: {result['depth_value']:.3f}m\n")
            f.write(f"    Error: {result['error']:.3f}m\n")
        
        # 에러 통계
        if len(verification_results) > 0:
            errors = [r['error'] for r in verification_results]
            f.write(f"\nError Statistics:\n")
            f.write(f"  Mean Error: {np.mean(errors):.4f}m\n")
            f.write(f"  Max Error: {np.max(errors):.4f}m\n")
        else:
            f.write(f"\nNo objects to verify.\n")
        
    print(f"  Saved: depth_info_{timestamp}.txt")
    
    # 결과 요약
    print("\n" + "="*60)
    print("VERIFICATION SUMMARY")
    print("="*60)
    
    if len(verification_results) > 0:
        avg_error = np.mean([r['error'] for r in verification_results])
        max_error = np.max([r['error'] for r in verification_results])
        print(f"Average Error: {avg_error:.4f}m")
        print(f"Maximum Error: {max_error:.4f}m")
        
        if avg_error < 0.01:  # 1cm 이내
            print("[PASS] Coordinate system is accurate!")
        elif avg_error < 0.05:  # 5cm 이내
            print("[WARNING] Coordinate system has small errors")
        else:
            print("[FAIL] Coordinate system has significant errors")
    else:
        print("[WARNING] No objects to verify. Please check object mesh directory.")
        print("  Expected: objects/blocks/*.obj")
    
    print(f"\nAll outputs saved to: {output_dir}/")
    print("="*60)


if __name__ == '__main__':
    try:
        test_camera_capture()
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()

