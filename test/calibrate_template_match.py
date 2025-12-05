"""
Template Matching for Marker Detection

ArUco 감지가 안되므로 템플릿 매칭으로 마커 중심 찾기
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

test_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, test_dir)

from robot_zmq_irb360 import RobotIRB360ZMQ


def find_marker_centers_template(image):
    """
    템플릿 매칭으로 검은색 사각형 마커 중심 찾기
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 검은색 사각형 영역 찾기 (더 낮은 임계값)
    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
    
    # Morphology로 정리
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    # Contours 찾기
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    print(f"  Total contours found: {len(contours)}")
    
    # 사각형 모양의 큰 영역 찾기 (마커 배경)
    marker_boxes = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # 적당한 크기의 사각형 (범위를 더 넓게)
        if area < 500 or area > 15000:
            continue
        
        # Bounding box
        x, y, w, h = cv2.boundingRect(contour)
        
        # 정사각형에 가까운지 확인
        aspect_ratio = float(w) / h
        if 0.6 < aspect_ratio < 1.4:  # 정사각형 (조금 더 관대하게)
            cx = x + w // 2
            cy = y + h // 2
            marker_boxes.append({
                'center': (cx, cy),
                'bbox': (x, y, w, h),
                'area': area,
                'aspect_ratio': aspect_ratio
            })
            print(f"  Found marker candidate: Center({cx}, {cy}), Size({w}x{h}), Area={area:.0f}, AR={aspect_ratio:.2f}")
    
    # 면적 순 정렬
    marker_boxes.sort(key=lambda m: m['area'], reverse=True)
    
    return marker_boxes, thresh


def calibrate_with_template():
    """템플릿 매칭 기반 캘리브레이션"""
    
    print("="*60)
    print("Template Matching-based Marker Calibration")
    print("="*60)
    
    # 출력 디렉토리
    output_dir = os.path.join('test', 'test_output')
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 로봇 초기화
    print("\n[1] Initializing robot...")
    workspace_limits = np.asarray([
        [-0.75, -0.25],
        [-0.25, 0.25],
        [0.001, 0.20]
    ])
    
    robot = RobotIRB360ZMQ(
        is_sim=True,
        obj_mesh_dir=None,
        num_obj=0,
        workspace_limits=workspace_limits,
        workspace_limit_place=None,
        is_testing=False,
        test_preset_cases=False,
        test_preset_file=None,
        place=False
    )
    
    print("Robot initialized!")
    time.sleep(1.0)
    
    # 마커 월드 좌표
    print("\n[2] Getting marker world positions...")
    marker_world_pos = {}
    
    for marker_idx in range(4):
        marker_name = f'arucoMarker[{marker_idx}]'
        try:
            handle = robot.sim.getObject(f'/{marker_name}')
            pos = robot.sim.getObjectPosition(handle, -1)
            marker_world_pos[marker_idx] = pos
            print(f"  Marker {marker_idx}: World({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f})")
        except Exception as e:
            print(f"  Marker {marker_idx}: NOT FOUND")
    
    if len(marker_world_pos) < 4:
        print(f"\n[ERROR] Found only {len(marker_world_pos)}/4 markers!")
        return
    
    # 카메라 이미지 캡처
    print("\n[3] Capturing camera image...")
    color_img, depth_img = robot.get_camera_data()
    
    if color_img is None:
        print("[ERROR] Failed to get camera image!")
        return
    
    # 마커 감지
    print("\n[4] Detecting markers via template matching...")
    color_bgr = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)
    marker_boxes, thresh = find_marker_centers_template(color_bgr)
    
    print(f"Found {len(marker_boxes)} marker candidates")
    
    if len(marker_boxes) < 4:
        print(f"[ERROR] Not enough markers detected ({len(marker_boxes)}/4)")
        
        # Debug 이미지
        debug_path = os.path.join(output_dir, f'template_debug_{timestamp}.png')
        cv2.imwrite(debug_path, thresh)
        print(f"Saved threshold image: {debug_path}")
        return
    
    # 상위 4개 선택
    top4_markers = marker_boxes[:4]
    
    print("\nTop 4 marker candidates:")
    for i, m in enumerate(top4_markers):
        print(f"  Candidate {i}: Center{m['center']}, BBox{m['bbox']}, Area={m['area']:.0f}")
    
    # 시각화
    img_marked = color_bgr.copy()
    for i, m in enumerate(top4_markers):
        cx, cy = m['center']
        x, y, w, h = m['bbox']
        
        cv2.rectangle(img_marked, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.circle(img_marked, (cx, cy), 5, (0, 0, 255), -1)
        cv2.putText(img_marked, f"C{i}", (cx+10, cy-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    
    marked_path = os.path.join(output_dir, f'template_detected_{timestamp}.png')
    cv2.imwrite(marked_path, img_marked)
    print(f"\nSaved detected markers: {marked_path}")
    
    # 위치별 매칭
    print("\n[5] Matching markers to world positions...")
    
    centers = [m['center'] for m in top4_markers]
    
    # X, Y로 정렬하여 매칭
    # 마커 0: (-0.65, -0.15) - 왼쪽 위
    # 마커 1: (-0.35, -0.15) - 오른쪽 위
    # 마커 2: (-0.35, 0.15) - 오른쪽 아래
    # 마커 3: (-0.65, 0.15) - 왼쪽 아래
    
    # 이미지 좌표계에서 분류
    img_center_x = 256
    img_center_y = 256
    
    left_markers = [c for c in centers if c[0] < img_center_x]
    right_markers = [c for c in centers if c[0] >= img_center_x]
    
    left_markers.sort(key=lambda p: p[1])   # Y로 정렬 (위->아래)
    right_markers.sort(key=lambda p: p[1])
    
    if len(left_markers) < 2 or len(right_markers) < 2:
        print("[ERROR] Cannot properly classify markers")
        print(f"Left: {len(left_markers)}, Right: {len(right_markers)}")
        return
    
    # 매칭 (상하 반전 수정)
    # Vision_sensor_ortho: 위쪽=+Y, 아래쪽=-Y
    # 이미지: 위쪽=작은Y, 아래쪽=큰Y
    # 따라서 World +Y는 이미지 작은Y(위쪽)에 매칭되어야 함
    marker_pixel_pos = {}
    marker_pixel_pos[3] = left_markers[0]    # 왼쪽 위 (World Y=+0.15)
    marker_pixel_pos[0] = left_markers[1]    # 왼쪽 아래 (World Y=-0.15)
    marker_pixel_pos[2] = right_markers[0]   # 오른쪽 위 (World Y=+0.15)
    marker_pixel_pos[1] = right_markers[1]   # 오른쪽 아래 (World Y=-0.15)
    
    print("\nMatched markers:")
    for marker_id in range(4):
        world_pos = marker_world_pos[marker_id]
        pixel_pos = marker_pixel_pos[marker_id]
        print(f"  Marker {marker_id}:")
        print(f"    World: ({world_pos[0]:.4f}, {world_pos[1]:.4f})")
        print(f"    Pixel: {pixel_pos}")
    
    # Homography 계산
    print("\n[6] Computing homography matrix...")
    
    world_pts = np.array([[marker_world_pos[i][0], marker_world_pos[i][1]] for i in range(4)], dtype=np.float32)
    pixel_pts = np.array([list(marker_pixel_pos[i]) for i in range(4)], dtype=np.float32)
    
    H, status = cv2.findHomography(world_pts, pixel_pts, method=0)
    
    print("\nHomography Matrix H:")
    print(H)
    
    H_inv = np.linalg.inv(H)
    
    # 검증
    print("\n[7] Verification:")
    total_error = 0
    max_error = 0
    
    for marker_id in range(4):
        world_pos = marker_world_pos[marker_id]
        world_xy = np.array([[world_pos[0], world_pos[1], 1.0]]).T
        
        pixel_h = H @ world_xy
        pred_px = int(pixel_h[0, 0] / pixel_h[2, 0])
        pred_py = int(pixel_h[1, 0] / pixel_h[2, 0])
        
        actual_px, actual_py = marker_pixel_pos[marker_id]
        error = np.sqrt((pred_px - actual_px)**2 + (pred_py - actual_py)**2)
        total_error += error
        max_error = max(max_error, error)
        
        print(f"Marker {marker_id}:")
        print(f"  Predicted pixel: ({pred_px}, {pred_py})")
        print(f"  Actual pixel: ({actual_px}, {actual_py})")
        print(f"  Error: {error:.2f} pixels")
    
    avg_error = total_error / 4
    print(f"\nAverage error: {avg_error:.2f} pixels")
    print(f"Maximum error: {max_error:.2f} pixels")
    
    # 저장
    print("\n[8] Saving calibration...")
    np.save('camera_calibration.npy', H)
    np.save('camera_calibration_inv.npy', H_inv)
    
    # 정보 파일
    with open('camera_calibration_info.txt', 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("Camera Calibration (Template Matching)\n")
        f.write("="*60 + "\n\n")
        
        for marker_id in range(4):
            world_pos = marker_world_pos[marker_id]
            pixel_pos = marker_pixel_pos[marker_id]
            f.write(f"Marker {marker_id}:\n")
            f.write(f"  World: ({world_pos[0]:.4f}, {world_pos[1]:.4f})\n")
            f.write(f"  Pixel: {pixel_pos}\n\n")
        
        f.write("Homography Matrix H:\n")
        for row in H:
            f.write(f"  {row}\n")
        
        f.write(f"\nAverage error: {avg_error:.2f} pixels\n")
        f.write(f"Maximum error: {max_error:.2f} pixels\n")
    
    print("  Saved: camera_calibration.npy")
    print("  Saved: camera_calibration_inv.npy")
    print("  Saved: camera_calibration_info.txt")
    
    print("\n" + "="*60)
    print("SUCCESS! Calibration complete")
    print("="*60)
    print(f"Average error: {avg_error:.2f} pixels")
    print("Next: Run python test/test_camera.py to verify")
    print("="*60)


if __name__ == '__main__':
    try:
        calibrate_with_template()
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()

