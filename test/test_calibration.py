"""
Camera Calibration Script - Dummy 기반 좌표 변환 캘리브레이션

Dummy2, Dummy1, Dummy3, Dummy4를 사용하여
월드 좌표 → 이미지 픽셀 좌표 변환 행렬을 계산합니다.
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


def test_calibration():
    """Dummy 기반 카메라 캘리브레이션"""
    
    print("="*60)
    print("Camera Calibration - Dummy-based Coordinate Transform")
    print("="*60)
    
    # 출력 디렉토리
    output_dir = os.path.join('test', 'test_output')
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 로봇 초기화 (물체 없이)
    print("\n[1] Initializing robot...")
    workspace_limits = np.asarray([
        [-0.7, -0.3],   # x
        [-0.2, 0.2],    # y
        [0.001, 0.20]   # z
    ])
    
    robot = RobotIRB360ZMQ(
        is_sim=True,
        obj_mesh_dir=None,  # 물체 없음
        num_obj=0,
        workspace_limits=workspace_limits,
        workspace_limit_place=None,
        is_testing=False,
        test_preset_cases=False,
        test_preset_file=None,
        place=False
    )
    
    print("Robot initialized successfully!")
    time.sleep(1.0)
    
    # Dummy 객체 찾기
    print("\n[2] Finding Dummy objects...")
    dummy_names = ['Dummy2', 'Dummy1', 'Dummy3', 'Dummy4']
    dummy_info = []
    
    for name in dummy_names:
        try:
            handle = robot.sim.getObject(f'/{name}')
            pos = robot.sim.getObjectPosition(handle, -1)
            dummy_info.append({
                'name': name,
                'handle': handle,
                'world_pos': pos
            })
            print(f"  {name}: handle={handle}, pos=({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")
        except Exception as e:
            print(f"  {name}: NOT FOUND - {e}")
    
    if len(dummy_info) < 4:
        print("\n[ERROR] Not all Dummy objects found!")
        print("Please create Dummy2, Dummy1, Dummy3, Dummy4 in CoppeliaSim")
        print("Recommended positions (workspace corners):")
        print("  Dummy2: (-0.7, -0.2, 0.05) - Left Top")
        print("  Dummy1: (-0.7, 0.2, 0.05) - Left Bottom")
        print("  Dummy3: (-0.3, -0.2, 0.05) - Right Top")
        print("  Dummy4: (-0.3, 0.2, 0.05) - Right Bottom")
        return
    
    # 카메라 이미지 캡처
    print("\n[3] Capturing camera image...")
    color_img, depth_img = robot.get_camera_data()
    
    if color_img is None:
        print("[ERROR] Failed to get camera image!")
        return
    
    print(f"Image shape: {color_img.shape}")
    
    # Dummy 위치에 마커 표시 (시각화용)
    color_marked = color_img.copy()
    img_height, img_width = color_img.shape[:2]
    
    print("\n[4] Marking Dummy positions on image...")
    print("\nPlease manually find pixel coordinates for each Dummy:")
    print("(Look at the saved image: calibration_marked_*.png)")
    print("")
    
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]  # RGB
    
    for i, info in enumerate(dummy_info):
        name = info['name']
        pos = info['world_pos']
        color = colors[i]
        
        # 예상 픽셀 위치 표시 (대략적)
        # 이 값들은 나중에 수동으로 확인하여 교체해야 함
        print(f"{name}:")
        print(f"  World: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")
        print(f"  Find this Dummy in the image (Color: {color})")
        print("")
    
    # 이미지 저장
    print("\n[5] Saving calibration image...")
    
    color_bgr = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)
    
    # Dummy 이름 표시 (참고용)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(color_bgr, "Find Dummy positions:", (10, 30), font, 0.7, (255, 255, 255), 2)
    for i, info in enumerate(dummy_info):
        y_pos = 60 + i * 30
        color_bgr_tuple = tuple(reversed(colors[i]))  # RGB to BGR
        cv2.putText(color_bgr, f"{info['name']}: ({info['world_pos'][0]:.2f}, {info['world_pos'][1]:.2f})", 
                    (10, y_pos), font, 0.5, color_bgr_tuple, 2)
    
    calib_img_path = os.path.join(output_dir, f'calibration_{timestamp}.png')
    cv2.imwrite(calib_img_path, color_bgr)
    print(f"  Saved: {calib_img_path}")
    
    # 좌표 기록 파일 생성
    coord_file = os.path.join(output_dir, f'calibration_coords_{timestamp}.txt')
    with open(coord_file, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("Camera Calibration - Dummy Coordinates\n")
        f.write("="*60 + "\n\n")
        f.write(f"Image: calibration_{timestamp}.png\n")
        f.write(f"Image size: {img_width} x {img_height}\n\n")
        
        f.write("Dummy World Coordinates:\n")
        for info in dummy_info:
            pos = info['world_pos']
            f.write(f"  {info['name']}: ({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f})\n")
        
        f.write("\n" + "="*60 + "\n")
        f.write("INSTRUCTIONS:\n")
        f.write("="*60 + "\n")
        f.write("1. Open calibration_*.png in an image viewer\n")
        f.write("2. Find each Dummy object in the image\n")
        f.write("3. Note the pixel coordinates (X, Y) for each Dummy\n")
        f.write("4. Fill in the coordinates below:\n\n")
        
        f.write("Pixel Coordinates (Fill this in):\n")
        for info in dummy_info:
            f.write(f"  {info['name']}: (???, ???)  # pixel_x, pixel_y\n")
        
        f.write("\n5. After filling, run:\n")
        f.write("   python test/compute_calibration.py\n")
        f.write("\n" + "="*60 + "\n")
    
    print(f"  Saved: {coord_file}")
    
    print("\n" + "="*60)
    print("NEXT STEPS:")
    print("="*60)
    print(f"1. Open: {calib_img_path}")
    print("2. Find each Dummy in the image")
    print(f"3. Fill pixel coordinates in: {coord_file}")
    print("4. Run: python test/compute_calibration.py")
    print("="*60)


if __name__ == '__main__':
    try:
        test_calibration()
    except KeyboardInterrupt:
        print("\n\nCalibration interrupted by user.")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()







