"""
Compute Calibration Matrix

calibration_coords_*.txt 파일에서 Dummy 좌표를 읽어
homography 행렬을 계산하고 저장합니다.
"""

import os
import numpy as np
import cv2
import re


def compute_calibration():
    """캘리브레이션 행렬 계산"""
    
    print("="*60)
    print("Computing Calibration Matrix")
    print("="*60)
    
    # 가장 최근 calibration_coords 파일 찾기
    output_dir = os.path.join('test', 'test_output')
    coord_files = [f for f in os.listdir(output_dir) if f.startswith('calibration_coords_')]
    
    if not coord_files:
        print("\n[ERROR] No calibration_coords_*.txt file found!")
        print("Please run: python test/test_calibration.py first")
        return
    
    # 가장 최근 파일 선택
    coord_files.sort(reverse=True)
    coord_file = os.path.join(output_dir, coord_files[0])
    
    print(f"\n[1] Reading: {coord_file}")
    
    # 파일 읽기
    with open(coord_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # World 좌표 파싱
    world_coords = {}
    world_section = re.search(r'Dummy World Coordinates:(.*?)=', content, re.DOTALL)
    if world_section:
        for line in world_section.group(1).strip().split('\n'):
            match = re.search(r'(Dummy\d+):\s*\(([-\d.]+),\s*([-\d.]+),\s*([-\d.]+)\)', line)
            if match:
                name = match.group(1)
                x, y, z = float(match.group(2)), float(match.group(3)), float(match.group(4))
                world_coords[name] = (x, y)
    
    print(f"\nWorld Coordinates:")
    for name, (x, y) in sorted(world_coords.items()):
        print(f"  {name}: ({x:.3f}, {y:.3f})")
    
    # Pixel 좌표 파싱
    pixel_coords = {}
    pixel_section = re.search(r'Pixel Coordinates \(Fill this in\):(.*?)5\. After', content, re.DOTALL)
    if pixel_section:
        for line in pixel_section.group(1).strip().split('\n'):
            match = re.search(r'(Dummy\d+):\s*\((\d+),\s*(\d+)\)', line)
            if match:
                name = match.group(1)
                px, py = int(match.group(2)), int(match.group(3))
                pixel_coords[name] = (px, py)
    
    if len(pixel_coords) < 4:
        print("\n[ERROR] Pixel coordinates not filled in!")
        print(f"Please edit {coord_file} and fill in the pixel coordinates.")
        print("\nExample:")
        print("  Dummy2: (50, 450)  # pixel_x, pixel_y")
        print("  Dummy1: (450, 450)")
        print("  Dummy3: (50, 50)")
        print("  Dummy4: (450, 50)")
        return
    
    print(f"\nPixel Coordinates:")
    for name, (px, py) in sorted(pixel_coords.items()):
        print(f"  {name}: ({px}, {py})")
    
    # 순서 맞춰 배열 생성
    dummy_names = ['Dummy2', 'Dummy1', 'Dummy3', 'Dummy4']
    
    world_pts = np.array([world_coords[name] for name in dummy_names], dtype=np.float32)
    pixel_pts = np.array([pixel_coords[name] for name in dummy_names], dtype=np.float32)
    
    print("\n[2] Computing homography matrix...")
    
    # Homography 계산
    H, status = cv2.findHomography(world_pts, pixel_pts, method=0)
    
    print("\nHomography Matrix H:")
    print(H)
    
    # 역방향 행렬 (pixel -> world)도 계산
    H_inv = np.linalg.inv(H)
    
    print("\nInverse Homography Matrix H_inv (pixel -> world):")
    print(H_inv)
    
    # 검증: world -> pixel -> world
    print("\n[3] Verification:")
    for name in dummy_names:
        world_xy = np.array([[world_coords[name][0], world_coords[name][1], 1.0]]).T
        pixel_h = H @ world_xy
        pred_px = int(pixel_h[0] / pixel_h[2])
        pred_py = int(pixel_h[1] / pixel_h[2])
        
        actual_px, actual_py = pixel_coords[name]
        error_px = abs(pred_px - actual_px)
        error_py = abs(pred_py - actual_py)
        
        print(f"{name}:")
        print(f"  World: ({world_coords[name][0]:.3f}, {world_coords[name][1]:.3f})")
        print(f"  Predicted pixel: ({pred_px}, {pred_py})")
        print(f"  Actual pixel: ({actual_px}, {actual_py})")
        print(f"  Error: ({error_px}, {error_py}) pixels")
    
    # 저장
    print("\n[4] Saving calibration matrices...")
    
    calib_file = 'camera_calibration.npy'
    calib_inv_file = 'camera_calibration_inv.npy'
    
    np.save(calib_file, H)
    np.save(calib_inv_file, H_inv)
    
    print(f"  Saved: {calib_file} (world -> pixel)")
    print(f"  Saved: {calib_inv_file} (pixel -> world)")
    
    # 정보 파일 저장
    info_file = 'camera_calibration_info.txt'
    with open(info_file, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("Camera Calibration Information\n")
        f.write("="*60 + "\n\n")
        
        f.write("World Coordinates:\n")
        for name in dummy_names:
            f.write(f"  {name}: ({world_coords[name][0]:.4f}, {world_coords[name][1]:.4f})\n")
        
        f.write("\nPixel Coordinates:\n")
        for name in dummy_names:
            f.write(f"  {name}: ({pixel_coords[name][0]}, {pixel_coords[name][1]})\n")
        
        f.write("\nHomography Matrix H (world XY -> pixel UV):\n")
        for row in H:
            f.write(f"  {row}\n")
        
        f.write("\nInverse Homography Matrix H_inv (pixel UV -> world XY):\n")
        for row in H_inv:
            f.write(f"  {row}\n")
    
    print(f"  Saved: {info_file}")
    
    print("\n" + "="*60)
    print("SUCCESS! Calibration matrices saved.")
    print("="*60)
    print("Next: Run python test/test_camera.py to verify accuracy")
    print("="*60)


if __name__ == '__main__':
    try:
        compute_calibration()
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()







