#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Heightmap 좌표 변환 디버깅 스크립트

물체 2개를 생성하고 heightmap과 카메라 이미지에서 위치를 비교합니다.
- 선형 변환 vs Homography 변환 비교
- 시각화로 정확성 확인
"""

import sys
import os
import numpy as np
import cv2
import time

# 프로젝트 루트 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from robot_zmq_irb360 import RobotIRB360ZMQ as Robot
import utils

# 설정
workspace_limits = np.asarray([[-0.7, -0.3], [-0.2, 0.2], [0.001, 0.20]])
workspace_limit_place = np.asarray([[-0.5, 0.5], [-0.6, -0.4], [0.001, 0.20]])
heightmap_resolution = 0.002  # 2mm per pixel
num_objects = 2  # 테스트용 물체 2개

print('='*80)
print('Heightmap Coordinate Transformation Debug')
print('='*80)

# 1. 캘리브레이션 로드
print('\n[1] Loading calibration...')
H_world_to_pixel, H_pixel_to_world = utils.load_camera_calibration('camera_calibration.npy')

# 2. 로봇 초기화
print('\n[2] Initializing robot...')
robot = Robot(
    is_sim=True,
    obj_mesh_dir=os.path.abspath('objects/blocks'),
    num_obj=num_objects,
    workspace_limits=workspace_limits,
    workspace_limit_place=workspace_limit_place,
    is_testing=False,
    test_preset_cases=False,
    test_preset_file=None,
    place=True
)

# 3. 시뮬레이션 리셋 및 물체 추가
print('\n[3] Resetting simulation and adding objects...')
robot.restart_sim()
robot.add_objects()
time.sleep(1.0)  # 물체 안정화 대기

# 4. 물체 정보 가져오기
print('\n[4] Getting object information...')
objects_info = robot.get_object_info()

print(f'\nTotal objects: {len(objects_info)}')
for obj in objects_info:
    print(f"  {obj['name']}: pos=({obj['position'][0]:.3f}, {obj['position'][1]:.3f}, {obj['position'][2]:.3f})")

# 5. 카메라 데이터 취득
print('\n[5] Capturing camera data...')
robot.move_to(robot.home_position, 0)
time.sleep(0.5)
color_img, depth_img = robot.get_camera_data()
depth_img = depth_img * robot.cam_depth_scale

print(f'  Color image shape: {color_img.shape}')
print(f'  Depth image shape: {depth_img.shape}')
print(f'  Depth range: {np.min(depth_img):.3f}m ~ {np.max(depth_img):.3f}m')

# 6. Heightmap 생성
print('\n[6] Generating heightmap...')
color_heightmap, depth_heightmap = utils.get_heightmap(
    color_img, depth_img, 
    robot.cam_intrinsics, robot.cam_pose, 
    workspace_limits, heightmap_resolution
)
valid_depth_heightmap = depth_heightmap.copy()
valid_depth_heightmap[np.isnan(valid_depth_heightmap)] = 0

print(f'  Heightmap shape: {valid_depth_heightmap.shape}')
print(f'  Heightmap depth range: {np.min(valid_depth_heightmap)*1000:.1f}mm ~ {np.max(valid_depth_heightmap)*1000:.1f}mm')
print(f'  Non-zero pixels (>10mm): {np.sum(valid_depth_heightmap > 0.010)}')

# 7. 카메라 포즈 확인
print('\n[7] Camera information:')
print(f'  Camera position: {robot.cam_pose[:3, 3]}')
print(f'  Camera pose matrix:')
print(robot.cam_pose)

# 8. 좌표 변환 비교
print('\n[8] Coordinate transformation comparison:')
print('-'*80)

heightmap_height, heightmap_width = valid_depth_heightmap.shape

for i, obj in enumerate(objects_info):
    obj_name = obj['name']
    obj_pos = obj['position']
    
    print(f'\n{obj_name}:')
    print(f'  World position: ({obj_pos[0]:.3f}, {obj_pos[1]:.3f}, {obj_pos[2]:.3f})')
    
    # 선형 변환
    pixel_x_linear = int((obj_pos[0] - workspace_limits[0][0]) / heightmap_resolution)
    pixel_y_linear = int((obj_pos[1] - workspace_limits[1][0]) / heightmap_resolution)
    print(f'  Linear transform → pixel: ({pixel_x_linear}, {pixel_y_linear})', end='')
    
    if 0 <= pixel_x_linear < heightmap_width and 0 <= pixel_y_linear < heightmap_height:
        depth_linear = valid_depth_heightmap[pixel_y_linear, pixel_x_linear]
        print(f' → depth: {depth_linear*1000:.1f}mm')
    else:
        print(f' → OUT OF BOUNDS!')
        depth_linear = 0
    
    # Homography 변환
    pixel_x_homo, pixel_y_homo = utils.world_to_heightmap_pixel(
        obj_pos[0], obj_pos[1],
        workspace_limits, heightmap_resolution, H_world_to_pixel
    )
    print(f'  Homography transform → pixel: ({pixel_x_homo}, {pixel_y_homo})', end='')
    
    if 0 <= pixel_x_homo < heightmap_width and 0 <= pixel_y_homo < heightmap_height:
        depth_homo = valid_depth_heightmap[pixel_y_homo, pixel_x_homo]
        print(f' → depth: {depth_homo*1000:.1f}mm')
    else:
        print(f' → OUT OF BOUNDS!')
        depth_homo = 0
    
    # 비교
    pixel_diff = np.sqrt((pixel_x_linear - pixel_x_homo)**2 + (pixel_y_linear - pixel_y_homo)**2)
    print(f'  Pixel difference: {pixel_diff:.1f}px')
    
    expected_depth = obj_pos[2] + obj['size'][2]/2  # 물체 상단
    print(f'  Expected depth: ~{expected_depth*1000:.1f}mm')
    
    # 어느 쪽이 더 정확한가?
    error_linear = abs(depth_linear - expected_depth)
    error_homo = abs(depth_homo - expected_depth)
    
    if depth_linear > 0.010 and depth_homo > 0.010:
        if error_linear < error_homo:
            print(f'  ✓ Linear is more accurate (error: {error_linear*1000:.1f}mm vs {error_homo*1000:.1f}mm)')
        else:
            print(f'  ✓ Homography is more accurate (error: {error_homo*1000:.1f}mm vs {error_linear*1000:.1f}mm)')
    elif depth_linear > 0.010:
        print(f'  ✓ Only linear detected object ({depth_linear*1000:.1f}mm)')
    elif depth_homo > 0.010:
        print(f'  ✓ Only homography detected object ({depth_homo*1000:.1f}mm)')
    else:
        print(f'  ✗ Neither method detected object!')

# 9. 시각화
print('\n[9] Creating visualizations...')

# Heightmap 시각화 (depth를 컬러맵으로)
depth_vis = valid_depth_heightmap.copy()
depth_vis = (depth_vis / np.max(depth_vis) * 255).astype(np.uint8) if np.max(depth_vis) > 0 else depth_vis.astype(np.uint8)
depth_vis_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

# 물체 위치에 마커 표시
for i, obj in enumerate(objects_info):
    obj_pos = obj['position']
    
    # 선형 변환 (초록색)
    pixel_x_linear = int((obj_pos[0] - workspace_limits[0][0]) / heightmap_resolution)
    pixel_y_linear = int((obj_pos[1] - workspace_limits[1][0]) / heightmap_resolution)
    
    if 0 <= pixel_x_linear < heightmap_width and 0 <= pixel_y_linear < heightmap_height:
        cv2.circle(depth_vis_color, (pixel_x_linear, pixel_y_linear), 7, (0, 255, 0), 2)
        cv2.putText(depth_vis_color, f'L{i}', (pixel_x_linear+10, pixel_y_linear), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    # Homography 변환 (빨간색)
    pixel_x_homo, pixel_y_homo = utils.world_to_heightmap_pixel(
        obj_pos[0], obj_pos[1],
        workspace_limits, heightmap_resolution, H_world_to_pixel
    )
    
    if 0 <= pixel_x_homo < heightmap_width and 0 <= pixel_y_homo < heightmap_height:
        cv2.circle(depth_vis_color, (pixel_x_homo, pixel_y_homo), 7, (0, 0, 255), 2)
        cv2.putText(depth_vis_color, f'H{i}', (pixel_x_homo+10, pixel_y_homo-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

# 범례 추가
cv2.putText(depth_vis_color, 'Green = Linear Transform', (10, 20), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
cv2.putText(depth_vis_color, 'Red = Homography Transform', (10, 40), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

# 저장
output_dir = 'test/test_output'
os.makedirs(output_dir, exist_ok=True)

cv2.imwrite(f'{output_dir}/debug_heightmap_comparison.png', depth_vis_color)
cv2.imwrite(f'{output_dir}/debug_color_image.png', cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR))
cv2.imwrite(f'{output_dir}/debug_color_heightmap.png', cv2.cvtColor(color_heightmap, cv2.COLOR_RGB2BGR))

print(f'\n  Saved: {output_dir}/debug_heightmap_comparison.png')
print(f'  Saved: {output_dir}/debug_color_image.png')
print(f'  Saved: {output_dir}/debug_color_heightmap.png')

# 10. 카메라 이미지에도 물체 위치 표시
print('\n[10] Marking objects on camera image...')

# 카메라 이미지는 512x512
camera_img_marked = color_img.copy()

for i, obj in enumerate(objects_info):
    obj_pos = obj['position']
    
    # 월드 좌표 -> 카메라 이미지 픽셀
    if H_world_to_pixel is not None:
        world_xy = np.array([[obj_pos[0], obj_pos[1], 1.0]]).T
        pixel_h = H_world_to_pixel @ world_xy
        camera_pixel_x = int(pixel_h[0, 0] / pixel_h[2, 0])
        camera_pixel_y = int(pixel_h[1, 0] / pixel_h[2, 0])
        
        if 0 <= camera_pixel_x < 512 and 0 <= camera_pixel_y < 512:
            cv2.circle(camera_img_marked, (camera_pixel_x, camera_pixel_y), 10, (255, 0, 0), 2)
            cv2.putText(camera_img_marked, f'Obj{i}', (camera_pixel_x+12, camera_pixel_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

cv2.imwrite(f'{output_dir}/debug_camera_image_marked.png', cv2.cvtColor(camera_img_marked, cv2.COLOR_RGB2BGR))
print(f'  Saved: {output_dir}/debug_camera_image_marked.png')

print('\n' + '='*80)
print('Debug complete! Check images in test/test_output/')
print('='*80)

