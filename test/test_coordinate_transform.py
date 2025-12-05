#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
좌표 변환 함수 테스트 스크립트

캘리브레이션된 Homography 행렬을 사용한 좌표 변환을 테스트합니다.
CoppeliaSim 없이 독립적으로 실행 가능합니다.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import utils

# 테스트 파라미터 (main_irb360.py와 동일)
workspace_limits = np.asarray([[-0.8, -0.2], [-0.3, 0.3], [0.0, 0.3]])
heightmap_resolution = 0.002  # 2mm per pixel

print('=' * 80)
print('좌표 변환 함수 테스트')
print('=' * 80)

# 1. 캘리브레이션 행렬 로드
print('\n[1] 캘리브레이션 행렬 로드...')
H, H_inv = utils.load_camera_calibration('camera_calibration.npy')

if H is None or H_inv is None:
    print('[ERROR] 캘리브레이션 파일을 로드할 수 없습니다.')
    print('캘리브레이션 없이 선형 변환만 테스트합니다.')
else:
    print(f'[SUCCESS] Homography 행렬 로드 완료')
    print(f'H shape: {H.shape}')
    print(f'H_inv shape: {H_inv.shape}')

# 2. 테스트 케이스 설정
print('\n[2] 테스트 케이스 설정...')

# 테스트할 월드 좌표 (workspace 내부)
test_world_positions = [
    [-0.5, 0.0, 0.05],   # 중앙
    [-0.7, -0.2, 0.03],  # 왼쪽 아래
    [-0.3, 0.2, 0.07],   # 오른쪽 위
    [-0.5, -0.15, 0.04], # 왼쪽
    [-0.5, 0.15, 0.06],  # 오른쪽
]

print(f'테스트할 월드 좌표 개수: {len(test_world_positions)}')

# 3. 변환 테스트
print('\n[3] 변환 테스트 시작...')
print('-' * 80)

for i, world_pos in enumerate(test_world_positions):
    print(f'\n테스트 케이스 {i+1}:')
    print(f'  원본 월드 좌표: ({world_pos[0]:.3f}, {world_pos[1]:.3f}, {world_pos[2]:.3f})')
    
    # 월드 -> Heightmap 픽셀
    pixel_x, pixel_y = utils.world_to_heightmap_pixel(
        world_pos[0], 
        world_pos[1], 
        workspace_limits, 
        heightmap_resolution, 
        H
    )
    print(f'  변환된 Heightmap 픽셀: ({pixel_x}, {pixel_y})')
    
    # Heightmap 픽셀 -> 월드 (역변환)
    depth_value = world_pos[2] - workspace_limits[2][0]  # workspace z_bottom 기준 상대 높이
    reconstructed_pos = utils.heightmap_pixel_to_world(
        pixel_x,
        pixel_y,
        depth_value,
        workspace_limits,
        heightmap_resolution,
        H_inv
    )
    print(f'  역변환된 월드 좌표: ({reconstructed_pos[0]:.3f}, {reconstructed_pos[1]:.3f}, {reconstructed_pos[2]:.3f})')
    
    # 오차 계산
    error_x = abs(world_pos[0] - reconstructed_pos[0])
    error_y = abs(world_pos[1] - reconstructed_pos[1])
    error_z = abs(world_pos[2] - reconstructed_pos[2])
    total_error = np.sqrt(error_x**2 + error_y**2 + error_z**2)
    
    print(f'  오차 X: {error_x*1000:.2f}mm, Y: {error_y*1000:.2f}mm, Z: {error_z*1000:.2f}mm')
    print(f'  총 오차: {total_error*1000:.2f}mm')
    
    # 픽셀 범위 확인
    heightmap_width = int((workspace_limits[0][1] - workspace_limits[0][0]) / heightmap_resolution)
    heightmap_height = int((workspace_limits[1][1] - workspace_limits[1][0]) / heightmap_resolution)
    
    if 0 <= pixel_x < heightmap_width and 0 <= pixel_y < heightmap_height:
        print(f'  픽셀 범위: OK (0-{heightmap_width-1}, 0-{heightmap_height-1})')
    else:
        print(f'  [WARNING] 픽셀이 범위를 벗어났습니다!')

print('\n' + '=' * 80)
print('테스트 완료')
print('=' * 80)

# 4. 캘리브레이션 유무 비교
if H is not None and H_inv is not None:
    print('\n[4] 캘리브레이션 유무 비교...')
    print('-' * 80)
    
    test_pos = [-0.5, 0.0, 0.05]
    print(f'\n테스트 좌표: ({test_pos[0]:.3f}, {test_pos[1]:.3f}, {test_pos[2]:.3f})')
    
    # Heightmap 픽셀 계산 (선형 변환)
    pixel_x_linear = int((test_pos[0] - workspace_limits[0][0]) / heightmap_resolution)
    pixel_y_linear = int((test_pos[1] - workspace_limits[1][0]) / heightmap_resolution)
    print(f'\n선형 변환 픽셀: ({pixel_x_linear}, {pixel_y_linear})')
    
    # 선형 변환으로 역변환
    depth_value = test_pos[2] - workspace_limits[2][0]
    pos_linear = utils.heightmap_pixel_to_world(
        pixel_x_linear,
        pixel_y_linear,
        depth_value,
        workspace_limits,
        heightmap_resolution,
        None  # 캘리브레이션 없음
    )
    print(f'선형 역변환 좌표: ({pos_linear[0]:.3f}, {pos_linear[1]:.3f}, {pos_linear[2]:.3f})')
    
    # 캘리브레이션 변환
    pixel_x_calib, pixel_y_calib = utils.world_to_heightmap_pixel(
        test_pos[0], 
        test_pos[1], 
        workspace_limits, 
        heightmap_resolution, 
        H
    )
    print(f'\n캘리브레이션 변환 픽셀: ({pixel_x_calib}, {pixel_y_calib})')
    
    # 캘리브레이션 역변환
    pos_calib = utils.heightmap_pixel_to_world(
        pixel_x_calib,
        pixel_y_calib,
        depth_value,
        workspace_limits,
        heightmap_resolution,
        H_inv
    )
    print(f'캘리브레이션 역변환 좌표: ({pos_calib[0]:.3f}, {pos_calib[1]:.3f}, {pos_calib[2]:.3f})')
    
    # 픽셀 차이
    pixel_diff_x = abs(pixel_x_linear - pixel_x_calib)
    pixel_diff_y = abs(pixel_y_linear - pixel_y_calib)
    print(f'\n픽셀 차이: X={pixel_diff_x}, Y={pixel_diff_y}')
    
    # 좌표 차이
    coord_diff_x = abs(pos_linear[0] - pos_calib[0])
    coord_diff_y = abs(pos_linear[1] - pos_calib[1])
    print(f'좌표 차이: X={coord_diff_x*1000:.2f}mm, Y={coord_diff_y*1000:.2f}mm')
    
    print('\n' + '=' * 80)






