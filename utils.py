#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
유틸리티 함수 모음

원본: D:/1.github/23.Learning-Pick-to-Place-Objects-in-a-cluttered-scene-using-deep-reinforcement-learning-main/utils.py

주요 함수:
- get_heightmap(): RGB-D 이미지에서 heightmap 생성
- get_pointcloud(): depth 이미지에서 3D 포인트 클라우드 생성
- euler2rotm(): 오일러 각도 → 회전 행렬 변환
- get_change_value(): depth 변화량 계산

수정사항 (Python 3 호환):
- Python 2 나눗셈 → Python 3 정수 나눗셈(//)
- tostring() → tobytes()
- NLLLoss2d → NLLLoss (deprecated)
"""

import struct
import math
import numpy as np
import cv2
import torch.nn as nn
import torch.nn.functional as F


def load_camera_calibration(calibration_file='camera_calibration.npy'):
    """
    캘리브레이션된 Homography 행렬 로드
    
    Args:
        calibration_file: 캘리브레이션 파일 경로
        
    Returns:
        H: Homography 행렬 (world XY -> pixel UV), 없으면 None
        H_inv: 역 Homography 행렬 (pixel UV -> world XY), 없으면 None
    """
    import os
    
    if not os.path.exists(calibration_file):
        print(f'[CALIBRATION] File not found: {calibration_file}')
        return None, None
    
    try:
        H = np.load(calibration_file)
        H_inv = np.load(calibration_file.replace('.npy', '_inv.npy'))
        print(f'[CALIBRATION] Loaded homography matrix from {calibration_file}')
        return H, H_inv
    except Exception as e:
        print(f'[CALIBRATION] Failed to load: {e}')
        return None, None


def heightmap_pixel_to_world(pixel_x, pixel_y, depth_value, workspace_limits, 
                              heightmap_resolution, H_inv=None):
    """
    Heightmap 픽셀 좌표를 월드 XYZ 좌표로 변환
    
    Args:
        pixel_x: Heightmap X 픽셀 좌표
        pixel_y: Heightmap Y 픽셀 좌표
        depth_value: Heightmap depth 값 (workspace z_bottom 기준 상대 높이)
        workspace_limits: [[x_min, x_max], [y_min, y_max], [z_min, z_max]]
        heightmap_resolution: Heightmap 해상도 (미터/픽셀)
        H_inv: 역 Homography 행렬 (pixel UV -> world XY), None이면 선형 변환
        
    Returns:
        [world_x, world_y, world_z]: 월드 좌표
    """
    # Z 좌표는 동일 (depth + workspace z_bottom)
    world_z = depth_value + workspace_limits[2][0]
    
    if H_inv is not None:
        # 캘리브레이션된 Homography 역변환 사용
        # Heightmap 픽셀 -> 카메라 이미지 픽셀 (스케일 조정 필요)
        # 주의: Heightmap과 카메라 이미지 해상도가 다를 수 있음
        
        # 카메라 이미지 크기 가정 (Vision_sensor_ortho: 512x512)
        camera_img_width = 512
        camera_img_height = 512
        
        # Heightmap 크기 계산
        heightmap_width = int((workspace_limits[0][1] - workspace_limits[0][0]) / heightmap_resolution)
        heightmap_height = int((workspace_limits[1][1] - workspace_limits[1][0]) / heightmap_resolution)
        
        # Heightmap 픽셀 좌표를 카메라 이미지 픽셀 좌표로 스케일링
        # Heightmap의 (0, 0)은 workspace 왼쪽 아래 (x_min, y_min)
        # 카메라 이미지와 workspace 범위가 일치한다고 가정
        scale_x = camera_img_width / heightmap_width
        scale_y = camera_img_height / heightmap_height
        
        camera_pixel_x = pixel_x * scale_x
        camera_pixel_y = pixel_y * scale_y
        
        # Homography 역변환: 카메라 픽셀 -> 월드 XY
        pixel_uv = np.array([[camera_pixel_x, camera_pixel_y, 1.0]]).T
        world_xy_h = H_inv @ pixel_uv
        world_x = world_xy_h[0, 0] / world_xy_h[2, 0]
        world_y = world_xy_h[1, 0] / world_xy_h[2, 0]
        
        return [world_x, world_y, world_z]
    else:
        # Fallback: 기존 선형 변환
        world_x = pixel_x * heightmap_resolution + workspace_limits[0][0]
        world_y = pixel_y * heightmap_resolution + workspace_limits[1][0]
        return [world_x, world_y, world_z]


def world_to_heightmap_pixel(world_x, world_y, workspace_limits, 
                              heightmap_resolution, H=None):
    """
    월드 XY 좌표를 Heightmap 픽셀 좌표로 변환 (시각화용)
    
    Args:
        world_x: 월드 X 좌표
        world_y: 월드 Y 좌표
        workspace_limits: [[x_min, x_max], [y_min, y_max], [z_min, z_max]]
        heightmap_resolution: Heightmap 해상도 (미터/픽셀)
        H: Homography 행렬 (world XY -> pixel UV), None이면 선형 변환
        
    Returns:
        (pixel_x, pixel_y): Heightmap 픽셀 좌표
    """
    if H is not None:
        # 캘리브레이션된 Homography 변환 사용
        camera_img_width = 512
        camera_img_height = 512
        
        # 월드 XY -> 카메라 픽셀
        world_xy = np.array([[world_x, world_y, 1.0]]).T
        pixel_h = H @ world_xy
        camera_pixel_x = int(pixel_h[0, 0] / pixel_h[2, 0])
        camera_pixel_y = int(pixel_h[1, 0] / pixel_h[2, 0])
        
        # 카메라 픽셀 -> Heightmap 픽셀 (스케일 조정)
        heightmap_width = int((workspace_limits[0][1] - workspace_limits[0][0]) / heightmap_resolution)
        heightmap_height = int((workspace_limits[1][1] - workspace_limits[1][0]) / heightmap_resolution)
        
        scale_x = heightmap_width / camera_img_width
        scale_y = heightmap_height / camera_img_height
        
        pixel_x = int(camera_pixel_x * scale_x)
        pixel_y = int(camera_pixel_y * scale_y)
        
        return (pixel_x, pixel_y)
    else:
        # Fallback: 기존 선형 변환
        pixel_x = int((world_x - workspace_limits[0][0]) / heightmap_resolution)
        pixel_y = int((world_y - workspace_limits[1][0]) / heightmap_resolution)
        return (pixel_x, pixel_y)


def get_pointcloud(color_img, depth_img, camera_intrinsics):
    """
    RGB-D 이미지에서 3D 포인트 클라우드 생성
    
    Args:
        color_img: RGB 이미지 (H x W x 3)
        depth_img: Depth 이미지 (H x W)
        camera_intrinsics: 카메라 내부 파라미터 (3 x 3)
    
    Returns:
        cam_pts: 카메라 좌표계의 3D 점 (N x 3)
        rgb_pts: 각 점의 RGB 색상 (N x 3)
    """
    # Get depth image size
    im_h = depth_img.shape[0]
    im_w = depth_img.shape[1]

    # Project depth into 3D point cloud in camera coordinates
    pix_x, pix_y = np.meshgrid(np.linspace(0, im_w-1, im_w), np.linspace(0, im_h-1, im_h))
    cam_pts_x = np.multiply(pix_x - camera_intrinsics[0][2], depth_img / camera_intrinsics[0][0])
    cam_pts_y = np.multiply(pix_y - camera_intrinsics[1][2], depth_img / camera_intrinsics[1][1])
    cam_pts_z = depth_img.copy()
    cam_pts_x = cam_pts_x.reshape((im_h * im_w, 1))
    cam_pts_y = cam_pts_y.reshape((im_h * im_w, 1))
    cam_pts_z = cam_pts_z.reshape((im_h * im_w, 1))

    # Reshape image into colors for 3D point cloud
    rgb_pts_r = color_img[:, :, 0].copy()
    rgb_pts_g = color_img[:, :, 1].copy()
    rgb_pts_b = color_img[:, :, 2].copy()
    rgb_pts_r = rgb_pts_r.reshape((im_h * im_w, 1))
    rgb_pts_g = rgb_pts_g.reshape((im_h * im_w, 1))
    rgb_pts_b = rgb_pts_b.reshape((im_h * im_w, 1))

    cam_pts = np.concatenate((cam_pts_x, cam_pts_y, cam_pts_z), axis=1)
    rgb_pts = np.concatenate((rgb_pts_r, rgb_pts_g, rgb_pts_b), axis=1)

    return cam_pts, rgb_pts


def get_heightmap(color_img, depth_img, cam_intrinsics, cam_pose, workspace_limits, heightmap_resolution):
    """
    RGB-D 이미지에서 heightmap 생성
    
    카메라 좌표계의 3D 포인트 클라우드를 로봇 좌표계로 변환 후,
    위에서 내려다보는 orthographic heightmap을 생성합니다.
    
    Args:
        color_img: RGB 이미지 (H x W x 3)
        depth_img: Depth 이미지 (H x W)
        cam_intrinsics: 카메라 내부 파라미터 (3 x 3)
        cam_pose: 카메라 포즈 (4 x 4 transformation matrix)
        workspace_limits: 작업 공간 제한 [[x_min, x_max], [y_min, y_max], [z_min, z_max]]
        heightmap_resolution: heightmap 해상도 (미터/픽셀)
    
    Returns:
        color_heightmap: RGB heightmap (H x W x 3)
        depth_heightmap: Depth heightmap (H x W), workspace z_bottom 기준 높이
    """
    # Compute heightmap size
    heightmap_size = np.round((
        (workspace_limits[1][1] - workspace_limits[1][0]) / heightmap_resolution,
        (workspace_limits[0][1] - workspace_limits[0][0]) / heightmap_resolution
    )).astype(int)

    # Get 3D point cloud from RGB-D images
    surface_pts, color_pts = get_pointcloud(color_img, depth_img, cam_intrinsics)

    # Transform 3D point cloud from camera coordinates to robot coordinates
    surface_pts = np.transpose(
        np.dot(cam_pose[0:3, 0:3], np.transpose(surface_pts)) + 
        np.tile(cam_pose[0:3, 3:], (1, surface_pts.shape[0]))
    )

    # Sort surface points by z value
    sort_z_ind = np.argsort(surface_pts[:, 2])
    surface_pts = surface_pts[sort_z_ind]
    color_pts = color_pts[sort_z_ind]

    # Filter out surface points outside heightmap boundaries
    heightmap_valid_ind = np.logical_and(
        np.logical_and(
            np.logical_and(
                np.logical_and(
                    surface_pts[:, 0] >= workspace_limits[0][0],
                    surface_pts[:, 0] < workspace_limits[0][1]
                ),
                surface_pts[:, 1] >= workspace_limits[1][0]
            ),
            surface_pts[:, 1] < workspace_limits[1][1]
        ),
        surface_pts[:, 2] < workspace_limits[2][1]
    )
    surface_pts = surface_pts[heightmap_valid_ind]
    color_pts = color_pts[heightmap_valid_ind]

    # Create orthographic top-down-view RGB-D heightmaps
    color_heightmap_r = np.zeros((heightmap_size[0], heightmap_size[1], 1), dtype=np.uint8)
    color_heightmap_g = np.zeros((heightmap_size[0], heightmap_size[1], 1), dtype=np.uint8)
    color_heightmap_b = np.zeros((heightmap_size[0], heightmap_size[1], 1), dtype=np.uint8)
    depth_heightmap = np.zeros(heightmap_size)
    
    heightmap_pix_x = np.floor((surface_pts[:, 0] - workspace_limits[0][0]) / heightmap_resolution).astype(int)
    heightmap_pix_y = np.floor((surface_pts[:, 1] - workspace_limits[1][0]) / heightmap_resolution).astype(int)
    
    # Clamp indices to valid range
    heightmap_pix_x = np.clip(heightmap_pix_x, 0, heightmap_size[1] - 1)
    heightmap_pix_y = np.clip(heightmap_pix_y, 0, heightmap_size[0] - 1)
    
    color_heightmap_r[heightmap_pix_y, heightmap_pix_x] = color_pts[:, [0]]
    color_heightmap_g[heightmap_pix_y, heightmap_pix_x] = color_pts[:, [1]]
    color_heightmap_b[heightmap_pix_y, heightmap_pix_x] = color_pts[:, [2]]
    color_heightmap = np.concatenate((color_heightmap_r, color_heightmap_g, color_heightmap_b), axis=2)
    depth_heightmap[heightmap_pix_y, heightmap_pix_x] = surface_pts[:, 2]
    
    # Subtract z_bottom to get height relative to workspace floor
    z_bottom = workspace_limits[2][0]
    depth_heightmap = depth_heightmap - z_bottom
    depth_heightmap[depth_heightmap < 0] = 0
    depth_heightmap[depth_heightmap == -z_bottom] = np.nan

    return color_heightmap, depth_heightmap


def get_heightmap_ortho(color_img, depth_img, cam_intrinsics, cam_pose, 
                        workspace_limits, heightmap_resolution,
                        H_world_to_cam=None, H_cam_to_world=None):
    """
    Orthographic 카메라용 heightmap 생성
    
    Orthographic 카메라는 이미 상단에서 수직으로 내려다보는 뷰를 제공하므로,
    복잡한 3D 포인트 클라우드 변환 없이 직접 리사이즈하여 heightmap을 생성합니다.
    
    Args:
        color_img: RGB 이미지 (H x W x 3)
        depth_img: Depth 이미지 (H x W), 카메라로부터의 거리
        cam_intrinsics: 카메라 내부 파라미터 (3 x 3) - 사용 안 함 (Ortho)
        cam_pose: 카메라 포즈 (4 x 4) - Z 위치 필요
        workspace_limits: 작업 공간 제한 [[x_min, x_max], [y_min, y_max], [z_min, z_max]]
        heightmap_resolution: heightmap 해상도 (미터/픽셀)
        H_world_to_cam: 월드→카메라 Homography (옵션)
        H_cam_to_world: 카메라→월드 Homography (옵션)
    
    Returns:
        color_heightmap: RGB heightmap (H x W x 3)
        depth_heightmap: Depth heightmap (H x W), workspace z_bottom 기준 높이
    """
    # 목표 heightmap 크기 계산 (기존 코드와 동일한 순서: height, width)
    heightmap_size = np.round((
        (workspace_limits[1][1] - workspace_limits[1][0]) / heightmap_resolution,  # Y (height)
        (workspace_limits[0][1] - workspace_limits[0][0]) / heightmap_resolution   # X (width)
    )).astype(int)
    
    heightmap_height = heightmap_size[0]
    heightmap_width = heightmap_size[1]
    
    # Orthographic 카메라는 이미 상단 뷰이므로 단순 리사이즈
    # cv2.resize는 (width, height) 순서 사용
    # Color 이미지 리사이즈
    color_heightmap = cv2.resize(color_img, (heightmap_width, heightmap_height), 
                                  interpolation=cv2.INTER_LINEAR)
    
    # Depth 이미지 리사이즈 (최근접 이웃 보간으로 depth 값 보존)
    depth_img_resized = cv2.resize(depth_img, (heightmap_width, heightmap_height),
                                    interpolation=cv2.INTER_NEAREST)
    
    # !! 중요: Orthographic 카메라의 depth는 "카메라로부터의 거리"
    # 실제 높이 = 카메라 Z - depth 값
    camera_z = cam_pose[2, 3]  # 카메라의 월드 Z 좌표
    object_heights = camera_z - depth_img_resized  # 물체의 실제 월드 Z 높이
    
    # Workspace z_bottom 기준으로 상대 높이 계산
    z_bottom = workspace_limits[2][0]
    depth_heightmap = object_heights - z_bottom
    
    # 음수 값은 0으로 클리핑 (바닥 아래)
    depth_heightmap[depth_heightmap < 0] = 0
    
    # 바닥 영역은 NaN으로 표시 (옵션)
    # depth_heightmap[depth_heightmap == 0] = np.nan
    
    print(f'[HEIGHTMAP_ORTHO] Created heightmap: {depth_heightmap.shape}')
    print(f'[HEIGHTMAP_ORTHO] Camera Z: {camera_z:.4f}m')
    print(f'[HEIGHTMAP_ORTHO] Depth range: {np.nanmin(depth_heightmap):.4f} ~ {np.nanmax(depth_heightmap):.4f}m (relative to floor)')
    print(f'[HEIGHTMAP_ORTHO] Non-zero pixels: {np.sum(depth_heightmap > 0.001)}')
    
    return color_heightmap, depth_heightmap


def pcwrite(xyz_pts, filename, rgb_pts=None):
    """
    3D 포인트 클라우드를 PLY 파일로 저장
    
    Args:
        xyz_pts: 3D 점 좌표 (N x 3)
        filename: 저장할 파일 경로
        rgb_pts: 각 점의 RGB 색상 (N x 3), 기본값 흰색
    """
    assert xyz_pts.shape[1] == 3, 'input XYZ points should be an Nx3 matrix'
    if rgb_pts is None:
        rgb_pts = np.ones(xyz_pts.shape).astype(np.uint8) * 255
    assert xyz_pts.shape == rgb_pts.shape, 'input RGB colors should be Nx3 matrix and same size as input XYZ points'

    # Write header for .ply file
    pc_file = open(filename, 'wb')
    pc_file.write(bytearray('ply\n', 'utf8'))
    pc_file.write(bytearray('format binary_little_endian 1.0\n', 'utf8'))
    pc_file.write(bytearray(('element vertex %d\n' % xyz_pts.shape[0]), 'utf8'))
    pc_file.write(bytearray('property float x\n', 'utf8'))
    pc_file.write(bytearray('property float y\n', 'utf8'))
    pc_file.write(bytearray('property float z\n', 'utf8'))
    pc_file.write(bytearray('property uchar red\n', 'utf8'))
    pc_file.write(bytearray('property uchar green\n', 'utf8'))
    pc_file.write(bytearray('property uchar blue\n', 'utf8'))
    pc_file.write(bytearray('end_header\n', 'utf8'))

    # Write 3D points to .ply file
    for i in range(xyz_pts.shape[0]):
        pc_file.write(bytearray(struct.pack(
            "fffccc",
            xyz_pts[i][0], xyz_pts[i][1], xyz_pts[i][2],
            rgb_pts[i][0].tobytes(), rgb_pts[i][1].tobytes(), rgb_pts[i][2].tobytes()
        )))
    pc_file.close()


def get_affordance_vis(grasp_affordances, input_images, num_rotations, best_pix_ind):
    """
    Grasp affordance 시각화
    
    Args:
        grasp_affordances: Grasp 예측 값 (num_rotations x H x W)
        input_images: 입력 이미지 (num_rotations x H x W x 3)
        num_rotations: 회전 각도 수
        best_pix_ind: 최적 픽셀 인덱스 (rotation, y, x)
    
    Returns:
        vis: 시각화 이미지
    """
    vis = None
    for vis_row in range(num_rotations // 4):
        tmp_row_vis = None
        for vis_col in range(4):
            rotate_idx = vis_row * 4 + vis_col
            affordance_vis = grasp_affordances[rotate_idx, :, :]
            affordance_vis[affordance_vis < 0] = 0  # assume probability
            affordance_vis[affordance_vis > 1] = 1  # assume probability
            affordance_vis.shape = (grasp_affordances.shape[1], grasp_affordances.shape[2])
            affordance_vis = cv2.applyColorMap((affordance_vis * 255).astype(np.uint8), cv2.COLORMAP_JET)
            input_image_vis = (input_images[rotate_idx, :, :, :] * 255).astype(np.uint8)
            input_image_vis = cv2.resize(input_image_vis, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
            affordance_vis = (0.5 * cv2.cvtColor(input_image_vis, cv2.COLOR_RGB2BGR) + 0.5 * affordance_vis).astype(np.uint8)
            if rotate_idx == best_pix_ind[0]:
                affordance_vis = cv2.circle(affordance_vis, (int(best_pix_ind[2]), int(best_pix_ind[1])), 7, (0, 0, 255), 2)
            if tmp_row_vis is None:
                tmp_row_vis = affordance_vis
            else:
                tmp_row_vis = np.concatenate((tmp_row_vis, affordance_vis), axis=1)
        if vis is None:
            vis = tmp_row_vis
        else:
            vis = np.concatenate((vis, tmp_row_vis), axis=0)

    return vis


def get_difference(color_heightmap, color_space, bg_color_heightmap):
    """
    두 heightmap 간의 색상 차이 계산
    
    Args:
        color_heightmap: 현재 color heightmap
        color_space: 물체 색상 팔레트
        bg_color_heightmap: 배경 color heightmap
    
    Returns:
        float: 색상 일치도 (0~1)
    """
    color_space = np.concatenate((color_space, np.asarray([[0.0, 0.0, 0.0]])), axis=0)
    color_space.shape = (color_space.shape[0], 1, 1, color_space.shape[1])
    color_space = np.tile(color_space, (1, color_heightmap.shape[0], color_heightmap.shape[1], 1))

    # Normalize color heightmaps
    color_heightmap = color_heightmap.astype(float) / 255.0
    color_heightmap.shape = (1, color_heightmap.shape[0], color_heightmap.shape[1], color_heightmap.shape[2])
    color_heightmap = np.tile(color_heightmap, (color_space.shape[0], 1, 1, 1))

    bg_color_heightmap = bg_color_heightmap.astype(float) / 255.0
    bg_color_heightmap.shape = (1, bg_color_heightmap.shape[0], bg_color_heightmap.shape[1], bg_color_heightmap.shape[2])
    bg_color_heightmap = np.tile(bg_color_heightmap, (color_space.shape[0], 1, 1, 1))

    # Compute nearest neighbor distances to key colors
    key_color_dist = np.sqrt(np.sum(np.power(color_heightmap - color_space, 2), axis=3))
    bg_key_color_dist = np.sqrt(np.sum(np.power(bg_color_heightmap - color_space, 2), axis=3))

    key_color_match = np.argmin(key_color_dist, axis=0)
    bg_key_color_match = np.argmin(bg_key_color_dist, axis=0)
    key_color_match[key_color_match == color_space.shape[0] - 1] = color_space.shape[0] + 1
    bg_key_color_match[bg_key_color_match == color_space.shape[0] - 1] = color_space.shape[0] + 2

    return np.sum(key_color_match == bg_key_color_match).astype(float) / np.sum(bg_key_color_match < color_space.shape[0]).astype(float)


# Get rotation matrix from euler angles
def euler2rotm(theta):
    """
    오일러 각도를 회전 행렬로 변환
    
    Args:
        theta: 오일러 각도 [rx, ry, rz] (라디안)
    
    Returns:
        R: 3x3 회전 행렬
    """
    R_x = np.array([
        [1, 0, 0],
        [0, math.cos(theta[0]), -math.sin(theta[0])],
        [0, math.sin(theta[0]), math.cos(theta[0])]
    ])
    R_y = np.array([
        [math.cos(theta[1]), 0, math.sin(theta[1])],
        [0, 1, 0],
        [-math.sin(theta[1]), 0, math.cos(theta[1])]
    ])
    R_z = np.array([
        [math.cos(theta[2]), -math.sin(theta[2]), 0],
        [math.sin(theta[2]), math.cos(theta[2]), 0],
        [0, 0, 1]
    ])
    R = np.dot(R_z, np.dot(R_y, R_x))
    return R


def isRotm(R):
    """회전 행렬 유효성 검사"""
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


def rotm2euler(R):
    """회전 행렬을 오일러 각도로 변환"""
    assert(isRotm(R))

    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])


def angle2rotm(angle, axis, point=None):
    """
    축-각도 표현을 회전 행렬로 변환
    
    Args:
        angle: 회전 각도 (라디안)
        axis: 회전축 (3-vector)
        point: 회전 중심점 (옵션)
    
    Returns:
        M: 4x4 변환 행렬
    """
    sina = math.sin(angle)
    cosa = math.cos(angle)
    axis = axis / np.linalg.norm(axis)

    # Rotation matrix around unit vector
    R = np.diag([cosa, cosa, cosa])
    R += np.outer(axis, axis) * (1.0 - cosa)
    axis *= sina
    R += np.array([
        [0.0, -axis[2], axis[1]],
        [axis[2], 0.0, -axis[0]],
        [-axis[1], axis[0], 0.0]
    ])
    M = np.identity(4)
    M[:3, :3] = R
    if point is not None:
        point = np.array(point[:3], dtype=np.float64, copy=False)
        M[:3, 3] = point - np.dot(R, point)
    return M


def rotm2angle(R):
    """회전 행렬을 축-각도 표현으로 변환"""
    epsilon = 0.01
    epsilon2 = 0.1

    assert(isRotm(R))

    if ((abs(R[0][1] - R[1][0]) < epsilon) and 
        (abs(R[0][2] - R[2][0]) < epsilon) and 
        (abs(R[1][2] - R[2][1]) < epsilon)):
        # Singularity found
        if ((abs(R[0][1] + R[1][0]) < epsilon2) and 
            (abs(R[0][2] + R[2][0]) < epsilon2) and 
            (abs(R[1][2] + R[2][1]) < epsilon2) and 
            (abs(R[0][0] + R[1][1] + R[2][2] - 3) < epsilon2)):
            return [0, 1, 0, 0]

        angle = np.pi
        xx = (R[0][0] + 1) / 2
        yy = (R[1][1] + 1) / 2
        zz = (R[2][2] + 1) / 2
        xy = (R[0][1] + R[1][0]) / 4
        xz = (R[0][2] + R[2][0]) / 4
        yz = (R[1][2] + R[2][1]) / 4
        
        if (xx > yy) and (xx > zz):
            if xx < epsilon:
                x, y, z = 0, 0.7071, 0.7071
            else:
                x = np.sqrt(xx)
                y = xy / x
                z = xz / x
        elif yy > zz:
            if yy < epsilon:
                x, y, z = 0.7071, 0, 0.7071
            else:
                y = np.sqrt(yy)
                x = xy / y
                z = yz / y
        else:
            if zz < epsilon:
                x, y, z = 0.7071, 0.7071, 0
            else:
                z = np.sqrt(zz)
                x = xz / z
                y = yz / z
        return [angle, x, y, z]

    s = np.sqrt(
        (R[2][1] - R[1][2]) * (R[2][1] - R[1][2]) + 
        (R[0][2] - R[2][0]) * (R[0][2] - R[2][0]) + 
        (R[1][0] - R[0][1]) * (R[1][0] - R[0][1])
    )
    if abs(s) < 0.001:
        s = 1

    angle = np.arccos((R[0][0] + R[1][1] + R[2][2] - 1) / 2)
    x = (R[2][1] - R[1][2]) / s
    y = (R[0][2] - R[2][0]) / s
    z = (R[1][0] - R[0][1]) / s
    return [angle, x, y, z]


def get_change_value(depth_diff):
    """
    Depth 변화량 계산
    
    이전 depth heightmap과 현재 depth heightmap의 차이에서
    유의미한 변화가 있는 픽셀 수를 계산합니다.
    
    Args:
        depth_diff: depth 차이 이미지
    
    Returns:
        change_value: 변화 픽셀 수
    """
    depth_diff[np.isnan(depth_diff)] = 0
    depth_diff[depth_diff > 0.3] = 0
    depth_diff[depth_diff < 0.01] = 0
    depth_diff[depth_diff > 0] = 1
    change_value = np.sum(depth_diff)
    return change_value


# Cross entropy loss for 2D outputs
class CrossEntropyLoss2d(nn.Module):
    """2D Cross Entropy Loss (deprecated NLLLoss2d → NLLLoss 사용)"""

    def __init__(self, weight=None, size_average=True):
        super(CrossEntropyLoss2d, self).__init__()
        # NLLLoss2d is deprecated, use NLLLoss instead
        self.nll_loss = nn.NLLLoss(weight=weight, reduction='mean' if size_average else 'sum')

    def forward(self, inputs, targets):
        return self.nll_loss(F.log_softmax(inputs, dim=1), targets)


def verify_heightmap_accuracy(objects_info, color_heightmap, depth_heightmap, 
                               workspace_limits, heightmap_resolution, H=None):
    """
    Heightmap과 실제 물체 위치의 정확성을 검증
    
    Args:
        objects_info: 물체 정보 리스트 [{'name', 'position', 'size'}, ...]
        color_heightmap: RGB heightmap
        depth_heightmap: Depth heightmap
        workspace_limits: [[x_min, x_max], [y_min, y_max], [z_min, z_max]]
        heightmap_resolution: Heightmap 해상도 (미터/픽셀)
        H: Homography 행렬 (world XY -> pixel UV)
        
    Returns:
        verification_passed: 검증 통과 여부 (bool)
        report: 검증 리포트 (dict)
    """
    print('\n' + '='*80)
    print('HEIGHTMAP ACCURACY VERIFICATION')
    print('='*80)
    
    valid_depth_heightmap = depth_heightmap.copy()
    valid_depth_heightmap[np.isnan(valid_depth_heightmap)] = 0
    
    verification_passed = True
    object_results = []
    
    # 1. 각 물체별 검증
    print(f'\n[1] Object Position Verification ({len(objects_info)} objects):')
    print('-'*80)
    
    for obj in objects_info:
        obj_name = obj['name']
        obj_pos = obj['position']  # [x, y, z]
        obj_size = obj['size']  # [size_x, size_y, size_z]
        
        # 물체 월드 좌표 -> Heightmap 픽셀
        pixel_x, pixel_y = world_to_heightmap_pixel(
            obj_pos[0], obj_pos[1], 
            workspace_limits, heightmap_resolution, H
        )
        
        # Heightmap 범위 확인
        heightmap_height, heightmap_width = valid_depth_heightmap.shape
        in_bounds = (0 <= pixel_x < heightmap_width and 
                     0 <= pixel_y < heightmap_height)
        
        if in_bounds:
            depth_value = valid_depth_heightmap[pixel_y, pixel_x]
            expected_height = obj_pos[2] + obj_size[2]/2  # 물체 상단
            
            # 검증: depth >= 20mm
            min_depth = 0.020  # 20mm
            depth_ok = depth_value >= min_depth
            
            status = '[OK]' if depth_ok else '[FAIL]'
            if not depth_ok:
                verification_passed = False
            
            print(f'\n  {status} {obj_name}:')
            print(f'    World pos: ({obj_pos[0]:.3f}, {obj_pos[1]:.3f}, {obj_pos[2]:.3f})')
            print(f'    Pixel pos: ({pixel_x}, {pixel_y})')
            print(f'    Depth value: {depth_value*1000:.1f}mm (expected ~{expected_height*1000:.1f}mm)')
            print(f'    Object size: ({obj_size[0]*1000:.0f}x{obj_size[1]*1000:.0f}x{obj_size[2]*1000:.0f}mm)')
            
            object_results.append({
                'name': obj_name,
                'world_pos': obj_pos,
                'pixel_pos': (pixel_x, pixel_y),
                'depth_value': depth_value,
                'expected_height': expected_height,
                'depth_ok': depth_ok,
                'in_bounds': True
            })
        else:
            print(f'\n  [OUT_OF_BOUNDS] {obj_name}:')
            print(f'    World pos: ({obj_pos[0]:.3f}, {obj_pos[1]:.3f}, {obj_pos[2]:.3f})')
            print(f'    Pixel pos: ({pixel_x}, {pixel_y}) - outside heightmap!')
            verification_passed = False
            
            object_results.append({
                'name': obj_name,
                'world_pos': obj_pos,
                'pixel_pos': (pixel_x, pixel_y),
                'in_bounds': False
            })
    
    # 2. 반경 외부 영역 검증
    print(f'\n[2] Object Radius Verification:')
    print('-'*80)
    
    for i, obj_result in enumerate(object_results):
        if not obj_result['in_bounds']:
            continue
        
        obj = objects_info[i]
        pixel_x, pixel_y = obj_result['pixel_pos']
        
        # 물체 반경 (픽셀 단위)
        radius_m = max(obj['size'][0], obj['size'][1]) / 2 * 1.5  # 1.5배 여유
        radius_px = int(radius_m / heightmap_resolution)
        
        # 반경 외부 샘플링 (4방향)
        sample_offsets = [
            (0, radius_px + 10),   # 위
            (0, -radius_px - 10),  # 아래
            (radius_px + 10, 0),   # 오른쪽
            (-radius_px - 10, 0),  # 왼쪽
        ]
        
        outside_clear = True
        for dx, dy in sample_offsets:
            sample_x = pixel_x + dx
            sample_y = pixel_y + dy
            
            if 0 <= sample_x < heightmap_width and 0 <= sample_y < heightmap_height:
                sample_depth = valid_depth_heightmap[sample_y, sample_x]
                # 바닥 = depth < 10mm
                if sample_depth > 0.010:
                    outside_clear = False
                    print(f'  [WARNING] {obj_result["name"]}: Non-zero depth ({sample_depth*1000:.1f}mm) at offset ({dx},{dy})')
        
        if outside_clear:
            print(f'  [OK] {obj_result["name"]}: Surrounding area is clear (radius ~{radius_px}px)')
    
    # 3. 전체 통계
    print(f'\n[3] Heightmap Statistics:')
    print('-'*80)
    
    total_pixels = valid_depth_heightmap.size
    object_pixels = np.sum(valid_depth_heightmap > 0.010)  # >10mm
    floor_pixels = np.sum(valid_depth_heightmap <= 0.010)
    
    print(f'  Total pixels: {total_pixels}')
    print(f'  Object pixels (>10mm): {object_pixels} ({object_pixels/total_pixels*100:.2f}%)')
    print(f'  Floor pixels (<=10mm): {floor_pixels} ({floor_pixels/total_pixels*100:.2f}%)')
    print(f'  Depth range: {np.min(valid_depth_heightmap)*1000:.1f}mm ~ {np.max(valid_depth_heightmap)*1000:.1f}mm')
    if object_pixels > 0:
        print(f'  Average depth (object area): {np.mean(valid_depth_heightmap[valid_depth_heightmap > 0.010])*1000:.1f}mm')
    else:
        print(f'  Average depth (object area): N/A (no objects detected)')
    
    # 4. 최종 결과
    print(f'\n[4] Verification Result:')
    print('='*80)
    
    if verification_passed:
        print('  STATUS: [PASSED] All objects are correctly represented in heightmap')
    else:
        print('  STATUS: [FAILED] Some objects have incorrect depth values or are out of bounds')
    
    print('='*80 + '\n')
    
    report = {
        'passed': verification_passed,
        'object_results': object_results,
        'total_pixels': total_pixels,
        'object_pixels': object_pixels,
        'floor_pixels': floor_pixels
    }
    
    return verification_passed, report

