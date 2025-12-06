#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
IRB360 델타 로봇 DQN 학습 메인 스크립트

=============================================================================
Original Code Attribution:
    Repository: https://github.com/Marwanon/Learning-Pick-to-Place-Objects-in-a-cluttered-scene-using-deep-reinforcement-learning
    Original Author: Marwan Qaid Mohammed
    Paper: "Learning Pick to Place Objects using Self-supervised Learning with Minimal Training Resources"
           International Journal of Advanced Computer Science and Applications (IJACSA), 12(10), 2021
    
    Based on: https://github.com/andyzeng/visual-pushing-grasping

Modifications for this project (My Contributions):
    ✨ IRB360 델타 로봇 통합 (UR5 → IRB360)
    ✨ 진공 컵 그리퍼 제어 (Parallel-jaw → Vacuum cup)
    ✨ CoppeliaSim ZMQ Remote API 연동
    ✨ Curriculum Learning 기반 Epsilon-greedy 탐색 전략
       - 초기(0~500): 물체 위치에서만 탐색
       - 중반(500~1000): 80% 물체 + 20% 전체 영역
       - 후반(1000+): 전체 영역 탐색 (바닥 회피 학습)
    ✨ 그리퍼 영역 내 바닥 감지 및 즉시 실패 처리
    ✨ 동일 이미지 연속 감지 (로봇에 물체 붙음 감지)
    ✨ Homography 기반 카메라 캘리브레이션 적용
    ✨ Double DQN / Dueling DQN 명령줄 옵션 지원
    ✨ 한국어 주석 추가
    
This code is used for educational purposes as part of a graduate project.
=============================================================================
"""

# OpenMP 중복 라이브러리 경고 해결
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import time
import threading
import argparse
import numpy as np
import cv2
import torch
from test.robot_zmq_irb360 import RobotIRB360ZMQ as Robot
from DQNTrainer import DQNTrainer
from logger import Logger
import utils

# 캘리브레이션된 Homography 행렬 로드
H_world_to_pixel, H_pixel_to_world = utils.load_camera_calibration('camera_calibration.npy')


def get_min_z_in_circular_area(depth_heightmap, center_x, center_y, radius_pixels, z_offset):
    """
    원형 영역 내 최소 Z값 계산 (그리퍼 지름 영역 바닥 검사용)
    
    Args:
        depth_heightmap: 깊이 heightmap (H, W)
        center_x, center_y: 중심 픽셀 좌표
        radius_pixels: 반지름 (픽셀 단위)
        z_offset: Z 오프셋 (workspace_limits[2][0])
    
    Returns:
        min_z: 영역 내 최소 Z값 (meters)
    """
    h, w = depth_heightmap.shape
    y_coords, x_coords = np.ogrid[:h, :w]
    
    # 원형 마스크 생성 (중심 기준 반지름 이내의 픽셀)
    dist_from_center = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
    circular_mask = dist_from_center <= radius_pixels
    
    # 영역 내 깊이값 추출 (Z 오프셋 적용)
    z_values = depth_heightmap[circular_mask] + z_offset
    
    if len(z_values) == 0:
        return 0.0  # 영역이 비어있으면 바닥으로 간주
    
    return np.min(z_values)


def main(args):

    # --------------- Setup options ---------------
    is_sim = args.is_sim # Run in simulation?
    obj_mesh_dir = os.path.abspath(args.obj_mesh_dir) if is_sim else None # Directory containing 3D mesh files (.obj) of objects to be added to simulation
    num_obj = args.num_obj if is_sim else None # Number of objects to add to simulation
    if is_sim:
        # IRB360 workspace 설정 (robot_zmq_irb360.py 기본값과 동일)
        workspace_limits = np.asarray([[-0.7, -0.3], [-0.2, 0.2], [0.001, 0.20]]) # Cols: min max, Rows: x y z
        workspace_limit_place = np.asarray([[-0.5, 0.5], [-0.6, -0.4], [0.001, 0.20]])

    heightmap_resolution = args.heightmap_resolution # Meters per pixel of heightmap
    random_seed = args.random_seed
    force_cpu = args.force_cpu

    # ------------- Algorithm options -------------
    method = args.method # 'reactive' (supervised learning) or 'reinforcement' (reinforcement learning ie Q-learning)
    grasp_rewards = args.grasp_rewards if method == 'reinforcement' else None  # Use immediate rewards (from change detection) for pushing?
    future_reward_discount = args.future_reward_discount
    experience_replay = args.experience_replay # Use prioritized experience replay?
    
    # Double DQN 옵션 (Q값 과대추정 방지)
    double_dqn = args.double_dqn
    target_update_freq = args.target_update_freq
    
    # Dueling DQN 옵션 (Value + Advantage streams)
    dueling_dqn = args.dueling_dqn
    
    # 그리퍼 지름 설정 (바닥 검사 영역 크기 결정)
    gripper_diameter = args.gripper_diameter  # 그리퍼 지름 (meters), 기본값 30mm
    
    # Epsilon-greedy exploration parameters (논문 기준: 초기 0.5 → 최종 0.1)
    epsilon_start = 0.5       # 초기 탐색률 (50% 확률로 랜덤 행동)
    epsilon_end = 0.1         # 최종 탐색률 (10% 확률로 랜덤 행동)
    epsilon_decay_steps = 2500  # 감소 구간 (약 2500 iteration 동안 선형 감소)

    # -------------- Testing options --------------
    is_testing = args.is_testing
    max_test_trials = args.max_test_trials # Maximum number of test runs per case/scenario
    test_preset_cases = args.test_preset_cases
    test_preset_file = os.path.abspath(args.test_preset_file) if test_preset_cases else None

    # ------ Pre-loading and logging options ------
    load_snapshot = args.load_snapshot # Load pre-trained snapshot of model?
    snapshot_file = os.path.abspath(args.snapshot_file)  if load_snapshot else None
    continue_logging = args.continue_logging # Continue logging from previous session
    logging_directory = os.path.abspath(args.logging_directory) if continue_logging else os.path.abspath('logs')
    save_visualizations = args.save_visualizations # Save visualizations of FCN predictions? Takes 0.6s per training step if set to True


    # Set random seed
    np.random.seed(random_seed)

    # Initialize pick-and-place system (camera and robot)
    robot = Robot(is_sim, obj_mesh_dir, num_obj, workspace_limits, workspace_limit_place,
                  is_testing, test_preset_cases, test_preset_file, place=True)
    
    # 시뮬레이션 초기 시작 시 물체 추가 (is_sim일 때만)
    if is_sim and not test_preset_cases:
        print('[INIT] Adding initial objects to scene...')
        robot.add_objects()

    # Initialize trainer
    trainer = DQNTrainer(method, grasp_rewards, future_reward_discount,
                         is_testing, load_snapshot, snapshot_file, force_cpu,
                         double_dqn=double_dqn, target_update_freq=target_update_freq,
                         dueling_dqn=dueling_dqn)

    # Initialize data logger
    logger = Logger(continue_logging, logging_directory)
    logger.save_camera_info(robot.cam_intrinsics, robot.cam_pose, robot.cam_depth_scale) # Save camera intrinsics and pose
    logger.save_heightmap_info(workspace_limits, heightmap_resolution) # Save heightmap parameters

    # Find last executed iteration of pre-loaded log, and load execution info and RL variables
    if continue_logging:
        trainer.preload(logger.transitions_directory)

    # Initialize variables for heuristic bootstrapping and exploration probability
    no_change_count = [2] if not is_testing else [0]
    
    # 동일 이미지 감지를 위한 변수 (물체가 로봇에 붙어있는 경우 감지)
    same_image_count = [0]  # 동일 이미지 연속 카운트
    prev_color_heightmap_for_stuck = [None]  # 이전 이미지 저장

    # Quick hack for nonlocal memory between threads in Python 2
    nonlocal_variables = {'executing_action' : False,
                          'primitive_action' : None,
                          'best_pix_ind' : None,
                          'place_success' : False,
                          'grasp_success' : False}


    # 동일 이미지 감지 함수 (물체가 로봇에 붙어있는 경우)
    def is_same_image(img1, img2, threshold=50):
        """두 이미지가 거의 동일한지 확인 (픽셀 차이 기반)
        
        Args:
            img1, img2: 비교할 이미지 (numpy array)
            threshold: 평균 픽셀 차이 임계값 (기본 50)
        
        Returns:
            bool: 동일 이미지 여부
        """
        if img1 is None or img2 is None:
            return False
        if img1.shape != img2.shape:
            return False
        diff = np.abs(img1.astype(float) - img2.astype(float))
        mean_diff = np.mean(diff)
        return mean_diff < threshold

    # Parallel thread to process network output and execute actions
    # -------------------------------------------------------------
    def process_actions():
        grasp_count = 0
        successful_grasp_count = 0
        grasp_failed_iterations = 0
        while True:
            if nonlocal_variables['executing_action']:
                grasp_count += 1
                grasp_attempt = 'action Count: %r' % (grasp_count)
                print(grasp_attempt)
                best_grasp_conf = np.max(grasp_predictions)
                print('Primitive confidence scores: %f (grasp)' % (best_grasp_conf))
                nonlocal_variables['primitive_action'] = 'grasp'
                
                #Get pixel location and rotation with highest affordance prediction from heuristic algorithms (rotation, y, x)
                # Epsilon-greedy 탐색 정책 적용 (논문 기준)
                if nonlocal_variables['primitive_action'] == 'grasp':
                    # 현재 epsilon 계산 (선형 감소: epsilon_start → epsilon_end)
                    current_epsilon = max(epsilon_end, epsilon_start - (epsilon_start - epsilon_end) * trainer.iteration / epsilon_decay_steps)
                    
                    # Epsilon-greedy 행동 선택
                    is_random_action = False
                    if np.random.random() < current_epsilon and not is_testing:
                        # 탐색(Exploration): Curriculum Learning 방식
                        # 초기(0~500): 물체 위치만, 중반(500~1000): 80% 물체 + 20% 전체, 후반(1000+): 전체 영역
                        iteration = trainer.iteration
                        
                        if iteration < 500:
                            # 초기: 물체가 있는 위치에서만 탐색 (빠른 기본 학습)
                            valid_indices = np.argwhere(valid_depth_heightmap > 0.01)  # depth > 1cm인 위치만
                            exploration_mode = "object_only"
                        elif iteration < 1000:
                            # 중반: 80% 물체 위치 + 20% 전체 영역 (바닥 회피 학습 시작)
                            if np.random.random() < 0.8:
                                valid_indices = np.argwhere(valid_depth_heightmap > 0.01)  # 물체만
                                exploration_mode = "object_only"
                            else:
                                valid_indices = np.argwhere(valid_depth_heightmap >= 0)  # 전체 (바닥 포함)
                                exploration_mode = "full_area"
                        else:
                            # 후반: 전체 영역에서 탐색 (바닥 = 낮은 Q값 학습)
                            valid_indices = np.argwhere(valid_depth_heightmap >= 0)  # 전체 (바닥 포함)
                            exploration_mode = "full_area"
                        
                        if len(valid_indices) > 0:
                            random_rot = np.random.randint(0, grasp_predictions.shape[0])  # 랜덤 회전 각도
                            random_idx = valid_indices[np.random.randint(len(valid_indices))]  # 랜덤 위치
                            nonlocal_variables['best_pix_ind'] = (random_rot, random_idx[0], random_idx[1])
                            is_random_action = True
                            print(f'[EXPLORATION] Curriculum mode: {exploration_mode} (iter={iteration})')
                        else:
                            # 유효한 위치가 없으면 최적 행동 선택
                            nonlocal_variables['best_pix_ind'] = np.unravel_index(np.argmax(grasp_predictions), grasp_predictions.shape)
                    else:
                        # 활용(Exploitation): 최적 행동 선택 (Q값이 가장 높은 위치)
                        nonlocal_variables['best_pix_ind'] = np.unravel_index(np.argmax(grasp_predictions), grasp_predictions.shape)
                    
                    predicted_value = np.max(grasp_predictions)
                    print(f'[EXPLORATION] epsilon={current_epsilon:.3f}, random_action={is_random_action}')
                # Save predicted confidence value
                trainer.predicted_value_log.append([predicted_value])
                logger.write_to_log('predicted-value', trainer.predicted_value_log)
                
                # Compute 3D position of pixel
                print('Action: %s at (%d, %d, %d)' % (nonlocal_variables['primitive_action'], nonlocal_variables['best_pix_ind'][0], nonlocal_variables['best_pix_ind'][1], nonlocal_variables['best_pix_ind'][2]))
                best_rotation_angle = np.deg2rad(nonlocal_variables['best_pix_ind'][0]*(360.0/trainer.model.num_rotations))
                best_pix_x = nonlocal_variables['best_pix_ind'][2]
                best_pix_y = nonlocal_variables['best_pix_ind'][1]
                
                # 캘리브레이션된 변환 사용
                primitive_position = utils.heightmap_pixel_to_world(
                    best_pix_x, 
                    best_pix_y, 
                    valid_depth_heightmap[best_pix_y][best_pix_x],
                    workspace_limits,
                    heightmap_resolution,
                    H_pixel_to_world
                )
                
                # 변환 과정 디버깅 로그
                print(f'[TRANSFORM] Heightmap pixel: ({best_pix_x}, {best_pix_y})')
                print(f'[TRANSFORM] World position: ({primitive_position[0]:.4f}, {primitive_position[1]:.4f}, {primitive_position[2]:.4f})')
                if H_pixel_to_world is None:
                    print(f'[TRANSFORM] Using linear scaling (no calibration)')
                else:
                    print(f'[TRANSFORM] Using calibrated Homography transform')
                
                # Selected pixel debug output
                print(f'[HEIGHTMAP] selected pixel ({best_pix_y}, {best_pix_x}) depth value: {valid_depth_heightmap[best_pix_y][best_pix_x]:.4f}')
                print(f'[HEIGHTMAP] primitive_position Z = {valid_depth_heightmap[best_pix_y][best_pix_x]:.4f} + {workspace_limits[2][0]:.4f} = {primitive_position[2]:.4f}')
                
                # Z값 검증: 그리퍼 지름 영역 내 최소 Z값이 바닥(1cm 미만)이면 즉시 실패 처리
                # (로봇 안 움직이지만 학습에는 사용)
                min_valid_z = 0.01  # 1cm (바닥으로 간주)
                max_valid_z = 0.5   # 50cm (물체 최대 높이)
                
                # 그리퍼 반지름을 픽셀 수로 변환 (예: 30mm 지름 → 15mm 반지름 → 12 pixels at 1.25mm/pixel)
                gripper_radius_pixels = int((gripper_diameter / 2) / heightmap_resolution)
                
                # 원형 영역 내 최소 Z값 계산
                min_z_in_area = get_min_z_in_circular_area(
                    valid_depth_heightmap, 
                    best_pix_x, best_pix_y, 
                    gripper_radius_pixels,
                    workspace_limits[2][0]
                )
                
                print(f'[VALIDATION] 그리퍼 영역 검사: 지름={gripper_diameter*1000:.0f}mm, 반지름={gripper_radius_pixels}pixels')
                print(f'[VALIDATION] 중심 Z={primitive_position[2]:.4f}m, 영역 내 최소 Z={min_z_in_area:.4f}m')
                
                z_validation_failed = False
                if min_z_in_area < min_valid_z:
                    print(f'[VALIDATION] 영역 내 최소 Z값이 너무 낮음 ({min_z_in_area:.4f}m < {min_valid_z}m) - 바닥 선택, 즉시 실패 (학습은 사용)')
                    z_validation_failed = True  # 로봇 움직이지 않음, 하지만 아래 학습 코드는 실행
                elif primitive_position[2] > max_valid_z:
                    print(f'[VALIDATION] Z값이 너무 높음 ({primitive_position[2]:.4f}m > {max_valid_z}m) - 이상값, 즉시 실패')
                    z_validation_failed = True
                else:
                    print(f'[VALIDATION] Z값 정상 (영역 최소: {min_z_in_area:.4f}m >= {min_valid_z}m) - Grasp 실행')

                # Save executed primitive
                if nonlocal_variables['primitive_action'] == 'grasp':
                    trainer.executed_action_log.append([1, nonlocal_variables['best_pix_ind'][0], nonlocal_variables['best_pix_ind'][1], nonlocal_variables['best_pix_ind'][2]]) # 1 - grasp
                logger.write_to_log('executed-action', trainer.executed_action_log)

                # Visualize executed primitive, and affordances
                if save_visualizations:
                    grasp_pred_vis = trainer.get_prediction_vis(grasp_predictions, color_heightmap, nonlocal_variables['best_pix_ind'])
                    logger.save_visualizations(trainer.iteration, grasp_pred_vis, 'grasp')
                    cv2.imwrite('visualization.grasp.png', grasp_pred_vis)
                    grasp_direction_vis = trainer.get_best_grasp_vis(nonlocal_variables['best_pix_ind'], color_heightmap)
                    logger.save_visualizations(trainer.iteration, grasp_direction_vis, 'best_grasp')
                    cv2.imwrite('visualization.best_grasp.png', grasp_direction_vis)
                
                # Initialize variables that influence reward
                nonlocal_variables['grasp_success'] = False
                change_detected = False

                # Execute primitive (Z값 검증 실패 시 스킵)
                if not z_validation_failed:
                    if nonlocal_variables['primitive_action'] == 'grasp':
                        nonlocal_variables['grasp_success'] = robot.grasp(primitive_position, best_rotation_angle, workspace_limits, trainer, workspace_limit_place)
                        print('Grasp successful: %r' % (nonlocal_variables['grasp_success']))
                        if nonlocal_variables['grasp_success'] == 1:  # 완전 성공만 카운트 (1만 True)
                            successful_grasp_count += 1
                        else:
                            grasp_failed_iterations +=1
                        grasp_rate = float(successful_grasp_count) / float(grasp_count)
                        grasp_str = 'Grasp Count: %r, grasp success rate: %.2f%%' % (grasp_count, grasp_rate * 100)  # % 표시 추가
                        print(grasp_str)
                else:
                    # Z값 검증 실패 시 grasp 실행 없이 실패로 처리
                    nonlocal_variables['grasp_success'] = -1  # 실패 코드
                    grasp_failed_iterations += 1
                    grasp_rate = float(successful_grasp_count) / float(grasp_count)
                    print(f'[VALIDATION] Z-validation failed - marked as failure (grasp_success=-1)')
                    print(f'[VALIDATION] Grasp Count: {grasp_count}, Success: {successful_grasp_count}, Failed: {grasp_failed_iterations}, Rate: {grasp_rate * 100:.2f}%')
                    print(f'[TRAINING] This failure will be used for learning in next iteration')
                    
                trainer.grasp_success_log.append([int(nonlocal_variables['grasp_success'])])


                nonlocal_variables['executing_action'] = False

            time.sleep(0.01)
    action_thread = threading.Thread(target=process_actions)
    action_thread.daemon = True
    action_thread.start()
    exit_called = False
    # -------------------------------------------------------------
    # -------------------------------------------------------------


    # Start main training/testing loop
    while True:
        print('\n%s iteration: %d' % ('Testing' if is_testing else 'Training', trainer.iteration))
        iteration_time_0 = time.time()

        # Make sure simulation is still stable (if not, reset simulation)
        if is_sim: robot.check_sim()

        # IRB360: 로봇을 홈 위치로 이동하여 깨끗한 카메라 뷰 확보 (특히 실패 후)
        # wait_arrival=True로 실제 도착 확인 (timeout 5초)
        robot.move_to(robot.home_position, 0, wait_arrival=True, timeout=15.0)

        # Get latest RGB-D image
        color_img, depth_img = robot.get_camera_data()
        depth_img = depth_img * robot.cam_depth_scale # Apply depth scale from calibration

        # Get heightmap from RGB-D image (Orthographic 카메라용 직접 리사이즈 방식)
        color_heightmap, depth_heightmap = utils.get_heightmap_ortho(
            color_img, depth_img, 
            robot.cam_intrinsics, robot.cam_pose, 
            workspace_limits, heightmap_resolution,
            H_world_to_pixel, H_pixel_to_world
        )
        valid_depth_heightmap = depth_heightmap.copy()
        valid_depth_heightmap[np.isnan(valid_depth_heightmap)] = 0

        # Heightmap debug output
        print(f'[HEIGHTMAP] depth_heightmap range: min={np.nanmin(depth_heightmap):.4f}, max={np.nanmax(depth_heightmap):.4f}')
        print(f'[HEIGHTMAP] non-zero pixels (>0.01): {np.sum(valid_depth_heightmap > 0.01)}')
        print(f'[HEIGHTMAP] heightmap shape: {depth_heightmap.shape}')

        # 첫 iteration에서 heightmap 정확도 검증
        if trainer.iteration == 0 and is_sim:
            print('\n[INIT] Performing initial heightmap verification...')
            objects_info = robot.get_object_info()
            verification_passed, report = utils.verify_heightmap_accuracy(
                objects_info, 
                color_heightmap, 
                depth_heightmap,
                workspace_limits, 
                heightmap_resolution, 
                H_world_to_pixel
            )
            
            if not verification_passed:
                print('[VERIFICATION] FAILED - Press Enter to continue or Ctrl+C to exit...')
                try:
                    input()
                except KeyboardInterrupt:
                    print('\n[VERIFICATION] User aborted.')
                    exit(1)
            else:
                print('[VERIFICATION] PASSED - Continuing training...')
        
        # 첫 iteration에서 Double DQN Target network 초기화 검증 (카메라 이미지로)
        if trainer.iteration == 0 and trainer.double_dqn:
            print('\n[INIT] Verifying Double DQN Target network with camera image...')
            trainer._verify_target_network_output(color_heightmap, valid_depth_heightmap)

        # Save RGB-D images and RGB-D heightmaps
        logger.save_images(trainer.iteration, color_img, depth_img, '0')
        logger.save_heightmaps(trainer.iteration, color_heightmap, valid_depth_heightmap, '0')

        # 동일 이미지 감지 (물체가 로봇에 붙어있는 경우)
        if is_sim and prev_color_heightmap_for_stuck[0] is not None:
            if is_same_image(color_heightmap, prev_color_heightmap_for_stuck[0]):
                same_image_count[0] += 1
                print(f'[STUCK] Same image detected ({same_image_count[0]}/30)')
                if same_image_count[0] >= 30:
                    print('[STUCK] Image stuck for 30 iterations! Object may be attached to robot. Restarting simulation...')
                    robot.restart_sim()
                    robot.add_objects()
                    same_image_count[0] = 0
                    prev_color_heightmap_for_stuck[0] = None
                    trainer.clearance_log.append([trainer.iteration])
                    logger.write_to_log('clearance', trainer.clearance_log)
                    continue  # 다음 iteration으로
            else:
                same_image_count[0] = 0  # 다른 이미지면 카운트 리셋
        prev_color_heightmap_for_stuck[0] = color_heightmap.copy()

        # Reset simulation or pause real-world training if table is empty
        stuff_count = np.zeros(valid_depth_heightmap.shape)
        stuff_count[valid_depth_heightmap > 0.02] = 1
        # IRB360: 물체 크기와 heightmap 해상도 고려하여 threshold 조정
        empty_threshold = 200  # 기존 300 → 200으로 낮춤
        if is_sim and is_testing:
            empty_threshold = 10
        
        # 실제 물체 개수 체크 (1개 이하면 씬 재시작)
        actual_object_count = len(robot.object_handles) if hasattr(robot, 'object_handles') else 0
        objects_too_few = is_sim and actual_object_count <= 1
        
        if objects_too_few:
            print(f'[RESET] Too few objects remaining ({actual_object_count} <= 1)! Restarting scene...')
        
        if np.sum(stuff_count) < empty_threshold or (is_sim and no_change_count[0] > 10) :
            no_change_count = [0]
            if is_sim:
                if not objects_too_few:
                    print('Not enough objects in view (value: %d)! Repositioning objects.' % (np.sum(stuff_count)))
                robot.restart_sim()
                robot.add_objects()
                if is_testing: # If at end of test run, re-load original weights (before test run)
                    trainer.model.load_state_dict(torch.load(snapshot_file))
            else:
                print('Not enough stuff on the table (value: %d)! Flipping over bin of objects...' % (np.sum(stuff_count)))
                robot.restart_real()

            trainer.clearance_log.append([trainer.iteration])
            logger.write_to_log('clearance', trainer.clearance_log)
            if is_testing and len(trainer.clearance_log) >= max_test_trials:
                exit_called = True # Exit after training thread (backprop and saving labels)
            continue

        # Run training iteration in current thread (aka training thread)
        if 'prev_color_img' in locals():

            # Detect changes
            depth_diff = abs(depth_heightmap - prev_depth_heightmap)
            change_threshold = 300
            change_value = utils.get_change_value(depth_diff)
            change_detected = change_value > change_threshold or prev_grasp_success == 1  # 완전 성공만 change로 인정
            print('Change detected: %r (value: %d)' % (change_detected, change_value))

            if change_detected:
                if prev_primitive_action == 'grasp':
                    no_change_count[0] = 0
            else:
                if prev_primitive_action == 'grasp':
                    no_change_count[0] += 1

            # Compute training labels
            label_value, prev_reward_value = trainer.get_label_value(prev_primitive_action, prev_grasp_success, change_detected, prev_grasp_predictions, color_heightmap, valid_depth_heightmap)
            trainer.label_value_log.append([label_value])
            logger.write_to_log('label-value', trainer.label_value_log)
            trainer.reward_value_log.append([prev_reward_value])
            logger.write_to_log('reward-value', trainer.reward_value_log)
            trainer.change_detected_log.append([change_detected])
            logger.write_to_log('change-detected', trainer.change_detected_log)
            logger.write_to_log('grasp-success', trainer.grasp_success_log)
            logger.write_to_log('place-success', trainer.place_success_log)
            
            # 균형 잡힌 Experience Replay를 위해 성공/실패 버퍼에 샘플 추가
            # prev_grasp_success: 1=성공, 0=일반실패, -1=바닥선택
            sample_idx = trainer.iteration - 1  # 현재 샘플의 iteration 인덱스
            if prev_grasp_success == 1:
                trainer.success_buffer.append(sample_idx)
                print(f'[BUFFER] Added to success_buffer (idx={sample_idx}, total={len(trainer.success_buffer)})')
            else:  # 0 또는 -1 (일반 실패 또는 바닥 선택)
                trainer.failure_buffer.append(sample_idx)
                print(f'[BUFFER] Added to failure_buffer (idx={sample_idx}, total={len(trainer.failure_buffer)})')
            
            # 학습 데이터 사용 로그 (특히 바닥 선택 실패 시)
            if prev_grasp_success != 1:
                print(f'[TRAINING] Using failure (grasp_success={prev_grasp_success}) for learning')
                print(f'[TRAINING] Reward: {prev_reward_value:.4f}, Label: {label_value:.4f}')

            # Backpropagate (nan label이면 건너뜀 - 네트워크 손상 방지)
            if np.isnan(label_value):
                print('[TRAINING] Warning: label_value is nan, skipping backprop to prevent network corruption')
            else:
                trainer.backprop(prev_color_heightmap, prev_valid_depth_heightmap, prev_primitive_action, prev_best_pix_ind, label_value)
            
            # Double DQN: Target network 주기적 업데이트 (현재 heightmap으로 출력 검증)
            if trainer.double_dqn and trainer.iteration % trainer.target_update_freq == 0:
                trainer.update_target_network(color_heightmap, valid_depth_heightmap)

            # Do sampling for experience replay (균형 샘플링: 성공:실패 = 1:1)
            # 성공 버퍼에서 2개 + 실패 버퍼에서 2개 = 총 4개 균형 샘플링
            if experience_replay and not is_testing:
                samples_to_train = []  # 학습할 샘플 iteration 인덱스 리스트
                
                # 유효한 샘플 범위: label_value_log에 있는 샘플만 사용 가능
                # 현재 iteration의 샘플은 제외 (방금 추가된 샘플)
                max_valid_idx = len(trainer.label_value_log) - 1
                current_sample_idx = trainer.iteration - 1  # 방금 추가된 샘플
                
                # 유효한 성공/실패 버퍼 분리
                valid_success = [idx for idx in trainer.success_buffer
                                if idx <= max_valid_idx and idx != current_sample_idx]
                valid_failure = [idx for idx in trainer.failure_buffer
                                if idx <= max_valid_idx and idx != current_sample_idx]
                
                # 최소 1개의 성공 샘플 보장 (가능하다면), 나머지는 실패 샘플
                # 총 4개 샘플링 (1:3 비율 지향, 데이터 부족 시 유동적)
                target_batch_size = 4
                
                if len(valid_success) > 0:
                    n_success = 1
                    n_failure = target_batch_size - n_success
                else:
                    n_success = 0
                    n_failure = target_batch_size
                
                # 실패 버퍼가 부족한 경우 조정
                n_failure = min(n_failure, len(valid_failure))
                
                if n_success > 0:
                    success_samples = np.random.choice(valid_success, size=n_success, replace=False)
                    for idx in success_samples:
                        samples_to_train.append(('success', idx))
                if n_failure > 0:
                    failure_samples = np.random.choice(valid_failure, size=n_failure, replace=False)
                    for idx in failure_samples:
                        samples_to_train.append(('failure', idx))
                
                # 샘플링 결과 로그
                success_count = sum(1 for s in samples_to_train if s[0] == 'success')
                failure_count = sum(1 for s in samples_to_train if s[0] == 'failure')
                total_valid = len(valid_success) + len(valid_failure)
                print(f'[REPLAY] Sampling: {success_count} success, {failure_count} failure (valid={total_valid})')
                print(f'[REPLAY] Buffer: success={len(valid_success)}/{len(trainer.success_buffer)}, failure={len(valid_failure)}/{len(trainer.failure_buffer)}')
                
                # 샘플링된 데이터로 학습
                for sample_type, sample_iteration in samples_to_train:
                    try:
                        # 인덱스 범위 재검증 (안전장치)
                        if sample_iteration >= len(trainer.label_value_log):
                            print(f'[REPLAY] Skipping sample {sample_iteration}: index out of range (log size={len(trainer.label_value_log)})')
                            continue
                        if sample_iteration >= len(trainer.executed_action_log):
                            print(f'[REPLAY] Skipping sample {sample_iteration}: action log index out of range')
                            continue
                        
                        # Load sample RGB-D heightmap
                        sample_color_heightmap = cv2.imread(os.path.join(logger.color_heightmaps_directory, '%06d.0.color.png' % (sample_iteration)))
                        if sample_color_heightmap is None:
                            print(f'[REPLAY] Warning: Could not load color heightmap for iteration {sample_iteration}')
                            continue
                        sample_color_heightmap = cv2.cvtColor(sample_color_heightmap, cv2.COLOR_BGR2RGB)
                        sample_depth_heightmap = cv2.imread(os.path.join(logger.depth_heightmaps_directory, '%06d.0.depth.png' % (sample_iteration)), -1)
                        if sample_depth_heightmap is None:
                            print(f'[REPLAY] Warning: Could not load depth heightmap for iteration {sample_iteration}')
                            continue
                        sample_depth_heightmap = sample_depth_heightmap.astype(np.float32)/100000

                        # Get stored label value and action
                        sample_label_value = trainer.label_value_log[sample_iteration][0]
                        sample_best_pix_ind = (np.asarray(trainer.executed_action_log)[sample_iteration,1:4]).astype(int)
                        
                        # nan label 건너뛰기 (네트워크 손상 방지)
                        if np.isnan(sample_label_value):
                            print(f'[REPLAY] Skipping sample {sample_iteration}: label is nan')
                            continue
                        
                        print(f'[REPLAY] Training {sample_type} sample: iteration {sample_iteration}, label={sample_label_value:.4f}')
                        
                        # Backpropagate with stored label
                        trainer.backprop(sample_color_heightmap, sample_depth_heightmap, 'grasp', sample_best_pix_ind, sample_label_value)
                        
                    except IndexError as e:
                        print(f'[REPLAY] IndexError for sample {sample_iteration}: {e} (log sizes: label={len(trainer.label_value_log)}, action={len(trainer.executed_action_log)})')
                        continue
                    except Exception as e:
                        print(f'[REPLAY] Error processing sample {sample_iteration}: {e}')
                        continue
                
                if len(samples_to_train) == 0:
                    print('[REPLAY] No samples available yet. Skipping experience replay.')

        if not exit_called:

            # Run forward pass with network to get affordances
            grasp_predictions, state_feat = trainer.forward(color_heightmap, valid_depth_heightmap, is_volatile=True)

            # Execute best primitive action on robot in another thread
            nonlocal_variables['executing_action'] = True

        # Save model snapshot
        if not is_testing:
                logger.save_backup_model(trainer.model, method)
                if trainer.iteration % 50 == 0:
                    logger.save_model(trainer.iteration, trainer.model, method)
                    if trainer.use_cuda:
                        trainer.model = trainer.model.cuda()

        # Sync both action thread and training thread
        while nonlocal_variables['executing_action']:
            time.sleep(0.01)

        if exit_called:
            break

        # Save information for next training step
        prev_color_img = color_img.copy()
        prev_depth_img = depth_img.copy()
        prev_color_heightmap = color_heightmap.copy()
        prev_depth_heightmap = depth_heightmap.copy()
        prev_valid_depth_heightmap = valid_depth_heightmap.copy()
        prev_grasp_success = nonlocal_variables['grasp_success']
        prev_primitive_action = nonlocal_variables['primitive_action']
        prev_grasp_predictions = grasp_predictions.copy()
        prev_best_pix_ind = nonlocal_variables['best_pix_ind']

        trainer.iteration += 1
        iteration_time_1 = time.time()
        print('Time elapsed: %f' % (iteration_time_1-iteration_time_0))


if __name__ == '__main__':

    # Parse arguments
    parser = argparse.ArgumentParser(description='Train IRB360 delta robot to learn pick and place with deep reinforcement learning in PyTorch.')

    # --------------- Setup options ---------------
    parser.add_argument('--is_sim', dest='is_sim', action='store_true', default=False,                                    help='run in simulation?')
    parser.add_argument('--obj_mesh_dir', dest='obj_mesh_dir', action='store', default='objects/blocks',                  help='directory containing 3D mesh files (.obj) of objects to be added to simulation')
    parser.add_argument('--num_obj', dest='num_obj', type=int, action='store', default=10,                                 help='number of objects to add to simulation')
    parser.add_argument('--heightmap_resolution', dest='heightmap_resolution', type=float, action='store', default=0.00125, help='meters per pixel of heightmap (0.00125 = 320x320 for 0.4m workspace)')
    parser.add_argument('--random_seed', dest='random_seed', type=int, action='store', default=1234,                      help='random seed for simulation and neural net initialization')
    parser.add_argument('--cpu', dest='force_cpu', action='store_true', default=False,                                    help='force code to run in CPU mode')

    # ------------- Algorithm options -------------
    parser.add_argument('--method', dest='method', action='store', default='reinforcement',                               help='set to \'reactive\' (supervised learning) or \'reinforcement\' (reinforcement learning ie Q-learning)')
    parser.add_argument('--grasp_rewards', dest='grasp_rewards', action='store_true', default=True,                        help='use immediate rewards (from change detection) for pushing?')
    parser.add_argument('--future_reward_discount', dest='future_reward_discount', type=float, action='store', default=0.2)  # 논문 기준: γ = 0.2
    parser.add_argument('--experience_replay', dest='experience_replay', action='store_true', default=True,              help='use prioritized experience replay?')
    parser.add_argument('--double_dqn', dest='double_dqn', action='store_true', default=False,                            help='use Double DQN for more stable learning (prevents Q-value overestimation)')
    parser.add_argument('--target_update_freq', dest='target_update_freq', type=int, action='store', default=100,         help='frequency of target network update for Double DQN (iterations)')
    parser.add_argument('--dueling_dqn', dest='dueling_dqn', action='store_true', default=False,                          help='use Dueling DQN architecture (Value + Advantage streams)')
    parser.add_argument('--gripper_diameter', dest='gripper_diameter', type=float, action='store', default=0.005,       help='gripper diameter in meters for floor check (default: 0.030m = 30mm)')

    # -------------- Testing options --------------
    parser.add_argument('--is_testing', dest='is_testing', action='store_true', default=False)
    parser.add_argument('--max_test_trials', dest='max_test_trials', type=int, action='store', default=30,                help='maximum number of test runs per case/scenario')
    parser.add_argument('--test_preset_cases', dest='test_preset_cases', action='store_true', default=False)
    parser.add_argument('--test_preset_file', dest='test_preset_file', action='store', default='test-cases/SCS-1.txt')

    # ------ Pre-loading and logging options ------
    parser.add_argument('--load_snapshot', dest='load_snapshot', action='store_true', default=False,                      help='load pre-trained snapshot of model?')
    parser.add_argument('--snapshot_file', dest='snapshot_file', action='store')
    parser.add_argument('--continue_logging', dest='continue_logging', action='store_true', default=False,                help='continue logging from previous session?')
    parser.add_argument('--logging_directory', dest='logging_directory', action='store')
    parser.add_argument('--save_visualizations', dest='save_visualizations', action='store_true', default=False,          help='save visualizations of FCN predictions?')

    # Run main program with specified arguments
    args = parser.parse_args()
    main(args)

