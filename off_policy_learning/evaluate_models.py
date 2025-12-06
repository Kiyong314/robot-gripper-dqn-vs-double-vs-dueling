#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
모델 평가 스크립트

학습된 모든 체크포인트를 CoppeliaSim 시뮬레이션에서 평가하고 JSON 로그로 저장합니다.
그래프 생성은 compare_results.py에서 수행합니다.

사용법:
    python evaluate_models.py --trials 10 --objects 10
"""

import os
import sys
import argparse
import json
import glob
import numpy as np
import torch
import time
from datetime import datetime

# 프로젝트 루트를 path에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from DQNModels import DQN
from config import (
    RESULTS_DIR, EVALUATION_DIR, MODEL_CONFIGS,
    EVALUATION_CONFIG, SIMULATION_CONFIG,
    get_model_result_dir, get_model_eval_dir
)


class ModelEvaluator:
    """
    모델 평가 클래스
    
    CoppeliaSim 시뮬레이션을 통해 학습된 모델의 Grasp 성공률을 평가합니다.
    """
    
    def __init__(self, model_name, model_path, use_cuda=True):
        """
        평가기 초기화
        
        Args:
            model_name: 모델 이름 (예: 'standard_dqn')
            model_path: 가중치 파일 경로
            use_cuda: CUDA 사용 여부
        """
        self.model_name = model_name
        self.model_path = model_path
        self.use_cuda = use_cuda and torch.cuda.is_available()
        
        # 모델 설정
        self.config = MODEL_CONFIGS.get(model_name, {})
        
        # 모델 로드
        self.model = self._load_model()
        
    def _load_model(self):
        """가중치 파일에서 모델 로드"""
        dueling = self.config.get('dueling_dqn', False)
        model = DQN(self.use_cuda, dueling=dueling)
        
        # 가중치 로드
        state_dict = torch.load(self.model_path, map_location='cpu')
        model.load_state_dict(state_dict)
        
        if self.use_cuda:
            model = model.cuda()
        
        model.eval()
        return model
    
    def evaluate_trial(self, robot, num_objects):
        """
        단일 시행 평가
        
        Args:
            robot: 로봇 인스턴스
            num_objects: 물체 개수
            
        Returns:
            dict: 시행 결과
        """
        from scipy import ndimage
        
        trial_start = time.time()
        
        # 물체 배치
        robot.add_objects(num_objects)
        time.sleep(1)  # 물체 안정화 대기
        
        grasp_attempts = 0
        grasp_successes = 0
        q_values = []
        
        # 최대 시도 횟수 = 물체 개수 * 2
        max_attempts = num_objects * 2
        
        for attempt in range(max_attempts):
            # RGB-D 이미지 획득
            color_img, depth_img = robot.get_camera_data()
            
            if color_img is None or depth_img is None:
                continue
            
            # Heightmap 생성
            color_heightmap, depth_heightmap = robot.get_heightmap(color_img, depth_img)
            
            # 모델 추론
            with torch.no_grad():
                input_color, input_depth, _ = self._preprocess(color_heightmap, depth_heightmap)
                output_prob, _ = self.model.forward(input_color, input_depth, is_volatile=True)
            
            # 최적 Grasp 위치 선택
            grasp_predictions = output_prob[0].cpu().data.numpy()
            best_pix_ind = np.unravel_index(np.argmax(grasp_predictions), grasp_predictions.shape)
            
            q_value = grasp_predictions[best_pix_ind]
            q_values.append(float(q_value))
            
            # Grasp 실행
            grasp_attempts += 1
            primitive_position = robot.pix_to_position(best_pix_ind[1], best_pix_ind[2])
            rotation_angle = best_pix_ind[0] * (np.pi / 8)  # 16개 회전 각도
            
            success = robot.grasp(primitive_position, rotation_angle)
            
            if success:
                grasp_successes += 1
                # 물체 제거 (성공 시)
                robot.remove_grasped_object()
            
            # 모든 물체 제거됐는지 확인
            remaining = robot.get_remaining_objects()
            if remaining == 0:
                break
        
        trial_time = time.time() - trial_start
        
        return {
            'grasp_attempts': grasp_attempts,
            'grasp_successes': grasp_successes,
            'success_rate': grasp_successes / grasp_attempts if grasp_attempts > 0 else 0,
            'avg_q_value': float(np.mean(q_values)) if q_values else 0,
            'max_q_value': float(np.max(q_values)) if q_values else 0,
            'trial_time': trial_time,
        }
    
    def _preprocess(self, color_heightmap, depth_heightmap):
        """이미지 전처리 (train_offline.py와 동일)"""
        from scipy import ndimage
        
        # 2x 스케일
        color_heightmap_2x = ndimage.zoom(color_heightmap, zoom=[2, 2, 1], order=0)
        depth_heightmap_2x = ndimage.zoom(depth_heightmap, zoom=[2, 2], order=0)
        
        # 패딩
        diag_length = float(color_heightmap_2x.shape[0]) * np.sqrt(2)
        diag_length = np.ceil(diag_length / 32) * 32
        padding_width = int((diag_length - color_heightmap_2x.shape[0]) / 2)
        
        color_heightmap_2x_r = np.pad(color_heightmap_2x[:, :, 0], padding_width, 'constant', constant_values=0)
        color_heightmap_2x_r.shape = (color_heightmap_2x_r.shape[0], color_heightmap_2x_r.shape[1], 1)
        color_heightmap_2x_g = np.pad(color_heightmap_2x[:, :, 1], padding_width, 'constant', constant_values=0)
        color_heightmap_2x_g.shape = (color_heightmap_2x_g.shape[0], color_heightmap_2x_g.shape[1], 1)
        color_heightmap_2x_b = np.pad(color_heightmap_2x[:, :, 2], padding_width, 'constant', constant_values=0)
        color_heightmap_2x_b.shape = (color_heightmap_2x_b.shape[0], color_heightmap_2x_b.shape[1], 1)
        color_heightmap_2x = np.concatenate((color_heightmap_2x_r, color_heightmap_2x_g, color_heightmap_2x_b), axis=2)
        depth_heightmap_2x = np.pad(depth_heightmap_2x, padding_width, 'constant', constant_values=0)
        
        # 정규화
        image_mean = [0.485, 0.456, 0.406]
        image_std = [0.229, 0.224, 0.225]
        input_color_image = color_heightmap_2x.astype(float) / 255
        for c in range(3):
            input_color_image[:, :, c] = (input_color_image[:, :, c] - image_mean[c]) / image_std[c]
        
        depth_heightmap_2x.shape = (depth_heightmap_2x.shape[0], depth_heightmap_2x.shape[1], 1)
        input_depth_image = (depth_heightmap_2x - 0.01) / 0.03
        
        # 텐서 변환
        input_color_image.shape = (input_color_image.shape[0], input_color_image.shape[1], input_color_image.shape[2], 1)
        input_depth_image.shape = (input_depth_image.shape[0], input_depth_image.shape[1], input_depth_image.shape[2], 1)
        input_color_data = torch.from_numpy(input_color_image.astype(np.float32)).permute(3, 2, 0, 1)
        input_depth_data = torch.from_numpy(input_depth_image.astype(np.float32)).permute(3, 2, 0, 1)
        
        return input_color_data, input_depth_data, padding_width
    
    def evaluate(self, num_trials, num_objects):
        """
        전체 평가 실행 (시뮬레이션 필요)
        
        Args:
            num_trials: 시행 횟수
            num_objects: 물체 개수
            
        Returns:
            dict: 평가 결과
        """
        # 로봇 초기화 (CoppeliaSim 연결)
        try:
            from test.robot_zmq_irb360 import RobotZMQ
            robot = RobotZMQ(
                workspace_limits=SIMULATION_CONFIG['workspace_limits'],
                obj_mesh_dir=SIMULATION_CONFIG['obj_mesh_dir']
            )
        except Exception as e:
            print(f'[ERROR] Cannot connect to simulation: {e}')
            return None
        
        trials = []
        total_successes = 0
        total_attempts = 0
        
        print(f'\n[EVAL] Evaluating: {os.path.basename(self.model_path)}')
        print(f'[EVAL] Trials: {num_trials}, Objects: {num_objects}')
        
        for trial_idx in range(num_trials):
            trial_result = self.evaluate_trial(robot, num_objects)
            trials.append(trial_result)
            
            total_successes += trial_result['grasp_successes']
            total_attempts += trial_result['grasp_attempts']
            
            print(f'  Trial {trial_idx+1}/{num_trials}: '
                  f'{trial_result["grasp_successes"]}/{trial_result["grasp_attempts"]} '
                  f'({trial_result["success_rate"]*100:.1f}%)')
            
            # 다음 시행 전 초기화
            robot.restart_sim()
        
        # 종합 결과
        overall_success_rate = total_successes / total_attempts if total_attempts > 0 else 0
        avg_q_values = [t['avg_q_value'] for t in trials]
        
        result = {
            'model_name': self.model_name,
            'model_path': self.model_path,
            'num_trials': num_trials,
            'num_objects': num_objects,
            'total_attempts': total_attempts,
            'total_successes': total_successes,
            'overall_success_rate': overall_success_rate,
            'avg_q_value': float(np.mean(avg_q_values)),
            'trials': trials,
            'timestamp': datetime.now().isoformat(),
        }
        
        print(f'[EVAL] Overall: {total_successes}/{total_attempts} ({overall_success_rate*100:.1f}%)')
        
        return result


def evaluate_without_simulation(model_name, model_path):
    """
    시뮬레이션 없이 모델 상태만 확인 (테스트용)
    
    Args:
        model_name: 모델 이름
        model_path: 가중치 파일 경로
        
    Returns:
        dict: 기본 정보
    """
    try:
        config = MODEL_CONFIGS.get(model_name, {})
        dueling = config.get('dueling_dqn', False)
        
        # 모델 로드 테스트
        model = DQN(use_cuda=False, dueling=dueling)
        state_dict = torch.load(model_path, map_location='cpu')
        model.load_state_dict(state_dict)
        
        return {
            'model_name': model_name,
            'model_path': model_path,
            'status': 'loaded',
            'parameters': sum(p.numel() for p in model.parameters()),
            'timestamp': datetime.now().isoformat(),
        }
    except Exception as e:
        return {
            'model_name': model_name,
            'model_path': model_path,
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat(),
        }


def find_checkpoints(model_name):
    """
    모델의 모든 체크포인트 파일 찾기
    
    Args:
        model_name: 모델 이름
        
    Returns:
        list: [(epoch, file_path), ...] 정렬된 리스트
    """
    result_dir = get_model_result_dir(model_name)
    models_dir = os.path.join(result_dir, 'models')
    
    if not os.path.exists(models_dir):
        return []
    
    checkpoints = []
    for filename in os.listdir(models_dir):
        if filename.endswith('.pth'):
            # epoch_000.pth 형식
            try:
                epoch = int(filename.split('_')[1].split('.')[0])
                filepath = os.path.join(models_dir, filename)
                checkpoints.append((epoch, filepath))
            except:
                continue
    
    # 에폭 순으로 정렬
    checkpoints.sort(key=lambda x: x[0])
    return checkpoints


def evaluate_all_models(num_trials, num_objects, use_simulation=True):
    """
    모든 모델의 모든 체크포인트 평가
    
    Args:
        num_trials: 시행 횟수
        num_objects: 물체 개수
        use_simulation: 시뮬레이션 사용 여부
        
    Returns:
        dict: 평가 결과
    """
    all_results = {}
    
    for model_name in MODEL_CONFIGS.keys():
        print(f'\n{"="*60}')
        print(f'EVALUATING: {MODEL_CONFIGS[model_name]["name"]}')
        print(f'{"="*60}')
        
        checkpoints = find_checkpoints(model_name)
        
        if not checkpoints:
            print(f'[WARNING] No checkpoints found for {model_name}')
            continue
        
        print(f'Found {len(checkpoints)} checkpoints')
        
        model_results = []
        eval_dir = get_model_eval_dir(model_name)
        
        for epoch, model_path in checkpoints:
            print(f'\n--- Epoch {epoch} ---')
            
            if use_simulation:
                evaluator = ModelEvaluator(model_name, model_path)
                result = evaluator.evaluate(num_trials, num_objects)
            else:
                result = evaluate_without_simulation(model_name, model_path)
            
            if result:
                result['epoch'] = epoch
                model_results.append(result)
                
                # JSON 파일 저장
                eval_path = os.path.join(eval_dir, f'epoch_{epoch:03d}_eval.json')
                with open(eval_path, 'w') as f:
                    json.dump(result, f, indent=2)
                print(f'[SAVE] {eval_path}')
        
        all_results[model_name] = model_results
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description='Model Evaluation')
    parser.add_argument('--trials', type=int, default=10,
                        help='체크포인트당 테스트 횟수')
    parser.add_argument('--objects', type=int, default=10,
                        help='시뮬레이션당 물체 개수')
    parser.add_argument('--no-sim', action='store_true',
                        help='시뮬레이션 없이 모델 로드만 테스트')
    parser.add_argument('--model', type=str, default='all',
                        help='평가할 모델 (all 또는 특정 모델명)')
    
    args = parser.parse_args()
    
    print('='*60)
    print('MODEL EVALUATION')
    print('='*60)
    print(f'Trials per checkpoint: {args.trials}')
    print(f'Objects per trial: {args.objects}')
    print(f'Use simulation: {not args.no_sim}')
    
    # 평가 실행
    if args.model == 'all':
        results = evaluate_all_models(args.trials, args.objects, not args.no_sim)
    else:
        # 특정 모델만 평가
        if args.model not in MODEL_CONFIGS:
            print(f'[ERROR] Unknown model: {args.model}')
            return
        
        checkpoints = find_checkpoints(args.model)
        if not checkpoints:
            print(f'[ERROR] No checkpoints found for {args.model}')
            return
        
        model_results = []
        eval_dir = get_model_eval_dir(args.model)
        
        for epoch, model_path in checkpoints:
            if not args.no_sim:
                evaluator = ModelEvaluator(args.model, model_path)
                result = evaluator.evaluate(args.trials, args.objects)
            else:
                result = evaluate_without_simulation(args.model, model_path)
            
            if result:
                result['epoch'] = epoch
                model_results.append(result)
                
                eval_path = os.path.join(eval_dir, f'epoch_{epoch:03d}_eval.json')
                with open(eval_path, 'w') as f:
                    json.dump(result, f, indent=2)
        
        results = {args.model: model_results}
    
    # 결과 요약
    print('\n' + '='*60)
    print('EVALUATION SUMMARY')
    print('='*60)
    
    for model_name, model_results in results.items():
        if model_results:
            # 마지막 체크포인트 결과
            last_result = model_results[-1]
            if 'overall_success_rate' in last_result:
                print(f'{model_name}: {last_result["overall_success_rate"]*100:.1f}% success rate (epoch {last_result["epoch"]})')
            else:
                print(f'{model_name}: {last_result.get("status", "unknown")} (epoch {last_result["epoch"]})')
    
    print(f'\nEvaluation logs saved to: {EVALUATION_DIR}')
    print('Run compare_results.py to generate comparison graphs.')


if __name__ == '__main__':
    main()


