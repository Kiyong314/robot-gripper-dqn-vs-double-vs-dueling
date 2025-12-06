#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
오프라인 학습 스크립트 (Offline Training)

기존 학습 데이터를 사용하여 시뮬레이션 없이 DQN 변형들을 학습합니다.

사용법:
    # Standard DQN
    python train_offline.py --data_dir logs/2025-12-02.22.21.35
    
    # Double DQN
    python train_offline.py --data_dir logs/2025-12-02.22.21.35 --double_dqn
    
    # Dueling DQN
    python train_offline.py --data_dir logs/2025-12-02.22.21.35 --dueling_dqn
    
    # Double + Dueling DQN
    python train_offline.py --data_dir logs/2025-12-02.22.21.35 --double_dqn --dueling_dqn

장점:
    - 시뮬레이션 실행 불필요 (빠른 학습)
    - 동일한 데이터로 여러 알고리즘 비교 가능
    - GPU 메모리만 사용 (시뮬레이션 메모리 X)
"""

import os
import argparse
import numpy as np
import cv2
import torch
import time
from datetime import datetime
import matplotlib.pyplot as plt
from scipy import ndimage

from DQNModels import DQN
from DQNTrainer import DQNTrainer


class OfflineDataset:
    """
    오프라인 학습용 데이터셋 클래스
    
    기존 학습 로그에서 데이터를 로드하여 제공합니다.
    """
    
    def __init__(self, data_dir):
        """
        데이터셋 초기화
        
        Args:
            data_dir: 학습 로그 폴더 경로 (예: logs/2025-12-02.22.21.35)
        """
        self.data_dir = data_dir
        self.color_heightmaps_dir = os.path.join(data_dir, 'data', 'color-heightmaps')
        self.depth_heightmaps_dir = os.path.join(data_dir, 'data', 'depth-heightmaps')
        self.transitions_dir = os.path.join(data_dir, 'transitions')
        
        # 로그 데이터 로드
        self._load_logs()
        
        print(f'[DATASET] Loaded {self.num_samples} samples from {data_dir}')
        print(f'[DATASET] Success rate: {self.success_rate:.2f}%')
    
    def _load_logs(self):
        """로그 파일 로드"""
        # executed-action.log.txt: [action_type, rotation, y, x]
        self.executed_actions = np.loadtxt(
            os.path.join(self.transitions_dir, 'executed-action.log.txt')
        )
        
        # label-value.log.txt: [label_value]
        self.label_values = np.loadtxt(
            os.path.join(self.transitions_dir, 'label-value.log.txt')
        )
        
        # reward-value.log.txt: [reward_value]
        self.reward_values = np.loadtxt(
            os.path.join(self.transitions_dir, 'reward-value.log.txt')
        )
        
        # grasp-success.log.txt: [success]
        self.grasp_success = np.loadtxt(
            os.path.join(self.transitions_dir, 'grasp-success.log.txt')
        )
        
        # 샘플 수 확인
        self.num_samples = len(self.label_values)
        
        # 성공률 계산
        success_count = np.sum(self.grasp_success == 1)
        self.success_rate = 100.0 * success_count / self.num_samples
    
    def get_sample(self, idx):
        """
        특정 인덱스의 샘플 반환
        
        Args:
            idx: 샘플 인덱스
            
        Returns:
            dict: {
                'color_heightmap': (H, W, 3) uint8,
                'depth_heightmap': (H, W) float32,
                'action': [action_type, rotation, y, x],
                'label_value': float,
                'reward_value': float,
                'grasp_success': int
            }
        """
        # 이미지 로드
        color_path = os.path.join(self.color_heightmaps_dir, f'{idx:06d}.0.color.png')
        depth_path = os.path.join(self.depth_heightmaps_dir, f'{idx:06d}.0.depth.png')
        
        color_heightmap = cv2.imread(color_path)
        if color_heightmap is None:
            return None
        color_heightmap = cv2.cvtColor(color_heightmap, cv2.COLOR_BGR2RGB)
        
        depth_heightmap = cv2.imread(depth_path, -1)
        if depth_heightmap is None:
            return None
        depth_heightmap = depth_heightmap.astype(np.float32) / 100000  # 저장 시 100000 곱함
        
        # 로그 데이터
        action = self.executed_actions[idx]
        label_value = self.label_values[idx]
        reward_value = self.reward_values[idx]
        grasp_success = self.grasp_success[idx]
        
        return {
            'color_heightmap': color_heightmap,
            'depth_heightmap': depth_heightmap,
            'action': action,
            'label_value': label_value,
            'reward_value': reward_value,
            'grasp_success': grasp_success
        }
    
    def get_batch_indices(self, batch_size, balanced=True):
        """
        배치 인덱스 생성
        
        Args:
            batch_size: 배치 크기
            balanced: True면 성공/실패 샘플 균형 맞춤
            
        Returns:
            list: 샘플 인덱스 리스트
        """
        if balanced:
            # 성공/실패 샘플 분리
            success_indices = np.where(self.grasp_success == 1)[0]
            failure_indices = np.where(self.grasp_success != 1)[0]
            
            # 성공:실패 = 1:1 비율로 샘플링
            n_success = min(batch_size // 2, len(success_indices))
            n_failure = min(batch_size - n_success, len(failure_indices))
            
            if len(success_indices) > 0 and n_success > 0:
                success_batch = np.random.choice(success_indices, n_success, replace=False)
            else:
                success_batch = np.array([], dtype=int)
            
            if len(failure_indices) > 0 and n_failure > 0:
                failure_batch = np.random.choice(failure_indices, n_failure, replace=False)
            else:
                failure_batch = np.array([], dtype=int)
            
            indices = np.concatenate([success_batch, failure_batch])
            np.random.shuffle(indices)
            return indices.tolist()
        else:
            # 무작위 샘플링
            return np.random.choice(self.num_samples, batch_size, replace=False).tolist()


class OfflineTrainer:
    """
    오프라인 학습 트레이너
    
    기존 데이터로 DQN 변형들을 학습합니다.
    """
    
    def __init__(self, args):
        """
        트레이너 초기화
        
        Args:
            args: 명령줄 인자
        """
        self.args = args
        self.use_cuda = torch.cuda.is_available() and not args.force_cpu
        
        # 데이터셋 로드
        self.dataset = OfflineDataset(args.data_dir)
        
        # 모델 초기화
        self._init_model()
        
        # 출력 디렉토리 생성
        self._init_output_dir()
        
        # 학습 통계
        self.losses = []
        self.iteration = 0
    
    def _init_model(self):
        """모델 및 옵티마이저 초기화"""
        # DQN 모델 생성
        self.model = DQN(self.use_cuda, dueling=self.args.dueling_dqn)
        
        if self.use_cuda:
            self.model = self.model.cuda()
        
        # 기존 모델 로드 (선택사항)
        if self.args.load_snapshot:
            print(f'[MODEL] Loading snapshot: {self.args.snapshot_file}')
            if self.use_cuda:
                self.model.load_state_dict(torch.load(self.args.snapshot_file))
            else:
                self.model.load_state_dict(
                    torch.load(self.args.snapshot_file, map_location=torch.device('cpu'))
                )
        
        self.model.train()
        
        # 옵티마이저
        self.optimizer = torch.optim.SGD(
            self.model.parameters(), 
            lr=self.args.learning_rate, 
            momentum=0.9, 
            weight_decay=2e-5
        )
        
        # Loss 함수
        self.criterion = torch.nn.SmoothL1Loss(reduce=False)
        if self.use_cuda:
            self.criterion = self.criterion.cuda()
        
        # Double DQN: Target Network
        self.target_model = None
        if self.args.double_dqn:
            self.target_model = DQN(self.use_cuda, dueling=self.args.dueling_dqn)
            if self.use_cuda:
                self.target_model = self.target_model.cuda()
            self.target_model.load_state_dict(self.model.state_dict())
            self.target_model.eval()
            print('[MODEL] Double DQN enabled with target network')
        
        # 모델 정보 출력
        model_type = []
        if self.args.double_dqn:
            model_type.append('Double')
        if self.args.dueling_dqn:
            model_type.append('Dueling')
        if not model_type:
            model_type.append('Standard')
        print(f'[MODEL] Type: {" + ".join(model_type)} DQN')
        print(f'[MODEL] CUDA: {self.use_cuda}')
    
    def _init_output_dir(self):
        """출력 디렉토리 초기화"""
        # 모델 타입 이름
        model_name = 'dqn'
        if self.args.double_dqn and self.args.dueling_dqn:
            model_name = 'double_dueling_dqn'
        elif self.args.double_dqn:
            model_name = 'double_dqn'
        elif self.args.dueling_dqn:
            model_name = 'dueling_dqn'
        
        # 출력 디렉토리
        timestamp = datetime.now().strftime('%Y-%m-%d.%H.%M.%S')
        self.output_dir = os.path.join('logs', f'{timestamp}_offline_{model_name}')
        self.models_dir = os.path.join(self.output_dir, 'models')
        
        os.makedirs(self.models_dir, exist_ok=True)
        print(f'[OUTPUT] Directory: {self.output_dir}')
    
    def forward(self, color_heightmap, depth_heightmap):
        """
        모델 forward pass (DQNTrainer.forward와 동일한 전처리)
        """
        # 2x 스케일 적용
        color_heightmap_2x = ndimage.zoom(color_heightmap, zoom=[2, 2, 1], order=0)
        depth_heightmap_2x = ndimage.zoom(depth_heightmap, zoom=[2, 2], order=0)
        
        # 패딩 추가 (회전 처리용)
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
        
        # 전처리: 정규화
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
        
        return input_color_data, input_depth_data, padding_width, color_heightmap_2x.shape[0]
    
    def backprop(self, color_heightmap, depth_heightmap, best_pix_ind, label_value):
        """
        역전파 수행
        
        Args:
            color_heightmap: 컬러 heightmap (H, W, 3)
            depth_heightmap: 깊이 heightmap (H, W)
            best_pix_ind: 최적 픽셀 인덱스 [rotation, y, x]
            label_value: 레이블 값 (target Q-value)
            
        Returns:
            float: loss 값
        """
        # NaN 체크
        if np.isnan(label_value):
            return 0.0
        
        # Forward pass
        input_color_data, input_depth_data, padding_width, img_size = self.forward(
            color_heightmap, depth_heightmap
        )
        
        # 특정 rotation으로 forward
        output_prob, _ = self.model.forward(
            input_color_data, input_depth_data, 
            is_volatile=False, 
            specific_rotation=int(best_pix_ind[0])
        )
        
        # 출력 크기 계산
        output_shape = self.model.output_prob[0][0].shape
        output_size = (int(output_shape[-2]), int(output_shape[-1]))
        
        # 레이블 생성
        input_size = (color_heightmap.shape[0], color_heightmap.shape[1])
        padding = (output_size[0] - input_size[0]) // 2
        
        label = np.zeros((1, output_size[0], output_size[1]))
        action_area = np.zeros(input_size)
        action_area[int(best_pix_ind[1])][int(best_pix_ind[2])] = 1
        tmp_label = np.zeros(input_size)
        tmp_label[action_area > 0] = label_value
        label[0, padding:(output_size[0] - padding), padding:(output_size[1] - padding)] = tmp_label
        
        # 레이블 가중치
        label_weights = np.zeros(label.shape)
        tmp_label_weights = np.zeros(input_size)
        tmp_label_weights[action_area > 0] = 1
        label_weights[0, padding:(output_size[0] - padding), padding:(output_size[1] - padding)] = tmp_label_weights
        
        # Loss 계산 및 역전파
        self.optimizer.zero_grad()
        
        if self.use_cuda:
            loss = self.criterion(
                self.model.output_prob[0][0].view(1, output_size[0], output_size[1]),
                torch.from_numpy(label).float().cuda()
            ) * torch.from_numpy(label_weights).float().cuda()
        else:
            loss = self.criterion(
                self.model.output_prob[0][0].view(1, output_size[0], output_size[1]),
                torch.from_numpy(label).float()
            ) * torch.from_numpy(label_weights).float()
        
        loss = loss.sum()
        loss.backward()
        self.optimizer.step()
        
        return loss.cpu().data.numpy()
    
    def train_epoch(self, epoch):
        """
        한 에폭 학습
        
        Args:
            epoch: 현재 에폭 번호
            
        Returns:
            float: 평균 loss
        """
        epoch_losses = []
        batch_size = self.args.batch_size
        num_batches = self.dataset.num_samples // batch_size
        
        print(f'\n[EPOCH {epoch+1}/{self.args.epochs}] Starting...')
        start_time = time.time()
        
        for batch_idx in range(num_batches):
            # 배치 인덱스 생성
            indices = self.dataset.get_batch_indices(batch_size, balanced=True)
            
            batch_loss = 0.0
            valid_samples = 0
            
            for idx in indices:
                sample = self.dataset.get_sample(idx)
                if sample is None:
                    continue
                
                # 역전파
                best_pix_ind = [
                    sample['action'][1],  # rotation
                    sample['action'][2],  # y
                    sample['action'][3]   # x
                ]
                
                loss = self.backprop(
                    sample['color_heightmap'],
                    sample['depth_heightmap'],
                    best_pix_ind,
                    sample['label_value']
                )
                
                batch_loss += loss
                valid_samples += 1
                self.iteration += 1
            
            if valid_samples > 0:
                avg_batch_loss = batch_loss / valid_samples
                epoch_losses.append(avg_batch_loss)
            
            # 진행 상황 출력 (10배치마다)
            if (batch_idx + 1) % 10 == 0:
                elapsed = time.time() - start_time
                print(f'  Batch {batch_idx+1}/{num_batches}, Loss: {avg_batch_loss:.4f}, Time: {elapsed:.1f}s')
            
            # Double DQN: Target network 업데이트
            if self.args.double_dqn and self.iteration % self.args.target_update_freq == 0:
                self.target_model.load_state_dict(self.model.state_dict())
                print(f'  [Double DQN] Target network updated at iteration {self.iteration}')
        
        # 에폭 종료
        epoch_time = time.time() - start_time
        avg_epoch_loss = np.mean(epoch_losses) if epoch_losses else 0.0
        
        print(f'[EPOCH {epoch+1}] Completed in {epoch_time:.1f}s, Avg Loss: {avg_epoch_loss:.4f}')
        
        return avg_epoch_loss
    
    def save_model(self, epoch):
        """모델 저장"""
        model_path = os.path.join(self.models_dir, f'snapshot-epoch{epoch+1:03d}.pth')
        torch.save(self.model.cpu().state_dict(), model_path)
        if self.use_cuda:
            self.model = self.model.cuda()
        print(f'[SAVE] Model saved: {model_path}')
        
        # 백업 모델도 저장
        backup_path = os.path.join(self.models_dir, 'snapshot-backup.pth')
        torch.save(self.model.state_dict(), backup_path)
    
    def plot_loss_curve(self):
        """학습 곡선 저장"""
        if not self.losses:
            return
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.losses)
        plt.xlabel('Epoch')
        plt.ylabel('Average Loss')
        plt.title('Training Loss Curve')
        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, 'loss_curve.png'))
        plt.close()
        print(f'[SAVE] Loss curve saved')
    
    def train(self):
        """전체 학습 실행"""
        print('\n' + '='*60)
        print('OFFLINE TRAINING START')
        print('='*60)
        
        total_start_time = time.time()
        
        for epoch in range(self.args.epochs):
            avg_loss = self.train_epoch(epoch)
            self.losses.append(avg_loss)
            
            # 모델 저장 (매 에폭마다)
            self.save_model(epoch)
            
            # 학습 곡선 업데이트
            self.plot_loss_curve()
        
        total_time = time.time() - total_start_time
        print('\n' + '='*60)
        print(f'TRAINING COMPLETED in {total_time/60:.1f} minutes')
        print(f'Final model: {self.models_dir}')
        print('='*60)


def main():
    parser = argparse.ArgumentParser(description='Offline DQN Training')
    
    # 데이터 옵션
    parser.add_argument('--data_dir', type=str, required=True,
                        help='학습 데이터 폴더 경로 (예: logs/2025-12-02.22.21.35)')
    
    # 알고리즘 옵션
    parser.add_argument('--double_dqn', action='store_true', default=False,
                        help='Double DQN 사용')
    parser.add_argument('--dueling_dqn', action='store_true', default=False,
                        help='Dueling DQN 사용')
    parser.add_argument('--target_update_freq', type=int, default=100,
                        help='Target network 업데이트 주기 (Double DQN)')
    
    # 학습 옵션
    parser.add_argument('--epochs', type=int, default=50,
                        help='학습 에폭 수')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='배치 크기')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='학습률')
    parser.add_argument('--force_cpu', action='store_true', default=False,
                        help='CPU 강제 사용')
    
    # 모델 로드 옵션
    parser.add_argument('--load_snapshot', action='store_true', default=False,
                        help='기존 모델에서 시작')
    parser.add_argument('--snapshot_file', type=str, default=None,
                        help='로드할 모델 파일 경로')
    
    args = parser.parse_args()
    
    # 학습 실행
    trainer = OfflineTrainer(args)
    trainer.train()


if __name__ == '__main__':
    main()


