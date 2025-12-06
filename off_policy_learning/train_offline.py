#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Off-Policy 오프라인 학습 스크립트 (병렬 프리패치 버전)

최적화:
1. PyTorch DataLoader - 병렬 데이터 로딩 (num_workers)
2. 프리패치 - GPU 처리 중 다음 배치 미리 준비
3. 에폭별 시각화 - 랜덤 샘플 Q값 히트맵

사용법:
    python train_offline.py --model all
    python train_offline.py --model standard_dqn --epochs 100 --batch_size 64
"""

import os
import sys
import argparse
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import time
import json
from datetime import datetime
import matplotlib.pyplot as plt
from scipy import ndimage
from tqdm import tqdm

# 프로젝트 루트를 path에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from DQNModels import DQN
from config import (
    DATA_SOURCES, RESULTS_DIR, MODEL_CONFIGS,
    TRAINING_CONFIG, SAMPLING_CONFIG, VISUALIZATION_CONFIG,
    get_model_result_dir
)


class OfflineTorchDataset(Dataset):
    """
    PyTorch Dataset - 병렬 로딩 지원
    
    DataLoader의 num_workers를 통해 여러 프로세스에서 병렬로 데이터를 로드합니다.
    """
    
    def __init__(self, data_sources):
        """
        데이터셋 초기화
        
        Args:
            data_sources: 학습 로그 폴더 경로 리스트
        """
        self.samples = []  # [(source_idx, sample_idx), ...]
        self.sources_data = []
        
        # 각 데이터 소스 로드 (메타데이터만)
        for source_path in data_sources:
            if os.path.exists(source_path):
                data = self._load_source_metadata(source_path)
                if data is not None:
                    self.sources_data.append(data)
                    print(f'[DATASET] Loaded {data["num_samples"]} samples from {os.path.basename(source_path)}')
        
        # 전체 샘플 인덱스 생성
        self._build_sample_index()
        
        # 전처리 파라미터 초기화
        self._init_preprocess_params()
        
        print(f'[DATASET] Total: {len(self.samples)} samples')
        print(f'[DATASET] Success: {self.total_success} ({100*self.total_success/len(self.samples):.1f}%)')
        print(f'[DATASET] Failure: {self.total_failure} ({100*self.total_failure/len(self.samples):.1f}%)')
    
    def _load_source_metadata(self, source_path):
        """데이터 소스의 메타데이터만 로드 (이미지는 나중에)"""
        transitions_dir = os.path.join(source_path, 'transitions')
        
        try:
            executed_actions = np.loadtxt(os.path.join(transitions_dir, 'executed-action.log.txt'))
            label_values = np.loadtxt(os.path.join(transitions_dir, 'label-value.log.txt'))
            grasp_success = np.loadtxt(os.path.join(transitions_dir, 'grasp-success.log.txt'))
            
            if label_values.ndim == 0:
                label_values = np.array([label_values])
            if grasp_success.ndim == 0:
                grasp_success = np.array([grasp_success])
            
            return {
                'path': source_path,
                'color_dir': os.path.join(source_path, 'data', 'color-heightmaps'),
                'depth_dir': os.path.join(source_path, 'data', 'depth-heightmaps'),
                'actions': executed_actions,
                'labels': label_values,
                'success': grasp_success,
                'num_samples': len(label_values),
            }
        except Exception as e:
            print(f'[DATASET] Error loading {source_path}: {e}')
            return None
    
    def _build_sample_index(self):
        """전체 샘플 인덱스 생성"""
        self.samples = []
        self.success_indices = []
        self.failure_indices = []
        
        global_idx = 0
        for source_idx, data in enumerate(self.sources_data):
            for sample_idx in range(data['num_samples']):
                self.samples.append((source_idx, sample_idx))
                
                success_val = data['success'][sample_idx]
                if success_val == 1:
                    self.success_indices.append(global_idx)
                else:
                    self.failure_indices.append(global_idx)
                
                global_idx += 1
        
        self.total_success = len(self.success_indices)
        self.total_failure = len(self.failure_indices)
    
    def _init_preprocess_params(self):
        """전처리 파라미터 초기화 (패딩 제거 버전)"""
        # 첫 이미지로 크기 계산
        source_idx, sample_idx = self.samples[0]
        data = self.sources_data[source_idx]
        color_path = os.path.join(data['color_dir'], f'{sample_idx:06d}.0.color.png')
        
        img = cv2.imread(color_path)
        if img is None:
            raise ValueError(f"Cannot load first image: {color_path}")
        
        h, w = img.shape[:2]
        
        # [패딩 제거] 2x 스케일만 적용 (320 → 640)
        self.input_size = (h, w)  # 원본 크기 (320, 320)
        self.output_size = h * 2   # 2x 스케일 크기 (640)
        
        # 정규화 파라미터
        self.image_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.image_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        단일 샘플 반환 (워커 프로세스에서 병렬 실행)
        
        Returns:
            dict: 전처리된 샘플 데이터
        """
        source_idx, sample_idx = self.samples[idx]
        data = self.sources_data[source_idx]
        
        # 이미지 로드
        color_path = os.path.join(data['color_dir'], f'{sample_idx:06d}.0.color.png')
        depth_path = os.path.join(data['depth_dir'], f'{sample_idx:06d}.0.depth.png')
        
        color_img = cv2.imread(color_path)
        depth_img = cv2.imread(depth_path, -1)
        
        if color_img is None or depth_img is None:
            # 로드 실패 시 빈 데이터 반환
            return self._get_empty_sample()
        
        color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
        depth_img = depth_img.astype(np.float32) / 100000
        
        # 전처리
        color_t, depth_t = self._preprocess(color_img, depth_img)
        
        # 메타데이터
        action = data['actions'][sample_idx]
        label_value = data['labels'][sample_idx]
        grasp_success = data['success'][sample_idx]
        
        return {
            'color': color_t,
            'depth': depth_t,
            'action': torch.tensor(action, dtype=torch.float32),
            'label_value': torch.tensor(label_value, dtype=torch.float32),
            'grasp_success': torch.tensor(grasp_success, dtype=torch.float32),
            'idx': idx,
            'valid': True,
        }
    
    def _get_empty_sample(self):
        """로드 실패 시 빈 샘플 반환"""
        h, w = self.input_size
        return {
            'color': torch.zeros(3, self.output_size, self.output_size),
            'depth': torch.zeros(1, self.output_size, self.output_size),
            'action': torch.zeros(4),
            'label_value': torch.tensor(0.0),
            'grasp_success': torch.tensor(0.0),
            'idx': -1,
            'valid': False,
        }
    
    def _preprocess(self, color_img, depth_img):
        """이미지 전처리 (패딩 제거 버전 - 640×640 직접 입력)"""
        # 2x 스케일 (320 → 640)
        color_2x = ndimage.zoom(color_img, zoom=[2, 2, 1], order=0)
        depth_2x = ndimage.zoom(depth_img, zoom=[2, 2], order=0)
        
        # [패딩 제거] 패딩 없이 640×640 그대로 사용
        
        # 정규화
        color_norm = color_2x.astype(np.float32) / 255.0
        for c in range(3):
            color_norm[:, :, c] = (color_norm[:, :, c] - self.image_mean[c]) / self.image_std[c]
        
        depth_norm = (depth_2x.astype(np.float32) - 0.01) / 0.03
        depth_norm = depth_norm[:, :, np.newaxis]
        
        # 텐서 변환 [C, H, W]
        color_t = torch.from_numpy(color_norm).permute(2, 0, 1).float()
        depth_t = torch.from_numpy(depth_norm).permute(2, 0, 1).float()
        
        return color_t, depth_t
    
    def get_raw_sample(self, idx):
        """원본 이미지 반환 (시각화용)"""
        source_idx, sample_idx = self.samples[idx]
        data = self.sources_data[source_idx]
        
        color_path = os.path.join(data['color_dir'], f'{sample_idx:06d}.0.color.png')
        depth_path = os.path.join(data['depth_dir'], f'{sample_idx:06d}.0.depth.png')
        
        color_img = cv2.imread(color_path)
        if color_img is not None:
            color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
        
        depth_img = cv2.imread(depth_path, -1)
        if depth_img is not None:
            depth_img = depth_img.astype(np.float32) / 100000
        
        action = data['actions'][sample_idx]
        label_value = data['labels'][sample_idx]
        
        return {
            'color': color_img,
            'depth': depth_img,
            'action': action,
            'label_value': label_value,
        }


class BalancedBatchSampler:
    """
    균형 잡힌 배치 샘플러 (성공:실패 = 1:1)
    """
    
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_batches = len(dataset) // batch_size
    
    def __iter__(self):
        for _ in range(self.num_batches):
            n_success = self.batch_size // 2
            n_failure = self.batch_size - n_success
            
            success_idx = np.random.choice(
                self.dataset.success_indices, n_success, 
                replace=len(self.dataset.success_indices) < n_success
            )
            failure_idx = np.random.choice(
                self.dataset.failure_indices, n_failure,
                replace=len(self.dataset.failure_indices) < n_failure
            )
            
            batch = np.concatenate([success_idx, failure_idx])
            np.random.shuffle(batch)
            yield batch.tolist()
    
    def __len__(self):
        return self.num_batches


class OfflineTrainer:
    """
    오프라인 학습 트레이너 (DataLoader 기반)
    """
    
    def __init__(self, model_name, dataset, config, num_workers=4):
        """트레이너 초기화"""
        self.model_name = model_name
        self.dataset = dataset
        self.config = config
        self.num_workers = num_workers
        self.use_cuda = torch.cuda.is_available()
        
        # 결과 저장 경로
        self.result_dir = get_model_result_dir(model_name)
        self.models_dir = os.path.join(self.result_dir, 'models')
        self.logs_dir = os.path.join(self.result_dir, 'logs')
        self.vis_dir = os.path.join(self.result_dir, 'visualizations')
        
        # 모델 초기화
        self._init_model()
        
        # DataLoader 생성
        self._init_dataloader()
        
        # 학습 통계
        self.epoch_losses = []
        self.training_log = []
        
        print(f'\n[TRAINER] Model: {config["name"]}')
        print(f'[TRAINER] Double DQN: {config["double_dqn"]}')
        print(f'[TRAINER] Dueling DQN: {config["dueling_dqn"]}')
        print(f'[TRAINER] CUDA: {self.use_cuda}')
        print(f'[TRAINER] DataLoader workers: {num_workers}')
    
    def _init_model(self):
        """모델 및 옵티마이저 초기화"""
        self.model = DQN(self.use_cuda, dueling=self.config['dueling_dqn'])
        
        if self.use_cuda:
            self.model = self.model.cuda()
        
        self.model.train()
        
        # 옵티마이저
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=TRAINING_CONFIG['learning_rate'],
            momentum=0.9,
            weight_decay=2e-5
        )
        
        # Loss 함수
        self.criterion = torch.nn.SmoothL1Loss(reduction='none')
        if self.use_cuda:
            self.criterion = self.criterion.cuda()
        
        # Double DQN: Target Network
        self.target_model = None
        if self.config['double_dqn']:
            self.target_model = DQN(self.use_cuda, dueling=self.config['dueling_dqn'])
            if self.use_cuda:
                self.target_model = self.target_model.cuda()
            self.target_model.load_state_dict(self.model.state_dict())
            self.target_model.eval()
    
    def _init_dataloader(self):
        """DataLoader 초기화"""
        batch_size = TRAINING_CONFIG['batch_size']
        
        if SAMPLING_CONFIG['strategy'] == 'balanced':
            # 균형 샘플링
            batch_sampler = BalancedBatchSampler(self.dataset, batch_size)
            self.dataloader = DataLoader(
                self.dataset,
                batch_sampler=batch_sampler,
                num_workers=self.num_workers,
                pin_memory=self.use_cuda,
                prefetch_factor=2 if self.num_workers > 0 else None,
            )
        else:
            # 무작위 샘플링
            self.dataloader = DataLoader(
                self.dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=self.use_cuda,
                prefetch_factor=2 if self.num_workers > 0 else None,
                drop_last=True,
            )
    
    def _compute_batch_loss(self, batch):
        """
        배치 손실 계산 (진짜 배치 처리 - GPU 병렬)
        
        전체 배치를 한 번에 forward하여 GPU 활용도를 극대화합니다.
        """
        # 유효한 샘플만 처리
        valid_mask = batch['valid']
        if not valid_mask.any():
            return torch.tensor(0.0)
        
        # GPU 전송
        color = batch['color']
        depth = batch['depth']
        actions = batch['action']
        labels = batch['label_value']
        
        if self.use_cuda:
            color = color.cuda()
            depth = depth.cuda()
            actions = actions.cuda()
            labels = labels.cuda()
            valid_mask = valid_mask.cuda()
        
        batch_size = color.shape[0]
        
        # ========================================
        # 진짜 배치 Forward (한 번에 전체 배치!)
        # ========================================
        output, _ = self.model.forward(color, depth, is_volatile=False)
        
        # output_prob[0][0]: [B, 1, H, W]
        predictions = self.model.output_prob[0][0]  # [B, 1, H, W]
        out_h, out_w = predictions.shape[-2], predictions.shape[-1]
        predictions = predictions.view(batch_size, out_h, out_w)  # [B, H, W]
        
        # ========================================
        # 배치 레이블 생성 (패딩 제거 버전)
        # ========================================
        label_batch = torch.zeros(batch_size, out_h, out_w, device=color.device)
        weights_batch = torch.zeros(batch_size, out_h, out_w, device=color.device)
        
        # [패딩 제거] 레이블 위치 = 원본 픽셀 × 2 (2x 스케일)
        for i in range(batch_size):
            if not valid_mask[i]:
                continue
            
            pix_y = int(actions[i, 2].item())
            pix_x = int(actions[i, 3].item())
            label_val = labels[i].item()
            
            # 원본 좌표(320)를 2x 스케일 좌표(640)로 변환
            scaled_y, scaled_x = pix_y * 2, pix_x * 2
            
            if 0 <= scaled_y < out_h and 0 <= scaled_x < out_w:
                label_batch[i, scaled_y, scaled_x] = label_val
                weights_batch[i, scaled_y, scaled_x] = 1.0
        
        # ========================================
        # 배치 Loss 계산
        # ========================================
        loss = self.criterion(predictions, label_batch) * weights_batch
        total_loss = loss.sum()
        
        return total_loss
    
    def train_epoch(self, epoch):
        """한 에폭 학습"""
        self.model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        # tqdm 진행률 바
        pbar = tqdm(self.dataloader,
                    desc=f'Epoch {epoch:3d}/{TRAINING_CONFIG["epochs"]}',
                    unit='batch',
                    ncols=100,
                    leave=True)
        
        for batch_idx, batch in enumerate(pbar):
            self.optimizer.zero_grad()
            
            # 배치 손실 계산
            loss = self._compute_batch_loss(batch)
            
            if loss > 0:
                loss.backward()
                self.optimizer.step()
            
            batch_loss = float(loss.item()) if isinstance(loss, torch.Tensor) else loss
            epoch_loss += batch_loss
            num_batches += 1
            
            # 진행률 바 업데이트
            avg_loss = batch_loss / TRAINING_CONFIG['batch_size']
            pbar.set_postfix({'loss': f'{avg_loss:.4f}'})
            
            # Double DQN: Target network 업데이트
            if self.config['double_dqn']:
                iteration = epoch * len(self.dataloader) + batch_idx
                if iteration % TRAINING_CONFIG['target_update_freq'] == 0 and iteration > 0:
                    self.target_model.load_state_dict(self.model.state_dict())
            
            # 50 step마다 시각화
            if (batch_idx + 1) % 50 == 0:
                global_step = (epoch - 1) * len(self.dataloader) + batch_idx + 1
                self.visualize_sample(epoch, step=global_step)
        
        avg_epoch_loss = epoch_loss / (num_batches * TRAINING_CONFIG['batch_size']) if num_batches > 0 else 0.0
        return avg_epoch_loss
    
    def visualize_sample(self, epoch, step=None):
        """
        랜덤 샘플 Q값 히트맵 시각화 (패딩 제거 버전)
        
        Args:
            epoch: 현재 에폭
            step: 현재 스텝 (None이면 에폭 종료 시점)
        """
        random_idx = np.random.randint(0, len(self.dataset))
        raw_sample = self.dataset.get_raw_sample(random_idx)
        
        if raw_sample['color'] is None:
            return None
        
        # 추론
        sample = self.dataset[random_idx]
        color_t = sample['color'].unsqueeze(0)
        depth_t = sample['depth'].unsqueeze(0)
        
        if self.use_cuda:
            color_t = color_t.cuda()
            depth_t = depth_t.cuda()
        
        self.model.eval()
        with torch.no_grad():
            output_prob, _ = self.model.forward(color_t, depth_t, is_volatile=True)
            # output_prob[0][0]: [1, 1, 640, 640] -> [640, 640]
            q_map_640 = output_prob[0][0].squeeze().cpu().numpy()
        self.model.train()
        
        # [패딩 제거 버전] 640×640 → 원본 크기(320×320)로 리사이즈
        h, w = raw_sample['color'].shape[:2]
        q_map = cv2.resize(q_map_640, (w, h), interpolation=cv2.INTER_LINEAR)
        
        # 시각화
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 입력 이미지
        axes[0].imshow(raw_sample['color'])
        axes[0].set_title('Input Color Heightmap')
        axes[0].axis('off')
        
        # Q값 히트맵 (원본 크기)
        im = axes[1].imshow(q_map, cmap='jet', vmin=0, vmax=max(1, q_map.max()))
        axes[1].set_title(f'Q-Value (max={q_map.max():.3f})')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1], fraction=0.046)
        
        # 오버레이 (동일 크기)
        q_norm = (q_map - q_map.min()) / (q_map.max() - q_map.min() + 1e-8)
        q_colored = plt.cm.jet(q_norm)[:, :, :3]
        
        overlay = 0.5 * raw_sample['color'].astype(float) / 255 + 0.5 * q_colored
        overlay = np.clip(overlay, 0, 1)
        
        pix_y = int(raw_sample['action'][2])
        pix_x = int(raw_sample['action'][3])
        
        axes[2].imshow(overlay)
        axes[2].scatter([pix_x], [pix_y], c='lime', s=100, marker='x', linewidths=3)
        axes[2].set_title(f'Overlay (label={raw_sample["label_value"]:.2f})')
        axes[2].axis('off')
        
        # 파일명: epoch{에폭}_{스텝} 형식
        if step is not None:
            vis_path = os.path.join(self.vis_dir, f'epoch{epoch}_{step:05d}.png')
        else:
            vis_path = os.path.join(self.vis_dir, f'epoch{epoch}_final.png')
        plt.tight_layout()
        plt.savefig(vis_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        return vis_path
    
    def save_model(self, epoch):
        """모델 저장"""
        model_path = os.path.join(self.models_dir, f'epoch_{epoch:03d}.pth')
        torch.save(self.model.cpu().state_dict(), model_path)
        if self.use_cuda:
            self.model = self.model.cuda()
        return model_path
    
    def save_training_log(self):
        """학습 로그 저장"""
        log_path = os.path.join(self.logs_dir, 'training_log.json')
        with open(log_path, 'w') as f:
            json.dump({
                'model_name': self.model_name,
                'config': self.config,
                'training_config': TRAINING_CONFIG,
                'epoch_losses': self.epoch_losses,
                'training_log': self.training_log,
            }, f, indent=2)
        return log_path
    
    def plot_loss_curve(self):
        """Loss 곡선 저장"""
        if not self.epoch_losses:
            return
        
        plt.figure(figsize=VISUALIZATION_CONFIG['figsize'])
        plt.plot(self.epoch_losses,
                 color=VISUALIZATION_CONFIG['colors'].get(self.model_name, '#1f77b4'),
                 linewidth=2)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Average Loss', fontsize=12)
        plt.title(f'{self.config["name"]} - Training Loss Curve', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        loss_path = os.path.join(self.vis_dir, 'loss_curve.png')
        plt.savefig(loss_path, dpi=VISUALIZATION_CONFIG['dpi'], bbox_inches='tight')
        plt.close()
        
        return loss_path
    
    def train(self, epochs=None):
        """전체 학습 실행"""
        if epochs is None:
            epochs = TRAINING_CONFIG['epochs']
        
        save_interval = TRAINING_CONFIG['save_interval']
        
        print(f'\n{"="*60}')
        print(f'TRAINING: {self.config["name"]}')
        print(f'{"="*60}')
        print(f'Epochs: {epochs}, Save interval: {save_interval}')
        print(f'Batch size: {TRAINING_CONFIG["batch_size"]}')
        
        start_time = time.time()
        
        # 초기 가중치 저장
        self.save_model(0)
        
        for epoch in range(1, epochs + 1):
            epoch_start = time.time()
            
            avg_loss = self.train_epoch(epoch)
            self.epoch_losses.append(float(avg_loss))
            
            epoch_time = time.time() - epoch_start
            
            self.training_log.append({
                'epoch': epoch,
                'loss': float(avg_loss),
                'time': epoch_time,
            })
            
            # 모델 저장
            if epoch % save_interval == 0:
                model_path = self.save_model(epoch)
                print(f'  [SAVE] {model_path}')
                self.plot_loss_curve()
        
        total_time = time.time() - start_time
        print(f'\nTraining completed in {total_time/60:.1f} minutes')
        
        self.save_training_log()
        self.plot_loss_curve()
        
        return self.epoch_losses


def train_all_models(dataset, models=None, num_workers=4):
    """모든 모델 학습"""
    if models is None:
        models = list(MODEL_CONFIGS.keys())
    
    results = {}
    
    for model_name in models:
        if model_name not in MODEL_CONFIGS:
            print(f'[WARNING] Unknown model: {model_name}')
            continue
        
        config = MODEL_CONFIGS[model_name]
        trainer = OfflineTrainer(model_name, dataset, config, num_workers)
        losses = trainer.train()
        
        results[model_name] = {
            'final_loss': losses[-1] if losses else None,
            'min_loss': min(losses) if losses else None,
            'losses': losses,
        }
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Off-Policy Offline Training (Parallel Prefetch)')
    parser.add_argument('--model', type=str, default='all',
                        help='학습할 모델 (all, standard_dqn, double_dqn, dueling_dqn, double_dueling_dqn)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='에폭 수 (기본값: 100)')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='배치 크기 (기본값: 24GB 메모리 기준 5)')
    parser.add_argument('--save_interval', type=int, default=None,
                        help='모델 저장 간격 (기본값: 1)')
    parser.add_argument('--workers', type=int, default=4,
                        help='DataLoader 워커 수 (기본값: 4)')
    
    args = parser.parse_args()
    
    print('='*60)
    print('OFF-POLICY OFFLINE TRAINING (PARALLEL PREFETCH)')
    print('='*60)
    
    # 데이터셋 생성
    dataset = OfflineTorchDataset(DATA_SOURCES)
    
    # 모델 선택
    if args.model == 'all':
        models = list(MODEL_CONFIGS.keys())
    else:
        models = [args.model]
    
    # 설정 적용
    if args.epochs:
        TRAINING_CONFIG['epochs'] = args.epochs
    if args.batch_size:
        TRAINING_CONFIG['batch_size'] = args.batch_size
    if args.save_interval:
        TRAINING_CONFIG['save_interval'] = args.save_interval
    
    print(f"\n[CONFIG] Epochs: {TRAINING_CONFIG['epochs']}")
    print(f"[CONFIG] Batch Size: {TRAINING_CONFIG['batch_size']}")
    print(f"[CONFIG] Save Interval: {TRAINING_CONFIG['save_interval']} epochs")
    print(f"[CONFIG] DataLoader Workers: {args.workers}")
    
    # 학습 실행
    results = train_all_models(dataset, models, args.workers)
    
    # 결과 요약
    print('\n' + '='*60)
    print('TRAINING SUMMARY')
    print('='*60)
    for model_name, result in results.items():
        if result['final_loss'] is not None:
            print(f'{model_name}: Final Loss={result["final_loss"]:.6f}, Min Loss={result["min_loss"]:.6f}')
        else:
            print(f'{model_name}: No results')
    
    print('\nAll models trained successfully!')
    print(f'Results saved to: {RESULTS_DIR}')


if __name__ == '__main__':
    main()
