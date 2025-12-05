#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 17:50:50 2021

@author: marwan
"""
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from network import FeatureTrunk

class DQN(nn.Module):

    def __init__(self, use_cuda, dueling=False):
        """
        DQN 모델 초기화
        
        Args:
            use_cuda: CUDA 사용 여부
            dueling: Dueling DQN 아키텍처 사용 여부 (Value + Advantage streams)
        """
        super(DQN, self).__init__()
        self.use_cuda = use_cuda
        self.dueling = dueling
        
        # Initialize network trunks with DenseNet pre-trained on ImageNet
        self.feature_tunk = FeatureTrunk()
        
        # IRB360 진공 컵 최적화: 회전 각도 1개만 사용
        # 이유: 진공 흡착은 물체 방향과 무관하게 동작 (UR5 gripper와 다름)
        # 효과: 학습 속도 약 30배 향상, 메모리 1/36 감소
        # 기존: self.num_rotations = 36  # 논문 기준: 36개 회전 각도 (10도 간격)
        self.num_rotations = 1  # IRB360용: 진공 컵은 회전 불필요
        
        if dueling:
            # Dueling DQN: Value Stream + Advantage Stream
            # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
            
            # Value Stream: 상태 가치 V(s) - Global Average Pooling → scalar
            self.value_stream = nn.Sequential(
                nn.Conv2d(1024, 256, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(256),
                nn.Conv2d(256, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(64),
                nn.AdaptiveAvgPool2d(1),  # Global Average Pooling → (B, 64, 1, 1)
                nn.Flatten(),              # → (B, 64)
                nn.Linear(64, 1)            # V(s): scalar → (B, 1)
            )
            
            # Advantage Stream: 행동 우위 A(s,a) - per-pixel advantage
            self.advantage_stream = nn.Sequential(
                nn.Conv2d(1024, 256, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(256),
                nn.Conv2d(256, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(64),
                nn.Conv2d(64, 1, kernel_size=1, stride=1)  # A(s,a): (B, 1, H, W)
            )
            print('[Dueling DQN] Value stream + Advantage stream initialized')
        else:
            # Standard DQN: 단일 graspnet
            self.graspnet = nn.Sequential(
                nn.Conv2d(1024, 256, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(256),
                nn.Conv2d(256, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(64),
                nn.Conv2d(64, 1, kernel_size=1, stride=1))
        
        # Initialize network weights
        self._initialize_weights()
        # Initialize output variable (for backprop)
        self.interm_feat = []
        self.output_prob = []

    def _compute_grasp_prediction(self, interm_feat):
        """
        중간 특징에서 grasp prediction 계산
        
        Dueling DQN: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        Standard DQN: Q(s,a) = graspnet(interm_feat)
        
        Args:
            interm_feat: 중간 특징 (B, 1024, H, W)
        
        Returns:
            grasp_prediction: Q값 예측 (B, 1, H, W)
        """
        if self.dueling:
            # Value Stream: 상태 가치 V(s) - scalar
            value = self.value_stream(interm_feat)  # (B, 1)
            value = value.view(-1, 1, 1, 1)  # (B, 1, 1, 1) - broadcast를 위해 reshape
            
            # Advantage Stream: 행동 우위 A(s,a) - per-pixel
            advantage = self.advantage_stream(interm_feat)  # (B, 1, H, W)
            
            # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
            # mean(A)를 빼서 Advantage의 합이 0이 되도록 함
            mean_advantage = advantage.mean(dim=[2, 3], keepdim=True)  # (B, 1, 1, 1)
            grasp_prediction = value + (advantage - mean_advantage)
        else:
            # Standard DQN
            grasp_prediction = self.graspnet(interm_feat)
        
        return grasp_prediction

    def forward(self, input_color_data, input_depth_data, is_volatile=False, specific_rotation=-1):
        if is_volatile:
            with torch.no_grad():
                output_prob = []

                # Apply rotations to images
                for rotate_idx in range(self.num_rotations):
                    rotate_theta = np.radians(rotate_idx * (360 / self.num_rotations))
                    # Compute intermediate features
                    interm_feat = self.layers_forward(rotate_theta, input_color_data, input_depth_data)
                    # Forward pass through branches (Standard or Dueling)
                    grasp_prediction = self._compute_grasp_prediction(interm_feat)
                    # Compute sample grid for rotation AFTER branches
                    affine_mat_after = np.asarray([[np.cos(rotate_theta), np.sin(rotate_theta), 0], [-np.sin(rotate_theta), np.cos(rotate_theta), 0]])
                    affine_mat_after.shape = (2, 3, 1)
                    affine_mat_after = torch.from_numpy(affine_mat_after).permute(2, 0, 1).float()
                    if self.use_cuda:
                        flow_grid_after = F.affine_grid(Variable(affine_mat_after, requires_grad=False).cuda(), grasp_prediction.data.size())
                    else:
                        flow_grid_after = F.affine_grid(Variable(affine_mat_after, requires_grad=False), grasp_prediction.data.size())
                    # Forward pass through branches, undo rotation on output predictions, upsample results
                    output_prob.append([F.interpolate(F.grid_sample(grasp_prediction, flow_grid_after), scale_factor=16, mode='bilinear', align_corners=True)])
            return output_prob, interm_feat

        else:
            self.output_prob = []
            # Apply rotations to intermediate features
            rotate_idx = specific_rotation
            rotate_theta = np.radians(rotate_idx * (360 / self.num_rotations))
            # Compute intermediate features
            self.interm_feat = self.layers_forward(rotate_theta, input_color_data, input_depth_data)
            # Forward pass through branches (Standard or Dueling)
            grasp_prediction = self._compute_grasp_prediction(self.interm_feat)
            # Compute sample grid for rotation AFTER branches
            affine_mat_after = np.asarray([[np.cos(rotate_theta), np.sin(rotate_theta), 0], [-np.sin(rotate_theta), np.cos(rotate_theta), 0]])
            affine_mat_after.shape = (2, 3, 1)
            affine_mat_after = torch.from_numpy(affine_mat_after).permute(2, 0, 1).float()
            if self.use_cuda:
                flow_grid_after = F.affine_grid(Variable(affine_mat_after, requires_grad=False).cuda(), grasp_prediction.data.size())
            else:
                flow_grid_after = F.affine_grid(Variable(affine_mat_after, requires_grad=False), grasp_prediction.data.size())
            # Forward pass through branches, undo rotation on output predictions, upsample results
            self.output_prob.append([F.interpolate(F.grid_sample(grasp_prediction, flow_grid_after), scale_factor=16, mode='bilinear', align_corners=True)])
            return self.output_prob, self.interm_feat
        
    
    def layers_forward(self, rotate_theta, input_color_data, input_depth_data):
#        """ Reduces the repetitive forward pass code across multiple model classes. See PixelNet forward() and responsive_net forward().
#        """
#        # Compute sample grid for rotation BEFORE neural network
        affine_mat_before = np.asarray([[np.cos(-rotate_theta), np.sin(-rotate_theta), 0],[-np.sin(-rotate_theta), np.cos(-rotate_theta), 0]])
        affine_mat_before.shape = (2, 3, 1)
        affine_mat_before = torch.from_numpy(affine_mat_before).permute(2,0,1).float()
        if self.use_cuda:
            flow_grid_before = F.affine_grid(Variable(affine_mat_before, requires_grad=False).cuda(), input_color_data.size())
        else:
            flow_grid_before = F.affine_grid(Variable(affine_mat_before, requires_grad=False), input_color_data.size())
        # Rotate images clockwise
        if self.use_cuda:
            rotate_color = F.grid_sample(Variable(input_color_data).cuda(), flow_grid_before)
            rotate_depth = F.grid_sample(Variable(input_depth_data).cuda(), flow_grid_before)
        else:
            rotate_color = F.grid_sample(Variable(input_color_data), flow_grid_before)
            rotate_depth = F.grid_sample(Variable(input_depth_data), flow_grid_before)
        # Compute intermediate features
        interm_feat = self.feature_tunk(rotate_color, rotate_depth )
        return interm_feat

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d): 
                nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
