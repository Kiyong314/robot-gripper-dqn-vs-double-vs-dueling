#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DQN 모델 정의 (Standard DQN + Dueling DQN 지원)

=============================================================================
Original Code Attribution:
    Repository: https://github.com/Marwanon/Learning-Pick-to-Place-Objects-in-a-cluttered-scene-using-deep-reinforcement-learning
    Original Author: Marwan Qaid Mohammed
    Paper: "Learning Pick to Place Objects using Self-supervised Learning with Minimal Training Resources"
           International Journal of Advanced Computer Science and Applications (IJACSA), 12(10), 2021
    
    Based on: https://github.com/andyzeng/visual-pushing-grasping

Modifications for this project (My Contributions):
    ✨ Dueling DQN 아키텍처 구현 (Value + Advantage streams)
       - Wang et al., "Dueling Network Architectures for Deep Reinforcement Learning", ICML 2016
       - Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
    ✨ IRB360 진공 컵 최적화 (num_rotations=1)
       - 진공 흡착은 회전 불필요 → 학습 속도 30배 향상
    ✨ 가중치 초기화 로직 개선 (pretrained backbone 보존)
    ✨ 한국어 주석 추가
    
This code is used for educational purposes as part of a graduate project.
=============================================================================

주요 클래스:
    - DQN: Deep Q-Network (Standard / Dueling 모드 지원)
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
        """
        Forward pass (num_rotations=1 고정, 회전 연산 완전 제거)
        
        Args:
            input_color_data: 컬러 이미지 텐서 [B, 3, H, W]
            input_depth_data: 깊이 이미지 텐서 [B, 1, H, W]
            is_volatile: 추론 모드 (gradient 계산 안 함)
            specific_rotation: 사용 안 함 (호환성 유지용, 무시됨)
        
        Returns:
            output_prob: Q값 예측 리스트 [[tensor]]
            interm_feat: 중간 특징 (B, 1024, H, W)
        """
        # CUDA 텐서 이동 (필요시)
        if self.use_cuda:
            input_color_data = input_color_data.cuda()
            input_depth_data = input_depth_data.cuda()
        
        if is_volatile:
            with torch.no_grad():
                interm_feat = self.feature_tunk(input_color_data, input_depth_data)
                grasp_prediction = self._compute_grasp_prediction(interm_feat)
                output_prob = [[F.interpolate(grasp_prediction, scale_factor=16, mode='bilinear', align_corners=True)]]
            return output_prob, interm_feat
        else:
            self.interm_feat = self.feature_tunk(input_color_data, input_depth_data)
            grasp_prediction = self._compute_grasp_prediction(self.interm_feat)
            self.output_prob = [[F.interpolate(grasp_prediction, scale_factor=16, mode='bilinear', align_corners=True)]]
            return self.output_prob, self.interm_feat
        
    
    def layers_forward(self, rotate_theta, input_color_data, input_depth_data):
        """
        호환성 유지용 wrapper (deprecated)
        num_rotations=1로 고정되어 회전 연산 불필요
        
        Args:
            rotate_theta: 회전 각도 (무시됨)
            input_color_data: 컬러 이미지 텐서
            input_depth_data: 깊이 이미지 텐서
            
        Returns:
            interm_feat: 중간 특징
        """
        # CUDA 텐서 이동 (필요시)
        if self.use_cuda:
            input_color_data = input_color_data.cuda()
            input_depth_data = input_depth_data.cuda()
        
        return self.feature_tunk(input_color_data, input_depth_data)


    def _initialize_weights(self):
        """
        새로 추가된 레이어만 초기화 (Pretrained DenseNet 백본 제외)
        
        주의: self.modules()는 모든 하위 모듈을 순회하므로,
        feature_tunk.dense121의 pretrained weights까지 덮어쓰게 됨.
        따라서 새로 추가된 레이어(graspnet, value_stream, advantage_stream)만 초기화.
        
        feature_tunk (FeatureTrunk)는 이미 __init__에서 올바르게 초기화되었으므로 여기서 제외.
        """
        # 초기화할 레이어 목록 (pretrained backbone 제외)
        layers_to_init = []
        
        if self.dueling:
            # Dueling DQN: value_stream, advantage_stream
            layers_to_init.extend([self.value_stream, self.advantage_stream])
        else:
            # Standard DQN: graspnet
            layers_to_init.append(self.graspnet)
        
        # feature_tunk는 이미 network.py에서 올바르게 초기화되었으므로 여기서 제외
        # (이전에는 feature_tunk의 레이어를 여기서도 초기화했으나, 중복이므로 제거)
        
        # 지정된 레이어만 초기화
        for layer in layers_to_init:
            for m in layer.modules():
                if isinstance(m, nn.Conv2d): 
                    nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight.data)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
        
        print('[DQN] Initialized new layers only (graspnet/value_stream/advantage_stream)')
        print('[DQN] DenseNet pretrained backbone preserved')
