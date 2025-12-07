#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Pretrained 가중치 로드 검증 스크립트
"""

import torch
import numpy as np
from DQNModels import DQN

def test_pretrained_preservation():
    """DenseNet pretrained 가중치가 올바르게 보존되는지 검증"""
    
    print("=" * 60)
    print("Pretrained Weights Preservation Test")
    print("=" * 60)
    
    # 1. 모델 생성
    print("\n[1] Creating new DQN model...")
    model = DQN(use_cuda=False, dueling=False)
    
    # 2. DenseNet 가중치 확인
    print("\n[2] Checking DenseNet-121 pretrained weights...")
    densenet_params = []
    for name, param in model.named_parameters():
        if 'feature_tunk.dense121' in name and 'conv0' not in name:
            densenet_params.append(param)
            if len(densenet_params) <= 3:  # 처음 3개만 출력
                print(f"  {name}: mean={param.mean():.6f}, std={param.std():.6f}")
    
    # 3. 새 레이어 가중치 확인
    print("\n[3] Checking newly initialized layers...")
    for name, param in model.named_parameters():
        if 'graspnet' in name and '0.weight' in name:
            print(f"  {name}: mean={param.mean():.6f}, std={param.std():.6f}")
            break
    
    # 4. Forward pass 테스트
    print("\n[4] Testing forward pass...")
    dummy_color = torch.randn(1, 3, 640, 640)
    dummy_depth = torch.randn(1, 1, 640, 640)
    
    output, _ = model.forward(dummy_color, dummy_depth, is_volatile=True)
    q_values = output[0][0].numpy()[0, 0]
    
    print(f"  Input shape: color={dummy_color.shape}, depth={dummy_depth.shape}")
    print(f"  Output shape: {output[0][0].shape}")
    print(f"  Q-value range: [{q_values.min():.4f}, {q_values.max():.4f}]")
    print(f"  Q-value mean: {q_values.mean():.4f}")
    print(f"  Q-value std: {q_values.std():.4f}")
    
    # 5. 결과 평가
    print("\n[5] Evaluation:")
    
    # DenseNet 가중치가 0이 아니면 pretrained 로드됨
    densenet_mean = np.mean([p.abs().mean().item() for p in densenet_params])
    if densenet_mean > 0.01:
        print(f"  ✅ DenseNet pretrained weights preserved (mean={densenet_mean:.4f})")
    else:
        print(f"  ❌ DenseNet weights not loaded! (mean={densenet_mean:.4f})")
    
    # 초기 Q-value가 너무 높지 않아야 함
    if abs(q_values.mean()) < 0.5 and q_values.std() < 2.0:
        print(f"  ✅ Initial Q-values reasonable (mean={q_values.mean():.4f}, std={q_values.std():.4f})")
    else:
        print(f"  ⚠️  Initial Q-values might be too high (mean={q_values.mean():.4f}, std={q_values.std():.4f})")
    
    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)

if __name__ == '__main__':
    test_pretrained_preservation()

