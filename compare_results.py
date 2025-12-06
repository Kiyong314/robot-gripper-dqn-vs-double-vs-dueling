#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DQN 변형들의 학습 결과 비교 스크립트

사용법:
    python compare_results.py --log_dirs logs/2025-12-05_offline_dqn logs/2025-12-05_offline_double_dqn ...
    
    또는 자동 검색:
    python compare_results.py --auto_find
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import re


def load_training_log(log_dir):
    """
    학습 로그 로드
    
    Args:
        log_dir: 로그 폴더 경로
        
    Returns:
        dict: 학습 통계
    """
    transitions_dir = os.path.join(log_dir, 'transitions')
    
    result = {
        'name': os.path.basename(log_dir),
        'log_dir': log_dir,
    }
    
    # reward-value.log.txt 로드
    reward_path = os.path.join(transitions_dir, 'reward-value.log.txt')
    if os.path.exists(reward_path):
        rewards = np.loadtxt(reward_path)
        result['rewards'] = rewards
        result['avg_reward'] = np.mean(rewards)
        result['total_reward'] = np.sum(rewards)
    
    # grasp-success.log.txt 로드
    success_path = os.path.join(transitions_dir, 'grasp-success.log.txt')
    if os.path.exists(success_path):
        success = np.loadtxt(success_path)
        result['success'] = success
        result['success_rate'] = 100.0 * np.sum(success == 1) / len(success)
        result['num_samples'] = len(success)
    
    # label-value.log.txt 로드
    label_path = os.path.join(transitions_dir, 'label-value.log.txt')
    if os.path.exists(label_path):
        labels = np.loadtxt(label_path)
        result['labels'] = labels
        result['avg_label'] = np.mean(labels[~np.isnan(labels)])
    
    # predicted-value.log.txt 로드
    predicted_path = os.path.join(transitions_dir, 'predicted-value.log.txt')
    if os.path.exists(predicted_path):
        predicted = np.loadtxt(predicted_path)
        result['predicted'] = predicted
        result['avg_predicted'] = np.mean(predicted[~np.isnan(predicted)])
    
    return result


def compute_moving_average(data, window=100):
    """이동 평균 계산"""
    if len(data) < window:
        window = max(1, len(data) // 10)
    return np.convolve(data, np.ones(window)/window, mode='valid')


def plot_comparison(results, output_dir='comparison_results'):
    """
    학습 결과 비교 그래프 생성
    
    Args:
        results: 각 모델의 결과 리스트
        output_dir: 출력 폴더
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 색상 팔레트
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    # 1. 성공률 비교 (이동 평균)
    plt.figure(figsize=(12, 6))
    for i, result in enumerate(results):
        if 'success' in result:
            success_ma = compute_moving_average(result['success'], window=100)
            plt.plot(success_ma * 100, label=result['name'], color=colors[i % len(colors)], linewidth=2)
    
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Success Rate (%) - Moving Avg', fontsize=12)
    plt.title('Grasp Success Rate Comparison', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'success_rate_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[SAVE] success_rate_comparison.png')
    
    # 2. 보상 비교 (이동 평균)
    plt.figure(figsize=(12, 6))
    for i, result in enumerate(results):
        if 'rewards' in result:
            reward_ma = compute_moving_average(result['rewards'], window=100)
            plt.plot(reward_ma, label=result['name'], color=colors[i % len(colors)], linewidth=2)
    
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Reward - Moving Avg', fontsize=12)
    plt.title('Reward Comparison', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'reward_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[SAVE] reward_comparison.png')
    
    # 3. Q값 비교 (예측 vs 실제)
    plt.figure(figsize=(12, 6))
    for i, result in enumerate(results):
        if 'predicted' in result and 'labels' in result:
            predicted_ma = compute_moving_average(result['predicted'], window=100)
            plt.plot(predicted_ma, label=f"{result['name']} (Predicted)", 
                    color=colors[i % len(colors)], linewidth=2, linestyle='-')
    
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Q-Value - Moving Avg', fontsize=12)
    plt.title('Predicted Q-Value Comparison', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'qvalue_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[SAVE] qvalue_comparison.png')
    
    # 4. 최종 성능 바 차트
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    names = [r['name'] for r in results]
    
    # 4.1 최종 성공률
    success_rates = [r.get('success_rate', 0) for r in results]
    axes[0].bar(names, success_rates, color=colors[:len(names)])
    axes[0].set_ylabel('Success Rate (%)')
    axes[0].set_title('Final Success Rate')
    axes[0].tick_params(axis='x', rotation=45)
    for i, v in enumerate(success_rates):
        axes[0].text(i, v + 1, f'{v:.1f}%', ha='center', fontsize=10)
    
    # 4.2 평균 보상
    avg_rewards = [r.get('avg_reward', 0) for r in results]
    axes[1].bar(names, avg_rewards, color=colors[:len(names)])
    axes[1].set_ylabel('Average Reward')
    axes[1].set_title('Average Reward')
    axes[1].tick_params(axis='x', rotation=45)
    for i, v in enumerate(avg_rewards):
        axes[1].text(i, v + 0.01, f'{v:.3f}', ha='center', fontsize=10)
    
    # 4.3 샘플 수
    num_samples = [r.get('num_samples', 0) for r in results]
    axes[2].bar(names, num_samples, color=colors[:len(names)])
    axes[2].set_ylabel('Number of Samples')
    axes[2].set_title('Training Samples')
    axes[2].tick_params(axis='x', rotation=45)
    for i, v in enumerate(num_samples):
        axes[2].text(i, v + 50, f'{v}', ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'final_performance.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[SAVE] final_performance.png')
    
    # 5. 성능 요약 테이블 출력
    print('\n' + '='*70)
    print('PERFORMANCE SUMMARY')
    print('='*70)
    print(f'{"Model":<30} {"Success Rate":<15} {"Avg Reward":<15} {"Samples":<10}')
    print('-'*70)
    for result in results:
        name = result['name'][:30]
        success = result.get('success_rate', 0)
        reward = result.get('avg_reward', 0)
        samples = result.get('num_samples', 0)
        print(f'{name:<30} {success:>10.2f}% {reward:>14.4f} {samples:>10}')
    print('='*70)


def find_offline_logs(base_dir='logs'):
    """오프라인 학습 로그 자동 검색"""
    pattern = os.path.join(base_dir, '*offline*')
    dirs = glob(pattern)
    return sorted(dirs)


def main():
    parser = argparse.ArgumentParser(description='Compare DQN Training Results')
    parser.add_argument('--log_dirs', nargs='+', default=None,
                        help='비교할 로그 폴더들')
    parser.add_argument('--auto_find', action='store_true', default=False,
                        help='오프라인 학습 로그 자동 검색')
    parser.add_argument('--output_dir', type=str, default='comparison_results',
                        help='결과 출력 폴더')
    
    args = parser.parse_args()
    
    # 로그 폴더 결정
    if args.auto_find:
        log_dirs = find_offline_logs()
        if not log_dirs:
            print('[ERROR] No offline training logs found in logs/ folder')
            return
        print(f'[FOUND] {len(log_dirs)} offline training logs')
    elif args.log_dirs:
        log_dirs = args.log_dirs
    else:
        print('[ERROR] Specify --log_dirs or use --auto_find')
        return
    
    # 결과 로드
    results = []
    for log_dir in log_dirs:
        if os.path.exists(log_dir):
            result = load_training_log(log_dir)
            if 'success' in result:
                results.append(result)
                print(f'[LOAD] {result["name"]}: {result.get("num_samples", 0)} samples, '
                      f'{result.get("success_rate", 0):.2f}% success')
    
    if not results:
        print('[ERROR] No valid results found')
        return
    
    # 비교 그래프 생성
    plot_comparison(results, args.output_dir)
    print(f'\n[DONE] Results saved to {args.output_dir}/')


if __name__ == '__main__':
    main()


