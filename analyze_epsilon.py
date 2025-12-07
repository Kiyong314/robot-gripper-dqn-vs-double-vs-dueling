#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Epsilon-greedy 학습 분석 스크립트
성공률과 탐색/활용 전략의 관계 분석
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def analyze_epsilon_learning(log_dir):
    """
    Epsilon-greedy 학습 분석
    
    Args:
        log_dir: 로그 디렉토리 경로
    """
    transitions_dir = os.path.join(log_dir, 'transitions')
    
    print("=" * 60)
    print(f"Analyzing: {log_dir}")
    print("=" * 60)
    
    # 1. 로그 로드
    print("\n[1] Loading logs...")
    try:
        epsilon_log = np.loadtxt(os.path.join(transitions_dir, 'epsilon.log.txt'))
        grasp_success = np.loadtxt(os.path.join(transitions_dir, 'grasp-success.log.txt'))
        
        # 로그 크기 맞추기 (epsilon은 action 선택 시, success는 결과 저장 시 기록)
        min_len = min(len(epsilon_log), len(grasp_success))
        epsilon_log = epsilon_log[:min_len]
        grasp_success = grasp_success[:min_len]
        
        epsilon_values = epsilon_log[:, 0]
        is_random = epsilon_log[:, 1]
        
        print(f"  Total iterations: {len(epsilon_values)}")
        print(f"  Epsilon range: [{epsilon_values.min():.4f}, {epsilon_values.max():.4f}]")
        print(f"  Random actions: {is_random.sum():.0f} / {len(is_random)} ({100*is_random.mean():.1f}%)")
    except FileNotFoundError as e:
        print(f"  Error: {e}")
        return
    
    # 2. 통계 분석
    print("\n[2] Statistics:")
    
    # 전체 성공률
    success_rate = (grasp_success > 0).mean()
    print(f"  Overall success rate: {100*success_rate:.2f}%")
    
    # Random vs Greedy 성공률
    random_mask = is_random == 1
    greedy_mask = is_random == 0
    
    if random_mask.sum() > 0:
        random_success = (grasp_success[random_mask] > 0).mean()
        print(f"  Random action success rate: {100*random_success:.2f}% ({random_mask.sum()} samples)")
    
    if greedy_mask.sum() > 0:
        greedy_success = (grasp_success[greedy_mask] > 0).mean()
        print(f"  Greedy action success rate: {100*greedy_success:.2f}% ({greedy_mask.sum()} samples)")
    
    # 3. 이동 평균 계산
    print("\n[3] Computing moving averages...")
    window = 100
    
    if len(epsilon_values) < window:
        print(f"  Warning: Not enough data for window size {window}")
        window = max(10, len(epsilon_values) // 10)
        print(f"  Using window size: {window}")
    
    epsilon_ma = np.convolve(epsilon_values, np.ones(window)/window, mode='valid')
    random_rate_ma = np.convolve(is_random, np.ones(window)/window, mode='valid')
    success_rate_ma = np.convolve((grasp_success > 0).astype(float), np.ones(window)/window, mode='valid')
    
    # 4. 시각화
    print("\n[4] Creating visualizations...")
    
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(3, 2, figure=fig)
    
    # 4-1. Epsilon decay
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(epsilon_values, alpha=0.3, label='Epsilon (raw)')
    ax1.plot(range(window//2, window//2 + len(epsilon_ma)), epsilon_ma, 
             linewidth=2, label=f'Epsilon (MA-{window})')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Epsilon')
    ax1.set_title('Epsilon Decay Over Training')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 4-2. Random vs Greedy actions
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.scatter(range(len(is_random)), is_random, alpha=0.1, s=5, label='Actions')
    ax2.plot(range(window//2, window//2 + len(random_rate_ma)), random_rate_ma,
             'r-', linewidth=2, label=f'Random Rate (MA-{window})')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Is Random (0=Greedy, 1=Random)')
    ax2.set_title('Exploration vs Exploitation')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 4-3. Success rate over time
    ax3 = fig.add_subplot(gs[1, 1])
    success_binary = (grasp_success > 0).astype(float)
    ax3.scatter(range(len(success_binary)), success_binary, 
                alpha=0.1, s=5, c=is_random, cmap='RdYlGn', label='Success')
    ax3.plot(range(window//2, window//2 + len(success_rate_ma)), success_rate_ma,
             'b-', linewidth=2, label=f'Success Rate (MA-{window})')
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Success (0=Fail, 1=Success)')
    ax3.set_title('Success Rate Over Training')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4-4. Epsilon vs Success correlation
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.plot(range(window//2, window//2 + len(epsilon_ma)), epsilon_ma, 
             label='Epsilon', linewidth=2)
    ax4_twin = ax4.twinx()
    ax4_twin.plot(range(window//2, window//2 + len(success_rate_ma)), success_rate_ma,
                  'g-', label='Success Rate', linewidth=2)
    ax4.set_xlabel('Iteration')
    ax4.set_ylabel('Epsilon', color='b')
    ax4_twin.set_ylabel('Success Rate', color='g')
    ax4.set_title('Epsilon vs Success Rate')
    ax4.grid(True, alpha=0.3)
    
    # 4-5. Random rate vs Success rate
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.plot(range(window//2, window//2 + len(random_rate_ma)), random_rate_ma,
             'r-', label='Random Rate', linewidth=2)
    ax5_twin = ax5.twinx()
    ax5_twin.plot(range(window//2, window//2 + len(success_rate_ma)), success_rate_ma,
                  'g-', label='Success Rate', linewidth=2)
    ax5.set_xlabel('Iteration')
    ax5.set_ylabel('Random Rate', color='r')
    ax5_twin.set_ylabel('Success Rate', color='g')
    ax5.set_title('Exploration Rate vs Success Rate')
    ax5.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 저장
    output_path = os.path.join(log_dir, 'epsilon_analysis.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    
    # Interactive mode: 그래프 창 표시 (선택적)
    import sys
    if '--show' in sys.argv:
        print("\n[5] Displaying interactive plot... (close window to continue)")
        plt.show()
    else:
        print("\n  Tip: Use '--show' flag to display interactive plot")
        plt.close()  # 메모리 해제
    
    print("\n" + "=" * 60)
    print("Analysis completed!")
    print("=" * 60)

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        log_dir = sys.argv[1]
    else:
        # 기본값: 가장 최근 로그 디렉토리
        import glob
        log_dirs = sorted(glob.glob('logs/202*'))
        if log_dirs:
            log_dir = log_dirs[-1]
            print(f"Using most recent log: {log_dir}")
        else:
            print("Usage: python analyze_epsilon.py <log_directory>")
            print("Example: python analyze_epsilon.py logs/2025-12-06.22.22.16")
            sys.exit(1)
    
    analyze_epsilon_learning(log_dir)

