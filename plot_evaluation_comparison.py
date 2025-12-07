#!/usr/bin/env python
"""
evaluation 폴더의 grasp-success.log.txt를 읽어서
DQN, Double DQN, Dueling DQN의 성공률을 비교하는 그래프 생성
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def extract_iteration(folder_name):
    """iter_000050 -> 50"""
    return int(folder_name.split('_')[1])

def calculate_success_rate(log_file_path):
    """
    grasp-success.log.txt 파일에서 성공률 계산
    1.0 = 성공, 0.0 = 일반 실패, -1.0 = 바닥 선택
    """
    try:
        data = np.loadtxt(log_file_path)
        if len(data) == 0:
            return None
        
        total = len(data)
        success = np.sum(data == 1.0)
        general_fail = np.sum(data == 0.0)
        floor_select = np.sum(data == -1.0)
        
        success_rate = (success / total) * 100
        
        return {
            'success_rate': success_rate,
            'total': total,
            'success': int(success),
            'general_fail': int(general_fail),
            'floor_select': int(floor_select)
        }
    except Exception as e:
        print(f"Warning: Failed to read {log_file_path}: {e}")
        return None

def collect_model_data(evaluation_dir, model_name):
    """
    특정 모델의 모든 iteration 결과 수집
    """
    model_path = os.path.join(evaluation_dir, model_name)
    if not os.path.exists(model_path):
        print(f"Warning: Model path not found: {model_path}")
        return None
    
    results = []
    
    # iter_XXXXXX 폴더들을 찾아서 정렬
    iter_folders = sorted([f for f in os.listdir(model_path) if f.startswith('iter_')],
                         key=extract_iteration)
    
    for iter_folder in iter_folders:
        iteration = extract_iteration(iter_folder)
        
        # iter_XXXXXX/YYYY-MM-DD.HH.MM.SS/transitions/grasp-success.log.txt
        iter_path = os.path.join(model_path, iter_folder)
        
        # 타임스탬프 폴더 찾기 (보통 1개만 있음)
        timestamp_folders = [f for f in os.listdir(iter_path) 
                           if os.path.isdir(os.path.join(iter_path, f))]
        
        if not timestamp_folders:
            print(f"Warning: No timestamp folder in {iter_folder}")
            continue
        
        timestamp_folder = timestamp_folders[0]  # 첫 번째 (최신) 폴더 사용
        log_file = os.path.join(iter_path, timestamp_folder, 'transitions', 'grasp-success.log.txt')
        
        if not os.path.exists(log_file):
            print(f"Warning: Log file not found: {log_file}")
            continue
        
        stats = calculate_success_rate(log_file)
        if stats:
            results.append({
                'iteration': iteration,
                **stats
            })
            print(f"  [{model_name}] iter_{iteration:06d}: {stats['success_rate']:.1f}% "
                  f"({stats['success']}/{stats['total']} success)")
    
    return results

def plot_comparison(dqn_data, double_dqn_data, dueling_dqn_data, output_file='evaluation_comparison.png'):
    """
    3개 모델의 성공률 비교 그래프 생성
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('DQN vs Double DQN vs Double+Dueling DQN - Evaluation Results', fontsize=16, fontweight='bold')
    
    # 색상 정의
    colors = {
        'DQN': '#1f77b4',
        'Double DQN': '#ff7f0e',
        'Double+Dueling DQN': '#2ca02c'
    }
    
    # 1. 성공률 비교 (메인 그래프)
    ax1 = axes[0, 0]
    if dqn_data:
        iterations = [r['iteration'] for r in dqn_data]
        success_rates = [r['success_rate'] for r in dqn_data]
        ax1.plot(iterations, success_rates, 'o-', label='DQN', color=colors['DQN'], linewidth=2, markersize=6)
    
    if double_dqn_data:
        iterations = [r['iteration'] for r in double_dqn_data]
        success_rates = [r['success_rate'] for r in double_dqn_data]
        ax1.plot(iterations, success_rates, 's-', label='Double DQN', color=colors['Double DQN'], linewidth=2, markersize=6)
    
    if dueling_dqn_data:
        iterations = [r['iteration'] for r in dueling_dqn_data]
        success_rates = [r['success_rate'] for r in dueling_dqn_data]
        ax1.plot(iterations, success_rates, '^-', label='Double+Dueling DQN', color=colors['Double+Dueling DQN'], linewidth=2, markersize=6)
    
    ax1.set_xlabel('Training Iteration', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Success Rate (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Grasp Success Rate Over Training', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11, loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 100)
    
    # 2. 성공 횟수 비교
    ax2 = axes[0, 1]
    if dqn_data:
        iterations = [r['iteration'] for r in dqn_data]
        success_counts = [r['success'] for r in dqn_data]
        ax2.plot(iterations, success_counts, 'o-', label='DQN', color=colors['DQN'], linewidth=2, markersize=6)
    
    if double_dqn_data:
        iterations = [r['iteration'] for r in double_dqn_data]
        success_counts = [r['success'] for r in double_dqn_data]
        ax2.plot(iterations, success_counts, 's-', label='Double DQN', color=colors['Double DQN'], linewidth=2, markersize=6)
    
    if dueling_dqn_data:
        iterations = [r['iteration'] for r in dueling_dqn_data]
        success_counts = [r['success'] for r in dueling_dqn_data]
        ax2.plot(iterations, success_counts, '^-', label='Double+Dueling DQN', color=colors['Double+Dueling DQN'], linewidth=2, markersize=6)
    
    ax2.set_xlabel('Training Iteration', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Successful Grasps (count)', fontsize=12, fontweight='bold')
    ax2.set_title('Number of Successful Grasps', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11, loc='best')
    ax2.grid(True, alpha=0.3)
    
    # 3. 실패 원인 분석 (바닥 선택 비율)
    ax3 = axes[1, 0]
    
    def calculate_floor_ratio(data):
        if not data:
            return [], []
        iterations = [r['iteration'] for r in data]
        floor_ratios = [(r['floor_select'] / r['total']) * 100 for r in data]
        return iterations, floor_ratios
    
    if dqn_data:
        iterations, ratios = calculate_floor_ratio(dqn_data)
        ax3.plot(iterations, ratios, 'o-', label='DQN', color=colors['DQN'], linewidth=2, markersize=6)
    
    if double_dqn_data:
        iterations, ratios = calculate_floor_ratio(double_dqn_data)
        ax3.plot(iterations, ratios, 's-', label='Double DQN', color=colors['Double DQN'], linewidth=2, markersize=6)
    
    if dueling_dqn_data:
        iterations, ratios = calculate_floor_ratio(dueling_dqn_data)
        ax3.plot(iterations, ratios, '^-', label='Double+Dueling DQN', color=colors['Double+Dueling DQN'], linewidth=2, markersize=6)
    
    ax3.set_xlabel('Training Iteration', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Floor Selection Rate (%)', fontsize=12, fontweight='bold')
    ax3.set_title('Floor Selection (Empty Space) Rate', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=11, loc='best')
    ax3.grid(True, alpha=0.3)
    
    # 4. 최종 성능 비교 (막대 그래프)
    ax4 = axes[1, 1]
    
    models = []
    final_success_rates = []
    model_colors = []
    
    if dqn_data:
        models.append('DQN')
        final_success_rates.append(dqn_data[-1]['success_rate'])
        model_colors.append(colors['DQN'])
    
    if double_dqn_data:
        models.append('Double DQN')
        final_success_rates.append(double_dqn_data[-1]['success_rate'])
        model_colors.append(colors['Double DQN'])
    
    if dueling_dqn_data:
        models.append('Double+Dueling\nDQN')
        final_success_rates.append(dueling_dqn_data[-1]['success_rate'])
        model_colors.append(colors['Double+Dueling DQN'])
    
    bars = ax4.bar(models, final_success_rates, color=model_colors, alpha=0.7, edgecolor='black', linewidth=2)
    
    # 막대 위에 값 표시
    for bar, rate in zip(bars, final_success_rates):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{rate:.1f}%',
                ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    ax4.set_ylabel('Success Rate (%)', fontsize=12, fontweight='bold')
    ax4.set_title('Final Performance Comparison', fontsize=14, fontweight='bold')
    ax4.set_ylim(0, 100)
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✅ Graph saved: {output_file}")
    plt.show()

def print_summary(dqn_data, double_dqn_data, dueling_dqn_data):
    """
    모델별 성능 요약 출력
    """
    print("\n" + "="*80)
    print("📊 EVALUATION SUMMARY")
    print("="*80)
    
    def print_model_summary(name, data):
        if not data:
            print(f"\n[{name}] No data available")
            return
        
        print(f"\n[{name}]")
        print(f"  Iterations evaluated: {len(data)}")
        print(f"  Range: iter_{data[0]['iteration']:06d} ~ iter_{data[-1]['iteration']:06d}")
        
        final = data[-1]
        print(f"  Final Success Rate: {final['success_rate']:.2f}%")
        print(f"  Final Stats: {final['success']}/{final['total']} success, "
              f"{final['general_fail']} fail, {final['floor_select']} floor")
        
        # 평균 성공률
        avg_success = np.mean([r['success_rate'] for r in data])
        print(f"  Average Success Rate: {avg_success:.2f}%")
        
        # 최고 성공률
        max_success = max([r['success_rate'] for r in data])
        max_iter = [r for r in data if r['success_rate'] == max_success][0]['iteration']
        print(f"  Peak Success Rate: {max_success:.2f}% at iter_{max_iter:06d}")
    
    print_model_summary("DQN", dqn_data)
    print_model_summary("Double DQN", double_dqn_data)
    print_model_summary("Double+Dueling DQN", dueling_dqn_data)
    
    print("\n" + "="*80 + "\n")

def main():
    evaluation_dir = 'evaluation'
    
    if not os.path.exists(evaluation_dir):
        print(f"Error: Evaluation directory not found: {evaluation_dir}")
        return
    
    print("🔍 Collecting evaluation data...\n")
    
    # 각 모델의 데이터 수집
    print("📂 DQN:")
    dqn_data = collect_model_data(evaluation_dir, '2025-12-07.12.29.59_dqn')
    
    print("\n📂 Double DQN:")
    double_dqn_data = collect_model_data(evaluation_dir, '2025-12-07.12.42.59_double_dqn')
    
    print("\n📂 Double+Dueling DQN:")
    dueling_dqn_data = collect_model_data(evaluation_dir, '2025-12-07.14.17.19_double_dueling')
    
    if not dqn_data and not double_dqn_data and not dueling_dqn_data:
        print("\n❌ No evaluation data found!")
        return
    
    # 요약 출력
    print_summary(dqn_data, double_dqn_data, dueling_dqn_data)
    
    # 그래프 생성
    print("📈 Generating comparison graph...")
    plot_comparison(dqn_data, double_dqn_data, dueling_dqn_data)

if __name__ == '__main__':
    main()

