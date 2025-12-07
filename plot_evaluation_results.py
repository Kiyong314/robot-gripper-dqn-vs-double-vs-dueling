"""
평가 결과 시각화 스크립트
CSV 파일을 읽어 여러 모델을 비교하는 그래프 생성

Usage:
    # 단일 모델
    python plot_evaluation_results.py --csv evaluation_results_dqn.csv
    
    # 여러 모델 비교
    python plot_evaluation_results.py \
        --csv evaluation_results_dqn.csv evaluation_results_double.csv evaluation_results_dueling.csv \
        --labels "Standard DQN" "Double DQN" "Dueling DQN"
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import numpy as np

# 한글 폰트 설정 (선택사항)
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def plot_single_model(df, output_dir='evaluation_plots', model_name='Model'):
    """단일 모델 결과 시각화"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Success Rate vs Iteration
    plt.figure(figsize=(10, 6))
    plt.plot(df['iteration'], df['success_rate'], marker='o', linewidth=2, markersize=4)
    plt.xlabel('Training Iteration', fontsize=12)
    plt.ylabel('Grasp Success Rate (%)', fontsize=12)
    plt.title(f'{model_name} Performance Over Training', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 100)
    plt.savefig(f'{output_dir}/success_rate.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Q-value Statistics
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].plot(df['iteration'], df['avg_q_value'], label='Average', marker='o', markersize=3)
    axes[0].plot(df['iteration'], df['max_q_value'], label='Maximum', marker='s', markersize=3)
    axes[0].plot(df['iteration'], df['min_q_value'], label='Minimum', marker='^', markersize=3)
    axes[0].set_xlabel('Iteration', fontsize=11)
    axes[0].set_ylabel('Q-value', fontsize=11)
    axes[0].set_title('Q-value Statistics', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].bar(df['iteration'], df['std_q_value'], width=max(df['iteration'])*0.02)
    axes[1].set_xlabel('Iteration', fontsize=11)
    axes[1].set_ylabel('Q-value Std Dev', fontsize=11)
    axes[1].set_title('Q-value Variability', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/qvalue_stats.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Failure Types Distribution
    plt.figure(figsize=(12, 6))
    width = max(df['iteration']) * 0.03
    plt.bar(df['iteration'], df['successful_grasps'], width=width, label='Success', alpha=0.8, color='green')
    plt.bar(df['iteration'], df['failed_grasps'], width=width, bottom=df['successful_grasps'], 
            label='Failed Grasp', alpha=0.8, color='orange')
    plt.bar(df['iteration'], df['floor_selections'], width=width,
            bottom=df['successful_grasps'] + df['failed_grasps'],
            label='Floor Selection', alpha=0.8, color='red')
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title('Outcome Distribution', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    plt.savefig(f'{output_dir}/outcome_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Failure Case Analysis
    if 'failure_with_many_objects' in df.columns:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        axes[0].bar(df['iteration'], df['failure_with_many_objects'], 
                   width=width, label='Many Objects (≥5)', alpha=0.8)
        axes[0].bar(df['iteration'], df['failure_with_few_objects'], 
                   width=width, label='Few Objects (≤2)', alpha=0.8)
        axes[0].set_xlabel('Iteration', fontsize=11)
        axes[0].set_ylabel('Failure Count', fontsize=11)
        axes[0].set_title('Failures by Object Count', fontsize=12, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3, axis='y')
        
        axes[1].plot(df['iteration'], df['avg_objects_per_scene'], marker='o', linewidth=2)
        axes[1].set_xlabel('Iteration', fontsize=11)
        axes[1].set_ylabel('Avg Objects per Scene', fontsize=11)
        axes[1].set_title('Object Count Trend', fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/failure_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Plots saved to {output_dir}/")

def plot_comparison(csv_files, labels, output_dir='comparison_plots'):
    """여러 모델 비교 시각화"""
    os.makedirs(output_dir, exist_ok=True)
    
    # CSV 파일 로드
    dfs = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        dfs.append(df)
    
    # 1. Success Rate Comparison
    plt.figure(figsize=(12, 7))
    for df, label in zip(dfs, labels):
        plt.plot(df['iteration'], df['success_rate'], marker='o', linewidth=2, 
                markersize=4, label=label, alpha=0.8)
    
    plt.xlabel('Training Iteration', fontsize=13)
    plt.ylabel('Grasp Success Rate (%)', fontsize=13)
    plt.title('Model Performance Comparison', fontsize=15, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 100)
    plt.savefig(f'{output_dir}/success_rate_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Q-value Average Comparison
    plt.figure(figsize=(12, 7))
    for df, label in zip(dfs, labels):
        plt.plot(df['iteration'], df['avg_q_value'], marker='o', linewidth=2,
                markersize=4, label=label, alpha=0.8)
    
    plt.xlabel('Training Iteration', fontsize=13)
    plt.ylabel('Average Q-value', fontsize=13)
    plt.title('Q-value Evolution Comparison', fontsize=15, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{output_dir}/qvalue_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Final Performance Bar Chart
    final_success_rates = [df.iloc[-1]['success_rate'] for df in dfs]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, final_success_rates, alpha=0.8, edgecolor='black')
    
    # 각 막대 위에 값 표시
    for i, (bar, val) in enumerate(zip(bars, final_success_rates)):
        plt.text(i, val + 2, f'{val:.2f}%', ha='center', fontsize=11, fontweight='bold')
    
    plt.ylabel('Success Rate (%)', fontsize=13)
    plt.title('Final Performance Comparison', fontsize=15, fontweight='bold')
    plt.ylim(0, 110)
    plt.grid(True, alpha=0.3, axis='y')
    plt.savefig(f'{output_dir}/final_performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Learning Curve (초기 수렴 속도)
    plt.figure(figsize=(12, 7))
    for df, label in zip(dfs, labels):
        # 처음 50% iteration만 표시
        cutoff = len(df) // 2
        plt.plot(df['iteration'][:cutoff], df['success_rate'][:cutoff], 
                marker='o', linewidth=2, markersize=4, label=label, alpha=0.8)
    
    plt.xlabel('Training Iteration', fontsize=13)
    plt.ylabel('Grasp Success Rate (%)', fontsize=13)
    plt.title('Learning Curve (Early Training)', fontsize=15, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 100)
    plt.savefig(f'{output_dir}/learning_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Statistics Table
    stats_data = []
    for df, label in zip(dfs, labels):
        stats = {
            'Model': label,
            'Final Success Rate (%)': df.iloc[-1]['success_rate'],
            'Max Success Rate (%)': df['success_rate'].max(),
            'Avg Success Rate (%)': df['success_rate'].mean(),
            'Iterations to 80%': get_iterations_to_threshold(df, 80),
            'Final Avg Q-value': df.iloc[-1]['avg_q_value']
        }
        stats_data.append(stats)
    
    stats_df = pd.DataFrame(stats_data)
    stats_df.to_csv(f'{output_dir}/comparison_statistics.csv', index=False)
    
    # 표를 이미지로 저장
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=stats_df.values, colLabels=stats_df.columns,
                    cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2)
    plt.title('Performance Statistics', fontsize=14, fontweight='bold', pad=20)
    plt.savefig(f'{output_dir}/statistics_table.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Comparison plots saved to {output_dir}/")
    print(f"Statistics saved to {output_dir}/comparison_statistics.csv")

def get_iterations_to_threshold(df, threshold):
    """성공률이 threshold를 처음 넘는 iteration 찾기"""
    idx = df[df['success_rate'] >= threshold].index
    if len(idx) > 0:
        return int(df.iloc[idx[0]]['iteration'])
    else:
        return -1  # 도달하지 못함

def main():
    parser = argparse.ArgumentParser(description='Plot evaluation results')
    parser.add_argument('--csv', type=str, nargs='+', required=True,
                       help='CSV file(s) with evaluation results')
    parser.add_argument('--labels', type=str, nargs='+', default=None,
                       help='Labels for each CSV file (for comparison mode)')
    parser.add_argument('--output_dir', type=str, default='evaluation_plots',
                       help='Output directory for plots')
    args = parser.parse_args()
    
    if len(args.csv) == 1:
        # 단일 모델 모드
        df = pd.read_csv(args.csv[0])
        model_name = args.labels[0] if args.labels else 'Model'
        plot_single_model(df, args.output_dir, model_name)
    else:
        # 비교 모드
        if args.labels and len(args.labels) != len(args.csv):
            print("ERROR: Number of labels must match number of CSV files")
            return
        
        labels = args.labels if args.labels else [f'Model {i+1}' for i in range(len(args.csv))]
        plot_comparison(args.csv, labels, args.output_dir)

if __name__ == '__main__':
    main()

