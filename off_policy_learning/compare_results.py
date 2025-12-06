#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
결과 비교 및 시각화 스크립트

evaluate_models.py에서 생성한 JSON 로그를 읽어 비교 그래프와 최종 리포트를 생성합니다.

사용법:
    python compare_results.py

출력:
    - evaluation/reports/success_rate_comparison.png
    - evaluation/reports/comparison_by_epoch.png
    - evaluation/reports/final_report.png
"""

import os
import sys
import json
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# 프로젝트 루트를 path에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    EVALUATION_DIR, RESULTS_DIR, MODEL_CONFIGS,
    VISUALIZATION_CONFIG, get_model_eval_dir, get_reports_dir
)


def load_evaluation_results():
    """
    모든 모델의 평가 결과 로드
    
    Returns:
        dict: {model_name: [result_dict, ...], ...}
    """
    all_results = {}
    
    for model_name in MODEL_CONFIGS.keys():
        eval_dir = get_model_eval_dir(model_name)
        
        if not os.path.exists(eval_dir):
            print(f'[WARNING] No evaluation dir for {model_name}')
            continue
        
        # JSON 파일 로드
        json_files = glob.glob(os.path.join(eval_dir, 'epoch_*_eval.json'))
        
        if not json_files:
            print(f'[WARNING] No evaluation files for {model_name}')
            continue
        
        results = []
        for filepath in sorted(json_files):
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    results.append(data)
            except Exception as e:
                print(f'[ERROR] Failed to load {filepath}: {e}')
        
        if results:
            all_results[model_name] = results
            print(f'[LOAD] {model_name}: {len(results)} evaluation files')
    
    return all_results


def load_training_logs():
    """
    모든 모델의 학습 로그 로드
    
    Returns:
        dict: {model_name: training_log_dict, ...}
    """
    all_logs = {}
    
    for model_name in MODEL_CONFIGS.keys():
        result_dir = os.path.join(RESULTS_DIR, model_name)
        log_path = os.path.join(result_dir, 'logs', 'training_log.json')
        
        if os.path.exists(log_path):
            try:
                with open(log_path, 'r') as f:
                    all_logs[model_name] = json.load(f)
                print(f'[LOAD] {model_name}: training log')
            except Exception as e:
                print(f'[ERROR] Failed to load training log for {model_name}: {e}')
    
    return all_logs


def plot_success_rate_comparison(all_results, output_path):
    """
    최종 성공률 비교 바 차트 생성
    
    Args:
        all_results: 평가 결과 딕셔너리
        output_path: 저장 경로
    """
    fig, ax = plt.subplots(figsize=VISUALIZATION_CONFIG['figsize'])
    
    models = []
    success_rates = []
    colors = []
    
    for model_name in MODEL_CONFIGS.keys():
        if model_name not in all_results:
            continue
        
        results = all_results[model_name]
        
        # 마지막 에폭 결과 사용
        last_result = results[-1]
        
        if 'overall_success_rate' in last_result:
            models.append(MODEL_CONFIGS[model_name]['name'])
            success_rates.append(last_result['overall_success_rate'] * 100)
            colors.append(VISUALIZATION_CONFIG['colors'].get(model_name, '#1f77b4'))
    
    if not models:
        print('[WARNING] No success rate data available')
        return
    
    x = np.arange(len(models))
    bars = ax.bar(x, success_rates, color=colors, alpha=0.8, edgecolor='black')
    
    # 값 표시
    for bar, rate in zip(bars, success_rates):
        height = bar.get_height()
        ax.annotate(f'{rate:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Grasp Success Rate (%)', fontsize=12)
    ax.set_title('Grasp Success Rate Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha='right')
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=VISUALIZATION_CONFIG['dpi'], bbox_inches='tight')
    plt.close()
    
    print(f'[SAVE] {output_path}')


def plot_comparison_by_epoch(all_results, output_path):
    """
    에폭별 성공률 변화 비교 그래프 생성
    
    Args:
        all_results: 평가 결과 딕셔너리
        output_path: 저장 경로
    """
    fig, ax = plt.subplots(figsize=VISUALIZATION_CONFIG['figsize'])
    
    for model_name in MODEL_CONFIGS.keys():
        if model_name not in all_results:
            continue
        
        results = all_results[model_name]
        
        epochs = []
        success_rates = []
        
        for result in results:
            if 'epoch' in result and 'overall_success_rate' in result:
                epochs.append(result['epoch'])
                success_rates.append(result['overall_success_rate'] * 100)
        
        if epochs:
            color = VISUALIZATION_CONFIG['colors'].get(model_name, '#1f77b4')
            ax.plot(epochs, success_rates, 
                    color=color, linewidth=2, marker='o', markersize=6,
                    label=MODEL_CONFIGS[model_name]['name'])
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Grasp Success Rate (%)', fontsize=12)
    ax.set_title('Success Rate by Epoch', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=VISUALIZATION_CONFIG['dpi'], bbox_inches='tight')
    plt.close()
    
    print(f'[SAVE] {output_path}')


def plot_loss_comparison(all_logs, output_path):
    """
    학습 Loss 비교 그래프 생성
    
    Args:
        all_logs: 학습 로그 딕셔너리
        output_path: 저장 경로
    """
    fig, ax = plt.subplots(figsize=VISUALIZATION_CONFIG['figsize'])
    
    for model_name in MODEL_CONFIGS.keys():
        if model_name not in all_logs:
            continue
        
        log_data = all_logs[model_name]
        
        if 'epoch_losses' in log_data and log_data['epoch_losses']:
            epochs = range(1, len(log_data['epoch_losses']) + 1)
            losses = log_data['epoch_losses']
            
            color = VISUALIZATION_CONFIG['colors'].get(model_name, '#1f77b4')
            ax.plot(epochs, losses, 
                    color=color, linewidth=2,
                    label=MODEL_CONFIGS[model_name]['name'])
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Average Loss', fontsize=12)
    ax.set_title('Training Loss Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=VISUALIZATION_CONFIG['dpi'], bbox_inches='tight')
    plt.close()
    
    print(f'[SAVE] {output_path}')


def plot_q_value_comparison(all_results, output_path):
    """
    평균 Q값 비교 그래프 생성
    
    Args:
        all_results: 평가 결과 딕셔너리
        output_path: 저장 경로
    """
    fig, ax = plt.subplots(figsize=VISUALIZATION_CONFIG['figsize'])
    
    for model_name in MODEL_CONFIGS.keys():
        if model_name not in all_results:
            continue
        
        results = all_results[model_name]
        
        epochs = []
        q_values = []
        
        for result in results:
            if 'epoch' in result and 'avg_q_value' in result:
                epochs.append(result['epoch'])
                q_values.append(result['avg_q_value'])
        
        if epochs:
            color = VISUALIZATION_CONFIG['colors'].get(model_name, '#1f77b4')
            ax.plot(epochs, q_values, 
                    color=color, linewidth=2, marker='s', markersize=6,
                    label=MODEL_CONFIGS[model_name]['name'])
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Average Q-Value', fontsize=12)
    ax.set_title('Average Q-Value by Epoch', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=VISUALIZATION_CONFIG['dpi'], bbox_inches='tight')
    plt.close()
    
    print(f'[SAVE] {output_path}')


def generate_final_report(all_results, all_logs, output_path):
    """
    최종 비교 리포트 생성 (종합 대시보드)
    
    Args:
        all_results: 평가 결과 딕셔너리
        all_logs: 학습 로그 딕셔너리
        output_path: 저장 경로
    """
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.25)
    
    # 1. 최종 성공률 바 차트 (상단 왼쪽)
    ax1 = fig.add_subplot(gs[0, 0])
    
    models = []
    success_rates = []
    colors = []
    
    for model_name in MODEL_CONFIGS.keys():
        if model_name in all_results and all_results[model_name]:
            last_result = all_results[model_name][-1]
            if 'overall_success_rate' in last_result:
                models.append(MODEL_CONFIGS[model_name]['name'].replace(' ', '\n'))
                success_rates.append(last_result['overall_success_rate'] * 100)
                colors.append(VISUALIZATION_CONFIG['colors'].get(model_name, '#1f77b4'))
    
    if models:
        x = np.arange(len(models))
        bars = ax1.bar(x, success_rates, color=colors, alpha=0.8, edgecolor='black')
        for bar, rate in zip(bars, success_rates):
            ax1.annotate(f'{rate:.1f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(models, fontsize=9)
        ax1.set_ylim(0, 100)
    ax1.set_title('Final Success Rate', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Success Rate (%)', fontsize=10)
    ax1.grid(axis='y', alpha=0.3)
    
    # 2. 에폭별 성공률 (상단 오른쪽)
    ax2 = fig.add_subplot(gs[0, 1])
    
    for model_name in MODEL_CONFIGS.keys():
        if model_name in all_results:
            results = all_results[model_name]
            epochs = [r['epoch'] for r in results if 'epoch' in r and 'overall_success_rate' in r]
            rates = [r['overall_success_rate'] * 100 for r in results if 'epoch' in r and 'overall_success_rate' in r]
            
            if epochs:
                color = VISUALIZATION_CONFIG['colors'].get(model_name, '#1f77b4')
                ax2.plot(epochs, rates, color=color, linewidth=2, marker='o', markersize=5,
                        label=MODEL_CONFIGS[model_name]['name'])
    
    ax2.set_title('Success Rate by Epoch', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=10)
    ax2.set_ylabel('Success Rate (%)', fontsize=10)
    ax2.legend(loc='lower right', fontsize=8)
    ax2.set_ylim(0, 100)
    ax2.grid(True, alpha=0.3)
    
    # 3. 학습 Loss 곡선 (중간 왼쪽)
    ax3 = fig.add_subplot(gs[1, 0])
    
    for model_name in MODEL_CONFIGS.keys():
        if model_name in all_logs and 'epoch_losses' in all_logs[model_name]:
            losses = all_logs[model_name]['epoch_losses']
            if losses:
                epochs = range(1, len(losses) + 1)
                color = VISUALIZATION_CONFIG['colors'].get(model_name, '#1f77b4')
                ax3.plot(epochs, losses, color=color, linewidth=2,
                        label=MODEL_CONFIGS[model_name]['name'])
    
    ax3.set_title('Training Loss', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Epoch', fontsize=10)
    ax3.set_ylabel('Loss', fontsize=10)
    ax3.legend(loc='upper right', fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # 4. 평균 Q값 (중간 오른쪽)
    ax4 = fig.add_subplot(gs[1, 1])
    
    for model_name in MODEL_CONFIGS.keys():
        if model_name in all_results:
            results = all_results[model_name]
            epochs = [r['epoch'] for r in results if 'epoch' in r and 'avg_q_value' in r]
            q_vals = [r['avg_q_value'] for r in results if 'epoch' in r and 'avg_q_value' in r]
            
            if epochs:
                color = VISUALIZATION_CONFIG['colors'].get(model_name, '#1f77b4')
                ax4.plot(epochs, q_vals, color=color, linewidth=2, marker='s', markersize=5,
                        label=MODEL_CONFIGS[model_name]['name'])
    
    ax4.set_title('Average Q-Value by Epoch', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Epoch', fontsize=10)
    ax4.set_ylabel('Q-Value', fontsize=10)
    ax4.legend(loc='lower right', fontsize=8)
    ax4.grid(True, alpha=0.3)
    
    # 5. 결과 요약 테이블 (하단)
    ax5 = fig.add_subplot(gs[2, :])
    ax5.axis('off')
    
    # 테이블 데이터 구성
    table_data = []
    headers = ['Model', 'Double DQN', 'Dueling DQN', 'Final Success Rate', 'Avg Q-Value', 'Min Loss']
    
    for model_name in MODEL_CONFIGS.keys():
        config = MODEL_CONFIGS[model_name]
        row = [
            config['name'],
            '✓' if config['double_dqn'] else '✗',
            '✓' if config['dueling_dqn'] else '✗',
        ]
        
        if model_name in all_results and all_results[model_name]:
            last = all_results[model_name][-1]
            row.append(f"{last.get('overall_success_rate', 0)*100:.1f}%")
            row.append(f"{last.get('avg_q_value', 0):.4f}")
        else:
            row.extend(['N/A', 'N/A'])
        
        if model_name in all_logs and 'epoch_losses' in all_logs[model_name]:
            losses = all_logs[model_name]['epoch_losses']
            if losses:
                row.append(f"{min(losses):.6f}")
            else:
                row.append('N/A')
        else:
            row.append('N/A')
        
        table_data.append(row)
    
    if table_data:
        table = ax5.table(
            cellText=table_data,
            colLabels=headers,
            loc='center',
            cellLoc='center',
            colColours=['#E6E6FA'] * len(headers)
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
    
    ax5.set_title('Model Comparison Summary', fontsize=12, fontweight='bold', pad=20)
    
    # 전체 제목
    fig.suptitle('Off-Policy Learning Results Comparison', fontsize=16, fontweight='bold', y=0.98)
    
    plt.savefig(output_path, dpi=VISUALIZATION_CONFIG['dpi'], bbox_inches='tight')
    plt.close()
    
    print(f'[SAVE] {output_path}')


def main():
    print('='*60)
    print('RESULTS COMPARISON')
    print('='*60)
    
    # 결과 디렉토리 확인
    reports_dir = get_reports_dir()
    os.makedirs(reports_dir, exist_ok=True)
    
    # 평가 결과 로드
    print('\n[1/2] Loading evaluation results...')
    all_results = load_evaluation_results()
    
    # 학습 로그 로드
    print('\n[2/2] Loading training logs...')
    all_logs = load_training_logs()
    
    if not all_results and not all_logs:
        print('\n[ERROR] No data to compare!')
        print('Run train_offline.py and evaluate_models.py first.')
        return
    
    # 그래프 생성
    print('\nGenerating comparison graphs...')
    
    if all_results:
        # 성공률 비교 바 차트
        plot_success_rate_comparison(
            all_results,
            os.path.join(reports_dir, 'success_rate_comparison.png')
        )
        
        # 에폭별 성공률 비교
        plot_comparison_by_epoch(
            all_results,
            os.path.join(reports_dir, 'comparison_by_epoch.png')
        )
        
        # Q값 비교
        plot_q_value_comparison(
            all_results,
            os.path.join(reports_dir, 'q_value_comparison.png')
        )
    
    if all_logs:
        # Loss 비교
        plot_loss_comparison(
            all_logs,
            os.path.join(reports_dir, 'loss_comparison.png')
        )
    
    # 최종 리포트 생성
    if all_results or all_logs:
        generate_final_report(
            all_results,
            all_logs,
            os.path.join(reports_dir, 'final_report.png')
        )
    
    print('\n' + '='*60)
    print('COMPARISON COMPLETE')
    print('='*60)
    print(f'Reports saved to: {reports_dir}')


if __name__ == '__main__':
    main()


