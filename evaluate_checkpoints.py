"""
체크포인트 모델 평가 스크립트
각 snapshot을 로드하여 성능 지표를 측정하고 CSV로 저장

Usage:
    python evaluate_checkpoints.py --log_dir logs/2025-12-07.17.45.26 --trials 30
"""

import os
import glob
import subprocess
import pandas as pd
import numpy as np
import argparse
from datetime import datetime

def evaluate_single_checkpoint(snapshot_path, num_trials=10, num_objects=10, model_name='model'):
    """
    단일 체크포인트 평가
    
    Args:
        snapshot_path: 모델 경로
        num_trials: 평가 시행 횟수
        num_objects: 초기 물체 개수
        model_name: 모델 이름 (폴더명으로 사용)
    
    Returns:
        dict: 평가 결과 딕셔너리
    """
    iteration = extract_iteration_from_path(snapshot_path)
    print(f"\n{'='*60}")
    print(f"Evaluating: {os.path.basename(snapshot_path)}")
    print(f"Iteration: {iteration}, Trials: {num_trials}")
    print(f"{'='*60}\n")
    
    # 체계적인 로그 디렉토리 구조: evaluation/model_name/iter_000000/
    temp_log_dir = f"evaluation/{model_name}/iter_{iteration:06d}"
    
    # main_irb360.py 실행 (테스트 모드)
    cmd = [
        'python', 'main_irb360.py',
        '--is_sim',
        '--obj_mesh_dir', 'objects/blocks',
        '--num_obj', str(num_objects),
        '--is_testing',
        '--load_snapshot',
        '--snapshot_file', snapshot_path,
        '--max_test_trials', str(num_trials),
        '--logging_directory', temp_log_dir,
        '--save_visualizations'
        #'--dueling_dqn',
        #'--double_dqn',
    ]
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Evaluation failed with return code {e.returncode}")
        return None
    
    # 로그 파일에서 결과 파싱
    results = parse_evaluation_logs(temp_log_dir, iteration)
    
    return results

def extract_iteration_from_path(path):
    """snapshot-001000.reinforcement.pth -> 1000"""
    basename = os.path.basename(path)
    iter_str = basename.split('-')[1].split('.')[0]
    return int(iter_str)

def parse_evaluation_logs(log_dir, iteration):
    """
    평가 로그에서 지표 추출
    
    Returns:
        dict: {
            'iteration': int,
            'success_rate': float,
            'total_grasps': int,
            'successful_grasps': int,
            'failed_grasps': int,
            'floor_selections': int,
            'avg_q_value': float,
            'max_q_value': float,
            'std_q_value': float,
            'avg_objects_per_scene': float,
            'failure_with_many_objects': int,
            'failure_with_few_objects': int
        }
    """
    results = {'iteration': iteration}
    
    transitions_dir = os.path.join(log_dir, 'transitions')
    
    try:
        # grasp-success.log.txt 읽기
        success_log = np.loadtxt(os.path.join(transitions_dir, 'grasp-success.log.txt'))
        
        total = len(success_log)
        successful = np.sum(success_log == 1.0)
        failed = np.sum(success_log == 0.0)
        floor = np.sum(success_log == -1.0)
        
        results['success_rate'] = (successful / total * 100) if total > 0 else 0
        results['total_grasps'] = int(total)
        results['successful_grasps'] = int(successful)
        results['failed_grasps'] = int(failed)
        results['floor_selections'] = int(floor)
        
        # predicted-value.log.txt (Q-value)
        q_values = np.loadtxt(os.path.join(transitions_dir, 'predicted-value.log.txt'))
        results['avg_q_value'] = float(np.mean(q_values))
        results['max_q_value'] = float(np.max(q_values))
        results['std_q_value'] = float(np.std(q_values))
        results['min_q_value'] = float(np.min(q_values))
        
        # object-count.log.txt 읽기 (물체 개수별 실패 분석)
        object_count_path = os.path.join(transitions_dir, 'object-count.log.txt')
        if os.path.exists(object_count_path):
            object_counts = np.loadtxt(object_count_path)
            results['avg_objects_per_scene'] = float(np.mean(object_counts))
            
            # 실패 시 물체 개수 분석
            failures = (success_log != 1.0)  # 실패 = 0.0 또는 -1.0
            failed_object_counts = object_counts[failures] if len(object_counts) == len(success_log) else []
            
            if len(failed_object_counts) > 0:
                # 물체가 5개 이상일 때 실패 (많은 물체)
                results['failure_with_many_objects'] = int(np.sum(failed_object_counts >= 5))
                # 물체가 2개 이하일 때 실패 (적은 물체)
                results['failure_with_few_objects'] = int(np.sum(failed_object_counts <= 2))
            else:
                results['failure_with_many_objects'] = 0
                results['failure_with_few_objects'] = 0
        else:
            results['avg_objects_per_scene'] = 0
            results['failure_with_many_objects'] = 0
            results['failure_with_few_objects'] = 0
        
        # clearance.log.txt (씬 재시작 횟수)
        clearance_path = os.path.join(transitions_dir, 'clearance.log.txt')
        if os.path.exists(clearance_path):
            clearance_log = np.loadtxt(clearance_path)
            results['scene_restarts'] = int(len(clearance_log) if clearance_log.ndim > 0 else 1)
        else:
            results['scene_restarts'] = 0
        
    except Exception as e:
        print(f"ERROR parsing logs from {log_dir}: {e}")
        return None
    
    return results

def save_summary_log(log_dir, results):
    """
    평가 결과를 요약하여 .log 파일로 저장
    각 모델 평가 후 해당 폴더에 evaluation_summary.log 저장
    """
    summary_path = os.path.join(log_dir, 'evaluation_summary.log')
    
    with open(summary_path, 'w') as f:
        f.write(f"={'='*60}\n")
        f.write(f"EVALUATION SUMMARY\n")
        f.write(f"={'='*60}\n\n")
        
        f.write(f"Iteration: {results['iteration']}\n")
        f.write(f"Total Grasps: {results['total_grasps']}\n")
        f.write(f"Success Rate: {results['success_rate']:.2f}%\n")
        f.write(f"  - Successful: {results['successful_grasps']}\n")
        f.write(f"  - Failed: {results['failed_grasps']}\n")
        f.write(f"  - Floor Selection: {results['floor_selections']}\n\n")
        
        f.write(f"Q-value Statistics:\n")
        f.write(f"  - Average: {results['avg_q_value']:.4f}\n")
        f.write(f"  - Maximum: {results['max_q_value']:.4f}\n")
        f.write(f"  - Minimum: {results['min_q_value']:.4f}\n")
        f.write(f"  - Std Dev: {results['std_q_value']:.4f}\n\n")
        
        f.write(f"Object Count Analysis:\n")
        f.write(f"  - Average objects per scene: {results['avg_objects_per_scene']:.2f}\n")
        f.write(f"  - Failures with many objects (>=5): {results['failure_with_many_objects']}\n")
        f.write(f"  - Failures with few objects (<=2): {results['failure_with_few_objects']}\n\n")
        
        f.write(f"Scene Restarts: {results['scene_restarts']}\n")
    
    print(f"Summary saved to: {summary_path}")

def main():
    parser = argparse.ArgumentParser(description='Evaluate all checkpoints')
    parser.add_argument('--log_dir', type=str, required=True,
                       help='Log directory containing models/ folder')
    parser.add_argument('--model_name', type=str, default=None,
                       help='Model name for evaluation folder (default: extracted from log_dir)')
    parser.add_argument('--trials', type=int, default=30,
                       help='Number of trials per checkpoint (default: 30)')
    parser.add_argument('--num_obj', type=int, default=10,
                       help='Number of objects to spawn (default: 10)')
    parser.add_argument('--output', type=str, default='evaluation_results.csv',
                       help='Output CSV file path (default: evaluation_results.csv)')
    parser.add_argument('--start_iter', type=int, default=0,
                       help='Start from this iteration (default: 0)')
    parser.add_argument('--end_iter', type=int, default=999999,
                       help='End at this iteration (default: 999999)')
    args = parser.parse_args()
    
    # 모델 이름 자동 추출 (log_dir의 마지막 폴더명 사용)
    if args.model_name is None:
        args.model_name = os.path.basename(os.path.normpath(args.log_dir))
        print(f"Auto-detected model name: {args.model_name}")
    
    # 모든 snapshot 찾기
    model_dir = os.path.join(args.log_dir, 'models')
    snapshots = sorted(glob.glob(os.path.join(model_dir, 'snapshot-*.pth')))
    
    # backup 파일 제외
    snapshots = [s for s in snapshots if 'backup' not in s]
    
    # iteration 범위 필터링
    snapshots = [s for s in snapshots if args.start_iter <= extract_iteration_from_path(s) <= args.end_iter]
    
    print(f"Found {len(snapshots)} checkpoints to evaluate")
    print(f"Iteration range: {args.start_iter} to {args.end_iter}")
    print(f"Evaluation directory: evaluation/{args.model_name}/")
    
    # 각 체크포인트 평가
    all_results = []
    for i, snapshot_path in enumerate(snapshots):
        print(f"\n[{i+1}/{len(snapshots)}] Processing {os.path.basename(snapshot_path)}")
        
        try:
            results = evaluate_single_checkpoint(snapshot_path, args.trials, args.num_obj, args.model_name)
            if results is not None:
                all_results.append(results)
                
                # 개별 폴더에 summary 저장
                eval_log_dir = f"evaluation/{args.model_name}/iter_{results['iteration']:06d}"
                if os.path.exists(eval_log_dir):
                    save_summary_log(eval_log_dir, results)
                
        except Exception as e:
            print(f"ERROR evaluating {snapshot_path}: {e}")
            continue
    
    if len(all_results) == 0:
        print("No results to save!")
        return
    
    # 결과를 DataFrame으로 변환 및 저장
    df = pd.DataFrame(all_results)
    df = df.sort_values('iteration')
    
    # CSV를 evaluation 폴더에 저장
    output_path = f"evaluation/{args.model_name}/{args.output}"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"\n{'='*60}")
    print(f"Evaluation complete! Results saved to: {output_path}")
    print(f"{'='*60}")
    
    # 간단한 통계 출력
    print("\nSummary Statistics:")
    print(f"  Best success rate: {df['success_rate'].max():.2f}% at iteration {df.loc[df['success_rate'].idxmax(), 'iteration']}")
    print(f"  Worst success rate: {df['success_rate'].min():.2f}% at iteration {df.loc[df['success_rate'].idxmin(), 'iteration']}")
    print(f"  Average success rate: {df['success_rate'].mean():.2f}%")
    print(f"  Final success rate (last iter): {df.iloc[-1]['success_rate']:.2f}%")

if __name__ == '__main__':
    main()

