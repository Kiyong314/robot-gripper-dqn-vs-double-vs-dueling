#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
로그 폴더 정리 스크립트

지정된 iteration까지만 유지하고 나머지는 삭제합니다.
"""

import os
import glob
import shutil

def trim_log_file(filepath, max_lines):
    """로그 파일을 max_lines까지만 유지"""
    if not os.path.exists(filepath):
        print(f'  [SKIP] Not found: {filepath}')
        return
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    original_count = len(lines)
    
    if original_count > max_lines:
        lines = lines[:max_lines]
        with open(filepath, 'w') as f:
            f.writelines(lines)
        print(f'  [TRIM] {os.path.basename(filepath)}: {original_count} -> {max_lines} lines')
    else:
        print(f'  [OK] {os.path.basename(filepath)}: {original_count} lines (no change needed)')

def delete_files_after(directory, max_iteration, pattern='*'):
    """지정된 iteration 이후의 파일 삭제"""
    if not os.path.exists(directory):
        print(f'  [SKIP] Not found: {directory}')
        return 0
    
    deleted_count = 0
    
    # 모든 파일 검색
    files = glob.glob(os.path.join(directory, pattern))
    
    for filepath in files:
        filename = os.path.basename(filepath)
        
        # 파일명에서 iteration 번호 추출 (예: 001501.0.color.png -> 1501)
        try:
            # 다양한 형식 처리
            if filename.startswith('snapshot-'):
                # snapshot-001500.reinforcement.pth -> 1500
                parts = filename.replace('snapshot-', '').split('.')
                iteration = int(parts[0])
            else:
                # 001501.0.color.png -> 1501
                iteration = int(filename.split('.')[0])
            
            if iteration > max_iteration:
                os.remove(filepath)
                deleted_count += 1
        except (ValueError, IndexError):
            # 파일명 파싱 실패 시 무시 (backup 파일 등)
            pass
    
    return deleted_count

def trim_logs(log_dir, max_iteration):
    """로그 폴더 정리 메인 함수"""
    print(f'\n{"="*60}')
    print(f'Trimming logs to iteration {max_iteration}')
    print(f'Directory: {log_dir}')
    print(f'{"="*60}\n')
    
    # 1. 로그 파일 정리 (transitions/)
    print('[1/6] Trimming log files...')
    transitions_dir = os.path.join(log_dir, 'transitions')
    log_files = [
        'executed-action.log.txt',
        'label-value.log.txt',
        'predicted-value.log.txt',
        'reward-value.log.txt',
        'grasp-success.log.txt',
        'place-success.log.txt',
        'change-detected.log.txt',
        'clearance.log.txt'
    ]
    
    for log_file in log_files:
        filepath = os.path.join(transitions_dir, log_file)
        trim_log_file(filepath, max_iteration)
    
    # 2. Color heightmaps 정리
    print('\n[2/6] Deleting color heightmaps...')
    color_hm_dir = os.path.join(log_dir, 'data', 'color-heightmaps')
    deleted = delete_files_after(color_hm_dir, max_iteration, '*.png')
    print(f'  Deleted {deleted} files')
    
    # 3. Depth heightmaps 정리
    print('\n[3/6] Deleting depth heightmaps...')
    depth_hm_dir = os.path.join(log_dir, 'data', 'depth-heightmaps')
    deleted = delete_files_after(depth_hm_dir, max_iteration, '*.png')
    print(f'  Deleted {deleted} files')
    
    # 4. Color images 정리
    print('\n[4/6] Deleting color images...')
    color_img_dir = os.path.join(log_dir, 'data', 'color-images')
    deleted = delete_files_after(color_img_dir, max_iteration, '*.png')
    print(f'  Deleted {deleted} files')
    
    # 5. Depth images 정리
    print('\n[5/6] Deleting depth images...')
    depth_img_dir = os.path.join(log_dir, 'data', 'depth-images')
    deleted = delete_files_after(depth_img_dir, max_iteration, '*.png')
    print(f'  Deleted {deleted} files')
    
    # 6. Visualizations 정리
    print('\n[6/6] Deleting visualizations...')
    vis_dir = os.path.join(log_dir, 'visualizations')
    deleted = delete_files_after(vis_dir, max_iteration, '*.png')
    print(f'  Deleted {deleted} files')
    
    # 7. Models 정리 (선택)
    print('\n[7/7] Checking models...')
    models_dir = os.path.join(log_dir, 'models')
    deleted = delete_files_after(models_dir, max_iteration, 'snapshot-*.pth')
    print(f'  Deleted {deleted} model files')
    
    print(f'\n{"="*60}')
    print('Trimming completed!')
    print(f'{"="*60}\n')


if __name__ == '__main__':
    import sys
    
    # 기본값
    log_dir = r'D:\1.github\24.Pick_And_Place_Robot\logs\2025-12-04.00.03.30_double_dueling'
    max_iteration = 1500
    
    # 명령줄 인자 처리
    if len(sys.argv) >= 2:
        log_dir = sys.argv[1]
    if len(sys.argv) >= 3:
        max_iteration = int(sys.argv[2])
    
    trim_logs(log_dir, max_iteration)


