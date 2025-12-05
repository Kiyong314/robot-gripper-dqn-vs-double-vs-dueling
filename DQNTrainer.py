import os
import numpy as np
import cv2
import torch
from torch.autograd import Variable
from scipy import ndimage
import matplotlib.pyplot as plt
from DQNModels import DQN

class DQNTrainer(object):
    def __init__(self, method, grasp_rewards, future_reward_discount,
                 is_testing, load_snapshot, snapshot_file, force_cpu,
                 double_dqn=False, target_update_freq=100, dueling_dqn=False):
        """
        DQN Trainer 초기화
        
        Args:
            double_dqn: Double DQN 사용 여부 (Q값 과대추정 방지)
            target_update_freq: Target network 업데이트 주기 (iterations)
            dueling_dqn: Dueling DQN 사용 여부 (Value + Advantage streams)
        """
        self.method = method
        self.use_cuda = torch.cuda.is_available() and not force_cpu
        self.double_dqn = double_dqn
        self.target_update_freq = target_update_freq
        self.dueling_dqn = dueling_dqn

        if self.use_cuda:
            print("CUDA detected. Running with GPU acceleration.")
        elif force_cpu:
            print("CUDA detected, but overriding with option '--cpu'. Running with only CPU.")
        else:
            print("CUDA is *NOT* detected. Running with only CPU.")

        if self.method == 'reinforcement':
            self._initialize_reinforcement_learning(grasp_rewards, future_reward_discount)

        if load_snapshot:
            self._load_snapshot(snapshot_file)

        self.model = self.model.cuda() if self.use_cuda else self.model
        self.model.train()
        
        # Double DQN: Main network CUDA 이동 및 snapshot 로드 완료 후 Target network 생성
        # 이 시점에서 Main network가 완전히 준비된 상태임
        if self.double_dqn:
            self._initialize_target_network()
            print(f'[Double DQN] Enabled with target update frequency: {self.target_update_freq}')
        
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-4, momentum=0.9, weight_decay=2e-5)
        self.iteration = 0
        self._initialize_logs()

    def _initialize_reinforcement_learning(self, grasp_rewards, future_reward_discount):
        """강화학습 모델 및 설정 초기화"""
        self.model = DQN(self.use_cuda, dueling=self.dueling_dqn)
        self.grasp_rewards = grasp_rewards
        self.future_reward_discount = future_reward_discount
        self.criterion = torch.nn.SmoothL1Loss(reduce=False)  # Huber loss
        if self.use_cuda:
            self.criterion = self.criterion.cuda()
        
        # Dueling DQN 활성화 로그
        if self.dueling_dqn:
            print('[Dueling DQN] Enabled: Q(s,a) = V(s) + (A(s,a) - mean(A))')
        
        # Double DQN: Target Network는 나중에 _initialize_target_network에서 생성
        # Main network가 CUDA로 이동하고 snapshot 로드 후에 생성해야 함
        self.target_model = None

    def _load_snapshot(self, snapshot_file):
        """스냅샷 파일에서 모델 가중치를 로드합니다.
        GPU에서 저장된 모델도 CPU에서 로드 가능하도록 map_location 처리
        """
        print(f'Loading snapshot from: {snapshot_file}')
        # CPU/GPU 호환을 위해 map_location 추가
        if self.use_cuda:
            self.model.load_state_dict(torch.load(snapshot_file))
        else:
            self.model.load_state_dict(torch.load(snapshot_file, map_location=torch.device('cpu')))
        print('Pre-trained model snapshot loaded successfully.')

    def _initialize_target_network(self):
        """
        Double DQN용 Target Network 초기화
        
        Main network가 CUDA로 이동하고 snapshot 로드가 완료된 후에 호출됨.
        Main network와 동일한 구조로 생성하고 가중치를 복사함.
        """
        print('[Double DQN] Initializing target network...')
        
        # 1. Target network를 Main network와 동일한 디바이스에서 직접 생성
        # Dueling DQN 옵션도 동일하게 적용
        self.target_model = DQN(self.use_cuda, dueling=self.dueling_dqn)
        
        # 2. CUDA 이동 (Main network와 동일한 디바이스)
        if self.use_cuda:
            self.target_model = self.target_model.cuda()
        
        # 3. Main network의 가중치를 Target network에 복사 (deep copy)
        self.target_model.load_state_dict(self.model.state_dict())
        
        # 4. Target network는 학습하지 않음 (eval 모드)
        self.target_model.eval()
        
        # 5. 검증: Target network가 정상적으로 초기화되었는지 확인
        self._verify_target_network()
        
        print('[Double DQN] Target network initialized and synchronized with main network')
    
    def _verify_target_network(self):
        """Target network가 Main network와 동일한 가중치를 가지는지 검증 (가중치 비교 방식)"""
        if self.target_model is None:
            print('[Double DQN] Warning: Target network is None')
            return False
        
        # 첫 번째 레이어의 가중치 비교
        main_params = list(self.model.parameters())
        target_params = list(self.target_model.parameters())
        
        if len(main_params) != len(target_params):
            print(f'[Double DQN] Warning: Parameter count mismatch ({len(main_params)} vs {len(target_params)})')
            return False
        
        # 가중치 차이 확인
        total_diff = 0.0
        for main_p, target_p in zip(main_params, target_params):
            diff = torch.abs(main_p.data - target_p.data).sum().item()
            total_diff += diff
        
        if total_diff > 1e-6:
            print(f'[Double DQN] Warning: Weight difference detected: {total_diff}')
            return False
        
        print(f'[Double DQN] Verification passed: Main and Target networks are identical (diff={total_diff:.2e})')
        return True
    
    def _verify_target_network_output(self, color_heightmap, depth_heightmap):
        """
        실제 heightmap 데이터를 Main/Target network에 넣어 출력 비교 검증
        출력이 다르면 경고 출력 후 Enter 입력 대기
        
        Args:
            color_heightmap: 현재 컬러 heightmap (H, W, 3)
            depth_heightmap: 현재 깊이 heightmap (H, W)
        
        Returns:
            bool: 검증 통과 여부 (True=일치, False=불일치)
        """
        if not self.double_dqn or self.target_model is None:
            print('[Double DQN] Warning: Target network not initialized for verification')
            return False
        
        print('[Double DQN] Verifying Main/Target network output consistency...')
        
        # 1. Main network forward (train 모드 유지 - BatchNorm running stats 초기화 전 eval 모드 사용 시 NaN 발생)
        was_training = self.model.training
        self.model.train()  # BatchNorm이 batch statistics 사용하도록 train 모드
        with torch.no_grad():
            main_output, _ = self.forward(color_heightmap, depth_heightmap, is_volatile=True)
        if not was_training:
            self.model.eval()
        
        # 2. Target network forward (forward_target 내부에서 train 모드로 전환)
        target_output, _ = self.forward_target(color_heightmap, depth_heightmap)
        
        # 3. NaN 검사
        main_has_nan = np.isnan(main_output).any()
        target_has_nan = np.isnan(target_output).any()
        
        if main_has_nan or target_has_nan:
            print(f'[Double DQN] WARNING: NaN detected in network output!')
            print(f'  - Main network has NaN: {main_has_nan} (count: {np.isnan(main_output).sum()})')
            print(f'  - Target network has NaN: {target_has_nan} (count: {np.isnan(target_output).sum()})')
            print(f'  - Main output range: [{np.nanmin(main_output):.4f}, {np.nanmax(main_output):.4f}]')
            print(f'  - Target output range: [{np.nanmin(target_output):.4f}, {np.nanmax(target_output):.4f}]')
            input('[Double DQN] Press Enter to continue...')
            return False
        
        # 4. 출력 비교 (최대 차이값, 평균 차이값)
        max_diff = np.abs(main_output - target_output).max()
        mean_diff = np.abs(main_output - target_output).mean()
        
        # 5. 검증 결과 판정 (허용 오차: 1e-5)
        if max_diff > 1e-5:
            print(f'[Double DQN] WARNING: Output mismatch detected!')
            print(f'  - Max diff: {max_diff:.6f}')
            print(f'  - Mean diff: {mean_diff:.6f}')
            print(f'  - Main output range: [{main_output.min():.4f}, {main_output.max():.4f}]')
            print(f'  - Target output range: [{target_output.min():.4f}, {target_output.max():.4f}]')
            input('[Double DQN] Press Enter to continue...')
            return False
        
        print(f'[Double DQN] Output verification PASSED (max diff: {max_diff:.2e}, mean diff: {mean_diff:.2e})')
        return True

    def _initialize_logs(self):
        self.executed_action_log = []
        self.label_value_log = []
        self.reward_value_log = []
        self.predicted_value_log = []
        self.clearance_log = []
        self.grasp_success_log = []
        self.place_success_log = []
        self.change_detected_log = []
        
        # 균형 잡힌 Experience Replay를 위한 분리된 버퍼
        # 성공/실패 샘플을 별도로 저장하여 3:1 비율로 샘플링
        self.success_buffer = []  # 성공 샘플의 iteration 인덱스 리스트
        self.failure_buffer = []  # 실패 샘플의 iteration 인덱스 리스트 (일반 실패 + 바닥 선택)

    def preload(self, transitions_directory):
        """
        이전 학습 세션에서 로그 데이터를 로드하고 Experience Replay 버퍼를 복원합니다.
        
        학습 재개 시 호출되어:
        1. 마지막 iteration 번호 계산
        2. 모든 로그 파일 로드
        3. grasp_success_log를 기반으로 success_buffer/failure_buffer 복원
        """
        # 먼저 executed-action.log.txt에서 마지막 iteration 계산
        action_log_path = os.path.join(transitions_directory, 'executed-action.log.txt')
        if os.path.exists(action_log_path):
            with open(action_log_path, 'r') as f:
                line_count = sum(1 for _ in f)
            self.iteration = line_count  # 마지막 iteration + 1 설정 (다음 iteration 번호)
            print(f'[PRELOAD] Resuming from iteration: {self.iteration}')
        
        # 로그 데이터 로드
        self._load_log_data(transitions_directory)
        
        # grasp_success_log를 기반으로 Experience Replay 버퍼 복원
        self._restore_replay_buffers()
    def _load_log_data(self, transitions_directory):
        """이전 학습 세션의 로그 데이터를 로드합니다."""
        # 파일명과 속성명 매핑 (파일명 -> 클래스 속성명)
        # 각 파일의 예상 열 수도 함께 지정 (1열=스칼라값, 4열=action 등)
        log_file_mapping = {
            'executed-action.log.txt': ('executed_action_log', 4),      # [action, rot, y, x]
            'label-value.log.txt': ('label_value_log', 1),              # [value]
            'predicted-value.log.txt': ('predicted_value_log', 1),      # [value]
            'reward-value.log.txt': ('reward_value_log', 1),            # [value]
            'clearance.log.txt': ('clearance_log', 1),                  # [value]
            'grasp-success.log.txt': ('grasp_success_log', 1),          # [success]
            'place-success.log.txt': ('place_success_log', 1),          # [success]
            'change-detected.log.txt': ('change_detected_log', 1)       # [detected]
        }
        for filename, (attr_name, expected_cols) in log_file_mapping.items():
            filepath = os.path.join(transitions_directory, filename)
            if not os.path.exists(filepath):
                print(f'Warning: {filename} not found, skipping...')
                continue
            
            try:
                data = np.loadtxt(filepath, delimiter=' ')
            except Exception as e:
                print(f'Warning: Failed to load {filename}: {e}')
                continue
            
            # 빈 배열 처리
            if data.size == 0:
                print(f'Warning: {filename} is empty, skipping...')
                continue
            
            # 1열짜리 2D 데이터가 1D로 로드된 경우 복원
            if data.ndim == 1 and expected_cols == 1:
                # 각 요소를 [value] 형태로 변환하여 2D처럼 사용
                #data = data[:self.iteration]
                result = [[val] for val in data.tolist()]
            elif data.ndim == 2:
                #data = data[:self.iteration, :]
                result = data.tolist()
            elif data.ndim == 1 and expected_cols > 1:
                # 단일 행 데이터인 경우
                result = [data.tolist()] if self.iteration > 0 else []
            elif data.ndim == 0:
                # 단일 값
                result = [[data.item()]] if self.iteration > 0 else []
            else:
                result = []
            
            setattr(self, attr_name, result)
            print(f'Loaded {filename}: {len(result)} entries')

    def _restore_replay_buffers(self):
        """
        grasp_success_log를 기반으로 Experience Replay 버퍼(success_buffer, failure_buffer) 복원
        
        학습 재개 시 이전 세션의 성공/실패 샘플들을 버퍼에 복원하여
        Experience Replay가 이전 데이터도 활용할 수 있도록 함.
        
        grasp_success 값:
        - 1: 성공 → success_buffer에 추가
        - 0: 일반 실패 → failure_buffer에 추가  
        - -1: 바닥 선택 실패 → failure_buffer에 추가
        """
        self.success_buffer = []
        self.failure_buffer = []
        
        if not hasattr(self, 'grasp_success_log') or len(self.grasp_success_log) == 0:
            print('[PRELOAD] No grasp_success_log found, starting with empty buffers')
            return
        
        # grasp_success_log를 순회하며 버퍼 복원
        for idx, success_entry in enumerate(self.grasp_success_log):
            success_value = int(success_entry[0]) if isinstance(success_entry, list) else int(success_entry)
            
            if success_value == 1:
                self.success_buffer.append(idx)
            else:  # 0 또는 -1
                self.failure_buffer.append(idx)
        
        print(f'[PRELOAD] Restored replay buffers: {len(self.success_buffer)} success, {len(self.failure_buffer)} failure samples')
        
        # 버퍼 통계 출력 (최근 100개 샘플 기준 성공률)
        if len(self.grasp_success_log) > 0:
            recent_samples = self.grasp_success_log[-min(100, len(self.grasp_success_log)):]
            recent_success = sum(1 for s in recent_samples if (s[0] if isinstance(s, list) else s) == 1)
            print(f'[PRELOAD] Recent success rate (last {len(recent_samples)} samples): {recent_success}/{len(recent_samples)} = {100*recent_success/len(recent_samples):.1f}%')

    # Compute forward pass through model to compute affordances/Q
    def forward(self, color_heightmap, depth_heightmap, is_volatile=False, specific_rotation=-1):

        # Apply 2x scale to input heightmaps
        color_heightmap_2x = ndimage.zoom(color_heightmap, zoom=[2,2,1], order=0)
        depth_heightmap_2x = ndimage.zoom(depth_heightmap, zoom=[2,2], order=0)
        assert(color_heightmap_2x.shape[0:2] == depth_heightmap_2x.shape[0:2])
        
        # Add extra padding (to handle rotations inside network)
        diag_length = float(color_heightmap_2x.shape[0]) * np.sqrt(2)
        diag_length = np.ceil(diag_length/32)*32
        padding_width = int((diag_length - color_heightmap_2x.shape[0])/2)
        color_heightmap_2x_r =  np.pad(color_heightmap_2x[:,:,0], padding_width, 'constant', constant_values=0)
        color_heightmap_2x_r.shape = (color_heightmap_2x_r.shape[0], color_heightmap_2x_r.shape[1], 1)
        color_heightmap_2x_g =  np.pad(color_heightmap_2x[:,:,1], padding_width, 'constant', constant_values=0)
        color_heightmap_2x_g.shape = (color_heightmap_2x_g.shape[0], color_heightmap_2x_g.shape[1], 1)
        color_heightmap_2x_b =  np.pad(color_heightmap_2x[:,:,2], padding_width, 'constant', constant_values=0)
        color_heightmap_2x_b.shape = (color_heightmap_2x_b.shape[0], color_heightmap_2x_b.shape[1], 1)
        color_heightmap_2x = np.concatenate((color_heightmap_2x_r, color_heightmap_2x_g, color_heightmap_2x_b), axis=2)
        depth_heightmap_2x =  np.pad(depth_heightmap_2x, padding_width, 'constant', constant_values=0)
        
        # Pre-process color image (scale and normalize)
        image_mean = [0.485, 0.456, 0.406]
        image_std = [0.229, 0.224, 0.225]
        input_color_image = color_heightmap_2x.astype(float)/255
        for c in range(3):
            input_color_image[:, :, c] = (input_color_image[:, :, c] - image_mean[c])/image_std[c]
        # Pre-process depth image (normalize)
        image_mean = 0.01
        image_std = 0.03
        depth_heightmap_2x.shape = (depth_heightmap_2x.shape[0], depth_heightmap_2x.shape[1], 1)
        input_depth_image = (depth_heightmap_2x - image_mean) / image_std
        
        # Construct minibatch of size 1 (b,c,h,w)
        input_color_image.shape = (input_color_image.shape[0], input_color_image.shape[1], input_color_image.shape[2], 1)
        input_depth_image.shape = (input_depth_image.shape[0], input_depth_image.shape[1], input_depth_image.shape[2], 1)
        input_color_data = torch.from_numpy(input_color_image.astype(np.float32)).permute(3,2,0,1)
        input_depth_data = torch.from_numpy(input_depth_image.astype(np.float32)).permute(3,2,0,1)
        # Pass input data through model
        output_prob, state_feat = self.model.forward(input_color_data, input_depth_data,  is_volatile, specific_rotation)

        if self.method == 'reinforcement':

            # Return Q values (and remove extra padding)
            for rotate_idx in range(len(output_prob)):
                if rotate_idx == 0:
                    grasp_predictions = output_prob[rotate_idx][0].cpu().data.numpy()[:,0,int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2),int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2)]

                else:
                    grasp_predictions = np.concatenate((grasp_predictions, output_prob[rotate_idx][0].cpu().data.numpy()[:,0,int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2),int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2)]), axis=0)

        return grasp_predictions, state_feat

    def forward_target(self, color_heightmap, depth_heightmap):
        """
        Target network를 통한 forward pass (Double DQN용)
        
        Double DQN에서 Q값 평가에 사용됨.
        Main network로 action 선택 후, Target network로 해당 action의 Q값 계산.
        
        Args:
            color_heightmap: 컬러 heightmap (H, W, 3)
            depth_heightmap: 깊이 heightmap (H, W)
        
        Returns:
            grasp_predictions: Target network의 Q값 예측 (num_rotations, H, W)
            state_feat: 상태 특징 (사용 안 함)
        """
        if not self.double_dqn or self.target_model is None:
            raise RuntimeError('[Double DQN] Target network not initialized')
        
        # Apply 2x scale to input heightmaps
        color_heightmap_2x = ndimage.zoom(color_heightmap, zoom=[2, 2, 1], order=0)
        depth_heightmap_2x = ndimage.zoom(depth_heightmap, zoom=[2, 2], order=0)
        assert(color_heightmap_2x.shape[0:2] == depth_heightmap_2x.shape[0:2])
        
        # Add extra padding (to handle rotations inside network)
        diag_length = float(color_heightmap_2x.shape[0]) * np.sqrt(2)
        diag_length = np.ceil(diag_length / 32) * 32
        padding_width = int((diag_length - color_heightmap_2x.shape[0]) / 2)
        color_heightmap_2x_r = np.pad(color_heightmap_2x[:, :, 0], padding_width, 'constant', constant_values=0)
        color_heightmap_2x_r.shape = (color_heightmap_2x_r.shape[0], color_heightmap_2x_r.shape[1], 1)
        color_heightmap_2x_g = np.pad(color_heightmap_2x[:, :, 1], padding_width, 'constant', constant_values=0)
        color_heightmap_2x_g.shape = (color_heightmap_2x_g.shape[0], color_heightmap_2x_g.shape[1], 1)
        color_heightmap_2x_b = np.pad(color_heightmap_2x[:, :, 2], padding_width, 'constant', constant_values=0)
        color_heightmap_2x_b.shape = (color_heightmap_2x_b.shape[0], color_heightmap_2x_b.shape[1], 1)
        color_heightmap_2x = np.concatenate((color_heightmap_2x_r, color_heightmap_2x_g, color_heightmap_2x_b), axis=2)
        depth_heightmap_2x = np.pad(depth_heightmap_2x, padding_width, 'constant', constant_values=0)
        
        # Pre-process color image (scale and normalize)
        image_mean = [0.485, 0.456, 0.406]
        image_std = [0.229, 0.224, 0.225]
        input_color_image = color_heightmap_2x.astype(float) / 255
        for c in range(3):
            input_color_image[:, :, c] = (input_color_image[:, :, c] - image_mean[c]) / image_std[c]
        
        # Pre-process depth image (normalize)
        image_mean = 0.01
        image_std = 0.03
        depth_heightmap_2x.shape = (depth_heightmap_2x.shape[0], depth_heightmap_2x.shape[1], 1)
        input_depth_image = (depth_heightmap_2x - image_mean) / image_std
        
        # Construct minibatch of size 1 (b,c,h,w)
        input_color_image.shape = (input_color_image.shape[0], input_color_image.shape[1], input_color_image.shape[2], 1)
        input_depth_image.shape = (input_depth_image.shape[0], input_depth_image.shape[1], input_depth_image.shape[2], 1)
        input_color_data = torch.from_numpy(input_color_image.astype(np.float32)).permute(3, 2, 0, 1)
        input_depth_data = torch.from_numpy(input_depth_image.astype(np.float32)).permute(3, 2, 0, 1)
        
        # Pass input data through TARGET model (not main model)
        # 일시적으로 train 모드로 전환 (BatchNorm이 eval 모드에서 nan 발생 방지)
        # eval 모드에서는 running statistics를 사용하는데, 초기 상태에서 제대로 설정되지 않을 수 있음
        was_training = self.target_model.training
        self.target_model.train()  # BatchNorm이 현재 입력의 통계 사용
        
        with torch.no_grad():
            output_prob, state_feat = self.target_model.forward(input_color_data, input_depth_data, is_volatile=True)
        
        # 원래 모드로 복원
        if not was_training:
            self.target_model.eval()
        
        if self.method == 'reinforcement':
            # Return Q values (and remove extra padding)
            for rotate_idx in range(len(output_prob)):
                if rotate_idx == 0:
                    grasp_predictions = output_prob[rotate_idx][0].cpu().data.numpy()[:, 0,
                                        int(padding_width / 2):int(color_heightmap_2x.shape[0] / 2 - padding_width / 2),
                                        int(padding_width / 2):int(color_heightmap_2x.shape[0] / 2 - padding_width / 2)]
                else:
                    grasp_predictions = np.concatenate((grasp_predictions,
                                        output_prob[rotate_idx][0].cpu().data.numpy()[:, 0,
                                        int(padding_width / 2):int(color_heightmap_2x.shape[0] / 2 - padding_width / 2),
                                        int(padding_width / 2):int(color_heightmap_2x.shape[0] / 2 - padding_width / 2)]), axis=0)
        
        return grasp_predictions, state_feat

    def update_target_network(self, color_heightmap=None, depth_heightmap=None):
        """
        Main network의 가중치를 Target network에 복사 (Hard Update)
        복사 후 실제 데이터로 출력 일치성 검증 수행
        
        Double DQN에서 주기적으로 호출되어 Target network를 업데이트.
        일반적으로 100~1000 iteration 간격으로 호출.
        
        Args:
            color_heightmap: 검증용 컬러 heightmap (선택, None이면 검증 생략)
            depth_heightmap: 검증용 깊이 heightmap (선택, None이면 검증 생략)
        """
        if self.double_dqn and self.target_model is not None:
            self.target_model.load_state_dict(self.model.state_dict())
            print(f'[Double DQN] Target network updated (iteration {self.iteration})')
            
            # 실제 데이터가 주어지면 출력 비교 검증 수행
            if color_heightmap is not None and depth_heightmap is not None:
                self._verify_target_network_output(color_heightmap, depth_heightmap)
        else:
            print('[Double DQN] Warning: Target network update called but Double DQN not enabled')

    def get_label_value(self, primitive_action, grasp_success, change_detected, prev_grasp_predictions, next_color_heightmap, next_depth_heightmap):

        if self.method == 'reinforcement':

            # Compute current reward
            # grasp_success: 1=성공, 0=일반실패, -1=바닥선택
            current_reward = 0
            if primitive_action == 'grasp':
                if grasp_success == 1:      # 성공: +1.0
                    current_reward = 1.0
                elif grasp_success == -1:   # 바닥 선택: -0.5 (더 나쁨)
                    current_reward = -0.5
                elif grasp_success == 0:    # 일반 실패: -0.25
                    current_reward = -0.25
            # Compute future reward
            # grasp_success: 1=성공, 0=일반실패, -1=바닥선택
            # 바닥 선택(-1)은 무조건 future_reward = 0 (bootstrapping 방지)
            # 일반 실패(0)이고 변화 없으면 future_reward = 0
            if grasp_success == -1:
                # 바닥 선택: 무조건 future_reward = 0 (양수 Q값 부여 방지)
                future_reward = 0
            elif not change_detected and grasp_success != 1:
                # 일반 실패이고 변화 없음: future_reward = 0
                future_reward = 0
            else:
                # 성공 또는 변화 있음: 미래 Q값 계산
                if self.double_dqn:
                    # Double DQN: Main network로 best action 선택, Target network로 Q값 평가
                    # Q_target = r + γ * Q_target(s', argmax_a' Q_main(s', a'))
                    next_grasp_predictions, _ = self.forward(next_color_heightmap, next_depth_heightmap, is_volatile=True)
                    
                    # nan 체크: Main network 출력이 nan이면 표준 방식으로 fallback
                    if np.isnan(next_grasp_predictions).any():
                        print('[Double DQN] Warning: Main network output contains nan, using future_reward=0')
                        future_reward = 0
                    else:
                        best_action_idx = np.unravel_index(np.argmax(next_grasp_predictions), next_grasp_predictions.shape)
                        
                        # Target network로 해당 action의 Q값 평가
                        target_predictions, _ = self.forward_target(next_color_heightmap, next_depth_heightmap)
                        future_reward = target_predictions[best_action_idx]
                        
                        # nan 체크: Target network 출력이 nan이면 0으로 처리
                        if np.isnan(future_reward):
                            print(f'[Double DQN] Warning: Target Q is nan at {best_action_idx}, using future_reward=0')
                            future_reward = 0
                        else:
                            print(f'[Double DQN] Best action idx: {best_action_idx}, Target Q: {future_reward:.4f}')
                else:
                    # 표준 DQN: Q_target = r + γ * max_a' Q(s', a')
                    next_grasp_predictions, next_state_feat = self.forward(next_color_heightmap, next_depth_heightmap, is_volatile=True)
                    future_reward = np.max(next_grasp_predictions)
                    
                    # nan 체크
                    if np.isnan(future_reward):
                        print('[DQN] Warning: future_reward is nan, using 0')
                        future_reward = 0
            print('Current reward: %f' % (current_reward))
            print('Future reward: %f' % (future_reward))
            if primitive_action == 'grasp' and not self.grasp_rewards:
                expected_reward = self.future_reward_discount * future_reward
                print('Expected reward: %f + %f x %f = %f' % (0.0, self.future_reward_discount, future_reward, expected_reward))
            else:
                expected_reward = current_reward + self.future_reward_discount * future_reward
                print('Expected reward: %f + %f x %f = %f' % (current_reward, self.future_reward_discount, future_reward, expected_reward))
            return expected_reward, current_reward

    # Compute labels and backpropagate
    def backprop(self, color_heightmap, depth_heightmap, primitive_action, best_pix_ind, label_value):

        if self.method == 'reinforcement':

            # Do forward pass first to get actual output size
            if primitive_action == 'grasp':
                grasp_predictions, state_feat = self.forward(color_heightmap, depth_heightmap, is_volatile=False, specific_rotation=best_pix_ind[0])
                # 동적으로 output 크기 가져오기 (num_rotations=1일 때 대응)
                output_shape = self.model.output_prob[0][0].shape
                output_size = (int(output_shape[-2]), int(output_shape[-1]))  # (H, W)를 정수 튜플로 변환
                print(f'[DEBUG] output_shape: {output_shape}, output_size: {output_size}')
                print(f'[DEBUG] input (heightmap) shape: {color_heightmap.shape}')
            else:
                # place action일 경우 (현재 프로젝트에서는 사용 안 함)
                output_size = (320, 320)
            
            # Compute labels with dynamic size
            label = np.zeros((1, output_size[0], output_size[1]))
            # Input size = output_size - 2*padding
            input_size = (color_heightmap.shape[0], color_heightmap.shape[1])
            padding = (output_size[0] - input_size[0]) // 2
            
            print(f'[DEBUG] label shape: {label.shape}, input_size: {input_size}, padding: {padding}')
            
            action_area = np.zeros(input_size)
            action_area[best_pix_ind[1]][best_pix_ind[2]] = 1
            tmp_label = np.zeros(input_size)
            tmp_label[action_area > 0] = label_value
            label[0, padding:(output_size[0]-padding), padding:(output_size[1]-padding)] = tmp_label
            
            # Compute label mask
            label_weights = np.zeros(label.shape)
            tmp_label_weights = np.zeros(input_size)
            tmp_label_weights[action_area > 0] = 1
            label_weights[0, padding:(output_size[0]-padding), padding:(output_size[1]-padding)] = tmp_label_weights
            
            # Compute loss and backward pass
            self.optimizer.zero_grad()
            loss_value = 0
            if primitive_action == 'grasp':
                
                if self.use_cuda:
                    grasp_loss = self.criterion(self.model.output_prob[0][0].view(1, output_size[0], output_size[1]), Variable(torch.from_numpy(label).float().cuda())) * Variable(torch.from_numpy(label_weights).float().cuda(),requires_grad=False)
                else:
                    grasp_loss = self.criterion(self.model.output_prob[0][0].view(1, output_size[0], output_size[1]), Variable(torch.from_numpy(label).float())) * Variable(torch.from_numpy(label_weights).float(),requires_grad=False)
                grasp_loss = grasp_loss.sum()
                grasp_loss.backward()
                loss_value = grasp_loss.cpu().data.numpy()

                opposite_rotate_idx = int((best_pix_ind[0] + self.model.num_rotations/2) % self.model.num_rotations)

                grasp_predictions, state_feat = self.forward(color_heightmap, depth_heightmap, is_volatile=False, specific_rotation=opposite_rotate_idx)

                if self.use_cuda:
                    grasp_loss = self.criterion(self.model.output_prob[0][0].view(1, output_size[0], output_size[1]), Variable(torch.from_numpy(label).float().cuda())) * Variable(torch.from_numpy(label_weights).float().cuda(),requires_grad=False)
                else:
                    grasp_loss = self.criterion(self.model.output_prob[0][0].view(1, output_size[0], output_size[1]), Variable(torch.from_numpy(label).float())) * Variable(torch.from_numpy(label_weights).float(),requires_grad=False)

                grasp_loss = grasp_loss.sum()
                grasp_loss.backward()
                loss_value = grasp_loss.cpu().data.numpy()
                
                loss_value = loss_value/2

            print('Training loss: %f' % (loss_value))
            self.optimizer.step()


    def get_prediction_vis(self, predictions, color_heightmap, best_pix_ind, scale_factor=4):

        canvas = None
        num_rotations = predictions.shape[0]
        
        # IRB360 최적화: num_rotations=1일 때 단순 시각화
        if num_rotations == 1:
            prediction_vis = predictions[0,:,:].copy()
            prediction_vis = prediction_vis/scale_factor
            prediction_vis = np.clip(prediction_vis, 0, 1)
            prediction_vis.shape = (predictions.shape[1], predictions.shape[2])
            prediction_vis = cv2.applyColorMap((prediction_vis*255).astype(np.uint8), cv2.COLORMAP_JET)
            prediction_vis = cv2.circle(prediction_vis, (int(best_pix_ind[2]), int(best_pix_ind[1])), 7, (221,211,238), 2)
            background_image = color_heightmap
            prediction_vis = (0.5*cv2.cvtColor(background_image, cv2.COLOR_RGB2BGR) + 0.5*prediction_vis).astype(np.uint8)
            return prediction_vis
        
        # 기존 코드: num_rotations > 1일 때 그리드 형식
        for canvas_row in range(int(num_rotations/4)):
            tmp_row_canvas = None
            for canvas_col in range(4):
                rotate_idx = canvas_row*4+canvas_col
                prediction_vis = predictions[rotate_idx,:,:].copy()
                prediction_vis = prediction_vis/scale_factor
                prediction_vis = np.clip(prediction_vis, 0, 1)
                prediction_vis.shape = (predictions.shape[1], predictions.shape[2])
                prediction_vis = cv2.applyColorMap((prediction_vis*255).astype(np.uint8), cv2.COLORMAP_JET)
                if rotate_idx == best_pix_ind[0]:
                    prediction_vis = cv2.circle(prediction_vis, (int(best_pix_ind[2]), int(best_pix_ind[1])), 7, (221,211,238), 2)
                prediction_vis = ndimage.rotate(prediction_vis, rotate_idx*(360.0/num_rotations), reshape=False, order=0)
                background_image = ndimage.rotate(color_heightmap, rotate_idx*(360.0/num_rotations), reshape=False, order=0)
                prediction_vis = (0.5*cv2.cvtColor(background_image, cv2.COLOR_RGB2BGR) + 0.5*prediction_vis).astype(np.uint8)
                if tmp_row_canvas is None:
                    tmp_row_canvas = prediction_vis
                else:
                    tmp_row_canvas = np.concatenate((tmp_row_canvas,prediction_vis), axis=1)
            if canvas is None:
                canvas = tmp_row_canvas
            else:
                canvas = np.concatenate((canvas,tmp_row_canvas), axis=0)

        return canvas
    
    
    def get_grasp_vis(self, grasp_predictions, color_heightmap, best_pix_ind):
        grasp_canvas = color_heightmap
        x = 0
        while x < grasp_predictions.shape[2]:
            y = 0
            while y < grasp_predictions.shape[1]:
                angle_idx = np.argmax(grasp_predictions[:, y, x])
                angel = np.deg2rad(angle_idx*(360.0/self.model.num_rotations))
                quality = np.max(grasp_predictions[:, y, x])
                
                color = (0, 0, (quality*255).astype(np.uint8))
                cv2.circle(grasp_canvas, (int(best_pix_ind[2]), int(best_pix_ind[1])), 7, (0,0,255), 2)
                y+=10
            x+=10

        plt.figure()
        plt.imshow(grasp_canvas)
        plt.show()
        return grasp_canvas

    def get_best_grasp_vis(self, best_pix_ind, color_heightmap):
        grasp_canvas = color_heightmap
        angle_idx = best_pix_ind[0]
        angel = np.deg2rad(angle_idx*(360.0/self.model.num_rotations))
        cv2.circle(grasp_canvas, (int(best_pix_ind[2]), int(best_pix_ind[1])), 4, (0,0,255), 1)

        return grasp_canvas

