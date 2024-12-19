from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
import numpy as np
import gymnasium as gym

class SimpleConstructionEnv(gym.Env):
    def __init__(self):
        super().__init__()
        
        # 상태 인덱스 상수 정의
        self.IDX_POS_X = 0
        self.IDX_POS_Y = 1
        self.IDX_RESOURCE_1 = 2
        self.IDX_RESOURCE_2 = 3
        self.IDX_DAY_TYPE = 4
        self.IDX_PROJECT_START = 5
        self.IDX_PROJECT_END = 13  # 프로젝트1의 2개 태스크 + 프로젝트2의 2개 태스크
        
        # 액션 공간 정의 (이동, 리소스 픽업/드롭, 태스크 실행)
        self.action_space = gym.spaces.Discrete(7)
        
        # 관찰 공간 정의 수정
        self.observation_space = gym.spaces.MultiDiscrete([
            5, 5,  # 에이전트 위치 (5x5 그리드)
            2, 2,  # 리소스1, 리소스2 보유 상태
            2,     # 현재 날짜 타입 (짝수/홀수)
            3, 3,  # 프로젝트1-태스크1 리소스1, 리소스2 상태
            3, 3,  # 프로젝트1-태스크2 리소스1, 리소스2 상태
            3, 3,  # 프로젝트2-태스크1 리소스1, 리소스2 상태
            3, 3   # 프로젝트2-태스크2 리소스1, 리소스2 상태
        ])
        
        # 위치 정보 설정
        self.agent_start_pos = [0, 0]
        self.project_positions = [[2, 0], [4, 4]]  # 2개의 프로젝트 위치
        self.resource1_pos = [4, 1]    # 리소스1 위치
        self.resource2_pos = [0, 4]    # 리소스2 위치
        
        # 일일 행동 제한 수정
        self.max_actions = 20
        self.action_count = 0
        self.current_day = 0  # 현재 날짜 추가
        self.available_resources = [True, False]  # 리소스 사용 가능 상태
        
        # 프로젝트별 리소스 현황 추적
        self.project_resources = [0, 0]  # 각 프로젝트의 리소스 보유 현황
        
        self.total_action_count = 0  # 총 액션 카운트 추가
        self.max_total_actions = 150  # 최대 총 액션 수 제한 추가
        self.reset()
        
    def reset(self, seed=None, options=None):
        self.agent_pos = self.agent_start_pos.copy()
        self.action_count = 0
        self.current_day = 0
        self.available_resources = [True, False]  # 첫날은 리소스1만 사용 가능
        self.total_action_count = 0  # 총 액션 카운트 초기화
        
        self.state = np.array([
            self.agent_pos[0], self.agent_pos[1],  # 에이전트 위치
            0, 0,  # 리소스1, 리소스2 보유 상태
            0,     # 현재 날짜 타입 (0: 짝수날, 1: 홀수날)
            0, 0,  # 프로젝트1-태스크1 리소스1, 리소스2 상태
            0, 0,  # 프로젝트1-태스크2 리소스1, 리소스2 상태
            0, 0,  # 프로젝트2-태스크1 리소스1, 리소스2 상태
            0, 0   # 프로젝트2-태스크2 리소스1, 리소스2 상태
        ])
        return self.state, {}

    def step(self, action):
        reward = -1
        done = False
        
        # 총 액션 카운트 제한 체크 추가
        if self.total_action_count >= self.max_total_actions:
            done = True
            reward = -300  # 제한 초과로 인한 페널티
            return self.state, reward, done, False, {}
            
        # 액션 처리
        if action < 4:  # 이동
            self.action_count += 1
            self.total_action_count += 1
            reward = self._move(action)
        elif action == 4:  # 리소스 픽업
            self.action_count += 5
            self.total_action_count += 5
            reward = self._pickup_resource()
        elif action == 5:  # 리소스 드롭
            self.action_count += 5
            self.total_action_count += 5
            reward = self._dropoff_resource()
        elif action == 6:  # 태스크 실행
            self.action_count += 10
            self.total_action_count += 10
            reward = self._execute_task()
            
        # 일일 행동 제한 초과 시 다음날로 전환
        if self.action_count >= self.max_actions:
            self.action_count = 0
            self.current_day += 1
            # 리소스 사용 가능 상태 전환
            self.available_resources = [not r for r in self.available_resources]
            # 날짜 타입 업데이트 (짝수날: 0, 홀수날: 1)
            self.state[self.IDX_DAY_TYPE] = self.current_day % 2
            
        # 모든 프로젝트가 완료되면 종료
        if self._all_projects_complete():
            done = True
            reward += 1000
            
        return self.state, reward, done, False, {}

    def _move(self, action):
        moves = {0: [-1, 0], 1: [0, 1], 2: [1, 0], 3: [0, -1]}
        new_pos = [
            self.agent_pos[0] + moves[action][0],
            self.agent_pos[1] + moves[action][1]
        ]
        
        if 0 <= new_pos[0] < 5 and 0 <= new_pos[1] < 5:  # 5x5 그리드 경계 체크
            self.agent_pos = new_pos
            self.state[self.IDX_POS_X:self.IDX_POS_Y+1] = self.agent_pos
            return -1
        return -5

    def _pickup_resource(self):
        # 리소스1 픽업
        if self.agent_pos == self.resource1_pos:
            if self.state[self.IDX_RESOURCE_1] == 0 and self.available_resources[0]:
                self.state[self.IDX_RESOURCE_1] = 1
                return 10
            return -5
            
        # 리소스2 픽업
        if self.agent_pos == self.resource2_pos:
            if self.state[self.IDX_RESOURCE_2] == 0 and self.available_resources[1]:
                self.state[self.IDX_RESOURCE_2] = 1
                return 10
            return -5
            
        return -5

    def _dropoff_resource(self):
        # 리소스를 보유하고 있지 않은 경우
        if self.state[self.IDX_RESOURCE_1] == 0 and self.state[self.IDX_RESOURCE_2] == 0:
            return -5
            
        # 프로젝트 위치 확인
        current_project = None
        for i, pos in enumerate(self.project_positions):
            if self.agent_pos == pos:
                current_project = i
                break
                
        # 프로젝트 위치가 아닌 경우 리소스 드롭
        if current_project is None:
            self.state[self.IDX_RESOURCE_1] = 0
            self.state[self.IDX_RESOURCE_2] = 0
            return -5
            
        reward = -5
        
        # 프로젝트1 처리
        if current_project == 0:
            # 리소스1을 드롭하는 경우
            if self.state[self.IDX_RESOURCE_1] == 1:
                # 태스크1과 태스크2 모두에 리소스1 적용
                if self.state[self.IDX_PROJECT_START] == 0:  # 태스크1 리소스1
                    self.state[self.IDX_PROJECT_START] = 1
                    self.state[self.IDX_RESOURCE_1] = 0
                    reward = 10
                if self.state[self.IDX_PROJECT_START + 2] == 0:  # 태스크2 리소스1
                    self.state[self.IDX_PROJECT_START + 2] = 1
                    self.state[self.IDX_RESOURCE_1] = 0
                    reward = 10
            
            # 리소스2를 드롭하는 경우 (태스크2에만 필요)
            if self.state[self.IDX_RESOURCE_2] == 1:
                if self.state[self.IDX_PROJECT_START + 3] == 0:  # 태스크2 리소스2
                    self.state[self.IDX_PROJECT_START + 3] = 1
                    self.state[self.IDX_RESOURCE_2] = 0
                    reward = 10
                
        # 프로젝트2 처리
        else:
            base_idx = self.IDX_PROJECT_START + 4
            # 리소스1을 드롭하는 경우
            if self.state[self.IDX_RESOURCE_1] == 1:
                # 태스크1과 태스크2 모두에 리소스1 적용
                if self.state[base_idx] == 0:  # 태스크1 리소스1
                    self.state[base_idx] = 1
                    self.state[self.IDX_RESOURCE_1] = 0
                    reward = 10
                if self.state[base_idx + 2] == 0:  # 태스크2 리소스1
                    self.state[base_idx + 2] = 1
                    self.state[self.IDX_RESOURCE_1] = 0
                    reward = 10
            
            # 리소스2를 드롭하는 경우 (태스크2에만 필요)
            if self.state[self.IDX_RESOURCE_2] == 1:
                if self.state[base_idx + 3] == 0:  # 태스크2 리소스2
                    self.state[base_idx + 3] = 1
                    self.state[self.IDX_RESOURCE_2] = 0
                    reward = 10
                
        return reward

    def _execute_single_task(self, start_idx, required_resources):
        """단일 태스크 실행을 처리하는 헬퍼 메서드
        
        Args:
            start_idx: 태스크의 시작 인덱스
            required_resources: 필요한 리소스 수 (1 또는 2)
        
        Returns:
            bool: 태스크 실행 성공 여부
        """
        # 모든 필요한 리소스가 준비되었는지 확인
        resources_ready = True
        for i in range(required_resources):
            if self.state[start_idx + i] != 1:
                resources_ready = False
                break
                
        if resources_ready:
            # 태스크 완료 상태로 변경
            for i in range(required_resources):
                self.state[start_idx + i] = 2
            return True
        return False

    def _execute_task(self):
        current_project = None
        for i, pos in enumerate(self.project_positions):
            if self.agent_pos == pos:
                current_project = i
                break
                
        if current_project is None:
            return -5
            
        # 프로젝트1 처리
        if current_project == 0:
            # 태스크1 실행 (리소스1만 필요)
            if self._execute_single_task(self.IDX_PROJECT_START, 1):
                return 100
            # 태스크2 실행 (리소스1과 리소스2 필요)
            if self._execute_single_task(self.IDX_PROJECT_START + 2, 2):
                return 100
                
        # 프로젝트2 처리
        else:
            base_idx = self.IDX_PROJECT_START + 4
            # 태스크1 실행 (리소스1만 필요)
            if self._execute_single_task(base_idx, 1):
                return 100
            # 태스크2 실행 (리소스1과 리소스2 필요)
            if self._execute_single_task(base_idx + 2, 2):
                return 100
                
        return -5

    def _all_projects_complete(self):
        # 프로젝트1의 두 태스크 모두 완료 확인
        proj1_task1_complete = self.state[self.IDX_PROJECT_START] == 2
        proj1_task2_complete = (self.state[self.IDX_PROJECT_START + 2] == 2 and 
                              self.state[self.IDX_PROJECT_START + 3] == 2)
        
        # 프로젝트2의 두 태스크 모두 완료 확인
        base_idx = self.IDX_PROJECT_START + 4
        proj2_task1_complete = self.state[base_idx] == 2
        proj2_task2_complete = (self.state[base_idx + 2] == 2 and 
                               self.state[base_idx + 3] == 2)
        
        return (proj1_task1_complete and proj1_task2_complete and 
                proj2_task1_complete and proj2_task2_complete)

if __name__ == "__main__":
    env = SimpleConstructionEnv()
    check_env(env)

    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=5e-4,
        buffer_size=10000,
        learning_starts=5000,
        batch_size=64,
        gamma=0.8,
        train_freq=1,
        gradient_steps=1,
        exploration_fraction=0.3,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.1,
        target_update_interval=1000,
        verbose=1,
        tensorboard_log="./crm_rl_iot/DQN/log/dqn_tensorboard/simple_construction_dqn"
    )
    model.learn(total_timesteps=120000)
    model.save("simple_construction_dqn") 