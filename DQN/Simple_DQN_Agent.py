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
        self.IDX_PROJECT_START = 4
        self.IDX_PROJECT_END = 8  # 프로젝트 상태 4개로 증가
        
        # 액션 공간 정의 (이동, 리소스 픽업/드롭, 태스크 실행)
        self.action_space = gym.spaces.Discrete(7)
        
        # 관찰 공간 정의 수정
        self.observation_space = gym.spaces.MultiDiscrete([
            3, 3,  # 에이전트 위치 (3x3 그리드)
            2, 2,  # 리소스1, 리소스2 보유 상태
            3, 3,  # 프로젝트1 리소스1 상태, 프로젝트2 리소스1 상태 (0: 미시작, 1: 진행중, 2: 완료)
            3, 3   # 프로젝트1 리소스2 상태, 프로젝트2 리소스2 상태 (0: 미시작, 1: 진행중, 2: 완료)
        ])
        
        # 위치 정보 설정
        self.agent_start_pos = [0, 0]
        self.project_positions = [[2, 1], [2, 2]]  # 2개의 프로젝트 위치
        self.resource1_pos = [1, 0]    # 리소스1 위치
        self.resource2_pos = [1, 2]    # 리소스2 위치
        
        # 일일 행동 제한
        self.max_actions = 20
        self.action_count = 0
        
        # 프로젝트별 리소스 현황 추적
        self.project_resources = [0, 0]  # 각 프로젝트의 리소스 보유 현황
        
        self.reset()
        
    def reset(self, seed=None, options=None):
        self.agent_pos = self.agent_start_pos.copy()
        self.action_count = 0
        
        self.state = np.array([
            self.agent_pos[0], self.agent_pos[1],  # 에이전트 위치
            0, 0,  # 리소스1, 리소스2 보유 상태
            0, 0,  # 프로젝트1 리소스1, 프로젝트2 리소스1
            0, 0   # 프로젝트1 리소스2, 프로젝트2 리소스2
        ])
        return self.state, {}

    def step(self, action):
        reward = -1  # 기본 리워드
        done = False
        
        # 액션 제한 확인
        if self.action_count >= self.max_actions:
            done = True
            return self.state, -10, done, False, {}
            
        # 액션 처리
        if action < 4:  # 이동
            reward = self._move(action)
        elif action == 4:  # 리소스 픽업
            reward = self._pickup_resource()
        elif action == 5:  # 리소스 드롭
            reward = self._dropoff_resource()
        elif action == 6:  # 태스크 실행
            reward = self._execute_task()
            
        self.action_count += 1
        
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
        
        if 0 <= new_pos[0] < 3 and 0 <= new_pos[1] < 3:
            self.agent_pos = new_pos
            self.state[self.IDX_POS_X:self.IDX_POS_Y+1] = self.agent_pos
            return -1
        return -5

    def _pickup_resource(self):
        # 리소스1 픽업
        if self.agent_pos == self.resource1_pos:
            if self.state[self.IDX_RESOURCE_1] == 0:
                self.state[self.IDX_RESOURCE_1] = 1
                return 10
            return -5
            
        # 리소스2 픽업
        if self.agent_pos == self.resource2_pos:
            if self.state[self.IDX_RESOURCE_2] == 0:
                self.state[self.IDX_RESOURCE_2] = 1
                return 10
            return -5
            
        return -5

    def _dropoff_resource(self):
        if self.state[self.IDX_RESOURCE_1] == 0 and self.state[self.IDX_RESOURCE_2] == 0:
            return -5
            
        current_project = None
        for i, pos in enumerate(self.project_positions):
            if self.agent_pos == pos:
                current_project = i
                break
                
        # 프로젝트 위치가 아닌 곳에서 드롭하면 리소스만 잃고 페널티
        if current_project is None:
            self.state[self.IDX_RESOURCE_1] = 0
            self.state[self.IDX_RESOURCE_2] = 0
            return -5
            
        # 리소스1 드롭
        if self.state[self.IDX_RESOURCE_1] == 1:
            self.state[self.IDX_RESOURCE_1] = 0  # 항상 리소스 상태 초기화
            if self.state[self.IDX_PROJECT_START + current_project] == 0:  # 리소스1 상태
                self.state[self.IDX_PROJECT_START + current_project] = 1
                return 10
            return -5
            
        # 리소스2 드롭
        if self.state[self.IDX_RESOURCE_2] == 1:
            self.state[self.IDX_RESOURCE_2] = 0  # 항상 리소스 상태 초기화
            if self.state[self.IDX_PROJECT_START + current_project + 2] == 0:  # 리소스2 상태
                self.state[self.IDX_PROJECT_START + current_project + 2] = 1
                return 10
            return -5
            
        return -5

    def _execute_task(self):
        current_project = None
        for i, pos in enumerate(self.project_positions):
            if self.agent_pos == pos:
                current_project = i
                break
                
        if current_project is None:
            return -5
            
        # 프로젝트1은 리소스1만 필요
        if current_project == 0:
            if self.state[self.IDX_PROJECT_START + current_project] == 1:
                self.state[self.IDX_PROJECT_START + current_project] = 2
                return 20
                
        # 프로젝트2는 리소스1과 리소스2 모두 필요
        elif current_project == 1:
            if (self.state[self.IDX_PROJECT_START + current_project] == 1 and 
                self.state[self.IDX_PROJECT_START + current_project + 2] == 1):
                self.state[self.IDX_PROJECT_START + current_project] = 2
                self.state[self.IDX_PROJECT_START + current_project + 2] = 2
                return 20
                
        return -5

    def _all_projects_complete(self):
        # 프로젝트1은 리소스1만 확인
        proj1_complete = self.state[self.IDX_PROJECT_START] == 2
        # 프로젝트2는 리소스1과 리소스2 모두 확인
        proj2_complete = (self.state[self.IDX_PROJECT_START + 1] == 2 and 
                         self.state[self.IDX_PROJECT_START + 3] == 2)
        return proj1_complete and proj2_complete

if __name__ == "__main__":
    env = SimpleConstructionEnv()
    check_env(env)

    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=5e-4,
        buffer_size=10000,
        learning_starts=1000,
        batch_size=64,
        gamma=0.8,
        train_freq=1,
        gradient_steps=1,
        exploration_fraction=0.3,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.1,
        target_update_interval=1000,
        verbose=1,
        tensorboard_log="./crm_rl_iot/DQN/log/dqn_tensorboard/"
    )
    model.learn(total_timesteps=70000)
    model.save("simple_construction_dqn") 