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
        self.IDX_RESOURCE = 2
        self.IDX_PROJECT_START = 3
        self.IDX_PROJECT_END = 5
        
        # 액션 공간 정의 (이동, 리소스 픽업/드롭, 태스크 실행)
        self.action_space = gym.spaces.Discrete(7)
        
        # 관찰 공간 정의
        # [에이전트 위치(x,y), 보유 리소스 상태, 프로젝트별 태스크 상태]
        self.observation_space = gym.spaces.MultiDiscrete([
            3, 3,  # 에이전트 위치 (3x3 그리드)
            2,     # 리소스 보유 상태
            3, 3   # 프로젝트 태스크 상태 (3단계: 준비안됨/준비됨/완료)
        ])
        
        # 위치 정보 설정
        self.agent_start_pos = [0, 0]
        self.project_positions = [[2, 1], [2, 2]]  # 2개의 프로젝트 위치
        self.resource_pos = [1, 1]    # 중앙에 리소스 위치
        
        # 일일 행동 제한
        self.max_actions = 20
        self.action_count = 0
        
        # 프로젝트별 리소스 현황 추적
        self.project_resources = [0, 0]  # 각 프로젝트의 리소스 보유 현황
        
        self.reset()
        
    def reset(self, seed=None, options=None):
        self.agent_pos = self.agent_start_pos.copy()
        self.action_count = 0
        self.project_resources = [0, 0]
        
        self.state = np.array([
            self.agent_pos[0], self.agent_pos[1],  # 에이전트 위치
            0,  # 리소스 보유 상태
            0, 0  # 프로젝트 태스크 상태
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
        if self.agent_pos != self.resource_pos:
            return -5
        if self.state[self.IDX_RESOURCE] == 1:
            return -5
            
        self.state[self.IDX_RESOURCE] = 1
        return 10

    def _dropoff_resource(self):
        if self.state[self.IDX_RESOURCE] == 0:
            return -5
            
        current_project = None
        for i, pos in enumerate(self.project_positions):
            if self.agent_pos == pos:
                current_project = i
                break
                
        if current_project is None:
            return -5
            
        # 이미 완료된 프로젝트인 경우 페널티
        if self.state[self.IDX_PROJECT_START + current_project] == 2:
            return -10
            
        self.project_resources[current_project] = 1
        self.state[self.IDX_RESOURCE] = 0
        
        if self.project_resources[current_project] == 1:
            self.state[self.IDX_PROJECT_START + current_project] = 1
            return 10
            
        return -5

    def _execute_task(self):
        current_project = None
        for i, pos in enumerate(self.project_positions):
            if self.agent_pos == pos:
                current_project = i
                break
                
        if current_project is None:
            return -5
            
        # 이미 완료된 프로젝트인 경우 페널티
        if self.state[self.IDX_PROJECT_START + current_project] == 2:
            return -10
            
        if self.state[self.IDX_PROJECT_START + current_project] == 1:
            self.state[self.IDX_PROJECT_START + current_project] = 2
            return 20
            
        return -5

    def _all_projects_complete(self):
        return all(self.state[self.IDX_PROJECT_START:self.IDX_PROJECT_END] == 2)

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
    model.learn(total_timesteps=50000)
    model.save("simple_construction_dqn") 