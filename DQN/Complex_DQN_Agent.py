from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
import random
import numpy as np
import gymnasium as gym
import json

class ComplexConstructionEnv(gym.Env):
    def __init__(self):
        super().__init__()
        
        # 액션 공간 정의 (이동, 리소스 픽업/드롭, 태스크 실행)
        self.action_space = gym.spaces.Discrete(7)
        
        # 일일 행동 제한 수정
        self.max_actions = 60 # 하루 액션 수
        self.action_count = 0
        self.current_day = 0  # 현재 날짜 추가
        self.available_resources = []  # 리소스 사용 가능 상태
        
        self.total_action_count = 0  # 총 액션 카운트 추가
        self.max_total_actions = self.max_actions * 30  # 총 액션 수: 총 30일 작업

        self._map_loader('data/construction_data.json')
        self.reset()
    
    def _map_loader(self, data_path):
        # 위치 정보 설정
        with open(data_path, encoding='utf-8-sig') as f:
            
            # 위치 정보 설정, 프로젝트 정보 설정
            data = json.load(f)
            self.meta_data = data['meta_data']
            project_count = len(self.meta_data['project_list'])
            self.resource_count = len(self.meta_data['resource_list'])

            self.field_width = self.meta_data['field_width']
            self.field_height = self.meta_data['field_height']
            self.agent_start_pos = self.meta_data['start_location']

            self.project_positions = [] # 프로젝트 위치 [x,y]
            self.projects_data = data['projects']
            for project in self.projects_data:
                self.project_positions.append(project['location'])

            self.resource_positions = [] # 리소스 위치 [x,y]
            resources = data['resources']
            for resource in resources:
                self.resource_positions.append(resource['location'])

            self.obstacle_positions = [] # 방해물 위치 [x,y]
            obstacles = data['obstacles']
            for obstacle in obstacles:
                self.obstacle_positions.append(obstacle['location'])

            # 상태 인덱스 상수 정의
            self.IDX_POS_X  = 0
            self.IDX_POS_Y = 1
            self.IDX_RESOURCE = [(2+i) for i in range(self.resource_count)]
            self.IDX_DAY_TYPE = self.IDX_RESOURCE[self.resource_count-1]+1
            self.IDX_PROJECT_START = [self.IDX_DAY_TYPE+1 for i in range(project_count)]# 초기화

            # 프로젝트 정보 정의
            self.task_resource = [] # 프로젝트 테스크 별 리소스 종류
            self.task_resource_sum = 0 # 총 테스크 개수
            for i, project in enumerate(self.projects_data):
                list1 = []
                for task in project['task']:
                    list2 = []
                    self.task_resource_sum  += len(task['resource'])
                    for resource in task['resource']:
                        resource_name, resource_num = resource.split('_')
                        resource_num = int(resource_num)-1 # 0부터 시작
                        list2.append(resource_num)
                    list1.append(list2)
                
                self.task_resource.append(list1)

                if i != len(self.projects_data)-1:
                    self.IDX_PROJECT_START[i+1] += self.task_resource_sum  # 프로젝트 리소스 개수에 따라 IDX_PROJECT_START 업데이트

            # 관찰 공간 설정
            agent_location = [self.field_width, self.field_height]
            resource_status = [2 for i in range(len(self.resource_positions))]
            date_type = [2] # 현재 날짜 타입 (짝수/홀수)
            project_status = [3 for i in range(self.task_resource_sum)]  # 프로젝트-테스크별 리소스 상태
            
            self.observation_space = gym.spaces.MultiDiscrete(
                agent_location # field width, field height
                + resource_status # 리소스별 보유 여부(true, false), 리소스 순서-resource list 순서
                + date_type # 짝수/홀수
                + project_status # 프로젝트 테스크별 리소스 상태 3
            )

    def reset(self, seed=None, options=None):
        self.agent_pos = self.agent_start_pos.copy()
        self.action_count = 0
        self.current_day = 0
        self.available_resources = [1]+[random.randint(0, 1) for _ in range(self.resource_count-1)]  # 매일 다른 리소스 할당
        random.shuffle(self.available_resources)

        self.total_action_count = 0  # 총 액션 카운트 초기화
        
        self.state = np.array(
            [self.agent_pos[0], self.agent_pos[1]]  # 에이전트 위치
            + [0 for i in range(len(self.IDX_RESOURCE))]  # 리소스별 보유 상태
            + [0]     # 현재 날짜 타입 (0: 짝수날, 1: 홀수날)
            + [0 for i in range(self.task_resource_sum)]  # 프로젝트-태스크 리소스별 상태(0:리소스 없음, 1:리소스 있음, 2:테스크 완료)
        )
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
            self.available_resources = [1]+[random.randint(0, 1) for _ in range(self.resource_count-1)]  # 매일 다른 리소스 할당
            random.shuffle(self.available_resources)
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
        
        if 0 <= new_pos[0] < self.field_width and 0 <= new_pos[1] <  self.field_height:  # 5x5 그리드 경계 체크
            if self.agent_pos in self.obstacle_positions: # 방해물 위치
                return -5
            self.agent_pos = new_pos
            self.state[self.IDX_POS_X] = self.agent_pos[0]
            self.state[self.IDX_POS_Y] = self.agent_pos[1]
            return -1
        return -5

    def _pickup_resource(self):
        # 리소스 픽업
        if self.agent_pos in self.resource_positions:
            resource_num = self.resource_positions.index(self.agent_pos) # resource 종류 확인
            if self.state[self.IDX_RESOURCE[resource_num]] == 0 and self.available_resources[resource_num]:
                self.state[self.IDX_RESOURCE[resource_num]] = 1
                return 10
            return -5
        
        return -5

    def _dropoff_resource(self):
        # 리소스를 보유하고 있지 않은 경우
        if all(resource == 0 for resource in self.state[self.IDX_RESOURCE[0]:self.IDX_RESOURCE[1]+1]):
            return -5
            
        # 프로젝트 위치 확인
        current_project_idx = None
        for i, pos in enumerate(self.project_positions):
            if self.agent_pos == pos:
                current_project_idx = i
                break
                
        # 프로젝트 위치가 아닌 경우 드롭 불가
        if current_project_idx is None:
            return -5
            
        reward = -5

        # 프로젝트 위치에 요구되는 리소스 처리
        for resource_idx in range(len(self.meta_data['resource_list'])): 

            if self.state[self.IDX_RESOURCE[resource_idx]] == 1: # resource_idx 보유했을 때
                
                term = -1
                for task_resource in self.task_resource[current_project_idx]: # 프로젝트 위치에서 테스크 정보 확인
                    for idx in task_resource: #테스크별 리소스 정보 확인
                        term += 1
                        if resource_idx == idx: # 요구되는 리소스와 소유한 리소스가 같으면
                            if self.state[self.IDX_PROJECT_START[current_project_idx]+term] == 0:
                                self.state[self.IDX_PROJECT_START[current_project_idx]+term] = 1
                                self.state[self.IDX_RESOURCE[resource_idx]] = 0
                                reward = 10
                
        return reward

    def _execute_single_task(self, start_idx, resource_count):
        """단일 태스크 실행을 처리하는 헬퍼 메서드
        
        Args:
            start_idx: 태스크의 시작 인덱스
            range(resource_count): 필요한 리소스 개수
        
        Returns:
            bool: 태스크 실행 성공 여부
        """
        # 모든 필요한 리소스가 준비되었는지 확인
        resources_ready = True
        for idx in range(resource_count):
            if self.state[start_idx + idx] != 1:
                resources_ready = False
                break
                
        if resources_ready:
            # 태스크 완료 상태로 변경
            for idx in range(resource_count):
                self.state[start_idx + idx] = 2
            return True
        return False

    def _execute_task(self):
        current_project_idx = None
        for i, pos in enumerate(self.project_positions):
            if self.agent_pos == pos:
                current_project_idx = i
                break
                
        if current_project_idx is None:
            return -5
        
        # 프로젝트 위치 프로젝트 처리
        term = [0]
        for i, task_resource in enumerate(self.task_resource[current_project_idx]):
            term.append(len(task_resource))
            if self._execute_single_task(self.IDX_PROJECT_START[current_project_idx]+term[i], len(task_resource)): # 각 테스크 시작 위치와 테스크별 자원 개수
                return 100
                
        return -5

    def _all_projects_complete(self):

        project_complete_list = []
        for task_state in self.state[self.IDX_PROJECT_START[0]:]:
            project_complete_list.append(task_state)
        
        if 0 in project_complete_list or 1 in project_complete_list:
            return False
        else:
            return True

if __name__ == "__main__":
    env = ComplexConstructionEnv()
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
        tensorboard_log="./crm_rl_iot/DQN/log/dqn_tensorboard/"
    )
    model.learn(total_timesteps=120000)
    model.save("simple_construction_dqn") 