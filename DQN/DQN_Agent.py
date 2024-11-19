from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
import numpy as np
import gymnasium as gym

class ConstructionEnv(gym.Env):
    def __init__(self):
        super().__init__()
        
        # 상태 인덱스 상수 정의
        self.IDX_POS_X = 0
        self.IDX_POS_Y = 1
        self.IDX_RESOURCE_START = 2
        self.IDX_RESOURCE_END = 4
        self.IDX_EQUIPMENT_START = 4
        self.IDX_EQUIPMENT_END = 6
        self.IDX_PROJECT_START = 6
        self.IDX_PROJECT_END = 9
        self.IDX_DAY = 9
        
        # 액션 공간 정의 (이동, 리소스 픽업/드롭, 장비 픽업/드롭, 태스크 실행)
        self.action_space = gym.spaces.Discrete(9)
        
        # 관찰 공간 정의
        # [에이전트 위치(x,y), 보유 리소스 상태, 보유 장비 상태, 프로젝트별 태스크 상태, 요일]
        self.observation_space = gym.spaces.MultiDiscrete([
            3, 3,  # 에이전트 위치 (3x3 그리드)
            2, 2,  # 리소스 보유 상태 (2종류)
            2, 2,  # 장비 보유 상태 (2종류)
            3, 3, 3,  # 프로젝트 태스크 상태 (3단계: 준비안됨/준비됨/완료)
            7  # 요일 (0-6: 월-일)
        ])
        
        # 일일 리소스/장비 내부 변수로 설정 (이름 변경)
        self.available_items = np.zeros(2)
        self.current_day = 0
        # 위치 정보 추가
        self.agent_start_pos = [0, 0]
        self.project_positions = [[1, 2], [2, 1], [2, 0]]
        self.resource_pos = [1, 1]
        
        # 일일 행동 제한
        self.max_daily_actions = 5
        self.daily_action_count = 0
        
        # 프로젝트별 리소스/장비 현황 추적
        self.project_resources = [
            {'resource': [0, 0], 'equipment': [0, 0]},  # 프로젝트 0: 첫번째 리소스/장비만 필요
            {'resource': [0, 0], 'equipment': [0, 0]},  # 프로젝트 1: 두번째 리소스/장비만 필요
            {'resource': [0, 0], 'equipment': [0, 0]}   # 프로젝트 2: 모든 리소스/장비 필요
        ]
        
        self.reset()
        
    def reset(self, seed=None, options=None):
        # 초기 상태 설정
        self.agent_pos = self.agent_start_pos.copy()
        self.daily_action_count = 0
        self.current_day = 0
        self.state = np.array([
            self.agent_pos[0], self.agent_pos[1],  # 에이전트 위치
            0, 0,  # 리소스 보유 상태
            0, 0,  # 장비 보유 상태
            0, 0, 0,  # 프로젝트 태스크 상태
            0  # 요일 (0: 월요일)
        ])
        self._update_available_resources()
        return self.state, {}
        
    def _update_available_resources(self):
        # 요일에 따라 사용 가능한 리소스와 장비 번갈아가며 업데이트
        # 짝수 날짜: 첫 번째 리소스/장비만 사용 가능 [1, 0]
        # 홀수 날짜: 두 번째 리소스/장비만 사용 가능 [0, 1]
        self.available_items = np.array([self.current_day % 2 == 0, self.current_day % 2 == 1], dtype=int)
        
    def step(self, action):
        reward = -1  # 기본 리워드
        done = False
        
        # 액션 처리 로직
        if action < 4:  # 이동
            reward = self._move(action)
        elif action == 4:  # 리소스 픽업
            reward = self._pickup_resource()
        elif action == 5:  # 리소스 드롭
            reward = self._dropoff_resource()
        elif action == 6:  # 장비 픽업
            reward = self._pickup_equipment()
        elif action == 7:  # 장비 드롭
            reward = self._dropoff_equipment()
        else:  # 태스크 실행
            reward = self._execute_task()
            
        # 일일 행동 카운트 증가
        self.daily_action_count += 1
        
        # 하루 최대 행동 수에 도달하면 다음 날로 이동
        if self.daily_action_count >= self.max_daily_actions:
            self.current_day = (self.current_day + 1) % 7
            self.daily_action_count = 0
            self._update_available_resources()
        # 상태 업데이트
        self.state = np.array([
            self.agent_pos[0], self.agent_pos[1],
            *self.state[self.IDX_RESOURCE_START:self.IDX_EQUIPMENT_END],  # 리소스와 장비 상태
            *self.state[self.IDX_PROJECT_START:self.IDX_PROJECT_END],  # 프로젝트 상태
            self.current_day
        ])
            
        # 모든 프로젝트가 완료되면 종료
        if self._all_projects_complete():
            done = True
            reward += 500
            
        return self.state, reward, done, False, {}

    def _move(self, action):
        # 이동 방향: 0=상, 1=우, 2=하, 3=좌
        moves = {
            0: [-1, 0],
            1: [0, 1],
            2: [1, 0],
            3: [0, -1]
        }
        
        new_pos = [
            self.agent_pos[0] + moves[action][0],
            self.agent_pos[1] + moves[action][1]
        ]
        
        # 그리드 범위 체크
        if 0 <= new_pos[0] < 3 and 0 <= new_pos[1] < 3:
            self.agent_pos = new_pos
            return -1  # 정상적인 이동은 -1의 보상
        else:
            return -10  # 벽에 부딪힐 경우 -10의 보상
        
    def _pickup_resource(self):
        # 리소스 위치에 있는지 확인
        if self.agent_pos != self.resource_pos:
            return -5  # 잘못된 위치에서 픽업 시도
            
        # 현재 운반 중인 총 아이템(리소스+장비) 개수 확인
        current_items = np.sum(self.state[self.IDX_RESOURCE_START:self.IDX_EQUIPMENT_END])
        if current_items >= 2:
            return -5  # 이미 최대 운반 개수에 도달
            
        # 해당 요일에 사용 가능한 리소스인지 확인
        picked_up = False
        for i in range(2):
            if self.available_items[i] == 1 and self.state[self.IDX_RESOURCE_START + i] == 0 and current_items < 2:
                self.state[self.IDX_RESOURCE_START + i] = 1  # 리소스 픽업
                picked_up = True
                break
                
        return 20 if picked_up else -5  # 성공/실패 여부에 따른 보상 반환
        
    def _dropoff_resource(self):
        # 리소스를 보유하고 있는지 확인
        if not np.any(self.state[self.IDX_RESOURCE_START:self.IDX_RESOURCE_END] == 1):
            return -5  # 드롭할 리소스가 없음
            
        # 프로젝트 위치에 있는지 확인
        current_project = None
        for i, pos in enumerate(self.project_positions):
            if self.agent_pos == pos:
                current_project = i
                break
                
        if current_project is None:
            return -5  # 잘못된 위치에서 드롭 시도
            
        # 리소스 드롭 및 프로젝트에 기록
        for i in range(2):
            if self.state[self.IDX_RESOURCE_START + i] == 1:
                self.project_resources[current_project]['resource'][i] = 1
                
        self.state[self.IDX_RESOURCE_START:self.IDX_RESOURCE_END] = 0  # 리소스 드롭
        
        # 프로젝트 준비 상태 체크
        self._check_project_ready(current_project)
        return 20
        
    def _pickup_equipment(self):
        # 장비 위치에 있는지 확인
        if self.agent_pos != self.resource_pos:
            return -5  # 잘못된 위치에서 픽업 시도
            
        # 현재 운반 중인 총 아이템(리소스+장비) 개수 확인
        current_items = np.sum(self.state[self.IDX_RESOURCE_START:self.IDX_EQUIPMENT_END])
        if current_items >= 2:
            return -5  # 이미 최대 운반 개수에 도달
            
        # 장비 픽업 시도
        picked_up = False
        for i in range(2):
            if self.available_items[i] == 1 and self.state[self.IDX_EQUIPMENT_START + i] == 0 and current_items < 2:
                self.state[self.IDX_EQUIPMENT_START + i] = 1
                picked_up = True
                break
        return 20 if picked_up else -5  # 성공/실패 여부에 따른 보상 반환
        
    def _dropoff_equipment(self):
        # 장비를 보유하고 있는지 확인
        if not np.any(self.state[self.IDX_EQUIPMENT_START:self.IDX_EQUIPMENT_END] == 1):
            return -5  # 드롭할 장비가 없음
            
        # 프로젝트 위치에 있는지 확인
        current_project = None
        for i, pos in enumerate(self.project_positions):
            if self.agent_pos == pos:
                current_project = i
                break
                
        if current_project is None:
            return -5  # 잘못된 위치에서 드롭 시도
            
        # 장비 드롭 및 프로젝트에 기록
        for i in range(2):
            if self.state[self.IDX_EQUIPMENT_START + i] == 1:
                self.project_resources[current_project]['equipment'][i] = 1
                
        self.state[self.IDX_EQUIPMENT_START:self.IDX_EQUIPMENT_END] = 0  # 장비 드롭
        
        # 프로젝트 준비 상태 체크
        self._check_project_ready(current_project)
        return 20
        
    def _check_project_ready(self, project_idx):
        # 프로젝트별 필요 리소스/장비 확인
        is_ready = False
        if project_idx == 0:  # 첫번째 프로젝트: 첫번째 리소스/장비만 필요
            is_ready = (self.project_resources[project_idx]['resource'][0] == 1 and 
                       self.project_resources[project_idx]['equipment'][0] == 1)
                       
        elif project_idx == 1:  # 두번째 프로젝트: 두번째 리소스/장비만 필요
            is_ready = (self.project_resources[project_idx]['resource'][1] == 1 and 
                       self.project_resources[project_idx]['equipment'][1] == 1)
                       
        else:  # 세번째 프로젝트: 모든 리소스/장비 필요
            is_ready = all(self.project_resources[project_idx]['resource']) and \
                      all(self.project_resources[project_idx]['equipment'])
            
        # 준비가 완료되면 프로젝트 상태를 1(준비됨)로 변경
        if is_ready:
            self.state[self.IDX_PROJECT_START + project_idx] = 1
            
    def _execute_task(self):
        # 프로젝트 위치에 있는지 확인
        current_project = None
        for i, pos in enumerate(self.project_positions):
            if self.agent_pos == pos:
                current_project = i
                break
                
        if current_project is None:
            return -5  # 프로젝트 위치가 아님
            
        # 프로젝트 상태 확인
        if self.state[self.IDX_PROJECT_START + current_project] == 0:
            return -5  # 아직 준비되지 않은 프로젝트
        elif self.state[self.IDX_PROJECT_START + current_project] == 2:
            return -5  # 이미 완료된 프로젝트
            
        # 태스크 실행 및 상태 업데이트
        self.state[self.IDX_PROJECT_START + current_project] = 2  # 완료 상태로 변경
        
        # 사용된 리소스/장비 초기화
        self.project_resources[current_project] = {'resource': [0, 0], 'equipment': [0, 0]}
        
        return 50  # 성공적인 태스크 실행
        
    def _all_projects_complete(self):
        return all(self.state[self.IDX_PROJECT_START:self.IDX_PROJECT_END] == 2)  # 모든 프로젝트가 2(완료)

if __name__ == "__main__":
    # 환경 생성 및 검증
    env = ConstructionEnv()
    check_env(env)

    # DQN 모델 생성 - 하이퍼파라미터 최적화
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=1e-4,           # 학습률 조정
        buffer_size=10000,           # 리플레이 버퍼 크기 증가
        learning_starts=1000,        # 더 많은 초기 데이터 수집
        batch_size=128,               # 배치 크기 증가
        gamma=0.98,                   # 감가율 약간 감소
        exploration_fraction=0.4,     # 탐색 기간 증가
        exploration_initial_eps=1.0,
        exploration_final_eps=0.1,    # 최종 탐색률 증가
        train_freq=1,                 # 매 스텝마다 학습
        gradient_steps=2,             # 그래디언트 스텝 증가
        target_update_interval=1000,  # 타겟 네트워크 업데이트 주기 조정
        verbose=1,
        tensorboard_log="./dqn_construction_tensorboard/"  # 텐서보드 로깅 추가
    )

    # 학습 시간 증가
    model.learn(
        total_timesteps=50000,  # 학습 스텝 대폭 증가
        log_interval=100
    )

    # 모델 저장
    model.save("construction_dqn")
