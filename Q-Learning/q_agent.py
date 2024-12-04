from simple_observation_space import SimpleObservationSpace as Observation
from environment import Env
import random
import numpy as np
from itertools import product

WORK_DATE_RANGE = 1 # 작업일자 0~3
WORK_TIME_RANGE = 1 # 작업시간 0~10
PROJECT_RANGE = 3 # project의 개수
TASK_STATUS_RANGE = 3 # task의 상태는 초기, 진행중, 완료
TASK_RESOURCE_RANGE = 2 # task당 필요 resource 개수
RESOURCE_STATUS_RANGE = 2 # resource의 상태는 준비됨 or 안됨
AGENT_X_RANGE = 3
AGENT_Y_RANGE = 3
ACTION_SPACE_SIZE = 10

class QAgent():
    def __init__(self):
        self.q_table = {}
        self.eps = 0.9
        self.init_q_table()

    # item_count = 3, item_range = 3을 받아서
    # [
    # [ 0, 0, 0 ], 
    # [ 0. 0. 1 ], 
    # [ 0, 0, 2 ],
    # [ 1, 0, 0 ],
    # ...
    # [ 2, 2, 2 ], 
    # ]
    # 반환
    # item_range는 list여도 된다.
    # 즉, 중요한 것은 길이가 3인 array에 각 item들에 대한 모든 경우의 수를 가지는 리스트들을 반환한다는 것이다.
    def generate_combination(self, item_count, item_range): 
        return [np.array(combination) for combination in product(item_range, repeat=item_count)]

    
    def init_q_table(self):
        project_task_status_range = self.generate_combination(PROJECT_RANGE, range(TASK_STATUS_RANGE))
        task_resource_status_range = self.generate_combination(TASK_RESOURCE_RANGE, range(RESOURCE_STATUS_RANGE))
        project_tasks_resource_range = self.generate_combination(PROJECT_RANGE, task_resource_status_range)           

        for work_date, work_time, agent_x, agent_y in product(
            range(WORK_DATE_RANGE), range(WORK_TIME_RANGE), range(AGENT_X_RANGE), range(AGENT_Y_RANGE)):
            for project_task_status in project_task_status_range:
                for project_tasks_resource in project_tasks_resource_range:
                    # 하나의 state 생성
                    observation = Observation()
                    observation.work_date = work_date
                    observation.available_work_time = work_time
                    observation.project_tasks_status = project_task_status
                    observation.project_tasks_resource = project_tasks_resource
                    observation.agent_location = [agent_x, agent_y]
                    action_space = []
                    for _ in range(ACTION_SPACE_SIZE):
                        action_space.append(1/ACTION_SPACE_SIZE) 
                    self.q_table[observation] = action_space # q[s] = a 추가 (a = [0.1, 0.1, .... 0.1])
            
            
    def select_action(self, s):
        # eps-greedy로 액션을 선택해준다
        coin = random.random()
        if coin < self.eps:
            action = random.randint(0, ACTION_SPACE_SIZE-1)
        else:
            action_val = self.q_table[s,:]
            action = np.argmax(action_val) # 그리디하게 선택
        return action

    def update_table(self, transition):
        s, a, r, s_prime = transition
        a_prime = self.select_action(s_prime) # S'에서 선택할 액션 (실제로 취한 액션이 아님)
        # Q러닝 업데이트 식을 이용 
        self.q_table[s,a] = self.q_table[s,a] + 0.1 * (r + np.amax(self.q_table[s_prime, :]) - self.q_table[s,a])

    def anneal_eps(self):
        self.eps -= 0.01  # Q러닝에선 epsilon 이 좀더 천천히 줄어 들도록 함.
        self.eps = max(self.eps, 0.2)

    def q_learning(self, episode):  
        for n_epi in range(episode):
            done = False

            s = self.env.reset()
            # 하나의 episode
            while not done:
                a = self.q_agent.select_action(s) # 입실론 그리디에 의해서 action을 선택한다.
                s_prime, r, done = self.env.step(a) # action을 수행하고 다음 상태, 보상, 종료 flag를 받아온다.
                self.q_agent.update_table((s,a,r,s_prime)) # 그리디에 의해서 table을 업데이트 한다.

                s = s_prime

            self.q_agent.anneal_eps() # epsilon을 점차 줄어들게 한다.