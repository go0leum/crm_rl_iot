from simple_observation_space import SimpleObservationSpace as Observation
from environment import Env
from state_adapter import StateAdapter
import random
import numpy as np
from itertools import product

PAYLOAD_RANGE = 11 # 0~50
EQUIPTMENT_RANGE = 4 # 0: 장비 미착용, 1~3: 장비 1~3
RESOURCE_RANGE = 6 # resource 3개, material 3개
AGENT_X_RANGE = 3 # 0, 1, 2
AGENT_Y_RANGE = 3 # 0, 1, 2
ACTION_SPACE_SIZE = 8

class QAgent():
    def __init__(self):
        self.q_table = {}
        self.eps = 0.9
        self.init_q_table()
        self.adapter = StateAdapter()

    def generate_combination(self, item_count, item_range): 
        return [np.array(combination) for combination in product(item_range, repeat=item_count)]
    
    def init_q_table(self):
        inventory_range = self.generate_combination(RESOURCE_RANGE, range(2))

        for ride, payload, agent_x, agent_y in product(
            range(EQUIPTMENT_RANGE), range(PAYLOAD_RANGE), range(AGENT_X_RANGE), range(AGENT_Y_RANGE)):
            for inventory in inventory_range:
                observation = Observation()
                observation.agent_inventory = inventory.tolist()
                observation.ride = ride
                observation.agent_payload = payload
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

            s = self.adapter.convert(self.env.reset())
            # 하나의 episode
            while not done:
                a = self.q_agent.select_action(s) # 입실론 그리디에 의해서 action을 선택한다.
                raw_s_prime, r, done = self.env.step(a) # action을 수행하고 다음 상태, 보상, 종료 flag를 받아온다.
                s_prime = self.adapter.convert(raw_s_prime)
                self.q_agent.update_table((s,a,r,s_prime)) # 그리디에 의해서 table을 업데이트 한다.

                s = s_prime

            self.q_agent.anneal_eps() # epsilon을 점차 줄어들게 한다.
