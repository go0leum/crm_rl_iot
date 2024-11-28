import random
import numpy as np

Action_Num = 9

class QAgent():
    def __init__(self, env):
        self.env = env
        self.q_table = np.zeros((self.env.field_height, self.env.field_width, Action_Num)) # 마찬가지로 Q 테이블을 0으로 초기화
        self.eps = 0.9

    def select_action(self, s):
        # eps-greedy로 액션을 선택해준다
        x, y = s
        coin = random.random()
        if coin < self.eps:
            action = random.randint(0,Action_Num-1)
        else:
            action_val = self.q_table[x,y,:]
            action = np.argmax(action_val) # 그리디하게 선택
        return action

    def update_table(self, transition):
        s, a, r, s_prime = transition
        x,y = s
        next_x, next_y = s_prime[0], s_prime[1]
        a_prime = self.select_action(s_prime) # S'에서 선택할 액션 (실제로 취한 액션이 아님)
        # Q러닝 업데이트 식을 이용 
        self.q_table[x,y,a] = self.q_table[x,y,a] + 0.1 * (r + np.amax(self.q_table[next_x,next_y,:]) - self.q_table[x,y,a])

    def anneal_eps(self):
        self.eps -= 0.01  # Q러닝에선 epsilon 이 좀더 천천히 줄어 들도록 함.
        self.eps = max(self.eps, 0.2) 

    def train(self):
        for n_epi in range(1000):
            done = False

            s = self.env.reset()
            # 하나의 episode
            while not done:
                a = self.select_action(s) # 입실론 그리디에 의해서 action을 선택한다.
                s_prime, r, done = self.env.step(a) # action을 수행하고 다음 상태, 보상, 종료 flag를 받아온다.
                self.update_table((s,a,r,s_prime)) # 그리디에 의해서 table을 업데이트 한다.
                s = s_prime
            self.anneal_eps() # epsilon을 점차 줄어들게 한다.
