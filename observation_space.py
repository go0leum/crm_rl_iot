import numpy as np
import tkinter as tk
import time
from PIL import ImageTk, Image

class ObservationSpace():
    def __init__(self):
        self.transition_probability = 
        self.work_date # 작업일자
        self.available_work_time # 작업시간 100 -> 0 (daily)
        self.project_tasks = np.zeros((10, 3)) # 프로젝트 task 정보
        # project1 : 0 (task1개)
        # ...
        # project7 : 0, 1, 2 (task3개)

        self.project_tasks_status = np.zeros((10, 3)) # 프로잭트 task 상태
        # project1 : 0 (task1 완료 안됨)
        # ...
        # project7 : 1, 0, 0 (task1만 완료됨)

        self.project_tasks_remains = np.zeros((10, 3)) # 프로젝트 task 남은 시간
        # project1 : 100 (task1 100시간 남음)
        # ...
        # project7 : 100, 200, 0 (task3만 완료됨)

        self.project_tasks_resource = np.zeros(((10, 3, 4)))
        # project1 : 1 (0, 0, 1, 1) (task1의 3번째 4번째 resource 준비됨)
        # ...

        self.agent_location = []
        # agent의 x,y 좌표

    def get_reward():

        return self.reward()
    
    def state_after_action

        return

    def 

