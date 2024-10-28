import numpy as np
class SimpleObservationSpace():
    def __init__(self):
        self.work_date = 0 # 작업일자 (0~3)
        self.available_work_time = 0 # 작업시간 10 -> 0 (daily)

        self.project_tasks_status = np.zeros(3) # 프로잭트 task 상태 -> 초기상태, 진행중, 완료됨
        # project1 : 0 (task 완료 안됨)
        # ...
        # project3 : 1 (task 완료됨)

        self.project_tasks_resource = np.zeros((3, 2))
        # project1 : (0, 1) (project1의 task의 2번째 resource 준비됨)
        # ...

        self.agent_location = [] # 10 x 10 = 100
        # agent의 x,y 좌표

    def __eq__(self, instance):
        if isinstance(instance, SimpleObservationSpace): # 동일한 class 인스턴스간의 비교만 True가 될 가능성이 존재
            if self.work_date == instance.work_date and \
                self.available_work_time == instance.available_work_time and \
                self.project_tasks_status.all() == instance.project_tasks_status.all() and \
                self.project_tasks_resource.all() == instance.project_tasks_resource.all() and \
                self.agent_location == instance.agent_location:
                return True
        return False

    def __hash__(self):
        hash_value = (self.work_date, 
                     self.available_work_time,
                     tuple(self.project_tasks_status), # np.array는 tuple로 변환해야 함
                     tuple(map(tuple, self.project_tasks_resource)), # 위와 동일하지만, 2차원 array이므로 이와 같이 변환함
                     tuple(self.agent_location)) # 마찬가지로 agent_location도 list로, tuple로 변환해야함.
        return hash(hash_value) 