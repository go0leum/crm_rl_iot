import numpy as np

class Obstacle:
    def __init__(self, name):
        self.name = name
        self.locations = []

class Resource:
    def __init__(self, name, resource_list):
        self.locations = []
        self.name = name
        self.resource_list = resource_list

class Project:
    def __init__(self, name, task_list):
        self.locations = []
        self.name = name
        self.task_list = task_list
        self.status = False # project 수행 여부

class Task:
    def __init__(self, name, resource_list, work_hour):
        self.name = name
        self.resource_list = resource_list # 필요 resource list
        self.resource_status = [False for i in range(len(resource_list))] # resource 보유 여부
        self.work_hour = work_hour # 작업해야할 시간
        self.status = False # task 수행 여부

class Material:
    def __init__(self, name, weight):
        self.name = name
        self.weight = weight

class Equipment:
    def __init__(self, name):
        self.location = np.zeros((1,2))
        self.name = name
