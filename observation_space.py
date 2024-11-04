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
        self.resource_status = [True for i in range(len(resource_list))]

class Project:
    def __init__(self, name, task_list):
        self.locations = []
        self.name = name
        self.task_list = task_list
        self.status=True

class Task:
    def __init__(self, name, resource_list, work_hour):
        self.name = name
        self.resource_list = resource_list
        self.work_hour = work_hour
        self.status = True

class Material:
    def __init__(self, name, weight):
        self.location = np.zeros((1,2))
        self.name = name
        self.weight = weight
        self.status = True

class Equipment:
    def __init__(self, name):
        self.location = np.zeros((1,2))
        self.name = name
        self.status = True