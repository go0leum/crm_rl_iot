import csv
import observation_space as OS
import numpy as np

class DataLoader:
    def __init__(self):
        self.resource_dict = self.load_resource()
        self.project_dict = self.load_project()
        self.task_dict = self.load_task()
        self.material_dict = self.load_material()
        self.equipment_dict = self.load_equipment()
        self.field_data, self.field_width, self.field_height, self.start, self.obstacle_dict = self.load_field()
    
    def load_field_data(self):
        return self.field_data, self.field_width, self.field_height
    
    def load_place_data(self):
        return self.start, self.resource_dict, self.obstacle_dict, self.project_dict
    
    def load_project_data(self):
        return self.task_dict, self.material_dict, self.equipment_dict
    
    def load_equipment(self):
        file = open('./data/equipment_data.CSV',"r")
        data = csv.reader(file)
        
        equipment_dict = dict()
        for row in data:
            equipment = OS.Equipment(row[0])
            equipment_dict[row[0]] = equipment
        
        return equipment_dict
    
    def load_material(self):
        file = open('./data/material_data.CSV',"r")
        data = csv.reader(file)
        
        material_dict = dict()
        for row in data:
            material = OS.Material(row[0], int(row[1]))
            material_dict[row[0]] = material
        
        return material_dict

    def load_task(self):
        file = open('./data/task_data.CSV',"r")
        data = csv.reader(file)

        task_dict = dict()
        for row in data:
            name = row[0]
            resource_list = []
            work_hour = row[len(row)-1]

            for idx in range(1, len(row)-1):
                resource = row[idx]

                if resource == '': continue
                
                resource_list.append(resource)

            task = OS.Task(name, resource_list, work_hour)
            task_dict[name] = task

        return task_dict
    
    def load_project(self):
        file = open('./data/project_data.CSV',"r")
        data = csv.reader(file)

        project_dict = dict()
        for row in data:
            name = row[0]
            task_list = []

            for idx in range(1, len(row)):
                task = row[idx]

                if task == '': continue
                
                task_list.append(task)

            project = OS.Project(name, task_list)
            project_dict[name] = project

        return project_dict

    def load_resource(self):
        file = open('./data/resource_data.CSV',"r")
        data = csv.reader(file)

        resource_dict = dict()
        for row in data:
            name = row[0]
            resource_list = []

            for idx in range(1, len(row)):
                resource = row[idx]

                if resource == '': continue
                
                resource_list.append(resource)

            resource = OS.Resource(name, resource_list)
            resource_dict[name] = resource

        return resource_dict
    
    def load_field(self):
        file = open('./data/field_map.CSV',"r")
        data = csv.reader(file)
        field_data = []
        width, height = 0, 0
        obstacle_dict = dict()

        for i, row in enumerate(data):
            row_data = []
            height += 1
            width = len(row)
            for j, item in enumerate(row):
                row_data.append(item)
                location = np.array((i, j))

                if 'start' in item:
                    start = location

                elif 'obstacle' in item:
                    if item in obstacle_dict:
                        obstacle_dict[item].locations.append(location)
                    else:
                        obstacle_dict[item] = OS.Obstacle(item)
                        obstacle_dict[item].locations.append(location)

                elif 'project' in item:
                    self.project_dict[item].locations.append(location)

                elif 'resource' in item:
                    self.resource_dict[item].locations.append(location)

            field_data.append(row_data)
        field_data = np.array(field_data)
        return field_data, width, height, start, obstacle_dict
