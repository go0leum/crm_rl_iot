import numpy as np
import tkinter as tk
import random

from data_loader import DataLoader

MaterialMaxQuota = 3 # material 종류 별 하루 최대 할당량
EquipmentMaxQuota = 1 # equipment 종류 별 하루 최대 할당량
WorkableWeight = 8 # agent의 payload가 일정값 이상일 때만 작업 가능
WorkDay = 30 # 총 작업 일수

MoveTime = 1 # 이동 소요 시간
MaterialLoadUnloadTime = 10 # material load, unload 소요 시간
EquipmentGetOnOffTime = 1 # Equipment get on, off 소요 시간
WorkTime = 20 # work 소요 시간

ResourceDone = 10 # material, equipment를 성공적으로 이동시켰을때 reward
TaskDone = 100 # task를 완료했을때 reward
ProjectDone = 300 # project를 완료했을때 reward


class Env(tk.Tk):
    def __init__(self):
        super(Env, self).__init__()
        self.title('Construction Resource Management')

        self.field_data, self.field_width, self.field_height = DataLoader.load_field_data()
        self.start, self.resource_dict, self.obstacle_dict, self.project_dict = DataLoader.load_place_data()
        self.task_dict, self.material_dict, self.equipment_dict = DataLoader.load_project_data()

        self.materials = list(self.material_dict.keys())
        self.resources = list(self.material_dict.keys()).extend(list(self.equipment_dict.keys()))

        self.agent_location = self.start
        self.agent_inventory = dict.fromkeys(self.materials, 0) # agent가 들고 있는 material 정보
        self.agent_ride = None # agent가 타고 있는 equipment 정보
        self.agent_payload = 10

        self.resource_day_quota = dict.fromkeys(self.resources, 0) # 하루에 할당된 자원의 양
        self.day_work_time = 100
        self.work_day = WorkDay
    
    def resource_reset(self):
        # 하루에 할당되는 자원 reset: random
        for key in self.resource_day_quota:
            if 'material' in key:
                self.resource_day_quota[key] = random.randrange(0, MaterialMaxQuota)
            elif 'equipment' in key:
                self.resource_day_quota[key] = random.randrange(0, EquipmentMaxQuota)
    
    def day_reset(self):
        # 하루 작업 시간이 끝난 후 reset: agent location, agent inventory, agent_ride, agent_payload, day_work_time
        self.agent_location = self.start
        self.agent_inventory = dict.fromkeys(self.materials, 0) # agent가 들고 있는 material 정보
        self.agent_ride = None # agent가 타고 있는 equipment 정보
        self.agent_payload = 10

        self.resource_day_quota = dict.fromkeys(self.resources, 0) # 하루에 할당된 자원의 양
        self.day_work_time = 100

    def step(self, action):
        # 0번 액션: 왼쪽, 1번 액션: 위, 2번 액션: 오른쪽, 3번 액션: 아래쪽

        # 남은 하루 작업시간 확인 & reset
        if self.day_work_time < 1:
            self.day_reset()
            self.resource_reset()
            self.work_day -= 1

        # move action
        if action==0:
           reward = Action.move_left()
        elif action==1:
            reward = Action.move_up()
        elif action==2:
            reward = Action.move_right()
        elif action==3:
            reward = Action.move_down()

        # material
        elif action==4:
            reward = Action.material_load()
        elif action==5:
            reward = Action.material_unload()

        # equipment
        elif action==6:
            reward = Action.equipment_get_on()
        elif action==7:
            reward = Action.equipment_get_off()
        
        # work
        elif action==8:
            reward = Action.work()
        
        done = self.is_done()

        return self.agent_location, reward, done

    def is_done(self):
        # 지정된 작업 일수가 지나면 종료
        if self.work_day < 1:
            return True
        else:
            False

class Action:
    # 소요시간: move- 1, material load-unload- 10, equipment get on-off- 1, work- 10
    def __init__(self):
        self.field_width = Env.field_width
        self.field_height = Env.field_height
        self.field_data = Env.field_data

    def move_left(self):
        new_x = Env.agent_location[0]-1
        new_y = Env.agent_location[1]
        
        Env.day_work_time -= MoveTime

        if new_x<0:
            pass
        elif 'obstacle' in self.field_data[new_x][new_y]:
            pass
        else:
            Env.agent_location = np.array((new_x, new_y))
        
        return MoveTime

    def move_right(self):
        new_x = Env.agent_location[0]+1
        new_y = Env.agent_location[1]
        
        Env.day_work_time -= MoveTime

        if 'obstacle' in self.field_data[new_x][new_y]:
            pass
        elif new_x>self.field_width:
            pass
        else:
            Env.agent_location = np.array((new_x, new_y))
        
        return MoveTime
      
    def move_up(self):
        new_x = Env.agent_location[0]
        new_y = Env.agent_location[1]-1
        
        Env.day_work_time -= MoveTime

        if new_y < 0:
            pass
        elif 'obstacle' in self.field_data[new_x][new_y]:
            pass
        else:
            Env.agent_location = np.array((new_x, new_y))
        
        return MoveTime

    def move_down(self):
        new_x = Env.agent_location[0]
        new_y = Env.agent_location[1]+1
        
        Env.day_work_time -= MoveTime

        if new_y > self.field_height:
            pass
        elif 'obstacle' in self.field_data[new_x][new_y]:
            pass
        else:
            Env.agent_location = np.array((new_x, new_y))
        
        return MoveTime
    
    def material_load(self):
        # resource 받는 위치에서만 load 가능
        x = Env.agent_location[0]
        y = Env.agent_location[1]
        
        Env.day_work_time -= MaterialLoadUnloadTime
        
        if Env.agnet_payload < 1: # 적재 가능 용량이 부족할 때
            return MaterialLoadUnloadTime
        elif 'resource' not in self.field_data[x][y]:
            return MaterialLoadUnloadTime
        else:
            resource_check = self.resource_check(self.field_data[x][y])
        
            material_list = [material for material in resource_check.keys() if 'material' in material]
            for material_name in material_list:
                material_status = resource_check[material_name]

                if material_status == False: # 자원이 고갈된 상태일 때
                    continue

                material = Env.material_dict[material_name]
                if Env.agent_payload > material.weight:
                    Env.agent_payload -= material.weight
                    Env.agent_inventory[material_name] += 1
                    Env.resource_day_quota[material_name] -= 1
                    return MaterialLoadUnloadTime
            
            return MaterialLoadUnloadTime
    
    def material_unload(self):
        # 위치한 프로젝트와 관련있는 material만 unload 가능
        x = Env.agent_location[0]
        y = Env.agent_location[1]
        
        Env.day_work_time -= MaterialLoadUnloadTime

        if 'project' not in self.field_data[x][y]: # project 위치가 아닐 때
            return MaterialLoadUnloadTime
        elif self.project_check(self.field_data[x][y]) == True: # 완료된 project 일 때
            return MaterialLoadUnloadTime
        else:
            task_list = Env.project_dict[self.field_data[x][y]].task_list
            for task_name in task_list:
                task_status, task_resource_dict = self.task_check(task_name)
                if task_status == True: # 완료된 task 일 때
                    continue

                for resource_name, resource_status in task_resource_dict:
                    if resource_status == True: # 이미 있는 resource 일 때
                        continue

                    if Env.agent_inventory[resource_name] > 0: # task에 필요한 resource가 있을 때
                        Env.agent_inventory[resource_name] -= 1
                        Env.agent_payload += Env.resource_dict[resource_name].weight
                        idx = Env.task_dict[task_name].resource_list.index(resource_name)
                        Env.task_dict.resource_status[idx] = True
                        return MaterialLoadUnloadTime + ResourceDone
            
            return MaterialLoadUnloadTime
    
    def equipment_get_on(self):
        x = Env.agent_location[0]
        y = Env.agent_location[1]
        
        Env.day_work_time -= EquipmentGetOnOffTime

        if Env.agent_ride != None: # 다른 장비를 타고 있는 상태일 때 탑승 불가
            return EquipmentGetOnOffTime
        elif Env.agent_payload < 10: # inventory에 material 있으면 탑승 불가
            return EquipmentGetOnOffTime
        elif 'resource' in self.field_data[x][y]: # 장비를 얻을 수 있는 위치일 때
            resource_check = self.resource_check(self.field_data[x][y])
            equipment_list = [equipment for equipment in resource_check.keys() if 'equipment' in equipment]

            for equipment_name in equipment_list:
                equipment_status = resource_check[equipment_name]

                if equipment_status == False: # 자원이 고갈된 상태일 때
                    continue
                else:
                    Env.agent_ride = equipment_name
                    Env.resource_day_quota[equipment_name] -= 1

                    return EquipmentGetOnOffTime
            
            return EquipmentGetOnOffTime

    def equipment_get_off(self):
        x = Env.agent_location[0]
        y = Env.agent_location[1]

        Env.day_work_time -= EquipmentGetOnOffTime

        if Env.agent_ride == None: # 장비 탑승 상태가 아닐 때
            return EquipmentGetOnOffTime
        
        elif 'resource' in self.field_data[x][y]:
            if Env.agnet_ride in Env.resource_dict(self.field_data[x][y]).resource_list: # 장비 반납
                Env.agent_ride = None
                Env.resource_day_quota[Env.agnet_ride] += 1
                return EquipmentGetOnOffTime
            
        elif 'project' in self.field_data[x][y]: # project 위치일 때
            if self.project_check(self.field_data[x][y]) == True: # 완료된 project 일 때
                return EquipmentGetOnOffTime
            
            task_list = Env.project_dict[self.field_data[x][y]].task_list

            for task_name in task_list:
                task_status, task_resource_dict = self.task_check(task_name)
                if task_status == True: # 완료된 task 일 때
                    continue

                for resource_name, resource_status in task_resource_dict:
                    if resource_status == True: # 이미 있는 resource 일 때
                        continue

                    if Env.agent_ride == resource_name: # task에 필요한 equipment를 타고 있을 때
                        idx = Env.task_dict[task_name].resource_list.index(resource_name)
                        Env.agent_ride = None
                        Env.task_dict.resource_status[idx] = True
                        return EquipmentGetOnOffTime + ResourceDone
            
            return EquipmentGetOnOffTime

    def work(self):
        x = Env.agent_location[0]
        y = Env.agent_location[1]

        Env.day_work_time -= WorkTime

        task_finish_count = 0
        task_reward = 0
        project_reward = 0

        if Env.agent_ride != None: # 장비 탑승 상태일 때
            return WorkTime
        elif Env.agent_payload < WorkableWeight: # inventory에 material이 일정량 이상 있으면 작업 불가
            return WorkTime
        elif 'project' not in self.field_data[x][y]: # project 위치가 아닐 때
            return WorkTime
        else:
            task_list = Env.project_dict[self.field_data[x][y]].task_list

            for task_name in task_list:
                task_status, task_resource_dict = self.task_check(task_name)

                if task_status == True: # 완료된 task 일 때
                    task_finish_count += 1
                    continue

                if sum(task_resource_dict.values()) == len(task_resource_dict): #resource를 모두 보유하고 있을 때
                    Env.task_dict(task_name).work_hour -= WorkTime # 수행해야할 작업 시간 감소

                    if Env.task_dict(task_name).work_hour < 1: # work action수행으로 작업시간을 모두 채웠을 때->task를 끝냈을 때
                        Env.task_dict(task_name).status = True
                        Env.task_dict(task_name).resource_status = [False for _ in Env.task_dict(task_name).resource_status]
                        task_finish_count += 1
                        task_reward = 100

                        if task_finish_count == len(task_list): # 모든 task를 완료해, project를 완료했을 때
                            Env.project_dict[self.field_data[x][y]].status = True
                            project_reward = 300

                    return WorkTime + task_reward + project_reward
        
            return WorkTime + task_reward + project_reward

    def resource_check(self, resource_name):
        # 해당 resource에서 load할 수 있는 재료나 장비가 남아있는지 확인
        resource = Env.resource_dict[resource_name]
        resource_check_dict = dict()

        for resource_name in resource.resource_list:
            if Env.resource_day_quota[resource_name] > 0:
                resource_status = True
            else: 
                resource_status = False
            
            resource_check_dict[resource_name] = resource_status
        
        return resource_check_dict

    def project_check(self, project_name):
        # task 상태, task별 resource 상태 확인
        project = Env.project_dict[project_name]

        return project.status
    
    def task_check(self, task_name):
        # task 작업 여부, resource 존재 여부 확인
        task = Env.task_dict[task_name]
        task_resource_dict = dict()

        for i, resource_name in enumerate(task.resource_list):
            task_resource_dict[resource_name] = task.resource_status[i]
        
        return task.status, task_resource_dict
