import numpy as np
class SimpleObservationSpace():
    def __init__(self):
        self.agent_inventory = [0, 0, 0, 0, 0, 0]
        # resource 종류 6개
        # 각 resource마다 1개만 가질 수 있음
        # 1, 0, 1, 0, 0, 0 => 1번과 3번 리소스를 1개 운반중
        # resource_1, resource_2, resource_3, material_1, material_2, material_3 로 대응됨
        self.ride = 0 # equintment 종류 (0~3) # equiptment_1, 2,3
        self.agent_payload = 0 # 최대 10
        self.agent_location = [0, 0] # 10 x 10 = 100
        # agent의 x,y 좌표

    def __eq__(self, instance):
        if isinstance(instance, SimpleObservationSpace): # 동일한 class 인스턴스간의 비교만 True가 될 가능성이 존재
            if ((np.array(self.agent_inventory) == np.array(instance.agent_inventory)).all()) and \
                self.ride == instance.ride and \
                self.agent_payload == instance.agent_payload and \
                self.agent_location == instance.agent_location:
                return True
        return False

    def __hash__(self):
        hash_value = (tuple(self.agent_inventory), 
                     self.ride, 
                     self.agent_payload,
                     tuple(self.agent_location)) 
        return hash(hash_value) 
    
    def __str__(self):
        return f"inventory: {self.agent_inventory}\n ride: {self.ride}\n payload: {self.agent_payload}\n location: {self.agent_location}\n"