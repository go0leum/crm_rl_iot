import unittest
from Q_complex_agent import QAgent
from simple_observation_space import SimpleObservationSpace as Observation
import numpy as np
from state_adapter import StateAdapter

PAYLOAD_RANGE = 11 # 0~50
EQUIPTMENT_RANGE = 4 # 0: 장비 미착용, 1~3: 장비 1~3
RESOURCE_RANGE = 6 # resource 3개, material 3개
AGENT_X_RANGE = 3 # 0, 1, 2
AGENT_Y_RANGE = 3 # 0, 1, 2

class QAgentTest(unittest.TestCase):
    def test_observation_equality(self):
        observation1 = Observation()
        observation2 = Observation()
        self.assertEqual(observation1, observation2)

    def test_q_table_init(self): 
        # q_table의 개수가 알맞게 생성되었는가? -> hash collision이 없는가?
        q_table_count = PAYLOAD_RANGE * EQUIPTMENT_RANGE * (2 ** RESOURCE_RANGE) * AGENT_X_RANGE * AGENT_Y_RANGE
        agent = QAgent()
        q_table = agent.q_table
        self.assertEqual(q_table_count, len(q_table))
        print(q_table_count)

        state = Observation()
        self.assertEqual(agent.q_table[state], [0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125])


    def test_q_table_equality(self):
        # q_table이 생성되고, 이를 올바르게 검색할 수 있는가?
        agent = QAgent()
        state1 = Observation()
        state1.agent_inventory = [1, 0, 1, 0, 1, 0]
        state1.agent_payload = 1
        state1.agent_location = [0, 0]
        state1.agent_ride = 2
        agent.q_table[state1] = [1, 2, 3, 4]

        state2 = Observation()
        state2.agent_inventory = [1, 0, 1, 0, 1, 0]
        state2.agent_payload = 1
        state2.agent_location = [0, 0]
        state2.agent_ride = 2

        self.assertTrue(agent.q_table[state2] == [1, 2, 3, 4])


    def test_adapter_conversion(self):
        adapter = StateAdapter()

        agent_location = np.array((4, 4))
        materials = ["resource_1", "resource_2", "resource_3", "material_1", "material_2", "material_3"]
        agent_inventory =  dict.fromkeys(materials, 0)
        agent_ride = None # string 1개
        agent_payload = 0 # integer

        converted_state = adapter.convert(agent_inventory, agent_ride, agent_payload, agent_location)

        same_state = Observation()
        same_state.agent_location = [4, 4]
        self.assertEqual(same_state, converted_state)


        # case 2
        agent_ride = "equipment_3"
        agent_payload = 10
        agent_inventory[materials[0]] = 1
        agent_inventory[materials[4]] = 1
        converted_state2 = adapter.convert(agent_inventory, agent_ride, agent_payload, agent_location)

        same_state2 = Observation()
        same_state2.agent_inventory = [1, 0, 0, 0, 1, 0]
        same_state2.ride = 3
        same_state2.agent_payload = 10
        same_state2.agent_location = [4, 4]


        self.assertEqual(same_state2, converted_state2)


    def test_select_action(self):
        # action이 정확히 선택되었는가?
        # -> 여러번 수행했을때, 탐색 action이 선택되는가?
        # -> 그리디 action이 선택되었을때, 올바르게 선택되었는가?
        return True
    
if __name__ == '__main__':
    unittest.main()