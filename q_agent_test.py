import unittest
from q_agent import QAgent
from simple_observation_space import SimpleObservationSpace as Observation

WORK_DATE_RANGE = 1
WORK_TIME_RANGE = 1
PROJECT_RANGE = 3
TASK_STATUS_RANGE = 3
TASK_RESOURCE_RANGE = 2
RESOURCE_STATUS_RANGE = 2
AGENT_X_RANGE = 3
AGENT_Y_RANGE = 3

class QAgentTest(unittest.TestCase):
    def test_q_table_init(self): 
        # q_table의 개수가 알맞게 생성되었는가? -> hash collision이 없는가?
        q_table_count = WORK_DATE_RANGE * WORK_TIME_RANGE * (TASK_STATUS_RANGE ** PROJECT_RANGE) * (
            ((TASK_RESOURCE_RANGE ** RESOURCE_STATUS_RANGE) ** PROJECT_RANGE) * (AGENT_X_RANGE*AGENT_Y_RANGE))
        agent = QAgent()
        q_table = agent.q_table
        self.assertEqual(q_table_count, len(q_table))
        print(q_table_count)

        agent = QAgent()
        state = Observation()
        state.work_date = 0
        state.available_work_time = 0
        state.agent_location = [0, 0]

        print(agent.q_table[state])


    def test_q_table_equality(self):
        # q_table이 생성되고, 이를 올바르게 검색할 수 있는가?
        agent = QAgent()
        state1 = Observation()
        state1.work_date = 0
        state1.available_work_time = 0
        state1.agent_location = [0, 0]
        agent.q_table[state1] = [1, 2, 3, 4]

        state2 = Observation()
        state2.work_date = 0
        state2.available_work_time = 0
        state2.agent_location = [0, 0]

        self.assertTrue(agent.q_table[state2] == [1, 2, 3, 4])


    def test_select_action(self):
        # action이 정확히 선택되었는가?
        # -> 여러번 수행했을때, 탐색 action이 선택되는가?
        # -> 그리디 action이 선택되었을때, 올바르게 선택되었는가?
        return True
    
if __name__ == '__main__':
    unittest.main()