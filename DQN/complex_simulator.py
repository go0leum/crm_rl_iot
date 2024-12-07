import tkinter as tk
from tkinter import Button, Entry, Label
import time
import numpy as np
from PIL import ImageTk, Image
from stable_baselines3 import DQN
from Complex_DQN_Agent import ComplexConstructionEnv

PhotoImage = ImageTk.PhotoImage
UNIT = 50

class GraphicDisplay(tk.Tk):
    def __init__(self, env):
        super(GraphicDisplay, self).__init__()
        self.env = env
        self.field_width = self.env.field_width
        self.field_height = self.env.field_width

        self.total_reward = 0

        # 상태 인덱스 상수
        self.IDX_POS_X = self.env.IDX_POS_X
        self.IDX_POS_Y = self.env.IDX_POS_Y

        self.title('crm rl iot')
        self.geometry('{0}x{1}'.format(self.field_width * UNIT, self.field_height * UNIT + 100))

        self.start_pos = self.env.agent_start_pos
        self.project_positions = self.env.project_positions
        self.resource_positions = self.env.resource_positions
        self.obstacle_positions = self.env.obstacle_positions

        self.project_status = self.project_state_check()

        self.icons, self.boxes = self.load_images() #icon:agent,equipment, box:start, resource, obstacle, on_project, off_project
        self.canvas = self._build_canvas()

    def load_images(self):
        agent_icon = PhotoImage(Image.open("./data/agent_icon.png").resize((UNIT, UNIT)))
        equipment_icon = PhotoImage(Image.open("./data/equipment_icon.png").resize((UNIT, UNIT)))
        obstacle_box = PhotoImage(Image.open("./data/obstacle.png").resize((UNIT, UNIT)))
        off_project_box = PhotoImage(Image.open("./data/off_project.png").resize((UNIT, UNIT)))
        on_project_box = PhotoImage(Image.open("./data/on_project.png").resize((UNIT, UNIT)))
        resource_box = PhotoImage(Image.open("./data/resource.png").resize((UNIT, UNIT)))
        start_box = PhotoImage(Image.open("./data/start.png").resize((UNIT, UNIT)))

        return (agent_icon, equipment_icon), (start_box, resource_box, obstacle_box, on_project_box, off_project_box)

    def _build_canvas(self):
        canvas = tk.Canvas(self, bg='white',
                        height=self.field_height * UNIT,
                        width=self.field_width * UNIT)

        # 그리드 생성
        for col in range(0, self.field_width * UNIT, UNIT):  # 0~400 by 80
            x0, y0, x1, y1 = col, 0, col, self.field_width * UNIT
            canvas.create_line(x0, y0, x1, y1)
        for row in range(0, self.field_height * UNIT, UNIT):  # 0~400 by 80
            x0, y0, x1, y1 = 0, row, self.field_width * UNIT, row
            canvas.create_line(x0, y0, x1, y1)

        # 캔버스에 이미지 추가
        # start, agent
        start_x, start_y = self.start_pos[0], self.start_pos[1]
        canvas.create_image(start_x* UNIT+ (UNIT/2), start_y* UNIT+ (UNIT/2), image=self.boxes[0])
        self.agent_icon = canvas.create_image(start_x* UNIT+ (UNIT/2), start_y* UNIT+ (UNIT/2), image=self.icons[0])
        # resource
        for resource_pos in self.resource_positions:
            location = resource_pos
            self.resource_icon = canvas.create_image(location[0]* UNIT+ (UNIT/2), location[1]* UNIT+ (UNIT/2), image=self.boxes[1])
        # project
        for project_pos in self.project_positions:
            location = project_pos
            self.project_icon = canvas.create_image(location[0]* UNIT+ (UNIT/2), location[1]* UNIT+ (UNIT/2), image=self.boxes[3])
        # obstacle
        for obstacle_pos in self.obstacle_positions:
            location = obstacle_pos
            self.obstacle_icon = canvas.create_image(location[0]* UNIT+ (UNIT/2), location[1]* UNIT+ (UNIT/2), image=self.boxes[2])

        canvas.pack()
        # project task 상태, reward 확인창, workday 확인창
        self.workday_string = tk.StringVar()
        self.workday_string.set(f'Remaining Work Day: {self.env.max_total_action//self.env.action_count}={self.env.max_total_action}/{self.env.action_count}')
        workday_label = Label(self, textvariable=self.workday_string)
        workday_label.place(x=self.field_width * UNIT * 0.05, y=(self.field_height * UNIT) + 25)

        self.reward_string = tk.StringVar()
        self.reward_string.set(f'Reward: {self.total_reward}')
        reward_label = Label(self, textvariable=self.reward_string)
        reward_label.place(x=self.field_width * UNIT * 0.05, y=(self.field_height * UNIT) + 45)
        
        self.project_status = self.project_state_check()
        self.project_string = []
        project_label = []
        for i in range(len(self.project_positions)):
            self.project_string.append(tk.StringVar())
            idx_start = self.env.IDX_PROJECT_START[i] - self.env.IDX_PROJECT_START[0]
            if i < len(self.project_positions)-1:
                idx_end = self.env.IDX_PROJECT_START[i+1] - self.env.IDX_PROJECT_START[i]
            else:
                idx_end = len(self.project_status)
            self.project_string[i].set(f'Project {i}: {self.project_status[idx_start:idx_end]}')
            project_label.append(Label(self, textvariable=self.project_string[i]))
            project_label[i].place(x=self.field_width * UNIT * 0.05, y=(self.field_height * UNIT) + 65+(i*20))


        # 버튼 초기화
        dqn_button = Button(self, text="simulator start",
                            command=self.simulation)
        dqn_button.configure(width=15, activebackground="#33B5E5")
        canvas.create_window(self.field_width * UNIT * 0.5, self.field_height * UNIT + 10,
                             window=dqn_button)

        return canvas
    
    def render(self):
        time.sleep(0.1)
        self.canvas.tag_raise(self.agent_icon)
        self.workday_string.set(f'Remaining Work Day: {self.env.action_count//self.env.action_count}={self.env.max_total_action}/{self.env.action_count}')
        self.reward_string.set(f'Reward: {self.total_reward}')
        for i in range(len(self.project_string)):
            idx_start = self.env.IDX_PROJECT_START[i] - self.env.IDX_PROJECT_START[0]
            if i < len(self.project_positions)-1:
                idx_end = self.env.IDX_PROJECT_START[i+1] - self.env.IDX_PROJECT_START[i]
            else:
                idx_end = len(self.project_status)
            self.project_string[i].set(f'Project {i}: {self.project_status[idx_start:idx_end]}')
        self.update()

    def model_load(self, model_path):
        try:
            model = DQN.load(
                model_path,
                env=self.env,
                # 필요한 경우 하이퍼파라미터 수정
                learning_rate=1e-4,
                exploration_fraction=0.2,
                exploration_initial_eps=0.5,
                exploration_final_eps=0.05
            )
            return model
        
        except Exception as e:
            print(f"모델 로딩 중 오류 발생: {e}")
            return None
    
    def project_state_check(self):
        project_state = []
        state = False
        for status in self.env.state[self.env.IDX_PROJECT_START[0]:]: # 0: resource없음, 1:resource 있음, 2: task 완료:
            if status == 0:
                state = 'False'
            elif status == 1:
                state = 'Ready'
            elif status == 2:
                state = 'Done'
            project_state.append(state)

        return project_state
        
    def simulation(self):
        # 모델 load 하기
        model_path = './weights/complex_construction_dqn'
        model = self.model_load(model_path)

        done = False
        # env의 observation space 가져오기
        obs, _ = self.env.reset()
        while not done:
            # agent의 action 가져오기
            agent_position_before = [obs[self.IDX_POS_X], obs[self.IDX_POS_Y]]
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)

            # env의 step을 통해 다음 상태 가져오기
            obs, reward, done, _, _ = self.env.step(action)
            agent_position_after = [obs[self.IDX_POS_X], obs[self.IDX_POS_Y]]
            self.total_reward += reward

            # agent icon move
            self.canvas.move(self.agent_icon, (agent_position_after[0]-agent_position_before[0])* UNIT, (agent_position_after[1]-agent_position_before[1])* UNIT)
            self.project_status = self.project_state_check()
            self.render()

if __name__ == "__main__":
    env = ComplexConstructionEnv()
    grid_world = GraphicDisplay(env)
    grid_world.mainloop()