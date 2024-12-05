import tkinter as tk
from tkinter import Button, Entry, Label
import time
import numpy as np
from PIL import ImageTk, Image
from stable_baselines3 import DQN
from Simple_DQN_Agent import SimpleConstructionEnv

PhotoImage = ImageTk.PhotoImage
UNIT = 50

class GraphicDisplay(tk.Tk):
    def __init__(self, env):
        super(GraphicDisplay, self).__init__()
        self.env = env
        self.field_width = 5
        self.field_height = 5

        # 상태 인덱스 상수
        self.IDX_POS_X = self.env.IDX_POS_X
        self.IDX_POS_Y = self.env.IDX_POS_Y

        self.title('crm rl iot')
        self.geometry('{0}x{1}'.format(self.field_width * UNIT, self.field_height * UNIT + 100))

        self.start_pos = self.env.agent_start_pos
        self.project_pos = self.env.project_positions
        self.resource_pos = [self.env.resource1_pos, self.env.resource2_pos]

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
        start_y, start_x = self.start_pos[0], self.start_pos[1]
        canvas.create_image(start_x* UNIT+ (UNIT/2), start_y* UNIT+ (UNIT/2), image=self.boxes[0])
        self.agent_icon = canvas.create_image(start_x* UNIT+ (UNIT/2), start_y* UNIT+ (UNIT/2), image=self.icons[0])
        # resource
        for resource_pos in self.resource_pos:
            location = resource_pos
            self.resource_icon = canvas.create_image(location[1]* UNIT+ (UNIT/2), location[0]* UNIT+ (UNIT/2), image=self.boxes[1])
        # project
        for project_pos in self.project_pos:
            location = project_pos
            self.project_icon = canvas.create_image(location[1]* UNIT+ (UNIT/2), location[0]* UNIT+ (UNIT/2), image=self.boxes[3])
        
        canvas.pack()

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
        self.update()

    def model_load(self, model_path):
        try:
            model = DQN.load(
                model_path,
                env=self.env,
                tensorboard_log="./crm_rl_iot/DQN/log/dqn_tensorboard/",
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
        
    def simulation(self):
        # 모델 load 하기
        try:
            model_path = '../weights/simple_construction_dqn'
            model = self.model_load(model_path)
        except FileNotFoundError:
            print(f'There is not model file. path: {model_path}.')

        done = False
        # env의 observation space 가져오기
        obs, _ = self.env.reset()
        while not done:
            # agent의 action 가져오기
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)

            # env의 step을 통해 다음 상태 가져오기
            obs_prime, reward, done, _, _ = self.env.step(action)
            agent_position = [obs[self.IDX_POS_X], obs[self.IDX_POS_Y]]

            # agent icon move
            self.canvas.move(self.agent_icon, (obs_prime[self.IDX_POS_X]-obs[self.IDX_POS_X])* UNIT, (obs_prime[self.IDX_POS_Y]-obs[self.IDX_POS_Y])* UNIT)
        
            obs = obs_prime
            self.render()

if __name__ == "__main__":
    env = SimpleConstructionEnv()
    grid_world = GraphicDisplay(env)
    grid_world.mainloop()