import tkinter as tk
from tkinter import Button, Entry, Label
import time
import numpy as np
from PIL import ImageTk, Image

PhotoImage = ImageTk.PhotoImage
UNIT = 50

class GraphicDisplay(tk.Tk):
    def __init__(self, env):
        super(GraphicDisplay, self).__init__()
        self.env = env
        self.field_width = self.env.field_width
        self.field_height = self.env.field_height

        self.title('crm rl iot')
        self.geometry('{0}x{1}'.format(self.field_width * UNIT, self.field_height * UNIT + 100))
        self.texts = []
        self.arrows = []

        self.field_data, self.box_data = self.env.get_field_data() #start, resource, obstacle, project
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
        start_y, start_x = self.box_data[0][0], self.box_data[0][1]
        canvas.create_image(start_x* UNIT+ (UNIT/2), start_y* UNIT+ (UNIT/2), image=self.boxes[0])
        self.agent_icon = canvas.create_image(start_x* UNIT+ (UNIT/2), start_y* UNIT+ (UNIT/2), image=self.icons[0])
        # resource
        resource_dict = self.box_data[1]
        for key in resource_dict:
            for location in resource_dict[key].locations:
                canvas.create_image(location[1]* UNIT+ (UNIT/2), location[0]* UNIT+ (UNIT/2), image=self.boxes[1])
        # obstacle
        obstacle_dict = self.box_data[2]
        for key in obstacle_dict:
            for location in obstacle_dict[key].locations:
                canvas.create_image(location[1]* UNIT+ (UNIT/2), location[0]* UNIT+ (UNIT/2), image=self.boxes[2])
        # project
        project_dict = self.box_data[3]
        self.project_box_dict = dict()
        for key in project_dict:
            for location in project_dict[key].locations:
                self.project_box_dict[key] = canvas.create_image(location[1]* UNIT+ (UNIT/2), location[0]* UNIT+ (UNIT/2), image=self.boxes[3])
        
        canvas.pack()

        # work day 확인 창
        self.workday = tk.StringVar()
        self.workday.set('Remaining working day: '+ str(self.env.work_day))
        workday_label = Label(self, textvariable=self.workday)
        workday_label.place(x=self.field_width * UNIT * 0.05, y=(self.field_height * UNIT) + 40)

        # 버튼 초기화
        dqn_button = Button(self, text="simulator start",
                            command=self.deep_q_learning)
        dqn_button.configure(width=15, activebackground="#33B5E5")
        canvas.create_window(self.field_width * UNIT * 0.5, self.field_height * UNIT + 10,
                             window=dqn_button)

        return canvas
    
    def render(self):
        time.sleep(0.1)
        self.canvas.tag_raise(self.agent_icon)
        self.workday.set('Remaining working day: '+ str(self.env.work_day))
        self.update()

if __name__ == "__main__":
    grid_world = GraphicDisplay()
    grid_world.mainloop()