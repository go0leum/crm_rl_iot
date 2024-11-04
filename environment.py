import numpy as np
import tkinter as tk
import time
from PIL import ImageTk, Image

from data_loader import DataLoader

PhotoImage = ImageTk.PhotoImage
UNIT = 100
WIDTH = 26
HEIGHT = 12

class Env(tk.Tk):
    def __init__(self):
        super(Env, self).__init__()
        self.agent_location = np.zeros((1,2))
        self.title('Construction Resource Management')
        self.field_width, self.field_height, self.start, self.end, self.resource_dict, self.obstacle_dict = DataLoader.load_field_data()
        self.project_dict, self.task_dict, self.material_dict, self.equipment_dict = DataLoader.load_project_data()

    def step(self, action):
        return 
    
    def get_obstacle_locations(self):
        obstacle_locations = []
        for key in self.obstacle_dict:
            for location in self.obstacle_dict[key].locations:
                obstacle_locations.append(location)
        
        return obstacle_locations

class Action:

    def __init__(self):
        self.field_width = Env.field_width
        self.field_height = Env.field_height
        self.obstacle_location = Env.get_obstacle_locations()

    def move_left(self, location):
        new_location = np.array((location[0]-1, location[1]))

        if new_location[0]<0:
            new_location = location
        elif new_location in self.obstacle_location:
            new_location = location
        
        return new_location

    def move_right(self, location):
        new_location = np.array((location[0]+1, location[1]))

        if new_location in self.obstacle_location:
            new_location = location
        elif new_location[0]>self.field_width:
            new_location = location

        return new_location
      
    def move_up(self, location):
        new_location = np.array((location[0], location[1]-1))

        if new_location[1]<0:
            new_location = location
        elif new_location in self.obstacle_location:
            new_location = location
        
        return new_location

    def move_down(self, location):
        new_location = np.array((location[0], location[1]+1))

        if new_location > self.field_height:
            new_location = location
        elif new_location in self.obstacle_location:
            new_location = location
        
        return new_location
    