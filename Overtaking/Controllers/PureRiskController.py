
import yaml
from PIL import Image
import numpy as np
import torch
from matplotlib import pyplot as plt
from ..MotionPrimitives.MotionPrimitives import MotionPrimitive
from .PrimitiveBasedControllerSuper import PrimitiveBasedControllerSuper
from ..Util import LocalField


class PureRiskController(PrimitiveBasedControllerSuper):

    def __init__(self, map_path, map_ext, speeds, angles, local_grid_world_size = 5, resolution=100):
        ## from renderer.py in f1tenth gym for loading map image
        super().__init__( map_path, map_ext, speeds, angles, local_grid_world_size = 5, resolution=100)


    def initialize_primitives(self, speeds, angles, resolution, grid_world_size):
        self.MP = MotionPrimitive(speeds, angles,L=.33, p=1, t_la=2.5, k1=.2, k2=.3, k3=.1, m=.1, c=.12, resolution = (resolution, resolution), local_grid_size=grid_world_size)


    def plan(self, x, y, theta):
        local_grid = LocalField.sample_map_obstacles(x,y,theta, self.map_img, self.local_grid_world_size, self.resolution, self.map_resolution, self.map_width, self.map_height, self.origin)

        risk = self.get_risks(local_grid)

        control_choice = torch.argmin(risk)
        speed, angle = self.MP.get_control_for(control_choice)


        return speed, angle, self.MP.primitives[control_choice]


