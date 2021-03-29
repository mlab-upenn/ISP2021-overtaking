
import yaml
from PIL import Image
import numpy as np
import torch
from matplotlib import pyplot as plt
from ..MotionPrimitives.MotionPrimitives import MotionPrimitive
from .PrimitiveBasedControllerSuper import PrimitiveBasedControllerSuper

from ..Util import LocalField


class DynamicObstacleController(PrimitiveBasedControllerSuper):

    def __init__(self, map_path, map_ext, speeds, angles, local_grid_world_size = 5, resolution=70):
        super().__init__( map_path, map_ext, speeds, angles, local_grid_world_size = 5, resolution=100)


    def initialize_primitives(self, speeds, angles, resolution, grid_world_size):
        self.MP = MotionPrimitive(speeds, angles, resolution = (resolution, resolution), local_grid_size=grid_world_size)


    def plan(self, x, y, theta):


        local_grid = LocalField.sample_map_obstacles(x,y,theta, self.map_img, self.local_grid_world_size, self.resolution, self.map_resolution, self.map_width, self.map_height, self.origin)

        time_thresh = .3
        dynamic_scaling_bonus = 5

        dynamic_risk = self.get_dynamic_risks(x, y, theta, time_thresh)*dynamic_scaling_bonus

        risk = self.get_risks(local_grid)

        cost = risk+dynamic_risk

        control_choice = torch.argmin(cost)
        speed, angle = self.MP.get_control_for(control_choice)

        return speed, angle, self.MP.primitives[control_choice]


