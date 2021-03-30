import torch
import numpy as np
from matplotlib import pyplot as plt
from ..MotionPrimitives.MotionPrimitives import MotionPrimitive
from .PrimitiveBasedControllerSuper import PrimitiveBasedControllerSuper

class DynamicObstacleController(PrimitiveBasedControllerSuper):

    def __init__(self, f1map, speeds, angles, local_grid_world_size = 5, resolution=70):
        super().__init__( f1map, speeds, angles, local_grid_size = 5, resolution=100)


    def initialize_primitives(self, speeds, angles):
        self.MP = MotionPrimitive(speeds, angles, resolution = (self.resolution, self.resolution), local_grid_size=self.local_grid_size)

        # construct motion primitive library



    def plan(self, pose):


        local_obstacles = self.map.sample_obstacles(pose, self.local_grid_size, self.resolution)
        time_thresh = .3
        dynamic_scaling_bonus = 5

        dynamic_risk = self.get_dynamic_risks(pose, time_thresh)*dynamic_scaling_bonus

        risk = self.get_risks(local_obstacles)

        cost = risk+dynamic_risk

        control_choice = torch.argmin(cost)
        speed, angle = self.MP.get_control_for(control_choice)

        return speed, angle, self.MP.primitives[control_choice]


