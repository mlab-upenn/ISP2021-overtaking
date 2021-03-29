import torch
import numpy as np
from matplotlib import pyplot as plt
from ..MotionPrimitives.MotionPrimitives import MotionPrimitive
from .PrimitiveBasedControllerSuper import PrimitiveBasedControllerSuper
from ..Util import LocalField


class PureRiskController(PrimitiveBasedControllerSuper):

    def __init__(self, f1map, speeds, angles, local_grid_world_size = 5, resolution=100):
        ## from renderer.py in f1tenth gym for loading map image
        super().__init__( f1map, speeds, angles, local_grid_size = 5, resolution=100)


    def initialize_primitives(self, speeds, angles):
        self.MP = MotionPrimitive(speeds, angles,L=.33, p=1, t_la=2.5, k1=.2, k2=.3, k3=.1, m=.1, c=.12, resolution = (self.resolution, self.resolution), local_grid_size=self.local_grid_size)


    def plan(self, pose):
        local_grid = self.map.sample_obstacles(pose, self.local_grid_size, self.resolution)

        risk = self.get_risks(local_grid)

        control_choice = torch.argmin(risk)
        speed, angle = self.MP.get_control_for(control_choice)



        return speed, angle, self.MP.primitives[control_choice]


