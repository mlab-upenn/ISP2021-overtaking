
import yaml
from PIL import Image
import numpy as np
import torch
from matplotlib import pyplot as plt
from ..MotionPrimitives.TreeMotionPrimitives import TreeMotionPrimitive
from .PrimitiveBasedControllerSuper import PrimitiveBasedControllerSuper
from ..Util import LocalField


class TreePrimitiveController(PrimitiveBasedControllerSuper):

    def __init__(self, map_path, map_ext, depth, speeds, angles, local_grid_world_size = 5, resolution=100):
        ## from renderer.py in f1tenth gym for loading map image
        self.depth = depth

        super().__init__( map_path, map_ext, speeds, angles, local_grid_world_size=5, resolution=100)

    def initialize_primitives(self, speeds, angles, resolution, grid_world_size):
        self.MP = TreeMotionPrimitive(speeds, angles, depth=self.depth, L=.33, p=.1, t_la=.2, k1=.0, k2=.0, k3=.0, m=.1, c=.12, resolution = (resolution, resolution), local_grid_size=grid_world_size)


    def plan(self, pose):
        local_obstacles = self.map.sample_obstacles(pose, self.local_grid_size, self.resolution)

        time_thresh = .1
        dyn_risk = self.get_dynamic_risks(pose, time_thresh)

        risk = self.get_risks(local_obstacles)

        dyn_risk_scale = 1
        cost = risk+dyn_risk*dyn_risk_scale
        control_choice = torch.argmin(cost)
        speed, angle = self.MP.get_control_for(control_choice)


        return speed, angle, self.MP.primitives[control_choice]


