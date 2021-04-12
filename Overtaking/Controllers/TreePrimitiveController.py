
import yaml
from PIL import Image
import numpy as np
import torch
from matplotlib import pyplot as plt
from ..MotionPrimitives.TreeMotionPrimitives import TreeMotionPrimitive
from .PrimitiveBasedControllerSuper import PrimitiveBasedControllerSuper
from ..Util import LocalField


class TreePrimitiveController(PrimitiveBasedControllerSuper):

    def __init__(self, f1map, depth, speeds, angles, local_grid_world_size = 10, resolution=120):
        ## from renderer.py in f1tenth gym for loading map image
        self.depth = depth


        super().__init__( f1map, speeds, angles, local_grid_world_size, resolution)

    def initialize_primitives(self, speeds, angles):
        self.MP = TreeMotionPrimitive(speeds, angles, depth=self.depth, L=.33, p=.1, t_la=1, k1=.0, k2=.0, k3=.0, m=.1, c=.12, resolution = (self.resolution, self.resolution), local_grid_size=self.local_grid_size)


    def plan(self, pose):
        local_obstacles = self.map.sample_obstacles(pose, self.local_grid_size, self.resolution)

        time_thresh = .17
        dyn_risk, opp_plan = self.get_dynamic_risks(pose, time_thresh)
        reward = self.get_rewards(self.map.sample_reward(pose, self.local_grid_size, self.resolution))
        risk = self.get_risks(local_obstacles)

        dyn_risk_scale = .05

        local_obstacles_scale = 10



        cost = risk*local_obstacles_scale+dyn_risk*dyn_risk_scale

        cost_scale = 1

        cost = cost*cost_scale-reward
        control_choice = torch.argmin(cost)
        print(reward[control_choice])
        speed, angle = self.MP.get_control_for(control_choice)


        return speed, angle, self.MP.primitives[control_choice]


