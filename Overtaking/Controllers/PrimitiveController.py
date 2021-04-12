

import yaml
from PIL import Image
import numpy as np
import torch
from matplotlib import pyplot as plt
from ..MotionPrimitives.TreeMotionPrimitives import TreeMotionPrimitive
from .PrimitiveBasedControllerSuper import PrimitiveBasedControllerSuper
from ..Util import LocalField


class PrimitiveController(PrimitiveBasedControllerSuper):

    def __init__(self, f1map, primitive_class, static_risk_factor=1, dynamic_risk_factor=0, reward_factor=0, dynamic_risk_thresh = .15, local_grid_world_size = 10, resolution=120):
        ## from renderer.py in f1tenth gym for loading map image
        self.MP = primitive_class
        self.static_risk_factor = static_risk_factor
        self.dynamic_risk_factor = dynamic_risk_factor
        self.reward_factor = reward_factor
        self.dynamic_risk_thresh = dynamic_risk_thresh
        super().__init__( f1map, None, None, local_grid_world_size, resolution)

    def initialize_primitives(self, speeds, angles):
        pass


    def plan(self, pose):
        cost = 0
        if(self.static_risk_factor > 0):

            local_obstacles = self.map.sample_obstacles(pose, self.local_grid_size, self.resolution)
            cost += self.get_risks(local_obstacles)*self.static_risk_factor

        if(self.dynamic_risk_factor > 0):

            # time_thresh = .17
            dyn_risk, opp_plan = self.get_dynamic_risks(pose, self.dynamic_risk_thresh)
            cost += dyn_risk*self.dynamic_risk_factor

        if(self.reward_factor > 0):
            reward = self.get_rewards(self.map.sample_reward(pose, self.local_grid_size, self.resolution))
            cost -= reward

        control_choice = torch.argmin(cost)
        speed, angle = self.MP.get_control_for(control_choice)


        return speed, angle, self.MP.primitives[control_choice]


