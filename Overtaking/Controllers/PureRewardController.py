import torch
import numpy as np
from matplotlib import pyplot as plt
from ..MotionPrimitives.MotionPrimitives import MotionPrimitive
from .PrimitiveBasedControllerSuper import PrimitiveBasedControllerSuper
from ..Util import LocalField


class PureRewardController(PrimitiveBasedControllerSuper):

    def __init__(self, f1map, speeds, angles, local_grid_world_size = 5, resolution=100):
        ## from renderer.py in f1tenth gym for loading map image
        super().__init__( f1map, speeds, angles, local_grid_size = 5, resolution=100)


    def initialize_primitives(self, speeds, angles):
        self.MP = MotionPrimitive(speeds, angles,L=.33, p=1, t_la=2.5, k1=.2, k2=.3, k3=.1, m=.1, c=.12, resolution = (self.resolution, self.resolution), local_grid_size=self.local_grid_size)

    def plan(self, pose):
        local_reward = self.map.sample_reward(pose, self.local_grid_size, self.resolution)
        cur_reward = self.map.sample_reward_at_pose(pose, self.local_grid_size)
        reward = self.get_rewards(local_reward- cur_reward)
        print('current reward: ', cur_reward)

        reward[reward < 0] += torch.max(reward)

        print(torch.max(reward))

        control_choice = torch.argmax(reward)
        speed, angle = self.MP.get_control_for(control_choice)



        return speed, angle, self.MP.primitives[control_choice]


