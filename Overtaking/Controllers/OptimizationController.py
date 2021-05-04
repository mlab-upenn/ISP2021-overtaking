
import yaml
from PIL import Image
import numpy as np
import torch
from matplotlib import pyplot as plt
from ..MotionPrimitives.TreeMotionPrimitives import TreeMotionPrimitive
from .PrimitiveBasedControllerSuper import PrimitiveBasedControllerSuper
from ..MotionPrimitives.ExplicitPrimitiveSet import ExplicitPrimitiveSet
from ..Util import LocalField


class OptimizationController(PrimitiveBasedControllerSuper):

    def __init__(self, f1map, speed_starting_set, steering_angle_starting_set, lr = .01, max_iter = 2,  static_risk_factor=1, dynamic_risk_factor=0, reward_factor=0, dynamic_risk_thresh = .15, local_grid_world_size = 10, resolution=120):
        ## from renderer.py in f1tenth gym for loading map image
        self.lr = lr
        self.max_iter = max_iter


        self.MP = ExplicitPrimitiveSet(torch.tensor(speed_starting_set,requires_grad=True),torch.tensor(steering_angle_starting_set,requires_grad=True), local_grid_size=local_grid_world_size, resolution=(resolution,resolution))
        self.static_risk_factor = torch.tensor(static_risk_factor)
        self.dynamic_risk_factor = torch.tensor(dynamic_risk_factor)
        self.reward_factor = torch.tensor(reward_factor)
        self.dynamic_risk_thresh = torch.tensor(dynamic_risk_thresh)
        super().__init__( f1map, None, None, local_grid_world_size, resolution)

    def initialize_primitives(self, speeds, angles):
        pass


    def plan(self, pose):

        pose = pose.to(self.device)
        cost = 0
        optim = torch.optim.SGD([self.MP.speeds, self.MP.steering_angles], lr= self.lr)
        for i in range(self.max_iter):
            optim.zero_grad()
            if(self.static_risk_factor > 0):

                local_obstacles = self.map.sample_obstacles(pose, self.local_grid_size, self.resolution)
                cost += self.get_risks(local_obstacles)*self.static_risk_factor

            if(self.dynamic_risk_factor > 0):

                # time_thresh = .17
                dyn_risk, opp_plan = self.get_dynamic_risks(pose, self.dynamic_risk_thresh)

                cost += dyn_risk*self.dynamic_risk_factor

            if(self.reward_factor > 0):
                reward_at_point = self.map.sample_reward_at_pose(pose, self.local_grid_size)
                rewards = self.map.sample_reward(pose, self.local_grid_size, self.resolution)
                # print('max at point: ', reward_at_point, 'max of rewards:', torch.max(rewards))
                # print('max diff: ', torch.max(rewards-reward_at_point))
                reward = self.get_rewards((rewards - reward_at_point))*self.reward_factor
                # print('integrated max reward: ', torch.max(reward), ' reward factor: ', self.reward_factor)

                cost -= reward

            total_cost= cost.sum()
            # total_cost.backward()
            # optim.step()


            #regenerate primitives with updated values
            self.MP.create_primitives()


        control_choice = torch.argmin(cost)
        speed, angle = self.MP.get_control_for(control_choice)

        if(self.reward_factor>0):
            print('chosen reward: ', reward[control_choice], 'highest reward: ', torch.max(reward))
        if(self.dynamic_risk_factor>0):
            print('chosen dynamic risk: ', dyn_risk[control_choice], 'lowest risk: ', torch.min(dyn_risk))


        return speed, angle, self.MP.primitives[control_choice]


