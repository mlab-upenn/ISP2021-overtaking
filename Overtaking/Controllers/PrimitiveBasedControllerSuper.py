import yaml
from PIL import Image
import numpy as np
import torch
from matplotlib import pyplot as plt
from ..MotionPrimitives.MotionPrimitives import MotionPrimitive

from ..Util import LocalField


class PrimitiveBasedControllerSuper():

    def __init__(self, f1map, speeds, angles, local_grid_size=5, resolution=70):
        self.map = f1map
        self.resolution = resolution
        self.local_grid_size = local_grid_size

        self.initialize_primitives(speeds, angles)

    def update_opponent_data(self, pose, plan, time_field, speed, turn, local_size):
        self.opp_pose = pose
        self.opp_plan = plan
        self.opp_time_field = time_field
        self.opp_speed = speed
        self.opp_turn = turn
        self.opp_local_size = local_size

    def plan(self, x, y, theta):
        pass
    #return speed, angle, primitive plan

    def initialize_primitives(self, speeds, angles):
        self.MP = MotionPrimitive(speeds, angles, resolution=(self.resolution, self.resolution),
                                  local_grid_size=self.local_grid_size)

    def get_risks(self, local_obstacles):
        risk = torch.sum(self.MP.primitives.transpose(1,2)*local_obstacles, dim=(1,2)) / (torch.sum(self.MP.primitives,dim=(1,2)))
        return risk

    def get_rewards(self, local_rewards):
        reward = torch.sum(self.MP.primitives.transpose(1,2)*local_rewards, dim=(1,2)) / (torch.sum(self.MP.primitives,dim=(1,2)))

        return reward

    def get_dynamic_risks(self, pose, time_threshold):
        sampled_opp_time_field = LocalField.sample_against_data(self.opp_time_field.unsqueeze(0).unsqueeze(0),
                                                                self.resolution, 1, self.opp_local_size,
                                                                pose-self.opp_pose, -np.Inf)
        sampled_opp_plan_field = LocalField.sample_against_data(self.opp_plan.unsqueeze(0).unsqueeze(0),
                                                                self.resolution, 1, self.opp_local_size,
                                                                pose-self.opp_pose, 0)

        ego_time_field = self.MP.time_field / self.MP.speeds.reshape((-1, 1, 1))

        time_mask = torch.abs(ego_time_field - sampled_opp_time_field) < time_threshold
        intersection = self.MP.primitives * time_mask.repeat((self.MP.primitives.shape[0]//time_mask.shape[0],1,1)) * sampled_opp_plan_field.repeat(
            (self.MP.primitives.shape[0], 1, 1))

        dynamic_risk = torch.sum(intersection, dim=(1, 2))

        return dynamic_risk

