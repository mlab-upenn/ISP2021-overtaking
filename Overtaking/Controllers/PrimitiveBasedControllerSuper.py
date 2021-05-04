import yaml
from PIL import Image
import numpy as np
import torch
from matplotlib import pyplot as plt
from ..MotionPrimitives.MotionPrimitives import MotionPrimitive

from ..Util import LocalField


class PrimitiveBasedControllerSuper():

    def __init__(self, f1map, speeds, angles, local_grid_size=5, resolution=70):

        if(torch.cuda.is_available()):
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')

        self.map = f1map
        self.resolution = torch.tensor(resolution)
        self.local_grid_size = torch.tensor(local_grid_size)

        self.initialize_primitives(speeds, angles)

        self.opp_time_field =None

    def update_opponent_data(self, pose, plan, time_field, speed, turn, local_size):
        self.opp_pose = pose.to(self.device)
        self.opp_plan = plan.to(self.device)
        self.opp_time_field = time_field.to(self.device)
        self.opp_speed = speed.to(self.device)
        self.opp_turn = turn.to(self.device)
        self.opp_local_size = torch.tensor(local_size)


    def plan(self, x, y, theta):
        pass
    #return speed, angle, primitive plan

    def initialize_primitives(self, speeds, angles):
        self.MP = MotionPrimitive(speeds, angles, resolution=(self.resolution, self.resolution),
                                  local_grid_size=self.local_grid_size)

    def get_risks(self, local_obstacles):
        risk = torch.sum(self.MP.primitives.transpose(1,2)*local_obstacles, dim=(1,2)) / (torch.sum(self.MP.primitives,dim=(1,2)))
        print('primitives: ')
        return risk

    def get_rewards(self, local_rewards):
        reward = torch.sum(self.MP.primitives.transpose(1,2)*local_rewards, dim=(1,2)) / (torch.sum(self.MP.primitives,dim=(1,2)))
        return reward

    def get_dynamic_risks(self, pose, time_threshold):
        if(self.opp_time_field is None):
            print('no opponent data')
            return 0
        sampled_opp_time_field = LocalField.sample_against_data2(self.opp_time_field.transpose(0,1).unsqueeze(0).unsqueeze(0),
                                                                self.resolution, self.local_grid_size, self.opp_local_size,
                                                                pose, self.opp_pose, -np.Inf)
        #need to transpose because primitives have transposed axis compared to car coordinates
        sampled_opp_plan_field = LocalField.sample_against_data2(self.opp_plan.transpose(0,1).unsqueeze(0).unsqueeze(0),
                                                                self.resolution, self.local_grid_size, self.opp_local_size,
                                                                pose, self.opp_pose, 0)
        local_speeds = self.MP.speeds if len(self.MP.speeds.shape)<2 else self.MP.speeds[:,0]
        ego_time_field = self.MP.time_field / local_speeds.reshape((-1, 1, 1))

        time_mask = torch.abs(ego_time_field - sampled_opp_time_field) < time_threshold
        intersection = self.MP.primitives* time_mask.repeat((self.MP.primitives.shape[0]//time_mask.shape[0],1,1)) * sampled_opp_plan_field.repeat(
            (self.MP.primitives.shape[0], 1, 1))

        dynamic_risk = torch.sum(intersection, dim=(1, 2))

        return dynamic_risk, sampled_opp_plan_field

