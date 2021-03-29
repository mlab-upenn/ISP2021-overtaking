import yaml
from PIL import Image
import numpy as np
import torch
from matplotlib import pyplot as plt
from ..MotionPrimitives.MotionPrimitives import MotionPrimitive

from ..Util import LocalField


class PrimitiveBasedControllerSuper():

    def __init__(self, map_path, map_ext, speeds, angles, local_grid_world_size=5, resolution=70):
        ## from renderer.py in f1tenth gym for loading map image
        with open(map_path + '.yaml', 'r') as yaml_stream:
            try:
                map_metadata = yaml.safe_load(yaml_stream)
                self.map_resolution = map_metadata['resolution']
                self.origin = map_metadata['origin']
            except yaml.YAMLError as ex:
                print(ex)

        # load map image
        self.map_img = torch.tensor(
            np.array(Image.open(map_path + map_ext).transpose(Image.FLIP_TOP_BOTTOM)).astype(np.float64)).unsqueeze(
            0).unsqueeze(0)
        self.map_height = self.map_img.shape[2]
        self.map_width = self.map_img.shape[3]
        print(self.origin)
        ##

        self.resolution = resolution
        self.local_grid_world_size = local_grid_world_size

        self.initialize_primitives(speeds, angles, resolution, local_grid_world_size)




    def update_opponent_data(self, x, y, theta, plan, time_field, speed, turn, local_size):
        self.opp_x = x
        self.opp_y = y
        self.opp_theta = theta
        self.opp_plan = plan
        self.opp_time_field = time_field
        self.opp_speed = speed
        self.opp_turn = turn
        self.opp_local_size = local_size

    def plan(self, x, y, theta):
        pass
    #return speed, angle, primitive plan

    def initialize_primitives(self, speeds, angles, resolution, grid_world_size):
        self.MP = MotionPrimitive(speeds, angles, resolution=(resolution, resolution),
                                  local_grid_size=grid_world_size)

    def get_risks(self, local_grid):
        risk = torch.sum(self.MP.primitives.transpose(1,2)*local_grid, dim=(1,2)) / (torch.sum(self.MP.primitives,dim=(1,2)))
        return risk

    def get_dynamic_risks(self, x, y, theta, time_threshold):
        sampled_opp_time_field = LocalField.sample_against_data(self.opp_time_field.unsqueeze(0).unsqueeze(0),
                                                                self.resolution, 1, self.opp_local_size,
                                                                torch.tensor([x - self.opp_x, y - self.opp_y]),
                                                                theta - self.opp_theta, -100)
        sampled_opp_plan_field = LocalField.sample_against_data(self.opp_plan.unsqueeze(0).unsqueeze(0),
                                                                self.resolution, 1, self.opp_local_size,
                                                                torch.tensor([x - self.opp_x, y - self.opp_y]),
                                                                theta - self.opp_theta, 0)

        ego_time_field = self.MP.time_field / self.MP.speeds.reshape((-1, 1, 1))

        time_mask = torch.abs(ego_time_field - sampled_opp_time_field) < time_threshold
        intersection = self.MP.primitives * time_mask.repeat((self.MP.primitives.shape[0]//time_mask.shape[0],1,1)) * sampled_opp_plan_field.repeat(
            (self.MP.primitives.shape[0], 1, 1))

        dynamic_risk = torch.sum(intersection, dim=(1, 2))

        return dynamic_risk

