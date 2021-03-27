
import yaml
from PIL import Image
import numpy as np
import torch
from matplotlib import pyplot as plt
from ..MotionPrimitives.MotionPrimitives import MotionPrimitive

from ..Util import LocalField


class DynamicObstacleController():

    def __init__(self, map_path, map_ext, local_grid_world_size = 5, resolution=70):
        ## from renderer.py in f1tenth gym for loading map image
        with open(map_path + '.yaml', 'r') as yaml_stream:
            try:
                map_metadata = yaml.safe_load(yaml_stream)
                self.map_resolution = map_metadata['resolution']
                self.origin = map_metadata['origin']
            except yaml.YAMLError as ex:
                print(ex)

        # load map image
        self.map_img = torch.tensor(np.array(Image.open(map_path + map_ext).transpose(Image.FLIP_TOP_BOTTOM)).astype(np.float64)).unsqueeze(0).unsqueeze(0)
        self.map_height = self.map_img.shape[2]
        self.map_width = self.map_img.shape[3]
        print(self.origin)
        ##

        self.resolution = resolution
        self.local_grid_world_size = local_grid_world_size

        speeds = [i for i in torch.arange(4.0,4.01,.3)]
        angles = [i for i in torch.arange(-.28,.2801, .005)]

        self.MP = MotionPrimitive(speeds, angles, resolution = (resolution, resolution), local_grid_size=local_grid_world_size)


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


        local_grid = LocalField.sample_map_obstacles(x,y,theta, self.map_img, self.local_grid_world_size, self.resolution, self.map_resolution, self.map_width, self.map_height, self.origin)

        #sample from opponent time and plan fields
        sampled_opp_time_field = LocalField.sample_against_data(self.opp_time_field.unsqueeze(0).unsqueeze(0), self.resolution, 1, self.opp_local_size, torch.tensor([x-self.opp_x, y-self.opp_y]), theta-self.opp_theta, -100)
        sampled_opp_plan_field = LocalField.sample_against_data(self.opp_plan.unsqueeze(0).unsqueeze(0), self.resolution, 1, self.opp_local_size, torch.tensor([x-self.opp_x, y-self.opp_y]), theta-self.opp_theta, 0)

        ego_time_field = self.MP.time_field / self.MP.speeds.reshape((-1,1,1))

        time_thresh = .3

        dynamic_scaling_bonus = 5

        time_mask = torch.abs(ego_time_field - sampled_opp_time_field) < time_thresh


        intersection = self.MP.primitives*time_mask* sampled_opp_plan_field.repeat((self.MP.primitives.shape[0], 1, 1))

        dynamic_risk = torch.sum(intersection, dim = (1,2))*dynamic_scaling_bonus

        normalization = torch.sum(self.MP.primitives,dim=(1,2)) * self.resolution * self.resolution

        dynamic_risk = dynamic_risk/normalization

        cost = torch.sum(self.MP.primitives.transpose(1,2)*local_grid, dim=(1,2)) / normalization

        cost = cost+dynamic_risk

        control_choice = torch.argmin(cost - self.MP.speeds)

        speed = self.MP.speeds[control_choice]
        angle = self.MP.steering_angles[control_choice]

        

        # print(cost[control_choice])



        # print(speed,angle)
        # print(self.MP.primitives[control_choice])
        # print(cost[control_choice])
        # print(local_grid)
        #
        # plt.imshow(self.MP.primitives[control_choice].transpose(0,1))
        # plt.colorbar()
        #
        # plt.figure()
        # plt.imshow(local_grid)
        # plt.colorbar()
        # #
        # plt.show()

        return speed, angle, self.MP.primitives[control_choice]


