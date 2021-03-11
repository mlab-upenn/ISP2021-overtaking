
import yaml
from PIL import Image
import numpy as np
import torch
from matplotlib import pyplot as plt
from ..MotionPrimitives.MotionPrimitives import MotionPrimitive

from ..Util import LocalField


class PureRiskController():

    def __init__(self, map_path, map_ext, local_grid_world_size = 5, resolution=100):
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

        speeds = [i for i in torch.arange(3.0,3.01,.3)]
        angles = [i for i in torch.arange(-.3,.3001, .005)]

        self.MP = MotionPrimitive(speeds, angles, resolution = (resolution, resolution), local_grid_size=local_grid_world_size)

    def plan(self, x, y, theta):


        local_grid = LocalField.sample_map_obstacles(x,y,theta, self.map_img, self.local_grid_world_size, self.resolution, self.map_resolution, self.map_width, self.map_height, self.origin)



        # print(self.MP.primitives.shape, local_grid.shape)

        cost = 10000* torch.sum(self.MP.primitives.transpose(1,2)*local_grid, dim=(1,2)) / (torch.sum(self.MP.primitives,dim=(1,2)) * self.resolution * self.resolution)

        cutoff = 1000

        legal_mask = cost<cutoff



        cost = cost[legal_mask]
        control_choice = torch.argmin(cost - self.MP.speeds[legal_mask])

        speed = self.MP.speeds[legal_mask][control_choice]
        angle = self.MP.steering_angles[legal_mask][control_choice]

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

        return speed, angle, self.MP.primitives[legal_mask][control_choice]


