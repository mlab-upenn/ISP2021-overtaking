import torch
import numpy as np
from matplotlib import pyplot as plt
from Overtaking.MotionPrimitives.MotionPrimitives import MotionPrimitive
from Overtaking.Controllers.Controller import Controller
from Overtaking.Util import LocalField


class PureRiskController(Controller):
    def __init__(self, f1map, local_grid_size=5, resolution=100):
        super().__init__(f1map, local_grid_size, resolution)

        # construct motion primitive library
        speeds = [i for i in torch.arange(3.0, 3.01, .3)]
        angles = [i for i in torch.arange(-.3, .3001, .005)]
        self.MP = MotionPrimitive(speeds, angles, L=.33, p=1, t_la=2.5, k1=.2, k2=.3, k3=.1, m=.1, c=.12, resolution =(resolution, resolution), local_grid_size=local_grid_size)

    def plan(self, pose):
        local_grid = self.map.sample_obstacles(pose, self.local_grid_size, self.resolution)

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


