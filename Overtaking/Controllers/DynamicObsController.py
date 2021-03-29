import torch
import numpy as np
from matplotlib import pyplot as plt
from Overtaking.MotionPrimitives.MotionPrimitives import MotionPrimitive
from Overtaking.Controllers.Controller import Controller
from Overtaking.Util.LocalField import sample_against_data


class DynamicObstacleController(Controller):
    def __init__(self, f1map, local_grid_size=5, resolution=70):
        super().__init__(f1map, local_grid_size, resolution)

        # construct motion primitive library
        speeds = [i for i in torch.arange(4.0,4.01,.3)]
        angles = [i for i in torch.arange(-.28,.2801, .005)]
        self.MP = MotionPrimitive(speeds, 
                                  angles, 
                                  resolution =(resolution, resolution), 
                                  local_grid_size=local_grid_size)

    def update_opponent_data(self, pose, plan, time_field, speed, turn, local_size):
        self.opp_pose = pose
        self.opp_plan = plan
        self.opp_time_field = time_field
        self.opp_speed = speed
        self.opp_turn = turn
        self.opp_local_size = local_size


    def plan(self, pose):


        local_grid = self.map.sample_obstacles(pose, self.local_grid_size, self.resolution)

        #sample from opponent time and plan fields
        sampled_opp_time_field = sample_against_data(self.opp_time_field.unsqueeze(0).unsqueeze(0), self.resolution, 1, self.opp_local_size, pose - self.opp_pose, -np.Inf)
        sampled_opp_plan_field = sample_against_data(self.opp_plan.unsqueeze(0).unsqueeze(0), self.resolution, 1, self.opp_local_size, pose - self.opp_pose, 0)

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


