import torch
import numpy as np
from matplotlib import pyplot as plt



class MotionPrimitiveSuper:
    def __init__(self, speed_list, steering_list, L=.33, p=1, t_la=2.5, k1=.2, k2=.3, k3=.1, m=.1, c=.12, local_grid_size = 7, resolution=(50, 50)):
        # speed and steering are lists of speed and steering values
        #L : wheelbase of the car
        # p : not currently used
        # t_la : lookahead time
        # k1 : interior std scaling with steering input
        # k2 : exterior std scaling with steering input
        # k3 : exterior std scaling with speed
        # m : path length std scaling
        # c : width of the car

        self.L = L
        self.p = p
        self.t_la = t_la
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
        self.m = m
        self.c = c
        self.local_grid_size = local_grid_size
        self.resolution = resolution


        #x goes from 0 to bounds, y goes from -bounds to bounds
        self.x, self.y = torch.meshgrid(torch.arange(0, local_grid_size+.0001, local_grid_size/resolution[0]), torch.arange(-local_grid_size/2,local_grid_size/2+.0001, local_grid_size/resolution[1]))
        # self.x = self.x-.3
        speed_vals = torch.tensor(speed_list)
        steering_vals = torch.tensor(steering_list)

        speed, steering = torch.meshgrid(speed_vals, steering_vals)


        speed = torch.flatten(speed)
        steering = torch.flatten(steering)

        self.speeds = speed
        self.steering_angles = steering

        self.create_primitives()

        self.time_field = self.create_time_field()

    def create_time_field(self):
        r = (self.x**2 + self.y**2)/(2*torch.abs(self.y))

        r = torch.clamp(r, max = 100)
        s = r*torch.atan2(self.x, r - torch.abs(self.y))

        # s[torch.abs(r) < .5] = -100

        return s

    def create_primitives(self):
        #self.primitives =
        raise NotImplementedError("create primitives method not defined, subclass needs to implement this method")

    def get_control_for(self, primitive_number):
        raise NotImplementedError("get_control_for method not defined, subclass needs to implement")
