import torch
import numpy as np
from matplotlib import pyplot as plt


# Superclass for defining different motion primitive classes
class MotionPrimitiveSuper:
    def __init__(self, speeds, steering_angles, L=.33, p=1, t_la=2.5, k1=.2, k2=.3, k3=.1, m=.1, c=.12, local_grid_size = 7, resolution=(50, 50)):
        # speed and steering are lists of speed and steering values
        #L : wheelbase of the car
        # p : not currently used
        # t_la : lookahead time
        # k1 : interior std scaling with steering input
        # k2 : exterior std scaling with steering input
        # k3 : exterior std scaling with speed
        # m : path length std scaling
        # c : width of the car
        # local_grid_size : size of the prediction area in front of the car
        # resolution ; resolution of the prediction area

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

        if(torch.cuda.is_available()):
            self.device = torch.device('cuda:0')

        else:
            self.device = torch.device('cpu')



        #x goes from 0 to bounds, y goes from -bounds to bounds
        self.x, self.y = torch.meshgrid(torch.arange(0, local_grid_size+.0001, local_grid_size/resolution[0]), torch.arange(-local_grid_size/2,local_grid_size/2+.0001, local_grid_size/resolution[1]))

        self.speeds = speeds
        self.steering_angles = steering_angles

        self.primitives = self.create_primitives()

        self.time_field = self.create_time_field()


    # calculates a for each point in the local grid (height scale of risk)
    def get_a(self, speed, steering, grid_x, grid_y, arc_length, straight):
        # input :
        # speed : 1d tensor of speeds
        # steering : 1d tensor of steering angles
        # grid_x : 2d tensor of x values at each point
        # grid_y : 2d tensor of y values at each point
        # arc_length : 1d tensor of the distance already traveled in each primitive
        # straight : boolean of whether or not these primitives are moving straight (this should be eliminated)

        # output :
        # a : tenosr of (N[straight/turn], res[0]+1, res[1]+1)

        s = self.get_s(speed, steering, grid_x, grid_y, arc_length, straight)

        a = (torch.clamp_min(1 - self.p * s, 0))

        a = torch.clamp_min(a ,0)
        a[s-arc_length.reshape(-1,1,1)<0] = 0

        a[(s-arc_length.reshape(-1,1,1))/speed.view(-1, 1, 1) > self.t_la] = 0
        return a

    def get_s(self, speed, steering, grid_x, grid_y, arc_length, straight):
        # input :
        # speed : 1d tensor of speeds
        # steering : 1d tensor of steering angles
        # grid_x : 2d tensor of x values at each point
        # grid_y : 2d tensor of y values at each point
        # arc_length : 1d tensor of the distance already traveled in each primitive
        # straight : boolean of whether or not these primitives are moving straight (this should be eliminated)

        # output :
        # s : tensor of (N[straight/turn], res[0], res[1])

        if(straight):
            return grid_x + arc_length.reshape(-1,1,1)
        else:
            R = self.get_R(speed,steering).view(-1,1,1)
            # radius of a point in relation to turn center times sweep angle gives arc length
            #not clear which formulation is best

            return torch.abs(R)*torch.atan2(grid_x, (R-grid_y)*torch.sign(R))+ arc_length.reshape(-1,1,1)


    def get_sig(self, speed, steering, grid_x, grid_y, arc_length, straight):
        # input :
        # speed : 1d tensor of speeds
        # steering : 1d tensor of steering angles
        # grid_x : 2d tensor of x values at each point
        # grid_y : 2d tensor of y values at each point
        # arc_length : 1d tensor of the distance already traveled in each primitive
        # straight : boolean of whether or not these primitives are moving straight (this should be eliminated)

        # output :
        # sig : tenosr of (N[straight/turn], res[0]+1, res[1]+1)

        s = self.get_s(speed, steering, grid_x, grid_y, arc_length, straight) #
        R = self.get_R(speed,steering).view(-1,1,1)
        real_R = torch.sqrt(grid_x.unsqueeze(0)**2 + (R-grid_y.unsqueeze(0))**2)
        k = torch.where(torch.abs(R)>real_R, torch.tensor(self.k1).view(-1,1,1), self.k2 + self.k3*speed.view(-1,1,1))
        sig = (self.m + k*torch.abs(steering.view(-1,1,1)))*s + self.c

        return sig

    # returns the radius of curvature for the associated
    def get_R(self, speed, steering):
        # only valid for steering[turn_mask]
        R = self.L / torch.tan(steering)
        return R

    # generates distance field (scale by inverse speed to get time field)
    def create_time_field(self):
        r = (self.x**2 + self.y**2)/(2*torch.abs(self.y))

        r = torch.clamp(r, max = 100)
        s = r*torch.atan2(self.x, r - torch.abs(self.y))

        # s[torch.abs(r) < .5] = -100

        return s


    # abstract methods for subclasses to customize behaviour
    def create_primitives(self):
        #self.primitives =
        raise NotImplementedError("create primitives method not defined, subclass needs to implement this method")

    def get_control_for(self, primitive_number):
        raise NotImplementedError("get_control_for method not defined, subclass needs to implement")
