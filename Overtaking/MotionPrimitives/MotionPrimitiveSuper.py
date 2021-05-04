import torch
import numpy as np
from matplotlib import pyplot as plt



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



    def get_a(self, speed, steering, grid_x, grid_y, arc_length, straight):
        # a should be (N[straight/turn], res[0], res[1])
        s = self.get_s(speed, steering, grid_x, grid_y, arc_length, straight)

        a = (torch.clamp_min(1 - self.p * s, 0))
        # a = self.p*(torch.clamp_max(speed.view(-1,1,1)*self.t_la - s, 0))

        # this is to keep tuning consistent, negates the impact of p
        # normalize = self.p * (speed.view(-1, 1, 1) * self.t_la)

        a = torch.clamp_min(a ,0)
        # print(a.shape, s.shape)
        a[s-arc_length.reshape(-1,1,1)<0] = 0

        a[(s-arc_length.reshape(-1,1,1))/speed.view(-1, 1, 1) > self.t_la] = 0
        return a

    def get_s(self, speed, steering, grid_x, grid_y, arc_length, straight):
        #s should be (N[straight/turn], res[0], res[1])

        if(straight):
            return grid_x + arc_length.reshape(-1,1,1)
        else:
            R = self.get_R(speed,steering).view(-1,1,1)
            # radius of a point in relation to turn center times sweep angle gives arc length
            #not clear which formulation is best

            return torch.abs(R)*torch.atan2(grid_x, (R-grid_y)*torch.sign(R))+ arc_length.reshape(-1,1,1)


    def get_sig(self, speed, steering, grid_x, grid_y, arc_length, straight):
        #sig should be (N, res[0], res[1])
        #just use K1 for now

        s = self.get_s(speed, steering, grid_x, grid_y, arc_length, straight) #
        R = self.get_R(speed,steering).view(-1,1,1)
        real_R = torch.sqrt(grid_x.unsqueeze(0)**2 + (R-grid_y.unsqueeze(0))**2)
        k = torch.where(torch.abs(R)>real_R, torch.tensor(self.k1).view(-1,1,1), self.k2 + self.k3*speed.view(-1,1,1))
        sig = (self.m + k*torch.abs(steering.view(-1,1,1)))*s + self.c

        return sig

    def get_R(self, speed, steering):
        # only valid for steering[turn_mask]
        R = self.L / torch.tan(steering)
        return R

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
