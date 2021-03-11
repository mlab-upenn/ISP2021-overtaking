
import torch
import numpy as np
from matplotlib import pyplot as plt


class MotionPrimitive:
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


        with torch.enable_grad():
            #x goes from 0 to bounds, y goes from -bounds to bounds
            self.x, self.y = torch.meshgrid(torch.arange(0, local_grid_size+.0001, local_grid_size/resolution[0]), torch.arange(-local_grid_size/2,local_grid_size/2+.0001, local_grid_size/resolution[1]))

            speed_vals = torch.tensor(speed_list)
            steering_vals = torch.tensor(steering_list)

            speed, steering = torch.meshgrid(speed_vals, steering_vals)


            speed = torch.flatten(speed)
            steering = torch.flatten(steering)

            self.speeds = speed
            self.steering_angles = steering

            primitives = torch.zeros((torch.numel(speed), resolution[0]+1, resolution[1]+1))

            turn_mask = torch.abs(steering) > 0.01
            straight_mask = torch.abs(steering) <= 0.01



            straight_sig = self.get_sig(speed[straight_mask], steering[straight_mask], True)
            straight_a = self.get_a(speed[straight_mask], steering[straight_mask], True)

            turn_sig = self.get_sig(speed[turn_mask], steering[turn_mask], False)
            turn_a = self.get_a(speed[turn_mask], steering[turn_mask], False)

            print(turn_a)

            if(torch.any(straight_mask)):
                primitives[straight_mask] = straight_a * torch.exp(-(self.y**2)/(2*straight_sig**2))

            #handle turn case:
            if(torch.any(turn_mask)):
                R = self.get_R(speed[turn_mask],steering[turn_mask]).view(-1,1,1)
                primitives[turn_mask] = turn_a * torch.exp(-(torch.sqrt(self.x**2 + (self.y - R)**2) - torch.abs(R))**2 / (2*turn_sig**2))

            self.primitives = primitives

            self.time_field = self.create_time_field()


        # self.primitives[:,2,1] = primitives[:,1,2]

    def get_a(self, speed, steering, straight):
        #a should be (N[straight/turn], res[0], res[1])
        s = self.get_s(speed,steering, straight)

        a = self.p*(torch.clamp_min(speed.view(-1,1,1)*self.t_la - s, 0))
        # a = self.p*(torch.clamp_max(speed.view(-1,1,1)*self.t_la - s, 0))


        #this is to keep tuning consistent, negates the impact of p
        normalize = self.p*(speed.view(-1,1,1)*self.t_la)
        return a/normalize


    def get_s(self, speed, steering, straight):
        #s should be (N[straight/turn], res[0], res[1])

        if(straight):
            return self.x
        else:
            R = self.get_R(speed,steering).view(-1,1,1)
            # radius of a point in relation to turn center times sweep angle gives arc length
            #not clear which formulation is best
            # return torch.sqrt(self.x.unsqueeze(0)**2 + (R-self.y.unsqueeze(0))**2)*torch.atan2(self.x.unsqueeze(0), R-self.y.unsqueeze(0))
            return torch.abs(R)*torch.min(torch.atan2(self.x.unsqueeze(0), self.y.unsqueeze(0)+R) , torch.atan2(self.x.unsqueeze(0), self.y.unsqueeze(0)-R))

        pass

    def get_sig(self, speed, steering, straight):
        #sig should be (N, res[0], res[1])
        #just use K1 for now

        s = self.get_s(speed, steering, straight)
        R = self.get_R(speed,steering).view(-1,1,1)
        real_R = torch.sqrt(self.x.unsqueeze(0)**2 + (R-self.y.unsqueeze(0))**2)
        k = torch.where(torch.abs(R)>real_R, torch.tensor(self.k1).view(-1,1,1), self.k2 + self.k3*speed.view(-1,1,1))
        sig = (self.m + k*torch.abs(steering.view(-1,1,1)))*s + self.c

        return sig

    def get_R(self, speed, steering):
        #only valid for steering[turn_mask]
        R = self.L/torch.tan(steering)
        return R

    def create_time_field(self):
        r = (self.x**2 + self.y**2)/(2*torch.abs(self.y))

        r = torch.clamp(r, max = 100)
        s = r*torch.atan2(self.x, r - torch.abs(self.y))

        # s[torch.abs(r) < .5] = -100

        return s

if __name__ == '__main__':
    import time
    t_b = time.time()
    # MP = MotionPrimitive(list(torch.arange(2,4,.05)),list(torch.arange(-.5,.5,.05)) )
    MP = MotionPrimitive([3.], [0.])

    t_a = time.time()

    # print(t_a-t_b)

    # print(MP.primitives.shape)
    plt.imshow(np.array(MP.primitives[0]))
    plt.colorbar()

    plt.show()


        


