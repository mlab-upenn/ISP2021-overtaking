
import torch
import numpy as np
from matplotlib import pyplot as plt
from .MotionPrimitiveSuper import MotionPrimitiveSuper


class MotionPrimitive(MotionPrimitiveSuper):
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
        super().__init__( speed_list, steering_list, L=L, p=p, t_la=t_la, k1=k1, k2=k2, k3=k3, m=m, c=c, local_grid_size = local_grid_size, resolution=resolution)


    def create_primitives(self):
        primitives = torch.zeros((torch.numel(self.speeds), self.resolution[0] + 1, self.resolution[1] + 1))

        turn_mask = torch.abs(self.steering_angles) > 0.01
        straight_mask = torch.abs(self.steering_angles) <= 0.01

        straight_sig = self.get_sig(self.speeds[straight_mask], self.steering_angles[straight_mask], True)
        straight_a = self.get_a(self.speeds[straight_mask], self.steering_angles[straight_mask], True)

        turn_sig = self.get_sig(self.speeds[turn_mask], self.steering_angles[turn_mask], False)
        turn_a = self.get_a(self.speeds[turn_mask], self.steering_angles[turn_mask], False)

        print(turn_a)

        if (torch.any(straight_mask)):
            primitives[straight_mask] = straight_a * torch.exp(-(self.y ** 2) / (2 * straight_sig ** 2))

        # handle turn case:
        if (torch.any(turn_mask)):
            R = self.get_R(self.speeds[turn_mask], self.steering_angles[turn_mask]).view(-1, 1, 1)
            primitives[turn_mask] = turn_a * torch.exp(
                -(torch.sqrt(self.x ** 2 + (self.y - R) ** 2) - torch.abs(R)) ** 2 / (2 * turn_sig ** 2))

        self.primitives = primitives/torch.max(primitives)

    def get_a(self, speed, steering, straight):
        #a should be (N[straight/turn], res[0], res[1])
        s = self.get_s(speed,steering, straight)

        a = self.p*(torch.clamp_min(speed.view(-1,1,1)*self.t_la - s, 0))
        # a = self.p*(torch.clamp_max(speed.view(-1,1,1)*self.t_la - s, 0))


        #this is to keep tuning consistent, negates the impact of p
        normalize = self.p*(speed.view(-1,1,1)*self.t_la)

        return a#/normalize


    def get_s(self, speed, steering, straight):
        #s should be (N[straight/turn], res[0], res[1])

        if(straight):
            return self.x
        else:
            R = self.get_R(speed,steering).view(-1,1,1)
            # radius of a point in relation to turn center times sweep angle gives arc length
            #not clear which formulation is best
            # return torch.sqrt(self.x.unsqueeze(0)**2 + (R-self.y.unsqueeze(0))**2)*torch.atan2(self.x.unsqueeze(0), R-self.y.unsqueeze(0))
            # return torch.abs(R)*torch.min(torch.atan2(self.x.unsqueeze(0), self.y.unsqueeze(0)+R) , torch.atan2(self.x.unsqueeze(0), self.y.unsqueeze(0)-R))
            return torch.abs(R) * torch.atan2(self.x, (R - self.y) * torch.sign(R))
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

    def get_control_for(self, primitive_number):
        return self.speeds[primitive_number], self.steering_angles[primitive_number]

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


        


