
import torch
import numpy as np
from matplotlib import pyplot as plt
from .MotionPrimitiveSuper import MotionPrimitiveSuper
# from .MotionPrimitiveSuper import MotionPrimitiveSuper


# Class for generating full set of primitives from discrete sets of speeds and steering angles.
# Primitives assume constant speed and steering angle for t_la time.
class MotionPrimitive(MotionPrimitiveSuper):
    def __init__(self, speed_list, steering_list, L=.33, p=1, t_la=2.5, k1=.2, k2=.3, k3=.1, m=.1, c=.12, local_grid_size = 7, resolution=(50, 50)):
        # speed and steering : lists of discrete speed and steering values
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
        self.arc_lengths_local = torch.zeros(len(speed_list)*len(steering_list))

        speed_vals = torch.tensor(speed_list)
        steering_vals = torch.tensor(steering_list)

        speeds, steering_angles = torch.meshgrid(speed_vals, steering_vals)

        speeds = speeds.flatten()
        steering_angles = steering_angles.flatten()

        super().__init__( speeds, steering_angles, L=L, p=p, t_la=t_la, k1=k1, k2=k2, k3=k3, m=m, c=c, local_grid_size = local_grid_size, resolution=resolution)


    # generates a set of primitives of tensor size[N, res+1, res+1] normalized 0 to 1
    def create_primitives(self):
        primitives = torch.zeros((torch.numel(self.speeds), self.resolution[0] + 1, self.resolution[1] + 1))

        turn_mask = torch.abs(self.steering_angles) > 0.01
        straight_mask = torch.abs(self.steering_angles) <= 0.01

        straight_sig = self.get_sig(self.speeds[straight_mask], self.steering_angles[straight_mask], True)
        straight_a = self.get_a(self.speeds[straight_mask], self.steering_angles[straight_mask], True)

        turn_sig = self.get_sig(self.speeds[turn_mask], self.steering_angles[turn_mask], False)
        turn_a = self.get_a(self.speeds[turn_mask], self.steering_angles[turn_mask], False)

        if (torch.any(straight_mask)):
            primitives[straight_mask] = straight_a * torch.exp(-(self.y ** 2) / (2 * straight_sig ** 2))

        # handle turn case:
        if (torch.any(turn_mask)):
            R = self.get_R(self.speeds[turn_mask], self.steering_angles[turn_mask]).view(-1, 1, 1)
            primitives[turn_mask] = turn_a * torch.exp(
                -(torch.sqrt(self.x ** 2 + (self.y - R) ** 2) - torch.abs(R)) ** 2 / (2 * turn_sig ** 2))

        return primitives/torch.max(primitives)

    # see superclass documentation for all remaining methods


    def get_a(self, speed, steering, straight):
        a = super().get_a(speed, steering, self.x, self.y, self.arc_lengths_local[:len(speed)], straight)
        return a#/normalize


    def get_s(self, speed, steering, grid_x, grid_y, arc_length, straight):
        # this signature just matches the most general case, this particular implementation only requires class vars
        return super(MotionPrimitive, self).get_s(speed, steering, self.x, self.y, self.arc_lengths_local[:len(speed)], straight)

    def get_sig(self, speed, steering, straight):

        sig = super(MotionPrimitive, self).get_sig(speed, steering, self.x, self.y, self.arc_lengths_local[:len(speed)], straight)
        return sig

    def get_R(self, speed, steering):
        #only valid for steering[turn_mask]
        R = super(MotionPrimitive, self).get_R(speed, steering)
        return R

    def get_control_for(self, primitive_number):
        return self.speeds[primitive_number], self.steering_angles[primitive_number]

#quick test for checking primitives
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


        


