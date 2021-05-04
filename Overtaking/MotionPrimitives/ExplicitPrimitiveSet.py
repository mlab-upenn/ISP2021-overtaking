
import torch
import numpy as np
from matplotlib import pyplot as plt
from .TreeMotionPrimitives import TreeMotionPrimitive
# import TreeMotionPrimitives
import time

import cProfile


class ExplicitPrimitiveSet(TreeMotionPrimitive):
    def __init__(self, speed_list, steering_list, L=.33, p=.2, t_la=1, k1=.0, k2=.0, k3=.0, m=.1, c=.12, local_grid_size = 7, resolution=(40, 40)):
        # speed and steering should be expressed as a 2d tensor [set_size, primitive length]
        # depth : depth of the primitive tree
        #L : wheelbase of the car
        # p : not currently used
        # t_la : lookahead time
        # k1 : interior std scaling with steering input
        # k2 : exterior std scaling with steering input
        # k3 : exterior std scaling with speed
        # m : path length std scaling
        # c : width of the car

        # check that both lists have the same number of time steps
        assert speed_list.shape[1] == steering_list.shape[1]

        super().__init__( speed_list, steering_list, depth=speed_list.shape[1],  L=L, p=p, t_la=t_la, k1=k1, k2=k2, k3=k3, m=m, c=c, local_grid_size = local_grid_size, resolution=resolution)


    def create_primitives(self):
        xy_offset = torch.zeros((self.speeds.shape[0], 2))
        theta_offset = torch.zeros(self.speeds.shape[0])


        arc_length = torch.zeros(self.speeds.shape[0])

        speeds = self.speeds
        steering_angles = self.steering_angles

        primitives = torch.zeros((self.speeds.shape[0], self.resolution[0]+1, self.resolution[1]+1))

        for d in range(self.depth):
            t_b = time.time()



            new_primitives, xy_offset, theta_offset, arc_length = self.generate_primitives(speeds[:,d], steering_angles[:,d], xy_offset, theta_offset, arc_length)


            primitives = torch.maximum(new_primitives, primitives)

            t_a = time.time()
            print('explict set level creation in: ', t_a-t_b, primitives.shape, primitives.device)

        return primitives

    def get_control_for(self, primitive_number):
        return self.speeds[primitive_number,0], self.steering_angles[primitive_number,0]

if __name__ == '__main__':
    import time
    t_b = time.time()
    # MP = MotionPrimitive(list(torch.arange(2,4,.05)),list(torch.arange(-.5,.5,.05)) )
    speeds = torch.ones((10,6))*4

    steering_angles = torch.range(-.2, .2, .4/9).unsqueeze(1).repeat((1,6))
    steering_angles[:,3:] *= -1
    MP = ExplicitPrimitiveSet(torch.tensor(speeds, requires_grad=True), torch.tensor(steering_angles,requires_grad=True),t_la = .25, p=.1)


    t_a = time.time()

    # print(t_a-t_b)

    # print(MP.primitives.shape)
    for i in range(min(MP.primitives.shape[0],10)):
        plt.figure()
        plt.imshow(np.array(MP.primitives[i]))
        plt.colorbar()

    plt.show()