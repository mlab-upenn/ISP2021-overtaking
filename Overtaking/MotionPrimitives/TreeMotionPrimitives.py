import torch
import numpy as np
from matplotlib import pyplot as plt
from .MotionPrimitives import MotionPrimitiveSuper


# generates a set of primitives consisting of all possible control combinations from discrete lists of speeds and steering angles up to certain depth.
class TreeMotionPrimitive(MotionPrimitiveSuper):
    def __init__(self, speed_list, steering_list, depth=3, L=.33, p=.2, t_la=1, k1=.0, k2=.0, k3=.0, m=.1, c=.12, local_grid_size = 7, resolution=(40, 40)):
        # speed and steering : lists of speed and steering values
        # depth : depth of the primitive tree
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
        self.depth = depth

        #check if enumeration of specific primitives has already been done (from ExplicitPrimitiveSet)
        if(len(speed_list.shape) > 1 and speed_list.shape[1] > 1):
            speeds = speed_list
            steering_angles = steering_list
        else:
            speed_vals = torch.tensor(speed_list)
            steering_vals = torch.tensor(steering_list)

            speeds, steering_angles = torch.meshgrid(speed_vals, steering_vals)
            speeds = speeds.flatten()
            steering_angles = steering_angles.flatten()

        super().__init__( speeds, steering_angles, L=L, p=p, t_la=t_la, k1=k1, k2=k2, k3=k3, m=m, c=c, local_grid_size = local_grid_size, resolution=resolution)


    # generates set of primitives in a tree up to certain depth
    def create_primitives(self):
        xy_offset = torch.zeros((1, 2))
        theta_offset = torch.zeros((1))


        arc_length = torch.zeros((1))

        speeds = self.speeds
        steering_angles = self.steering_angles

        primitives = torch.zeros((1, self.resolution[0]+1, self.resolution[1]+1))

        for d in range(self.depth):
            primitives, speeds, steering_angles, xy_offset, theta_offset, arc_length = self.create_next_level_input(
                speeds, steering_angles, primitives, xy_offset, theta_offset, arc_length)

            if(d==0):
                speeds = self.speeds
                steering_angles = self.steering_angles

            new_primitives, xy_offset, theta_offset, arc_length = self.generate_primitives(speeds, steering_angles, xy_offset, theta_offset, arc_length)

            primitives = torch.maximum(new_primitives, primitives)



        return primitives

    # helper method for expanding appropriate tensors to accomadate new tree level
    def create_next_level_input(self, speed, steering, primitives, xy_offset, theta_offset, arc_length):
        primitives = torch.repeat_interleave(primitives, self.speeds.shape[0], dim=0)
        arc_length = torch.repeat_interleave(arc_length, self.speeds.shape[0], dim=0)
        theta_offset = torch.repeat_interleave(theta_offset, self.speeds.shape[0], dim=0)
        xy_offset = torch.repeat_interleave(xy_offset, self.speeds.shape[0], dim=0)

        steering = steering.repeat(self.speeds.shape[0])
        speed = speed.repeat(self.speeds.shape[0])

        return primitives, speed, steering, xy_offset, theta_offset, arc_length

    # helper method for generating a section of primitive at certain tree depth given offsets and distance travelled by that point in the primitive
    def generate_primitives(self, speed, steering, xy_offset, theta_offset, arc_length):

        primitives = torch.zeros((torch.numel(speed), self.resolution[0] + 1, self.resolution[1] + 1))
        turn_mask = torch.abs(steering) > 0.01
        straight_mask = torch.abs(steering) <= 0.01

        grid_x, grid_y = self.translate_xy(self.x, self.y, xy_offset, theta_offset)

        local_theta = torch.zeros((torch.numel(speed)))
        local_xy = torch.zeros((torch.numel(speed), 2))



        if (torch.any(straight_mask)):
            straight_sig = self.get_sig(speed[straight_mask], steering[straight_mask], grid_x[straight_mask], grid_y[straight_mask], arc_length[straight_mask], True)
            straight_a = self.get_a(speed[straight_mask], steering[straight_mask], grid_x[straight_mask], grid_y[straight_mask], arc_length[straight_mask], True)

            primitives[straight_mask] = straight_a * torch.exp(-(grid_y[straight_mask] ** 2) / (2 * straight_sig ** 2))
            local_xy[straight_mask,0] = speed[straight_mask]*self.t_la


        # handle turn case:
        if (torch.any(turn_mask)):
            turn_sig = self.get_sig(speed[turn_mask], steering[turn_mask], grid_x[turn_mask], grid_y[turn_mask], arc_length[turn_mask], False)
            turn_a = self.get_a(speed[turn_mask], steering[turn_mask], grid_x[turn_mask], grid_y[turn_mask], arc_length[turn_mask], False)
            R = self.get_R(speed[turn_mask], steering[turn_mask]).view(-1, 1, 1)

            primitives[turn_mask] = turn_a * torch.exp(
                -(torch.sqrt(grid_x[turn_mask] ** 2 + (grid_y[turn_mask] - R) ** 2) - torch.abs(R)) ** 2 / (2 * turn_sig ** 2))
            # print('TreeMotionprimitve turning risk comp in: ', t_a-t_b)

            R = R.flatten()
            turn_angle = speed[turn_mask]*self.t_la/R

            local_xy[turn_mask, 0] = torch.sin(turn_angle)*R

            local_xy[turn_mask, 1] = (1-torch.cos(turn_angle))*R

            local_theta[turn_mask] = turn_angle

        arc_length += speed*self.t_la
        new_xy_offset, new_theta_offset = self.calculate_new_distance(xy_offset, theta_offset, local_xy, local_theta)
        return primitives, new_xy_offset, new_theta_offset, arc_length


    # translates grid of x and y points by offsets for use in generating a primitive segment given initial offsets (like would be the case in tree depth > 1)
    def translate_xy(self, x, y, xy_offset, theta_offset):
        # x : 3d tensor of size [N, res+1, res+1] of x grid values
        # y : 3d tensor of size [N, res+1, res+1] of y grid values
        # xy_offset : tensor of size [N, 2] of the current x, y (respectively) offsets from 0,0 in local space
        # theta_offset : 1d tensor of the current angular offset

        # outputs:
        # translated_x : updated x grid values
        # translated_y : updated y grid values

        x = x.unsqueeze(0)
        y = y.unsqueeze(0)
        theta_offset = theta_offset.reshape(-1,1,1)
        shift_x = x - xy_offset[:,0].reshape(-1,1,1)
        shift_y = y - xy_offset[:,1].reshape(-1,1,1)
        translated_x = (torch.cos(-theta_offset)*shift_x - torch.sin(-theta_offset)*shift_y)
        translated_y = (torch.sin(-theta_offset)*shift_x + torch.cos(-theta_offset)*shift_y)
        return translated_x, translated_y
        # return (torch.cos(-theta_offset)*x - torch.sin(-theta_offset)*y)-xy_offset[:,0].reshape(-1,1,1), (torch.sin(-theta_offset)*x + torch.cos(-theta_offset)*y)-xy_offset[:,1].reshape(-1,1,1)

    # updates the distance and angular offset
    def calculate_new_distance(self, xy_offset, theta_offset, local_xy, local_theta):
        # xy_offset : tensor of size [N, 2] of the current x, y (respectively) offsets from 0,0 in local space
        # theta_offset : 1d tensor of the current angular offset
        # local_xy : 2d tensor of size [N, 2] of x, y offsets for single primitive segment
        # local_theta : 1d tensor for the angular offset for single primitve segment

        # outputs:
        # xy_offset : updated x and y offsets from 0, 0 in local space
        # theta_offset : updated theta_offset in local space


        xy_offset[:,0] = xy_offset[:,0]+torch.cos(theta_offset)*local_xy[:,0] - torch.sin(theta_offset)*local_xy[:,1]
        xy_offset[:,1] = xy_offset[:,1] + torch.sin(theta_offset)*local_xy[:,0] + torch.cos(theta_offset)*local_xy[:,1]

        theta_offset = theta_offset+ local_theta

        return xy_offset, theta_offset

    def get_control_for(self, primitive_number):
        return self.speeds[primitive_number//self.speeds.shape[0]**(self.depth-1)], self.steering_angles[primitive_number//self.speeds.shape[0]**(self.depth-1)]

if __name__ == '__main__':
    import time
    t_b = time.time()
    # MP = MotionPrimitive(list(torch.arange(2,4,.05)),list(torch.arange(-.5,.5,.05)) )
    MP = TreeMotionPrimitive([2.], [0.0,.2])

    test_offset = torch.tensor([[0.,0.]])
    test_theta_offset = torch.tensor([0.])
    # print(MP.translate_xy(MP.x, MP.y, test_offset, test_theta_offset))


    t_a = time.time()

    # print(t_a-t_b)

    # print(MP.primitives.shape)
    for i in range(min(MP.primitives.shape[0],20)):
        plt.figure()
        plt.imshow(np.array(MP.primitives[i]))
        plt.colorbar()

    plt.show()
